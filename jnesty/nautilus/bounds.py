"""JAX bounds mirroring :mod:`nautilus.bounds`.

The classes here follow the reference implementation closely:

* :class:`JaxEllipsoid` - minimum-volume enclosing ellipsoid.
* :class:`JaxUnitCubeEllipsoidMixture` - per-dimension mix of cube and ellipsoid.
* :class:`JaxUnion` - multi-ellipsoid decomposition with GMM splitting and
  low-density trimming.
* :class:`JaxNeuralBound` - ellipsoid outer bound plus neural emulator rejection.
* :class:`JaxNautilusBound` - union of non-overlapping neural bounds plus a
  separate outer bound used for proposals.

Structural decisions (how many bounds exist, when to split) stay in Python;
everything numeric runs through JAX kernels.  All randomness is threaded as an
explicit JAX PRNG key so that a run is reproducible from a single seed and can
be checkpointed.
"""

import itertools
from collections import OrderedDict

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp as scipy_logsumexp
from scipy.stats import rankdata

from ._jit import bucket_size, padded_eval, stack_eval
from .fitting import fit_gmm, fit_mvee, normalise_backend
from .geometry import (contains_unified_bounds, sample_unified_bounds,
                       unified_log_v, unified_sample)
from .neural import ReferenceEmulator, train_emulator
from .random import finish_key, take_key

__all__ = ['JaxEllipsoid', 'JaxNautilusBound', 'JaxNeuralBound', 'JaxUnion',
           'JaxUnitCube', 'JaxUnitCubeEllipsoidMixture', 'PhaseShift']


class PhaseShift:
    """Phase shift centring periodic dimensions (host implementation)."""

    @classmethod
    def compute(cls, points, periodic):
        """Compute the phase shift for the given periodic dimension indices."""
        shift = cls()
        shift.periodic = np.asarray(periodic, dtype=int)
        shift.centers = np.zeros(len(shift.periodic))
        points = np.asarray(points)
        for i, dim in enumerate(shift.periodic):
            x = np.sort(points[:, dim])
            dx = np.append(np.diff(x), x[0] - (x[-1] - 1))
            shift.centers[i] = (
                x[np.argmax(dx)] + np.amax(dx) / 2.0 + 0.5) % 1
        return shift

    def transform(self, points, inverse=False):
        """Apply (or undo) the phase shift."""
        points = jnp.atleast_2d(jnp.asarray(points))
        cols = []
        for i in range(points.shape[1]):
            if i in self.periodic:
                j = int(np.flatnonzero(self.periodic == i)[0])
                offset = -self.centers[j] + 0.5
                if inverse:
                    offset = -offset
                cols.append(jnp.mod(points[:, i] + offset, 1.0))
            else:
                cols.append(points[:, i])
        return jnp.stack(cols, axis=1)


class JaxUnitCube:
    """The unit hypercube bound."""

    @classmethod
    def compute(cls, n_dim, rng=None):
        """Create a unit-cube bound for ``n_dim`` dimensions."""
        bound = cls()
        bound.n_dim = int(n_dim)
        return bound

    def contains(self, points):
        """Whether each point lies in ``[0, 1)`` along every dimension."""
        points = jnp.atleast_2d(jnp.asarray(points))
        return padded_eval(
            'unit_cube_contains',
            lambda pts: jnp.all((pts >= 0.0) & (pts < 1.0), axis=-1), points)

    def sample(self, n_points=100, key=None):
        """Draw ``n_points`` uniformly from the unit cube."""
        key = take_key(self, key)
        return jax.random.uniform(key, (int(n_points), self.n_dim))

    @property
    def log_v(self):
        """Log volume of the unit cube."""
        return 0.0

    def reset(self, rng=None):
        """Reset sampling counters (no-op)."""

    def unified(self):
        """Return the unified low-level representation of the bound."""
        d = self.n_dim
        return (jnp.full((d,), 0.5),
                jnp.zeros((d, d)),
                jnp.zeros((d, d)),
                jnp.ones((d,), dtype=bool),
                d, 0.0)


class JaxEllipsoid:
    """Minimum-volume enclosing ellipsoid bound."""

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, rng=None, backend='numpy'):
        """Fit the bounding ellipsoid to ``points``."""
        points = jnp.asarray(points)
        if enlarge_per_dim < 1.0:
            raise ValueError(
                "The 'enlarge_per_dim' factor cannot be smaller than unity.")
        if not points.shape[0] > points.shape[1]:
            raise ValueError(
                'Number of points must be larger than number dimensions.')

        bound = cls()
        bound.n_dim = int(points.shape[1])
        c, a, a_inv = fit_mvee(points, backend=backend)
        enlarge = enlarge_per_dim ** 2.0
        bound.c = c
        bound.A = a / enlarge
        a_inv = a_inv * enlarge
        bound.A_inv = a_inv
        bound.backend = backend
        bound._log_v_host = None
        if backend != 'jax':
            # A few tens of rows at most: a device Cholesky plus inverse costs
            # more than the enclosing-ellipsoid fit that produced them.
            b_host = np.linalg.cholesky(np.asarray(a_inv))
            bound.B = jnp.asarray(b_host)
            bound.B_inv = jnp.asarray(np.linalg.inv(b_host))
            d = bound.n_dim
            bound._log_v_host = float(
                np.linalg.slogdet(b_host)[1] + d * np.log(2.0)
                + d * gammaln(1.5) - gammaln(d / 2.0 + 1.0))
        else:
            bound.B = jnp.linalg.cholesky(a_inv)
            bound.B_inv = jnp.linalg.inv(bound.B)
        return bound

    def transform(self, points, inverse=False):
        """Transform points to (or from) the ellipsoid frame."""
        points = jnp.asarray(points)
        if not inverse:
            return jnp.einsum('ij,...j->...i', self.B_inv, points - self.c)
        return jnp.einsum('ij,...j->...i', self.B, points) + self.c

    def contains(self, points):
        """Whether each point lies inside the ellipsoid."""
        points = jnp.atleast_2d(jnp.asarray(points))
        return padded_eval(
            'ellipsoid_contains',
            lambda c, b_inv, pts: jnp.sum(
                jnp.einsum('ij,nj->ni', b_inv, pts - c) ** 2, axis=-1) < 1.0,
            points, self.c, self.B_inv)

    def sample(self, n_points=100, key=None):
        """Draw ``n_points`` uniformly from the ellipsoid."""
        key = take_key(self, key)
        points = unified_sample(self.c, self.B, jnp.zeros(self.n_dim, bool),
                                self.n_dim, n_points, key)
        finish_key(self, jax.random.fold_in(key, 1))
        return points

    @property
    def log_v(self):
        """Log volume of the ellipsoid."""
        if getattr(self, '_log_v_host', None) is not None:
            return self._log_v_host
        return unified_log_v(self.B, self.n_dim)

    def reset(self, rng=None):
        """Reset sampling counters (no-op)."""

    def unified(self):
        """Return the unified low-level representation of the bound."""
        return (self.c, self.B, self.B_inv, jnp.zeros(self.n_dim, bool),
                self.n_dim, self.log_v)


class JaxUnitCubeEllipsoidMixture:
    """Per-dimension mixture of the unit cube and an ellipsoid."""

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, rng=None, backend='numpy',
                ellipsoid=None):
        """Fit the cube/ellipsoid mixture to ``points``.

        ``ellipsoid`` may supply an already fitted minimum-volume enclosing
        ellipsoid of exactly these ``points`` with the same
        ``enlarge_per_dim`` and ``backend``.  It is used as the initial
        candidate and is never modified, so a caller can share one fitted
        ellipsoid with another bound.
        """
        points = np.asarray(points)
        n_dim = points.shape[1]
        bound = cls()
        bound.n_dim = n_dim
        dim_cube = np.zeros(n_dim, dtype=bool)
        if ellipsoid is None:
            ellipsoid = JaxEllipsoid.compute(points, enlarge_per_dim,
                                             backend=backend)

        while np.sum(~dim_cube) > 1:
            c = np.asarray(ellipsoid.c)
            a_inv = np.asarray(jnp.linalg.inv(ellipsoid.A))

            log_v = np.zeros(int(np.sum(~dim_cube)))
            for i in range(int(np.sum(~dim_cube))):
                pts_proj = np.delete(points[:, ~dim_cube], i, axis=1)
                c_proj = np.delete(c, i)
                a_inv_proj = np.delete(np.delete(a_inv, i, axis=0), i, axis=1)
                a_proj = np.linalg.inv(a_inv_proj)
                scale = np.amax(np.einsum(
                    '...i,ij,...j', pts_proj - c_proj, a_proj,
                    pts_proj - c_proj))
                a_proj /= scale
                log_v[i] = np.linalg.slogdet(np.linalg.inv(a_proj))[1]

            dim = int(np.arange(n_dim)[~dim_cube][np.argmin(log_v)])
            dim_cube[dim] = True
            ellipsoid_proj = JaxEllipsoid.compute(
                points[:, ~dim_cube], enlarge_per_dim, backend=backend)
            if float(ellipsoid_proj.log_v) < float(ellipsoid.log_v):
                ellipsoid = ellipsoid_proj
            else:
                dim_cube[dim] = False
                break

        if float(ellipsoid.log_v) > 0:
            dim_cube = np.ones(n_dim, dtype=bool)
            ellipsoid = None
            tested = np.zeros(n_dim, dtype=bool)
            while not np.all(tested):
                for dim in np.arange(n_dim)[~tested]:
                    dim_cube[dim] = False
                    tested[dim] = True
                    ellipsoid_test = JaxEllipsoid.compute(
                        points[:, ~dim_cube], enlarge_per_dim, backend=backend)
                    current = 0.0 if ellipsoid is None else float(
                        ellipsoid.log_v)
                    if current > float(ellipsoid_test.log_v):
                        ellipsoid = ellipsoid_test
                        tested[dim_cube] = False
                    else:
                        dim_cube[dim] = True

        bound.dim_cube = jnp.asarray(dim_cube)
        bound.cube = (JaxUnitCube.compute(int(np.sum(dim_cube)))
                      if np.any(dim_cube) else None)
        bound.ellipsoid = ellipsoid
        return bound

    def _center(self):
        d = self.n_dim
        center = np.full(d, 0.5)
        if self.ellipsoid is not None:
            idx = np.arange(d)[np.asarray(~self.dim_cube)]
            center[idx] = np.asarray(self.ellipsoid.c)
        return jnp.asarray(center)

    def _transform_matrix(self):
        d = self.n_dim
        t = 2.0 * np.diag(np.asarray(self.dim_cube, dtype=float))
        if self.ellipsoid is not None:
            idx = np.arange(d)[np.asarray(~self.dim_cube)]
            t[np.ix_(idx, idx)] = np.asarray(self.ellipsoid.B_inv)
        return jnp.asarray(t)

    def transform(self, points):
        """Transform points into the frame of the mixture."""
        points = jnp.atleast_2d(jnp.asarray(points))
        return jnp.einsum('ij,...j->...i', self._transform_matrix(),
                          points - self._center())

    def contains(self, points):
        """Whether each point lies in the cube part and ellipsoid part."""
        points = jnp.atleast_2d(jnp.asarray(points))
        center, _, b_inv, dim_cube, _, _ = self.unified()

        def program(center, b_inv, dim_cube, pts):
            z = jnp.einsum('ij,nj->ni', b_inv, pts - center)
            z = jnp.where(dim_cube[None, :], 0.0, z)
            in_ellipsoid = jnp.sum(z ** 2, axis=-1) < 1.0
            in_cube = jnp.all(
                jnp.where(dim_cube[None, :], (pts >= 0.0) & (pts < 1.0), True),
                axis=-1)
            return in_cube & in_ellipsoid

        return padded_eval('mixture_contains', program, points, center, b_inv,
                           dim_cube)

    def sample(self, n_points=100, key=None):
        """Draw ``n_points`` uniformly from the mixture."""
        key = take_key(self, key)
        center, b, _, dim_cube, n_ell, _ = self.unified()
        points = unified_sample(center, b, dim_cube, n_ell, n_points, key)
        finish_key(self, jax.random.fold_in(key, 1))
        return points

    @property
    def log_v(self):
        """Log volume of the mixture (zero if all dimensions use the cube)."""
        if self.ellipsoid is None:
            return 0.0
        return self.ellipsoid.log_v

    def reset(self, rng=None):
        """Reset sampling counters (no-op)."""

    def unified(self):
        """Return the unified low-level representation of the bound."""
        d = self.n_dim
        b = np.zeros((d, d))
        b_inv = np.zeros((d, d))
        if self.ellipsoid is not None:
            idx = np.arange(d)[np.asarray(~self.dim_cube)]
            b[np.ix_(idx, idx)] = np.asarray(self.ellipsoid.B)
            b_inv[np.ix_(idx, idx)] = np.asarray(self.ellipsoid.B_inv)
        n_ell = int(np.sum(~np.asarray(self.dim_cube)))
        return (self._center(), jnp.asarray(b), jnp.asarray(b_inv),
                jnp.asarray(self.dim_cube), n_ell, float(self.log_v))


def _ellipsoids_overlap(ellipsoids):
    """Whether any pair of ellipsoids overlaps (reference algorithm)."""
    c = [np.asarray(e.c) for e in ellipsoids]
    a_inv = [np.asarray(jnp.linalg.inv(e.A)) for e in ellipsoids]

    for i_1, i_2 in itertools.combinations(range(len(c)), 2):
        d = c[i_1] - c[i_2]

        def k(s):
            m = np.linalg.inv(a_inv[i_1] / (1 - s) + a_inv[i_2] / s)
            return 1 - np.dot(np.dot(d, m), d)

        if minimize(k, 0.5, bounds=[(1e-9, 1 - 1e-9)]).fun > 0:
            return True
    return False


class JaxUnion:
    """Union of multiple ellipsoids or cube/ellipsoid mixtures."""

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, n_points_min=None,
                unit=True, bound_class=JaxEllipsoid, rng=None,
                backend='numpy', key=None, initial_ellipsoid=None):
        """Create a union with a single bound fitted to ``points``.

        ``initial_ellipsoid`` is forwarded to a ``JaxUnitCubeEllipsoidMixture``
        ``bound_class`` as its starting ellipsoid; it must have been fitted to
        exactly these ``points`` with the same enlargement and backend.  The
        object is not modified.
        """
        points = jnp.asarray(points)
        bound = cls()
        bound.n_dim = int(points.shape[1])
        bound.enlarge_per_dim = enlarge_per_dim

        if n_points_min is None:
            n_points_min = bound.n_dim + 1
        if n_points_min < bound.n_dim + 1:
            raise ValueError('The number of points per bound must be '
                             'larger than the number of dimensions.')
        bound.n_points_min = int(n_points_min)
        bound.cube = JaxUnitCube.compute(bound.n_dim) if unit else None
        bound.bound_class = bound_class
        bound.backend = normalise_backend(backend)

        bound.points_bounds = [points]
        first_kwargs = {}
        if initial_ellipsoid is not None:
            first_kwargs['ellipsoid'] = initial_ellipsoid
        first = bound_class.compute(points, enlarge_per_dim=enlarge_per_dim,
                                    backend=bound.backend, **first_kwargs)
        bound.bounds = [first]
        bound.log_v_all = np.array([float(first.log_v)])
        bound.block = np.atleast_1d(len(points) < 2 * bound.n_points_min)

        bound.points = np.zeros((0, bound.n_dim))
        bound.n_sample = 0
        bound.n_reject = 0
        bound._sample_key = key
        bound._invalidate()
        return bound

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _invalidate(self):
        self._stack = None
        # Strong-reference mutation token: replacing it is what tells the
        # multi-bound pack cache that this union's geometry changed.
        self._geometry_token = object()

    def _stacks(self):
        """Stacked device arrays for sampling and containment checks."""
        if self._stack is None:
            unified = [b.unified() for b in self.bounds]
            self._stack = (
                jnp.stack([u[0] for u in unified]),
                jnp.stack([u[1] for u in unified]),
                jnp.stack([u[2] for u in unified]),
                jnp.stack([u[3] for u in unified]),
                jnp.asarray([u[4] for u in unified]),
                jnp.asarray([float(u[5]) for u in unified]),
            )
            self._geometry_token = object()
        return self._stack

    def split(self, allow_overlap=True, key=None):
        """Split the largest bound; returns whether a split happened."""
        if not allow_overlap and not isinstance(self.bounds[0], JaxEllipsoid):
            raise ValueError("'allow_overlap' can only be False if "
                             "bounds are ellipsoids.")
        key = take_key(self, key)
        if not np.any(~self.block):
            return False

        index = int(np.argmax(np.where(~self.block, self.log_v_all, -np.inf)))
        points = np.asarray(
            self.bounds[index].transform(self.points_bounds[index]))

        if key is None:
            key = jax.random.PRNGKey(0)
        weights, means, covs, log_p = fit_gmm(
            np.asarray(points), key, backend=getattr(self, 'backend', 'numpy'))
        labels = np.argmax(log_p, axis=1)

        counts = np.bincount(labels, minlength=2)
        if not np.all(counts >= self.n_points_min):
            label = int(np.argmin(counts))
            order = np.argsort(-log_p[:, label])[:self.n_points_min]
            labels[order] = label

        points = np.asarray(self.points_bounds[index])
        new_bounds = [
            self.bound_class.compute(points[labels == label],
                                     enlarge_per_dim=self.enlarge_per_dim,
                                     backend=getattr(self, 'backend', 'numpy'))
            for label in (0, 1)]

        if not allow_overlap and _ellipsoids_overlap(
                self.bounds[:index] + self.bounds[index + 1:] + new_bounds):
            return False

        if (scipy_logsumexp([float(new_bounds[0].log_v),
                             float(new_bounds[1].log_v)])
                > float(self.bounds[index].log_v)):
            self.block[index] = True
            return self.split(allow_overlap=allow_overlap, key=key)

        self.points_bounds.pop(index)
        self.points_bounds.append(points[labels == 0])
        self.points_bounds.append(points[labels == 1])
        self.bounds.pop(index)
        self.bounds = self.bounds + new_bounds
        self.log_v_all = np.array([float(b.log_v) for b in self.bounds])
        block = np.delete(self.block, index)
        self.block = np.concatenate(
            (block,
             [len(self.points_bounds[-2]) < 2 * self.n_points_min,
              len(self.points_bounds[-1]) < 2 * self.n_points_min]))
        self.reset()
        return True

    def trim(self, threshold=1e3):
        """Drop the lowest-density bound if it is an outlier."""
        if len(self.bounds) == 1:
            return False
        log_n = np.array([np.log(len(p)) for p in self.points_bounds])
        log_v = np.array([float(b.log_v) for b in self.bounds])
        log_r = log_n - log_v
        index = int(np.argmin(log_r))
        if log_r[index] - np.median(np.delete(log_r, index)) < -np.log(
                threshold):
            self.points_bounds.pop(index)
            self.bounds.pop(index)
            self.log_v_all = np.array([float(b.log_v) for b in self.bounds])
            self.reset()
            return True
        return False

    def contains(self, points):
        """Whether each point lies in any bound (and in the cube)."""
        points = jnp.atleast_2d(jnp.asarray(points))
        center, _, b_inv, dim_cube, _, _ = self._stacks()
        in_bound = padded_eval(
            'union_contains',
            lambda center, b_inv, dim_cube, pts: jnp.any(
                contains_unified_bounds(b_inv, center, dim_cube, pts), axis=0),
            points, center, b_inv, dim_cube)
        if self.cube is not None:
            in_bound = in_bound & self.cube.contains(points)
        return in_bound

    def sample(self, n_points=100, key=None):
        """Draw ``n_points`` uniformly from the union."""
        key = take_key(self, key)
        n_points = int(n_points)
        center, b, b_inv, dim_cube, n_ell, log_v = self._stacks()

        while self.points.shape[0] < n_points:
            key, sub = jax.random.split(key)
            n_draw = 1000
            kernel = (self._draw_chunk if self.cube is not None
                      else self._draw_chunk_without_cube)
            points, keep = stack_eval(
                'union_sample' if self.cube is not None else 'union_sample_free',
                kernel, b, center, b_inv, dim_cube, n_ell, log_v, sub)
            # Compact on the host: device-side compaction has a data-dependent
            # output shape, and JAX recompiles per distinct shape (~20 ms on
            # this box), which dwarfs this 1000 x n_dim transfer.
            points = np.asarray(points)[np.asarray(keep)]

            self.points = np.concatenate(
                [self._points_np(), points], axis=0)
            self.n_sample += n_draw
            self.n_reject += n_draw - len(points)

        points = self.points[:n_points]
        self.points = self.points[n_points:]
        finish_key(self, key)
        return points

    @staticmethod
    def _draw_chunk(b, center, b_inv, dim_cube, n_ell, log_v, key, n_draw=1000):
        """Draw one chunk of union proposals and mark the accepted rows.

        Boolean masking would produce a data-dependent shape, which cannot be
        compiled; the kernel therefore returns every proposal together with a
        keep mask and the host does the compaction.
        """
        k_count, k_draw, k_perm, k_keep = jax.random.split(key, 4)
        # ``jax.random.multinomial`` is unexpectedly slow (it simulates the
        # trials one by one); drawing the component of every point with
        # ``categorical`` and counting is equivalent and stays on device.
        assignment = jax.random.categorical(k_count, log_v, shape=(n_draw,))
        counts = jnp.bincount(assignment, length=b.shape[0])
        points, _ = sample_unified_bounds(b, center, dim_cube, n_ell, counts,
                                          k_draw, n_total=n_draw)
        points = jax.random.permutation(k_perm, points)
        # Both masks must refer to the same, permuted rows.  Computing the
        # cube mask before the shuffle and applying it afterwards returns a
        # random subsample, including points outside the prior box.
        in_cube = jnp.all((points >= 0.0) & (points < 1.0), axis=-1)
        multiplicity = jnp.sum(
            contains_unified_bounds(b_inv, center, dim_cube, points), axis=0)
        prob_keep = 1.0 - 1.0 / jnp.maximum(multiplicity, 1)
        keep = jax.random.uniform(k_keep, (n_draw,)) > prob_keep
        return points, keep & in_cube

    @staticmethod
    def _draw_chunk_without_cube(b, center, b_inv, dim_cube, n_ell, log_v, key,
                                 n_draw=1000):
        """Draw a chunk when the union is not restricted to the unit cube."""
        k_count, k_draw, k_perm, k_keep = jax.random.split(key, 4)
        assignment = jax.random.categorical(k_count, log_v, shape=(n_draw,))
        counts = jnp.bincount(assignment, length=b.shape[0])
        points, _ = sample_unified_bounds(b, center, dim_cube, n_ell, counts,
                                          k_draw, n_total=n_draw)
        points = jax.random.permutation(k_perm, points)
        multiplicity = jnp.sum(
            contains_unified_bounds(b_inv, center, dim_cube, points), axis=0)
        prob_keep = 1.0 - 1.0 / jnp.maximum(multiplicity, 1)
        keep = jax.random.uniform(k_keep, (n_draw,)) > prob_keep
        return points, keep

    @property
    def log_v(self):
        """Log volume of the union with the Monte Carlo correction."""
        if self.n_sample == 0:
            self.sample(100)
        return (scipy_logsumexp(np.asarray(self.log_v_all))
                + np.log(1.0 - self.n_reject / self.n_sample))

    def _points_np(self):
        """Buffered proposals as a NumPy array (cheap host-side appends)."""
        return (np.zeros((0, self.n_dim)) if self.points is None
                else np.asarray(self.points))

    def reset(self, rng=None):
        """Drop cached proposals and reset sampling counters."""
        self.points = np.zeros((0, self.n_dim))
        self.n_sample = 0
        self.n_reject = 0
        self._invalidate()


class JaxNeuralBound:
    """Ellipsoid outer bound plus a neural emulator rejection step."""

    @classmethod
    def compute(cls, points, log_l, log_l_min, enlarge_per_dim=1.1,
                n_networks=4, neural_network_kwargs=None, key=None,
                backend='numpy', neural_backend='jax', outer_bound=None):
        """Fit the neural bound to points and their likelihoods.

        ``outer_bound`` is an internal fast path: an already fitted
        :class:`JaxEllipsoid` for exactly ``points[log_l >= log_l_min]`` whose
        ``contains`` mask selected ``points``.  Under that proven precondition
        the inner containment test is all True and is skipped.  The supplied
        ellipsoid is used as-is and is never modified.
        """
        points = jnp.asarray(points)
        log_l_np = np.asarray(log_l)
        bound = cls()
        bound.n_dim = int(points.shape[1])
        if key is None:
            key = jax.random.PRNGKey(0)

        if outer_bound is None:
            bound.outer_bound = JaxEllipsoid.compute(
                points[log_l_np >= float(log_l_min)],
                enlarge_per_dim=enlarge_per_dim, backend=backend)
        else:
            bound.outer_bound = outer_bound

        if n_networks == 0:
            bound.emulator = None
            bound.score_predict_min = 0.0
            return bound

        neural_network_kwargs = dict(neural_network_kwargs or {})
        if outer_bound is None:
            select = np.array(bound.outer_bound.contains(points), dtype=bool)
        else:
            # The caller selected these points with this exact ellipsoid, so
            # the containment mask cannot reject any of them.
            select = np.ones(int(points.shape[0]), dtype=bool)
        points_t = jnp.asarray(
            bound.outer_bound.transform(points[select]))
        log_l_sel = log_l_np[select]

        score = np.zeros(len(log_l_sel))
        above = log_l_sel >= float(log_l_min)
        if np.any(above):
            score[above] = 0.5 * (1 + (rankdata(log_l_sel[above]) - 0.5)
                                  / np.sum(above))
        if np.any(~above):
            score[~above] = 0.5 * ((rankdata(log_l_sel[~above]) - 0.5)
                                   / np.sum(~above))

        key, sub = jax.random.split(key)
        if neural_backend == 'reference':
            bound.emulator = ReferenceEmulator.train(
                np.asarray(points_t), score, n_networks=n_networks,
                neural_network_kwargs=neural_network_kwargs)
        else:
            bound.emulator = train_emulator(
                points_t, jnp.asarray(score), sub,
                n_networks=n_networks, **neural_network_kwargs)
        bound.neural_backend = neural_backend

        predictions = np.asarray(bound.emulator.predict(points_t))
        coeffs = np.polyfit(score, predictions, 3)
        anchor = np.amin(score[above]) if np.any(above) else np.amin(score)
        bound.score_predict_min = float(np.polyval(coeffs, anchor))
        return bound

    def contains(self, points):
        """Whether each point is inside the ellipsoid and above the score cut."""
        points = jnp.atleast_2d(jnp.asarray(points))
        if self.emulator is None:
            return self.outer_bound.contains(points)

        if getattr(self, 'neural_backend', 'jax') == 'reference':
            # The reference emulator predicts on the host; keep the ellipsoid
            # test on device and only ship the surviving points across.
            in_ellipsoid = np.array(self.outer_bound.contains(points), dtype=bool)
            idx = np.flatnonzero(in_ellipsoid)
            if len(idx):
                transformed = np.asarray(self.outer_bound.transform(points[idx]))
                keep = np.asarray(self.emulator.predict(transformed)
                                  > self.score_predict_min - 1e-9, dtype=bool)
                in_ellipsoid[idx[~keep]] = False
            return jnp.asarray(in_ellipsoid)

        return padded_eval(
            'neural_contains',
            _neural_contains_kernel,
            points, self.outer_bound.c, self.outer_bound.B_inv,
            self.emulator.params, self.emulator.mean, self.emulator.scale,
            float(self.score_predict_min))

    @property
    def n_networks(self):
        """Number of networks in the emulator."""
        return 0 if self.emulator is None else self.emulator.n_networks


def _neural_contains_kernel(c, b_inv, params, mean, scale, threshold, points):
    """Ellipsoid containment plus ensemble rejection, one compiled program."""
    z = jnp.einsum('ij,nj->ni', b_inv, points - c)
    in_ellipsoid = jnp.sum(z ** 2, axis=-1) < 1.0
    h = (z - mean) / scale
    n_networks = params[0][0].shape[0]
    h = jnp.broadcast_to(h, (n_networks,) + h.shape)
    n_layers = len(params)
    for i, (w, b) in enumerate(params):
        h = jnp.einsum('nio,npi->npo', w, h) + b[:, None, :]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    prediction = jnp.mean(h[:, :, 0], axis=0)
    return in_ellipsoid & (prediction > threshold - 1e-9)


def _neural_contains_stacked(c_all, b_inv_all, params_all, mean_all, scale_all,
                             threshold_all, points):
    """OR of :func:`_neural_contains_kernel` over a batch of neural bounds.

    Every array carries an extra leading axis of length ``n_bounds``: all
    neural bounds of one :class:`JaxNautilusBound` are trained with identical
    shapes, so they stack without padding.  This is one compiled program and
    one dispatch for the whole OR-chain (the loop version uploaded the points
    once per bound as well).
    """
    z = jnp.einsum('bij,bnj->bni', b_inv_all, points[None] - c_all[:, None, :])
    in_ellipsoid = jnp.sum(z ** 2, axis=-1) < 1.0
    h = (z - mean_all[:, None, :]) / scale_all[:, None, :]
    n_networks = params_all[0][0].shape[1]
    # ``h`` is (n_bounds, n_points, n_dim); add the ensemble axis explicitly
    # (broadcasting alone would align the bound axis with the network axis).
    h = jnp.broadcast_to(h[:, None], (h.shape[0], n_networks) + h.shape[1:])
    n_layers = len(params_all)
    for i, (w, b) in enumerate(params_all):
        h = jnp.einsum('bnio,bnpi->bnpo', w, h) + b[:, :, None, :]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    prediction = jnp.mean(h[:, :, :, 0], axis=1)
    inside = in_ellipsoid & (prediction > threshold_all[:, None] - 1e-9)
    return jnp.any(inside, axis=0)


def _multi_contains_kernel(u_center, u_b_inv, u_dim_cube,
                           n_c, n_b_inv, params, n_mean, n_scale,
                           n_threshold, has_neural, points):
    """Per-bound masks for many nautilus bounds in one compiled program.

    Every bound occupies a fixed slot count: ``u_*`` carry
    ``(B, E, ...)`` union rows and ``n_*`` carry ``(B, K, ...)`` neural
    bounds, with inactive slots padded so that they can never contain a
    point (NaN centres, +inf thresholds).  ``B`` is a compile-time constant
    chosen once per run, which is what keeps the number of compiled
    programs at a handful instead of one per distinct shape combination.
    Returns ``(n_points, B)``: column ``i`` is the full membership test
    (outer union AND unit cube AND neural OR-chain) of bound ``i``; padded
    columns are all False.  The dtype arithmetic mirrors
    :func:`_neural_contains_stacked` and
    :func:`~jnesty.nautilus.geometry.contains_unified_bounds` exactly.
    """
    in_cube = jnp.all((points >= 0.0) & (points < 1.0), axis=-1)

    # Outer union rows: (B, E, n_points).
    delta = points[None, None, :, :] - u_center[:, :, None, :]
    z = jnp.einsum('beij,benj->beni', u_b_inv, delta)
    z = jnp.where(u_dim_cube[:, :, None, :], 0.0, z)
    in_ellipsoid = jnp.sum(z ** 2, axis=-1) < 1.0
    in_cube_rows = jnp.all(
        jnp.where(u_dim_cube[:, :, None, :],
                  (points[None, None, :, :] >= 0.0)
                  & (points[None, None, :, :] < 1.0),
                  True), axis=-1)
    u_ok = jnp.any(in_ellipsoid & in_cube_rows, axis=1)  # (B, n_points)

    # Neural bounds: (B, K, n_points).
    z2 = jnp.einsum('bkij,bknj->bkni', n_b_inv,
                    points[None, None, :, :] - n_c[:, :, None, :])
    in_ell2 = jnp.sum(z2 ** 2, axis=-1) < 1.0
    h = (z2 - n_mean[:, :, None, :]) / n_scale[:, :, None, :]
    n_networks = params[0][0].shape[2]
    h = jnp.broadcast_to(
        h[:, :, None], h.shape[:2] + (n_networks,) + h.shape[2:])
    n_layers = len(params)
    for i, (w, b) in enumerate(params):
        h = jnp.einsum('bknio,bknpi->bknpo', w, h) + b[:, :, :, None, :]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    prediction = jnp.mean(h[:, :, :, :, 0], axis=2)  # (B, K, n_points)
    n_ok = in_ell2 & (prediction > n_threshold[:, :, None] - 1e-9)
    n_any = jnp.any(n_ok, axis=1)  # (B, n_points)
    # A bound without neural bounds is its outer union alone.
    neural_ok = jnp.where(has_neural[:, None], n_any, True)

    masks = (u_ok & neural_ok) & in_cube[None, :]
    return jnp.transpose(masks)  # (n_points, B)


# Fixed width of the bound axis in the compiled program.  Bounds beyond it
# are processed in chunks, which keeps the shapes (and thus the compiled
# programs) constant for an entire run.
_MULTI_BOUND_WIDTH = 128


def _multi_bound_contains(bounds, points):
    """Per-bound containment masks for many nautilus bounds, few dispatches.

    Parameters
    ----------
    bounds : sequence of bound objects
        The bounds to test, in order.  Their individual results are NOT
        OR-combined; the caller receives one mask column per bound.
    points : array (n_points, n_dim)
        Unit-cube points, exactly what would be passed to each bound's
        ``contains``.

    Returns
    -------
    masks : array (n_points, len(bounds))
        Column ``i`` equals ``bounds[i].contains(points)``.  All stackable
        bounds run in one compiled program with fixed slot shapes; the
        sampler's shell-exclusion and association paths use this instead of
        a per-bound dispatch loop.
    """
    n_bounds = len(bounds)
    points = jnp.atleast_2d(jnp.asarray(points))
    n_points = points.shape[0]
    if n_bounds == 0 or n_points == 0:
        return jnp.zeros((n_points, n_bounds), dtype=bool)

    # The sampler's bound list starts with a JaxUnitCube; later bounds are
    # JaxNautilusBounds.  Non-nautilus and phase-shifted bounds keep their
    # own (cheap) evaluation and are merged back column-wise below.
    columns = [None] * n_bounds
    nautilus = []
    for i, b in enumerate(bounds):
        if isinstance(b, JaxNautilusBound) and b.shift is None:
            nautilus.append((i, b))
        else:
            columns[i] = np.asarray(
                jnp.atleast_1d(jnp.asarray(b.contains(points))))

    if not nautilus:
        return jnp.asarray(np.stack(columns, axis=-1))

    # The reference emulator predicts on the host; fall back there.
    stackable = all(
        b._stacked_neural_tensors() is not None
        or len(b.neural_bounds) == 0 for _, b in nautilus)
    if not stackable:
        for i, b in nautilus:
            columns[i] = np.asarray(
                jnp.atleast_1d(jnp.asarray(b.contains(points))))
        return jnp.asarray(np.stack(columns, axis=-1))

    width = _MULTI_BOUND_WIDTH
    all_masks = []
    for start in range(0, len(nautilus), width):
        chunk = nautilus[start:start + width]
        all_masks.append(_multi_bound_chunk(chunk, points))
    stacked = (all_masks[0] if len(all_masks) == 1
               else jnp.concatenate(all_masks, axis=1))

    if len(nautilus) == n_bounds:
        return stacked
    stacked_np = np.asarray(stacked)
    for new_i, (old_i, _) in enumerate(nautilus):
        columns[old_i] = stacked_np[:, new_i]
    return np.stack(columns, axis=-1)


# Packed kernel arguments are immutable once a bound has been computed, so a
# small LRU keyed by the actual bound objects removes the per-call
# device -> host -> device round trip.  Bound objects are hashable by identity
# and the cache keeps strong references to them, so identities cannot be
# recycled while a live entry refers to them.  The point-dependent masks are
# never cached: only the padded, point-independent arguments are.
_MULTI_BOUND_PACK_CACHE_SIZE = 4
_MULTI_BOUND_PACK_CACHE = OrderedDict()
_MULTI_BOUND_PACK_STATS = {'builds': 0, 'hits': 0, 'misses': 0}


def _multi_bound_slots(nautilus):
    """Power-of-two slot counts for one chunk of (index, bound) pairs."""
    e_slot = bucket_size(max(
        (b.outer_bound._stacks()[0].shape[0] for _, b in nautilus),
        default=1), minimum=1)
    k_slot = bucket_size(max(
        (len(b.neural_bounds) for _, b in nautilus), default=1), minimum=1)
    return e_slot, k_slot


def _multi_bound_pack_key(nautilus, e_slot, k_slot, n_dim):
    """Identity and mutation key for the packed arguments of one bound chunk.

    The bound objects themselves form the key (not their ``id`` integers), and
    the slot counts and dimensionality keep a change in padded shapes from
    reusing an incompatible pack.  Each bound contributes the strong-reference
    mutation tokens of its outer geometry and neural tensors: ``split``,
    ``trim`` or an emulator update replaces one of them, so a pack built from
    the old state can never be reused.
    """
    tokens = []
    for _, b in nautilus:
        b.outer_bound._stacks()      # ensures the geometry token exists
        b._stacked_neural_tensors()  # ensures the neural token exists
        tokens.append((getattr(b.outer_bound, '_geometry_token', None),
                       getattr(b, '_neural_token', None)))
    return (tuple(b for _, b in nautilus), tuple(tokens),
            int(e_slot), int(k_slot), int(n_dim))


def _build_multi_bound_pack(nautilus, e_slot, k_slot, n_dim):
    """Pad the immutable bound state into the fixed-width kernel arguments.

    Returns the arguments of :func:`_multi_contains_kernel` in its exact
    parameter order, with every array already resident on the device.
    """
    n_real = len(nautilus)
    width = _MULTI_BOUND_WIDTH
    d = int(n_dim)

    u_center = np.full((width, e_slot, d), np.nan)
    u_b_inv = np.zeros((width, e_slot, d, d))
    u_dim_cube = np.zeros((width, e_slot, d), dtype=bool)
    n_c = np.zeros((width, k_slot, d))
    n_b_inv = np.zeros((width, k_slot, d, d))
    n_mean = np.zeros((width, k_slot, d))
    n_scale = np.ones((width, k_slot, d))
    n_threshold = np.full((width, k_slot), np.inf)
    has_neural = np.zeros(width, dtype=bool)

    # Collect one stacked-params template; all neural bounds share shapes.
    params_template = None
    slot_params = [None] * width

    for slot, (_, b) in enumerate(nautilus):
        center, _, b_inv, dim_cube, _, _ = b.outer_bound._stacks()
        e_real = center.shape[0]
        u_center[slot, :e_real] = np.asarray(center)
        u_b_inv[slot, :e_real] = np.asarray(b_inv)
        u_dim_cube[slot, :e_real] = np.asarray(dim_cube)

        tensors = b._stacked_neural_tensors()
        if tensors is None:
            continue
        has_neural[slot] = True
        (c_all, b_inv_all, params_all, mean_all, scale_all,
         threshold_all) = tensors
        k_real = c_all.shape[0]
        n_c[slot, :k_real] = np.asarray(c_all)
        n_b_inv[slot, :k_real] = np.asarray(b_inv_all)
        n_mean[slot, :k_real] = np.asarray(mean_all)
        n_scale[slot, :k_real] = np.asarray(scale_all)
        n_threshold[slot, :k_real] = np.asarray(threshold_all)
        slot_params[slot] = params_all
        if params_template is None:
            params_template = jax.tree_util.tree_map(
                lambda leaf: np.zeros(
                    (width, k_slot) + tuple(leaf.shape[1:]),
                    dtype=np.asarray(leaf).dtype),
                params_all)

    if params_template is None:
        # No neural bounds at all: keep the kernel structurally valid with
        # network stacks that ``has_neural`` (all False) bypasses.
        params_template = [(np.zeros((width, k_slot, 1, d, 1)),
                            np.zeros((width, k_slot, 1, 1)))]

    # Assemble the parameter pytree: (width, k_slot, n_networks, ...).
    params = []
    for layer in range(len(params_template)):
        w_template, b_template = params_template[layer]
        w = np.zeros(w_template.shape, dtype=w_template.dtype)
        bias = np.zeros(b_template.shape, dtype=b_template.dtype)
        for slot in range(n_real):
            stacked = slot_params[slot]
            if stacked is None:
                continue
            w[slot, :stacked[layer][0].shape[0]] = np.asarray(
                stacked[layer][0])
            bias[slot, :stacked[layer][1].shape[0]] = np.asarray(
                stacked[layer][1])
        params.append((jnp.asarray(w), jnp.asarray(bias)))

    return (jnp.asarray(u_center), jnp.asarray(u_b_inv),
            jnp.asarray(u_dim_cube), jnp.asarray(n_c), jnp.asarray(n_b_inv),
            params, jnp.asarray(n_mean), jnp.asarray(n_scale),
            jnp.asarray(n_threshold), jnp.asarray(has_neural))


def _multi_bound_pack(nautilus, e_slot, k_slot, n_dim):
    """Return the cached fixed-slot arguments for this chunk, if present."""
    key = _multi_bound_pack_key(nautilus, e_slot, k_slot, n_dim)
    cached = _MULTI_BOUND_PACK_CACHE.get(key)
    if cached is not None:
        _MULTI_BOUND_PACK_CACHE.move_to_end(key)
        _MULTI_BOUND_PACK_STATS['hits'] += 1
        return cached

    _MULTI_BOUND_PACK_STATS['misses'] += 1
    packed = _build_multi_bound_pack(nautilus, e_slot, k_slot, n_dim)
    _MULTI_BOUND_PACK_STATS['builds'] += 1
    _MULTI_BOUND_PACK_CACHE[key] = packed
    while len(_MULTI_BOUND_PACK_CACHE) > _MULTI_BOUND_PACK_CACHE_SIZE:
        _MULTI_BOUND_PACK_CACHE.popitem(last=False)
    return packed


def _clear_multi_bound_pack_cache():
    """Drop cached packed arguments (tests and diagnostics only)."""
    _MULTI_BOUND_PACK_CACHE.clear()
    for name in _MULTI_BOUND_PACK_STATS:
        _MULTI_BOUND_PACK_STATS[name] = 0


def _multi_bound_chunk(nautilus, points):
    """Run the fixed-slot kernel on one chunk of (index, bound) pairs.

    Packing (padding, host copies, device uploads) is cached by bound identity;
    only the point-dependent kernel evaluation runs per call.
    """
    e_slot, k_slot = _multi_bound_slots(nautilus)
    packed = _multi_bound_pack(nautilus, e_slot, k_slot, points.shape[1])
    masks = padded_eval(
        'multi_contains', _multi_contains_kernel, points, *packed)
    return masks[:, :len(nautilus)]


def _neural_source_snapshot(neural_bounds):
    """Snapshot every tensor source behind the neural stack, by identity.

    The snapshot keeps strong references to the source arrays/objects, so their
    ``id`` values cannot be recycled while it is cached.  Comparison is done
    with ``is`` (never with ``==``, which would invoke elementwise JAX array
    equality); only the small scalar signature is compared by value.
    """
    refs = []
    scalars = []
    for b in neural_bounds:
        outer = getattr(b, 'outer_bound', None)
        emulator = getattr(b, 'emulator', None)
        params = getattr(emulator, 'params', None)
        c = getattr(outer, 'c', None)
        b_inv = getattr(outer, 'B_inv', None)
        mean = getattr(emulator, 'mean', None)
        scale = getattr(emulator, 'scale', None)
        refs.extend((b, outer, c, b_inv, emulator, mean, scale))
        if params is not None:
            for layer in params:
                refs.extend(layer)
        threshold = getattr(b, 'score_predict_min', None)
        scalars.append((
            getattr(b, 'neural_backend', 'jax'),
            None if threshold is None else repr(float(threshold)),
            None if params is None else tuple(len(layer) for layer in params),
        ))
    return (len(neural_bounds), tuple(refs), tuple(scalars))


def _neural_sources_match(cached, current):
    """Whether two source snapshots describe the exact same objects."""
    if cached is None or current is None:
        return False
    if cached[0] != current[0] or cached[2] != current[2]:
        return False
    cached_refs, current_refs = cached[1], current[1]
    return (len(cached_refs) == len(current_refs)
            and all(old is new for old, new in zip(cached_refs, current_refs)))


def _region_ellipsoid_reusable(points, log_l_np, log_l_min, select,
                               region_points):
    """Whether a neural region's outer MVEE is its existing union bound.

    The exact, host-side test mirrors the fit that
    :meth:`JaxNeuralBound.compute` would otherwise perform: the above-threshold
    rows of ``points[select]`` must equal the union bound's stored point set
    element for element (same order, values and dtype).  Any difference -- an
    extra point caught by the enlarged ellipsoid, a missing row, or a changed
    value -- falls back to the normal fit.
    """
    select_np = np.asarray(select, dtype=bool)
    selected = np.asarray(points[select])
    selected_log_l = np.asarray(log_l_np)[select_np]
    candidate = selected[selected_log_l >= float(log_l_min)]
    expected = np.asarray(region_points)
    return (candidate.shape == expected.shape
            and candidate.dtype == expected.dtype
            and np.array_equal(candidate, expected))


class JaxNautilusBound:
    """Union of non-overlapping neural bounds with a separate outer bound."""

    @classmethod
    def compute(cls, points, log_l, log_l_min, log_v_target,
                enlarge_per_dim=1.1, n_points_min=None, split_threshold=100,
                periodic=None, n_networks=4, neural_network_kwargs=None,
                key=None, rng=None, backend='numpy', neural_backend='jax'):
        """Build the neural bounds and the proposal bound."""
        backend = normalise_backend(backend)
        points = jnp.asarray(points)
        log_l_np = np.asarray(log_l)
        bound = cls()
        bound.n_dim = int(points.shape[1])
        if key is None:
            key = jax.random.PRNGKey(0)

        if periodic is not None:
            bound.shift = PhaseShift.compute(
                np.asarray(points)[log_l_np >= float(log_l_min)], periodic)
            points = jnp.asarray(bound.shift.transform(points))
        else:
            bound.shift = None

        bound.neural_bounds = []
        above = jnp.asarray(log_l_np >= float(log_l_min))

        multi_ellipsoid = JaxUnion.compute(
            points[above], enlarge_per_dim=enlarge_per_dim,
            n_points_min=n_points_min, bound_class=JaxEllipsoid,
            backend=backend, key=key)
        # Keep the pre-split, global MVEE for the outer mixture below.  The
        # object is immutable and is shared, never copied or mutated.
        initial_ellipsoid = multi_ellipsoid.bounds[0]
        while multi_ellipsoid.split(allow_overlap=False, key=key):
            pass

        for region_index, ellipsoid in enumerate(multi_ellipsoid.bounds):
            select = ellipsoid.contains(points)
            key, sub = jax.random.split(key)
            # Reusing the union's MVEE is valid only when the above-threshold
            # rows of this selection are exactly the rows it was fitted to.
            reuse = _region_ellipsoid_reusable(
                points, log_l_np, log_l_min, select,
                multi_ellipsoid.points_bounds[region_index])
            bound.neural_bounds.append(JaxNeuralBound.compute(
                points[select], log_l[select], log_l_min,
                enlarge_per_dim=enlarge_per_dim, n_networks=n_networks,
                neural_network_kwargs=neural_network_kwargs, key=sub,
                backend=backend, neural_backend=neural_backend,
                outer_bound=ellipsoid if reuse else None))

        bound.outer_bound = JaxUnion.compute(
            points[above], enlarge_per_dim=enlarge_per_dim,
            n_points_min=n_points_min,
            bound_class=JaxUnitCubeEllipsoidMixture, backend=backend, key=key,
            initial_ellipsoid=initial_ellipsoid)

        threshold = np.log(split_threshold * enlarge_per_dim ** points.shape[1])
        while float(bound.outer_bound.log_v) - log_v_target > threshold:
            if not bound.outer_bound.split(key=key):
                break
        while float(bound.outer_bound.log_v) - log_v_target > threshold:
            if not bound.outer_bound.trim():
                break

        bound.points = np.zeros((0, bound.n_dim))
        bound.n_sample = 0
        bound.n_reject = 0
        bound.rng = rng
        bound._sample_key = key
        return bound

    def _stacked_neural_tensors(self):
        """Stack the per-bound neural tensors for the batched containment test.

        Returns ``(c_all, b_inv_all, params_all, mean_all, scale_all,
        threshold_all)`` with a leading axis of length ``len(neural_bounds)``,
        or ``None`` when the bounds cannot share one kernel (the reference
        emulator predicts on the host; an emulator-less bound is ellipsoid
        only).  The stack is cached, but the source snapshot is checked on
        every call: replacing an emulator tensor (for example a deferred
        calibration) rebuilds the stack and replaces the mutation token, so no
        packed kernel arguments can outlive the tensors they were built from.
        The snapshot keeps the source objects alive, so identity comparison
        cannot be fooled by a recycled ``id``.
        """
        source = _neural_source_snapshot(self.neural_bounds)
        n_bounds = len(self.neural_bounds)
        cached = getattr(self, '_neural_stack', None)
        if (cached is not None and cached[0] == n_bounds
                and _neural_sources_match(
                    getattr(self, '_neural_sources', None), source)):
            return cached[1]

        stackable = n_bounds > 0 and all(
            getattr(b, 'emulator', None) is not None
            and getattr(b, 'neural_backend', 'jax') == 'jax'
            for b in self.neural_bounds)
        tensors = None
        if stackable:
            per_bound = [
                (b.outer_bound.c, b.outer_bound.B_inv, b.emulator.params,
                 b.emulator.mean, b.emulator.scale,
                 float(b.score_predict_min))
                for b in self.neural_bounds]
            # Identical shapes for every bound, so no padding is involved.
            tensors = jax.tree_util.tree_map(
                lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
                *per_bound)
        self._neural_stack = (n_bounds, tensors)
        self._neural_sources = source
        # Strong-reference mutation token for the multi-bound pack cache.
        self._neural_token = object()
        return tensors

    def _stacked_neural_mask(self, points):
        """OR over every neural bound of this nautilus bound, one dispatch."""
        points = jnp.atleast_2d(jnp.asarray(points))
        tensors = self._stacked_neural_tensors()
        if tensors is None:
            any_neural = jnp.zeros(points.shape[0], dtype=bool)
            for neural in self.neural_bounds:
                any_neural = any_neural | neural.contains(points)
            return any_neural
        (c_all, b_inv_all, params_all, mean_all, scale_all,
         threshold_all) = tensors
        return padded_eval(
            'neural_contains_stacked', _neural_contains_stacked, points,
            c_all, b_inv_all, params_all, mean_all, scale_all, threshold_all)

    def contains(self, points):
        """Whether each point lies in the outer bound and any neural bound."""
        return self._stacked_contains(points)

    def _stacked_contains(self, points):
        """Full membership test of this bound (outer bound plus neural OR-chain).

        This is the sampler-facing entry point: it applies the phase shift,
        tests the outer bound and, when neural bounds exist, ORs their tests
        through a single batched kernel (one ``padded_eval`` dispatch) instead
        of one dispatch per neural bound.  Results are identical to
        :meth:`contains`, which simply forwards here.
        """
        points = jnp.atleast_2d(jnp.asarray(points))
        if self.shift is not None:
            points = self.shift.transform(points)
        in_bound = self.outer_bound.contains(points)
        if len(self.neural_bounds) > 0:
            in_bound = in_bound & self._stacked_neural_mask(points)
        return in_bound

    def sample(self, n_points=100, key=None):
        """Draw ``n_points`` uniformly from the bound."""
        key = take_key(self, key)
        n_points = int(n_points)
        while self.points.shape[0] < n_points:
            key, sub = jax.random.split(key)
            n_draw = 1000
            points = self.outer_bound.sample(n_draw, key=sub)

            if len(self.neural_bounds) > 0:
                # The points are already in the shifted frame, so only the
                # neural masks apply (``_stacked_contains`` would shift again).
                # One stacked dispatch replaces the per-bound loop; the
                # compaction stays on the host for the reason given in
                # ``JaxUnion.sample``.
                neural_mask = getattr(self, '_stacked_neural_mask', None)
                if neural_mask is not None:
                    in_bound = neural_mask(points)
                else:
                    in_bound = jnp.zeros(points.shape[0], dtype=bool)
                    for neural in self.neural_bounds:
                        in_bound = in_bound | neural.contains(points)
                points = np.asarray(points)[np.asarray(in_bound)]

            points = np.asarray(points)
            self.points = np.concatenate([self.points, points], axis=0)
            self.n_sample += n_draw
            self.n_reject += n_draw - len(points)

        points = self.points[:n_points]
        self.points = self.points[n_points:]
        finish_key(self, key)
        if self.shift is not None:
            points = self.shift.transform(points, inverse=True)
        else:
            points = np.asarray(points)
        return points

    @property
    def log_v(self):
        """Log volume with the neural Monte Carlo correction."""
        if self.n_sample == 0:
            self.sample(100)
        return (float(self.outer_bound.log_v)
                + np.log(1.0 - self.n_reject / self.n_sample))

    @property
    def n_ell(self):
        """Number of ellipsoids in the outer bound."""
        return int(np.sum([not np.all(np.asarray(b.dim_cube))
                           for b in self.outer_bound.bounds]))

    @property
    def n_net(self):
        """Number of neural networks across all neural bounds."""
        return int(np.sum([b.n_networks for b in self.neural_bounds]))

    def reset(self, rng=None):
        """Drop cached proposals and reset counters."""
        self.points = np.zeros((0, self.n_dim))
        self.n_sample = 0
        self.n_reject = 0
        self.outer_bound.reset(rng)
        self.rng = rng