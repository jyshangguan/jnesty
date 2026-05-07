"""
Bounding methods for nested sampling.

Provides a Bound base class and concrete implementations:
- UnitCube: no bounding (sample uniformly from unit cube)
- SingleEllipsoid: single bounding ellipsoid
- MultiEllipsoid: multi-ellipsoid decomposition via k-means splitting

All bounds follow the same interface: fit(), sample(), get_axes(), contains().
Selected by string name via get_bound() factory function.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.scipy.special import logsumexp
from typing import Optional

from .utils import randsphere
from .multi_ellipsoid import (
    fit_multi_ellipsoid,
    MultiEllipsoidState,
)


class Bound:
    """Base class for bounding distributions."""

    def __init__(self, ndim, **kwargs):
        self.ndim = ndim

    def fit(self, points):
        """Fit the bound to a set of live points. Returns self."""
        raise NotImplementedError

    def sample(self, key, n=1):
        """Sample point(s) from the bound."""
        raise NotImplementedError

    def get_axes(self):
        """Return (ndim, ndim) axes matrix for proposal generation."""
        raise NotImplementedError

    def contains(self, point):
        """Check if point is inside the bound."""
        raise NotImplementedError


class UnitCube(Bound):
    """No bounding — sample uniformly from unit cube."""

    def fit(self, points):
        return self

    def sample(self, key, n=1):
        return random.uniform(key, shape=(n, self.ndim))

    def get_axes(self):
        return jnp.eye(self.ndim)

    def contains(self, point):
        return jnp.all((point >= 0.0) & (point <= 1.0))


class SingleEllipsoid(Bound):
    """Single bounding ellipsoid fitted to live points."""

    def __init__(self, ndim, scale=1.0, **kwargs):
        super().__init__(ndim, **kwargs)
        self.center = jnp.zeros(ndim)
        self.axes = jnp.eye(ndim)
        self.eigenvals = jnp.ones(ndim)
        self.scale = scale

    def fit(self, points):
        """Fit ellipsoid using covariance matrix eigenvalue decomposition."""
        center = jnp.mean(points, axis=0)
        centered = points - center
        cov = jnp.cov(centered.T)
        eigenvals, eigenvecs = jnp.linalg.eigh(cov)
        eigenvals = jnp.maximum(eigenvals, 1e-10)
        axes = eigenvecs @ jnp.diag(jnp.sqrt(eigenvals))
        self.center = center
        self.axes = axes
        self.eigenvals = eigenvals
        return self

    def sample(self, key, n=1):
        """Sample uniformly from the ellipsoid."""
        key1, key2 = random.split(key)
        direction = random.normal(key1, shape=(self.ndim,))
        direction = direction / jnp.linalg.norm(direction)
        radius = jnp.power(random.uniform(key2), 1.0 / self.ndim)
        point = self.center + self.scale * radius * (self.axes @ direction)
        return point

    def get_axes(self):
        return self.axes

    def contains(self, point):
        delta = point - self.center
        maha = delta @ jnp.linalg.solve(self.axes @ self.axes.T, delta)
        return maha < 1.0


class MultiEllipsoidBound(Bound):
    """
    Multi-ellipsoid decomposition using recursive k-means splitting.

    Wraps the JIT-compiled multi-ellipsoid fitting from multi_ellipsoid.py.
    Stores state as a MultiEllipsoidState NamedTuple for JAX compatibility.
    """

    def __init__(self, ndim, max_ellipsoids=20, **kwargs):
        super().__init__(ndim, **kwargs)
        self.max_ellipsoids = max_ellipsoids
        self.state: Optional[MultiEllipsoidState] = None

    def fit(self, points):
        """Fit multi-ellipsoid decomposition to live points."""
        self.state = fit_multi_ellipsoid(points, max_ellipsoids=self.max_ellipsoids)
        return self

    def sample(self, key, n=1):
        """Sample uniformly from the union of ellipsoids (with overlap correction)."""
        if self.state is None:
            return random.uniform(key, shape=(n, self.ndim))

        results = []
        for i in range(n):
            key, subkey = random.split(key)
            point, _ = _sample_from_union(subkey, self.state)
            results.append(point)
        return jnp.stack(results)

    def get_axes(self):
        """Return axes for the largest-volume ellipsoid (for rwalk proposals)."""
        if self.state is None:
            return jnp.eye(self.ndim)
        n = self.state.n_active
        logvols = self.state.logvol_ells[:n]
        idx = jnp.argmax(logvols)
        return self.state.axes[idx]

    def get_walk_schedule(self, rwalk_K):
        """
        Compute a Bresenham interleaving schedule for rwalk proposals.

        Returns an array of length rwalk_K with ellipsoid indices,
        proportional to each ellipsoid's volume.
        """
        if self.state is None:
            return jnp.zeros(rwalk_K, dtype=jnp.int32)

        n = self.state.n_active
        log_probs = self.state.logvol_ells[:n] - logsumexp(self.state.logvol_ells[:n])
        probs = jnp.exp(log_probs)

        def _sched_step(acc, _):
            acc = acc + probs
            best = jnp.argmax(acc)
            acc = acc.at[best].add(-1.0)
            return acc, best

        _, schedule = lax.scan(_sched_step, jnp.zeros(n), None, length=rwalk_K)
        return schedule

    def contains(self, point):
        if self.state is None:
            return True
        n = self.state.n_active
        delta = point[None, :] - self.state.centers[:n]
        maha = jnp.einsum('ij,ijk,ik->i', delta, self.state.precision[:n], delta)
        return jnp.any(maha < 1.0)


def _sample_from_union(key, state):
    """Sample uniformly from the union of ellipsoids with overlap correction."""
    n = state.n_active
    ndim = state.centers.shape[1]
    logvols = state.logvol_ells[:n]
    log_probs = logvols - logsumexp(logvols)
    probs = jnp.exp(log_probs)

    key, k1, k2 = random.split(key, 3)
    ell_idx = random.choice(k1, n, p=probs)
    center = state.centers[ell_idx]
    axes = state.axes[ell_idx]

    z = random.normal(k2, shape=(ndim,))
    z_norm = jnp.linalg.norm(z)
    z_norm = jnp.where(z_norm > 0, z_norm, 1.0)
    radius = random.uniform(random.fold_in(key, 0)) ** (1.0 / ndim)
    xhat = z * (radius / z_norm)
    point = center + axes @ xhat
    return point, ell_idx


# Factory
BOUND_REGISTRY = {
    'none': UnitCube,
    'single': SingleEllipsoid,
    'multi': MultiEllipsoidBound,
}


def get_bound(name, ndim, **kwargs):
    """
    Instantiate a bound by name string.

    Parameters
    ----------
    name : str
        One of 'none', 'single', 'multi'.
    ndim : int
        Number of dimensions.
    **kwargs
        Additional arguments passed to the bound constructor.

    Returns
    -------
    Bound
        Instantiated bound object.
    """
    if name not in BOUND_REGISTRY:
        raise ValueError(f"Unknown bound '{name}'. Choose from {list(BOUND_REGISTRY.keys())}")
    return BOUND_REGISTRY[name](ndim=ndim, **kwargs)
