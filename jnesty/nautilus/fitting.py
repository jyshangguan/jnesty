"""Backends for fitting bound geometry.

Four backends are available:

``'numpy'`` (default)
    Uses the reference implementations from :mod:`nautilus.bounds.basic` and
    ``sklearn`` for the Gaussian mixture.  Ellipsoid fitting is a sequence of
    tiny matrix operations (a few thousand of them for one minimum-volume
    ellipsoid), which is dominated by kernel-launch latency on a GPU and is
    faster on the host.  This backend also reproduces the reference results
    exactly, which makes it the oracle-friendly default.

``'numpy_rank1'``
    The same host Khachiyan iteration as ``'numpy'``, but the inverse design
    matrix is maintained with a Sherman--Morrison rank-1 update instead of a
    fresh Cholesky factorisation after every accepted point.  It returns the
    same ellipsoid as ``'numpy'`` up to floating-point roundoff and is
    supported only for minimum-volume-ellipsoid fits.

``'cpu_rank1'``
    The same rank-1 iteration compiled with :func:`jax.jit` and executed on
    the CPU device, with a program cached per input shape and dtype.  It pays
    a one-off compilation per shape but is much faster per fit than the host
    implementations.  Results are returned on the default JAX device.

``'jax'``
    Uses the JAX ports in :mod:`jnesty.nautilus.geometry` and
    :mod:`jnesty.nautilus.gmm`.  These are useful for very large point sets or
    when the whole pipeline must stay on device.

The expensive parts of a nautilus run - likelihood evaluation, bound sampling,
containment tests and neural emulator training/prediction - are on the device
in both cases.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .geometry import (invert_spd as _jax_invert_spd,
                       minimum_volume_enclosing_ellipsoid as _jax_mvee)
from .gmm import gmm2_fit as _jax_gmm, gmm2_log_prob as _jax_gmm_log_prob

__all__ = ['fit_gmm', 'fit_mvee', 'normalise_backend']

_BACKENDS = ('numpy', 'numpy_rank1', 'cpu_rank1', 'jax')

# Compiled CPU programs and the CPU device, keyed by input shape and dtype.
_CPU_RANK1_PROGRAMS = {}
_CPU_RANK1_DEVICE = None


def normalise_backend(backend):
    """Validate a backend name."""
    if backend is None:
        return 'numpy'
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown fit backend {backend!r}; expected one of "
                         f"{_BACKENDS}.")
    return backend


def _minimum_volume_enclosing_ellipsoid_rank1(points, n_max=100, n_batch=20):
    """Approximate the MVEE while updating the inverse design matrix by rank 1.

    This mirrors
    :func:`nautilus.bounds.basic.minimum_volume_enclosing_ellipsoid` exactly,
    including the final centre/covariance computation and scaling.  The only
    difference is that an accepted update

    ``V_new = (1 - a) * (V + alpha * q q^T)``, ``alpha = a / (1 - a)``

    is inverted with the Sherman--Morrison formula instead of a fresh
    Cholesky factorisation of ``V_new``.  The two formulas are algebraically
    identical, so the results differ only by accumulated floating-point
    roundoff (on the order of 1e-12 in the centre and 1e-11 relative in the
    returned matrix in a probe over real MVEE inputs).

    ``V`` itself is not updated: the reference only rebuilds it every 1000
    iterations, while ``n_max`` is 100 here.
    """
    from nautilus.bounds.basic import (
        invert_symmetric_positive_semidefinite_matrix)

    points = np.asarray(points)
    n_points, n_dim = points.shape
    q = np.append(points, np.ones(shape=(n_points, 1)), axis=1)
    u = np.repeat(1.0 / n_points, n_points)
    q_outer = np.array([np.outer(q_i, q_i) for q_i in q])
    v = np.einsum('ji,j,jk', q, u, q)
    v_inv = invert_symmetric_positive_semidefinite_matrix(v)
    target = n_dim + 1
    indices = np.arange(n_points)

    for _ in range(n_max):
        g_all = np.einsum('ijk,jk', q_outer, v_inv)
        for j in np.argsort(g_all)[-n_batch:][::-1]:
            g = np.einsum('jk,jk', q_outer[j], v_inv)
            if g < target:
                continue
            a = (g - target) / (target * (g - 1.0))
            # v_new = (1 - a) * (v + alpha * q q^T).  Sherman--Morrison
            # inverts it without another Cholesky factorisation.
            vq = v_inv @ q[j]
            alpha = a / (1.0 - a)
            v_inv = (v_inv - alpha / (1.0 + alpha * g)
                     * np.outer(vq, vq)) / (1.0 - a)
            u = u * (1.0 - a) + a * (indices == j)

    c = np.atleast_1d(np.average(points, weights=u, axis=0))
    a_inv = np.atleast_2d(np.cov(points, aweights=u, rowvar=False, bias=True))
    a = np.linalg.inv(a_inv)
    scale = np.amax(np.einsum('...i,ij,...j', points - c, a, points - c))
    a /= scale
    a_inv *= scale
    return c, a, a_inv


def _mvee_cpu_rank1_impl(points, n_max=100, n_batch=20):
    """Rank-1 Khachiyan iteration as a pure-JAX function for CPU use."""
    n_points, n_dim = points.shape
    dtype = points.dtype
    target = n_dim + 1.0
    q = jnp.concatenate([points, jnp.ones((n_points, 1), dtype)], axis=1)
    q_outer = q[:, :, None] * q[:, None, :]
    u0 = jnp.full((n_points,), 1.0 / n_points, dtype)
    v0 = jnp.einsum('ni,nj->ij', q * u0[:, None], q)
    v_inv0 = _jax_invert_spd(v0)

    def outer(_, carry):
        u, v_inv = carry
        g = jnp.einsum('nij,ij->n', q_outer, v_inv)
        order = jnp.argsort(g)[::-1][:n_batch]

        def inner(carry_inner, node):
            u, v_inv = carry_inner
            g_node = jnp.einsum('ij,ij->', q_outer[node], v_inv)

            def update(carry_update):
                u, v_inv = carry_update
                a = (g_node - target) / (target * (g_node - 1.0))
                one_hot = (jnp.arange(n_points) == node).astype(dtype)
                u = u * (1.0 - a) + a * one_hot
                vq = v_inv @ q[node]
                alpha = a / (1.0 - a)
                v_inv = (v_inv - alpha / (1.0 + alpha * g_node)
                         * jnp.outer(vq, vq)) / (1.0 - a)
                return u, v_inv

            return jax.lax.cond(g_node > target, update,
                                lambda c: c, (u, v_inv)), None

        return jax.lax.scan(inner, (u, v_inv), order)[0]

    u, _ = jax.lax.fori_loop(0, n_max, outer, (u0, v_inv0))
    c = jnp.einsum('n,ni->i', u, points)
    diff = points - c
    a_inv = jnp.einsum('n,ni,nj->ij', u, diff, diff)
    a = jnp.linalg.inv(a_inv)
    scale = jnp.max(jnp.einsum('ni,ij,nj->n', diff, a, diff))
    return c, a / scale, a_inv * scale


def _cpu_rank1_device():
    """Return the CPU device used by the compiled rank-1 backend."""
    global _CPU_RANK1_DEVICE
    if _CPU_RANK1_DEVICE is None:
        _CPU_RANK1_DEVICE = jax.devices('cpu')[0]
    return _CPU_RANK1_DEVICE


def _cpu_rank1_program(n_points, n_dim, dtype):
    """Return the compiled CPU rank-1 program for one shape and dtype."""
    key = (int(n_points), int(n_dim), np.dtype(dtype).str)
    program = _CPU_RANK1_PROGRAMS.get(key)
    if program is None:
        # The input is placed on the CPU device in the caller, which is what
        # selects the CPU target for this program.
        program = jax.jit(_mvee_cpu_rank1_impl)
        _CPU_RANK1_PROGRAMS[key] = program
    return program


def _minimum_volume_enclosing_ellipsoid_cpu_rank1(points):
    """Run the compiled CPU rank-1 MVEE and return NumPy arrays.

    The first call for a given shape and dtype compiles a program; subsequent
    calls reuse it.  Returning NumPy lets ``fit_mvee`` place the result on the
    default JAX device.
    """
    points = np.asarray(points)
    if not np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float64)
    n_points, n_dim = points.shape
    program = _cpu_rank1_program(n_points, n_dim, points.dtype)
    points_cpu = jax.device_put(points, _cpu_rank1_device())
    c, a, a_inv = program(points_cpu)
    return np.asarray(c), np.asarray(a), np.asarray(a_inv)


def _valid_mvee_result(c, a, a_inv, n_dim):
    """Whether a fit produced finite, correctly shaped ellipsoid arrays."""
    try:
        c = np.asarray(c)
        a = np.asarray(a)
        a_inv = np.asarray(a_inv)
        return bool(
            c.shape == (n_dim,)
            and a.shape == (n_dim, n_dim)
            and a_inv.shape == (n_dim, n_dim)
            and np.all(np.isfinite(c))
            and np.all(np.isfinite(a))
            and np.all(np.isfinite(a_inv))
            and np.all(np.diag(a) > 0)
            and np.all(np.diag(a_inv) > 0))
    except Exception:
        return False


def _fit_mvee_rank1_with_fallback(points, fit):
    """Run a rank-1 MVEE backend, falling back to the reference on failure."""
    points = np.asarray(points)
    n_dim = points.shape[1]
    c = a = a_inv = None
    try:
        c, a, a_inv = fit(points)
    except Exception:
        pass
    if _valid_mvee_result(c, a, a_inv, n_dim):
        return np.asarray(c), np.asarray(a), np.asarray(a_inv)
    from nautilus.bounds.basic import minimum_volume_enclosing_ellipsoid
    return minimum_volume_enclosing_ellipsoid(points)


def fit_mvee(points, backend='numpy'):
    """Return ``(c, A, A_inv)`` of the minimum-volume enclosing ellipsoid."""
    backend = normalise_backend(backend)
    if backend == 'numpy':
        from nautilus.bounds.basic import minimum_volume_enclosing_ellipsoid
        c, a, a_inv = minimum_volume_enclosing_ellipsoid(np.asarray(points))
        return jnp.asarray(c), jnp.asarray(a), jnp.asarray(a_inv)

    if backend == 'numpy_rank1':
        c, a, a_inv = _fit_mvee_rank1_with_fallback(
            points, _minimum_volume_enclosing_ellipsoid_rank1)
        return jnp.asarray(c), jnp.asarray(a), jnp.asarray(a_inv)

    if backend == 'cpu_rank1':
        c, a, a_inv = _fit_mvee_rank1_with_fallback(
            points, _minimum_volume_enclosing_ellipsoid_cpu_rank1)
        return jnp.asarray(c), jnp.asarray(a), jnp.asarray(a_inv)

    return _jax_mvee(jnp.asarray(points))


def fit_gmm(points, key, backend='numpy'):
    """Fit a two-component full-covariance mixture and label every point.

    Returns
    -------
    weights, means, covs : jax.numpy.ndarray
        Mixture parameters.
    log_p : numpy.ndarray, shape (n_points, 2)
        Log responsibility of each component for each point.
    """
    backend = normalise_backend(backend)
    points_np = np.asarray(points)
    if backend in ('numpy', 'numpy_rank1', 'cpu_rank1'):
        from sklearn.mixture import GaussianMixture
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # pragma: no cover - threadpoolctl is a dependency
            threadpool_limits = None

        seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
        fit = lambda: GaussianMixture(  # noqa: E731
            n_components=2, n_init=10, random_state=seed).fit(points_np)
        # The fit is a long chain of 2x2 linear algebra calls; letting a
        # multi-threaded BLAS spread each of them over every core costs ~30 %
        # (measured), so hold it to one thread for the duration.
        if threadpool_limits is not None:
            with threadpool_limits(limits=1):
                gmm = fit()
        else:
            gmm = fit()
        log_p = gmm._estimate_weighted_log_prob(points_np)
        return (jnp.asarray(gmm.weights_), jnp.asarray(gmm.means_),
                jnp.asarray(gmm.covariances_), np.asarray(log_p))

    weights, means, covs, _ = _jax_gmm(jnp.asarray(points_np), key)
    log_p = np.asarray(_jax_gmm_log_prob(jnp.asarray(points_np), weights,
                                        means, covs))
    return weights, means, covs, log_p