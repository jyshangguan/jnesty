"""JAX geometry primitives for the GPU nautilus port.

The reference implementation lives in :mod:`nautilus.bounds.basic`.  This
module reproduces the same mathematics with JAX arrays so that the expensive
parts (minimum-volume enclosing ellipsoid fitting, coordinate transforms,
containment tests and sampling of whole unions) run on the device.

A single unified representation is used for every bound type:

``center``   (d,)     centre of the ellipsoid (0.5 along unit-cube dimensions)
``B``        (d, d)   ``x_ell = center + B @ z``  (zero rows/cols for cube dims)
``B_inv``    (d, d)   inverse of ``B`` (zero rows/cols for cube dims)
``dim_cube`` (d,)     boolean mask of dimensions sampled from the unit cube
``log_v``    scalar   natural log of the bound volume

The transform used by the multi-ellipsoid decomposition maps a point to

``z = T @ (x - center)`` with ``T = B_inv + 2 * diag(dim_cube)``,

so cube dimensions map to ``[-1, 1]`` and ellipsoid dimensions to the
whitened ellipsoid frame, exactly like ``UnitCubeEllipsoidMixture.transform``.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

__all__ = [
    'invert_spd',
    'minimum_volume_enclosing_ellipsoid',
    'unified_transform',
    'unified_log_v',
    'unified_contains',
    'unified_sample',
    'sample_unified_bounds',
    'contains_unified_bounds',
    'log_volume_of_ball',
]


def invert_spd(matrix):
    """Invert a symmetric positive definite matrix.

    Equivalent to ``dpotri(dpotrf(matrix))`` in
    :func:`nautilus.bounds.basic.invert_symmetric_positive_semidefinite_matrix`
    but expressed through a Cholesky solve, which is stable on GPU.
    """
    n = matrix.shape[-1]
    chol = jnp.linalg.cholesky(matrix)
    chol_inv = jax.scipy.linalg.solve_triangular(
        chol, jnp.eye(n, dtype=matrix.dtype), lower=True)
    return chol_inv.T @ chol_inv


def minimum_volume_enclosing_ellipsoid(points, n_max=100, n_batch=20):
    """Approximate minimum-volume enclosing ellipsoid (Khachiyan algorithm).

    Parameters
    ----------
    points : jax.numpy.ndarray, shape (n_points, n_dim)
        Points to enclose.
    n_max : int, optional
        Maximum number of outer iterations. Default is 100.
    n_batch : int, optional
        Number of violated points handled per iteration. Default is 20.

    Returns
    -------
    c : jax.numpy.ndarray, shape (n_dim,)
        Centre.
    a : jax.numpy.ndarray, shape (n_dim, n_dim)
        Matrix defining ``(x - c)^T A (x - c) <= 1``.
    a_inv : jax.numpy.ndarray, shape (n_dim, n_dim)
        Inverse of ``a``.

    Notes
    -----
    Mirrors :func:`nautilus.bounds.basic.minimum_volume_enclosing_ellipsoid`,
    including the batch of ``n_batch`` largest violations per iteration and the
    ``g < n_dim + 1`` skipping rule.
    """
    points = jnp.asarray(points)
    n_points, n_dim = points.shape
    dtype = points.dtype
    target = n_dim + 1.0

    q = jnp.concatenate([points, jnp.ones((n_points, 1), dtype)], axis=1)
    q_outer = q[:, :, None] * q[:, None, :]

    u0 = jnp.full((n_points,), 1.0 / n_points, dtype)
    v0 = jnp.einsum('ni,nj->ij', q * u0[:, None], q)
    v_inv0 = invert_spd(v0)

    def outer(_, carry):
        u, v, v_inv = carry
        g = jnp.einsum('nij,ij->n', q_outer, v_inv)
        order = jnp.argsort(g)[::-1][:n_batch]

        def inner(carry_inner, j):
            u, v, v_inv = carry_inner
            node = order[j]
            # The violation is re-evaluated against the *current* inverse:
            # accepting an update shrinks the ellipsoid, so a node selected
            # from the stale ordering may no longer violate the constraint.
            g_node = jnp.einsum('ij,ij->', q_outer[node], v_inv)
            a = (g_node - target) / (target * (g_node - 1.0))

            def do_update(carry_update):
                u, v, _ = carry_update
                one_hot = (jnp.arange(n_points) == node).astype(dtype)
                u_new = u * (1.0 - a) + a * one_hot
                v_new = v * (1.0 - a) + a * q_outer[node]
                return u_new, v_new, invert_spd(v_new)

            return jax.lax.cond(g_node > target, do_update,
                                lambda c: c, (u, v, v_inv)), None

        return jax.lax.scan(inner, (u, v, v_inv), jnp.arange(n_batch))[0]

    u, _, _ = jax.lax.fori_loop(0, n_max, outer, (u0, v0, v_inv0))

    c = jnp.einsum('n,ni->i', u, points)
    diff = points - c
    a_inv = jnp.einsum('n,ni,nj->ij', u, diff, diff)
    a = jnp.linalg.inv(a_inv)
    scale = jnp.max(jnp.einsum('ni,ij,nj->n', diff, a, diff))
    a = a / scale
    a_inv = a_inv * scale
    return c, a, a_inv


def log_volume_of_ball(n_dim):
    """Log volume of the free part of a bound (unit ball prefactor)."""
    return (n_dim * jnp.log(2.0) + n_dim * gammaln(1.5)
            - gammaln(n_dim / 2.0 + 1.0))


def unified_log_v(b, n_ell):
    """Log volume of a unified bound from its ``B`` matrix."""
    return jnp.linalg.slogdet(b)[1] + log_volume_of_ball(n_ell)


def unified_transform(center, b_inv, dim_cube, points):
    """Transform points into the bound frame ``z = T @ (x - center)``."""
    t = b_inv + 2.0 * jnp.diag(dim_cube.astype(center.dtype))
    return jnp.einsum('ij,...j->...i', t, points - center)


def unified_contains(center, b_inv, dim_cube, points, tol=0.0):
    """Containment test for a unified bound.

    Matches ``UnitCube.contains & Ellipsoid.contains`` of the reference
    implementation: the unit-cube dimensions must lie in ``[0, 1)`` and the
    remaining dimensions inside the ellipsoid.
    """
    points = jnp.atleast_2d(points)
    z = unified_transform(center, b_inv, dim_cube, points)
    z = jnp.where(dim_cube[None, :], 0.0, z)
    in_ellipsoid = jnp.sum(z ** 2, axis=-1) < (1.0 + tol)
    in_cube = jnp.all(
        jnp.where(dim_cube[None, :],
                  (points >= 0.0) & (points < 1.0),
                  True), axis=-1)
    return in_cube & in_ellipsoid


def unified_sample(center, b, dim_cube, n_ell, n_points, key):
    """Draw ``n_points`` uniformly from a unified bound."""
    key_dir, key_radius, key_cube = jax.random.split(key, 3)
    n_points = int(n_points)
    n_dim = center.shape[0]

    # The direction is normalised over the *ellipsoid* dimensions only.  The
    # bound is a product of a unit cube (for the cube dimensions) and an
    # n_ell-dimensional ball, so including the cube dimensions in the norm
    # would shrink the ellipsoid coordinate towards its centre by a factor
    # sqrt(n_dim / n_ell) -- the proposals then cluster in the middle of the
    # bound, never leave the unit cube, and the Monte-Carlo estimate of the
    # bound volume comes out far too large.
    z = jax.random.normal(key_dir, (n_points, n_dim))
    z = jnp.where(dim_cube[None, :], 0.0, z)
    norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
    norm = jnp.where(norm > 0, norm, 1.0)
    radius = jax.random.uniform(key_radius, (n_points, 1)) ** (
        1.0 / jnp.maximum(n_ell, 1))
    y = z / norm * radius

    points = jnp.einsum('ij,nj->ni', b, y) + center

    cube_points = jax.random.uniform(key_cube, (n_points, n_dim))
    points = jnp.where(dim_cube[None, :], cube_points, points)
    return points


def sample_unified_bounds(b, center, dim_cube, n_ell, counts, key,
                          n_total=None):
    """Sample from several unified bounds in a single device call.

    Parameters
    ----------
    b : array (n_bounds, n_dim, n_dim)
    center : array (n_bounds, n_dim)
    dim_cube : array (n_bounds, n_dim) bool
    n_ell : array (n_bounds,)
    counts : array (n_bounds,) int - points to draw from each bound
    key : PRNG key

    Returns
    -------
    points : array (sum(counts), n_dim)
    ids : array (sum(counts),) int - index of the bound that produced a point
    """
    counts = jnp.asarray(counts, dtype=jnp.int32)
    if n_total is None:
        n_total = int(jnp.sum(counts))
    n_total = int(n_total)
    ids = jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts,
                     total_repeat_length=n_total)

    d = center.shape[1]
    z = jax.random.normal(key, (n_total, d))
    # Normalise over the ellipsoid dimensions only (see ``unified_sample``).
    z = jnp.where(dim_cube[ids], 0.0, z)
    norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
    norm = jnp.where(norm > 0, norm, 1.0)
    radius = jax.random.uniform(
        jax.random.fold_in(key, 1), (n_total, 1)) ** (
            1.0 / jnp.maximum(n_ell[ids], 1))[:, None]
    y = z / norm * radius

    points = jnp.einsum('nij,nj->ni', b[ids], y) + center[ids]
    cube_points = jax.random.uniform(jax.random.fold_in(key, 2), (n_total, d))
    points = jnp.where(dim_cube[ids], cube_points, points)
    # Bounds near the unit-cube edge can scatter points outside the cube;
    # the union-level sampler filters against the cube afterwards.
    return points, ids


def contains_unified_bounds(b_inv, center, dim_cube, points):
    """Containment of ``points`` (n_points, d) in every bounded bound.

    Returns an ``(n_bounds, n_points)`` boolean array.  Bounds are expected to
    be padded with inactive entries whose ``center`` is ``nan``; the caller
    masks them out.
    """
    delta = points[None, :, :] - center[:, None, :]
    z = jnp.einsum('bij,bnj->bni', b_inv, delta)
    z = jnp.where(dim_cube[:, None, :], 0.0, z)
    in_ellipsoid = jnp.sum(z ** 2, axis=-1) < 1.0
    in_cube = jnp.all(
        jnp.where(dim_cube[:, None, :],
                  (points[None, :, :] >= 0.0) & (points[None, :, :] < 1.0),
                  True), axis=-1)
    ok = in_cube & in_ellipsoid
    # Inactive padded bounds have nan centres; comparisons yield False above,
    # but make that explicit for safety.
    return jnp.where(jnp.isnan(center[:, None, 0]), False, ok)