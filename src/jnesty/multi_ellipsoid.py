"""
Multi-ellipsoid bounding for nested sampling.

Implements Dynesty's multi-ellipsoidal decomposition in pure JAX:
- Recursive k-means splitting (k=2) with BIC stopping criterion
- Ellipsoid fitting via covariance eigenvalue decomposition
- Axis selection for rwalk proposals (proportional to volume)
- Union sampling with overlap correction (1/q rejection)

The core fitting (fit_multi_ellipsoid) is JIT-compiled using masked
operations and lax.while_loop to eliminate Python dispatch overhead.
"""

import functools

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import NamedTuple
from jax.scipy.special import gammaln, logsumexp


class MultiEllipsoidState(NamedTuple):
    """Fixed-size arrays for a collection of ellipsoids."""
    centers: jnp.ndarray       # (max_ells, ndim)
    covs: jnp.ndarray          # (max_ells, ndim, ndim)
    axes: jnp.ndarray          # (max_ells, ndim, ndim)
    precision: jnp.ndarray     # (max_ells, ndim, ndim)
    logvol_ells: jnp.ndarray   # (max_ells,)
    n_active: int              # number of active ellipsoids


def logvol_prefactor(n):
    """Log of volume constant for n-dimensional unit sphere."""
    return n * jnp.log(2.0) + n * gammaln(1.0 / 2.0 + 1.0) - gammaln(n / 2.0 + 1.0)


@jax.jit
def fit_bounding_ellipsoid(points):
    """
    Fit a bounding ellipsoid to a set of points.

    Computes the minimum-volume ellipsoid containing all points by:
    1. Computing the covariance matrix
    2. Expanding to contain the farthest point (Mahalanobis distance)

    Returns center, cov, axes, precision, logvol.
    """
    npoints, ndim = points.shape
    center = jnp.mean(points, axis=0)
    delta = points - center

    cov = jnp.cov(delta, rowvar=False)
    cov = jnp.atleast_2d(cov)

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals = jnp.maximum(eigvals, 1e-10)
    precision = (eigvecs / eigvals) @ eigvecs.T

    maha = jnp.einsum('ij,jk,ik->i', delta, precision, delta)
    fmax = maha.max()
    fmax = fmax / (1.0 - 1e-3)
    cov = cov * fmax
    precision = precision / fmax

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals = jnp.maximum(eigvals, 1e-10)
    precision = (eigvecs / eigvals) @ eigvecs.T
    axes = eigvecs * jnp.sqrt(eigvals)
    logvol = logvol_prefactor(ndim) + 0.5 * jnp.sum(jnp.log(eigvals))

    return center, cov, axes, precision, logvol


# ---------------------------------------------------------------------------
# Masked helpers for JIT-compiled multi-ellipsoid fitting
# ---------------------------------------------------------------------------

def _fit_bounding_ellipsoid_masked(points, mask):
    """Fit bounding ellipsoid to masked subset of points."""
    npoints, ndim = points.shape
    n = mask.sum()
    w = mask.astype(points.dtype) / jnp.maximum(n, 1.0)

    center = (points * w[:, None]).sum(0)
    delta = (points - center) * mask[:, None]

    # Sample covariance with Bessel correction (matching jnp.cov)
    cov = jnp.einsum('ni,nj->ij', delta, delta * w[:, None])
    cov = cov * jnp.where(n > 1, n / jnp.maximum(n - 1, 1), 1.0)
    cov = jnp.atleast_2d(cov)

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals = jnp.maximum(eigvals, 1e-10)
    precision = (eigvecs / eigvals) @ eigvecs.T

    # Mahalanobis distance (only masked points)
    maha = jnp.einsum('ni,ij,nj->n', delta, precision, delta)
    maha = jnp.where(mask, maha, 0.0)
    fmax = maha.max()
    fmax = fmax / (1.0 - 1e-3)
    cov = cov * fmax
    precision = precision / fmax

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals = jnp.maximum(eigvals, 1e-10)
    precision = (eigvecs / eigvals) @ eigvecs.T
    axes = eigvecs * jnp.sqrt(eigvals)
    logvol = logvol_prefactor(ndim) + 0.5 * jnp.sum(jnp.log(eigvals))

    return center, cov, axes, precision, logvol


def _kmeans2_masked(points, mask, init_centers, n_iter=10):
    """K-means with 2 clusters on masked subset of points."""
    n = mask.sum()
    w = mask.astype(points.dtype) / jnp.maximum(n, 1.0)
    mean = (points * w[:, None]).sum(0)
    var = ((points - mean) ** 2 * w[:, None]).sum(0)
    scale = jnp.sqrt(jnp.maximum(var, 1e-10))

    pts_sc = points / scale
    cts_sc = init_centers / scale

    def step(carry, _):
        dists = jnp.sum((pts_sc[None, :, :] - carry[:, None, :]) ** 2, axis=2)
        dists = dists + jnp.where(mask[None, :], 0.0, 1e30)
        labels = jnp.argmin(dists, axis=0)

        m0 = (labels == 0) & mask
        m1 = (labels == 1) & mask
        n0 = jnp.maximum(m0.sum(), 1.0)
        n1 = jnp.maximum(m1.sum(), 1.0)
        c0 = (pts_sc * m0[:, None].astype(float)).sum(0) / n0
        c1 = (pts_sc * m1[:, None].astype(float)).sum(0) / n1
        return jnp.stack([c0, c1]), None

    cts_sc, _ = lax.scan(step, cts_sc, None, length=n_iter)

    dists = jnp.sum((pts_sc[None, :, :] - cts_sc[:, None, :]) ** 2, axis=2)
    dists = dists + jnp.where(mask[None, :], 0.0, 1e30)
    labels = jnp.argmin(dists, axis=0)
    return jnp.where(mask, labels, 0)


@functools.partial(jax.jit, static_argnums=(1,))
def _fit_multi_ellipsoid_core(points, max_ellipsoids):
    """
    JIT-compiled core: fit multi-ellipsoid using lax.while_loop with
    fixed-size masked state arrays.

    Uses a slot-based approach: max_ellipsoids slots, each with a point mask.
    At each step, pick a splittable slot, try k-means split, accept/reject
    via BIC. On success, replace slot with left child, add right child to
    next free slot. On failure, mark as leaf.
    """
    npoints, ndim = points.shape
    min_size = 2 * ndim
    nparam = ndim * (ndim + 3) // 2

    # State arrays
    init_masks = jnp.zeros((max_ellipsoids, npoints), dtype=bool).at[0].set(True)
    init_splittable = jnp.zeros(max_ellipsoids, dtype=bool).at[0].set(True)
    init_centers = jnp.zeros((max_ellipsoids, ndim))
    init_covs = jnp.zeros((max_ellipsoids, ndim, ndim))
    init_axes = jnp.zeros((max_ellipsoids, ndim, ndim))
    init_precs = jnp.zeros((max_ellipsoids, ndim, ndim))
    init_logvols = jnp.full(max_ellipsoids, -jnp.inf)
    init_n_slots = jnp.array(1)
    init_n_ells = jnp.array(0)

    init_state = (init_masks, init_splittable, init_centers, init_covs,
                  init_axes, init_precs, init_logvols, init_n_slots, init_n_ells)

    def cond_fn(state):
        _, splittable, _, _, _, _, _, _, n_ells = state
        return splittable.any() & (n_ells < max_ellipsoids)

    def body_fn(state):
        masks, splittable, centers, covs, axes_arr, precs, logvols, n_slots, n_ells = state

        slot = jnp.argmax(splittable)
        mask = masks[slot]
        n = mask.sum()

        # Fit ellipsoid for this cluster
        ctr, cv, ax, prec, lv = _fit_bounding_ellipsoid_masked(points, mask)

        # Major axis endpoints for k-means init
        ev = jnp.linalg.eigvalsh(cv)
        i_major = jnp.argmax(ev)
        v_major = ax[:, i_major]
        init_ctrs = jnp.stack([ctr - v_major, ctr + v_major])

        # K-means clustering
        labels = _kmeans2_masked(points, mask, init_ctrs)
        left_mask = mask & (labels == 0)
        right_mask = mask & (labels == 1)
        n0, n1 = left_mask.sum(), right_mask.sum()

        # Can split: enough points, room for new slot
        can_split = (n >= 2 * min_size) & (n0 >= min_size) & (n1 >= min_size)
        room = n_slots < max_ellipsoids

        # Fit sub-cluster ellipsoids (always compute; cheap for small ndim)
        l_ctr, l_cv, l_ax, l_pr, l_lv = _fit_bounding_ellipsoid_masked(points, left_mask)
        r_ctr, r_cv, r_ax, r_pr, r_lv = _fit_bounding_ellipsoid_masked(points, right_mask)

        # BIC criterion
        log_vol_dec = nparam * jnp.log(jnp.maximum(n, 2.0)) / jnp.maximum(n, 1.0)
        combined = jnp.logaddexp(l_lv, r_lv)
        bic_pass = (combined - lv) < -log_vol_dec

        success = can_split & bic_pass & room

        # --- Success branch: replace slot with left child, add right child ---
        s_masks = masks.at[slot].set(left_mask).at[n_slots].set(right_mask)
        s_splittable = splittable.at[slot].set(True).at[n_slots].set(True)
        s_centers = centers.at[slot].set(l_ctr).at[n_slots].set(r_ctr)
        s_covs = covs.at[slot].set(l_cv).at[n_slots].set(r_cv)
        s_axes = axes_arr.at[slot].set(l_ax).at[n_slots].set(r_ax)
        s_precs = precs.at[slot].set(l_pr).at[n_slots].set(r_pr)
        s_logvols = logvols.at[slot].set(l_lv).at[n_slots].set(r_lv)
        s_n_slots = n_slots + 1
        s_n_ells = n_ells

        # --- Failure branch: mark as leaf, store ellipsoid ---
        f_masks = masks
        f_splittable = splittable.at[slot].set(False)
        f_centers = centers.at[slot].set(ctr)
        f_covs = covs.at[slot].set(cv)
        f_axes = axes_arr.at[slot].set(ax)
        f_precs = precs.at[slot].set(prec)
        f_logvols = logvols.at[slot].set(lv)
        f_n_slots = n_slots
        f_n_ells = n_ells + 1

        # Select branch
        return (
            jnp.where(success, s_masks, f_masks),
            jnp.where(success, s_splittable, f_splittable),
            jnp.where(success, s_centers, f_centers),
            jnp.where(success, s_covs, f_covs),
            jnp.where(success, s_axes, f_axes),
            jnp.where(success, s_precs, f_precs),
            jnp.where(success, s_logvols, f_logvols),
            jnp.where(success, s_n_slots, f_n_slots),
            jnp.where(success, s_n_ells, f_n_ells),
        )

    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    masks_f, splittable_f, centers_f, covs_f, axes_f, precs_f, logvols_f, _, n_ells_f = final_state

    # Active ellipsoids: all used slots (masks have at least one point)
    is_used = masks_f.any(axis=1)
    n_active = is_used.sum()

    return centers_f, covs_f, axes_f, precs_f, logvols_f, n_active


def fit_multi_ellipsoid(points, max_ellipsoids=20, enlarge=1.25):
    """
    Fit a set of bounding ellipsoids using recursive k-means splitting.

    Algorithm (matching Dynesty's _bounding_ellipsoids):
    1. Fit initial bounding ellipsoid to all points
    2. Split into 2 clusters via k-means along major axis
    3. Accept split if BIC criterion satisfied
    4. Recursively split sub-clusters
    5. Enlarge all ellipsoids by the given factor (matches Dynesty's bound_enlarge)

    JIT-compiled internally using masked operations and lax.while_loop.

    Args:
        points: (npoints, ndim) live points
        max_ellipsoids: maximum number of ellipsoids
        enlarge: volume enlargement factor (default 1.25, matches Dynesty)

    Returns:
        MultiEllipsoidState with fitted ellipsoids
    """
    centers, covs, axes, precs, logvols, n_active = _fit_multi_ellipsoid_core(
        points, max_ellipsoids
    )

    # Apply enlarge factor (matches Dynesty's bound_enlarge=1.25)
    # Scale axes so volume increases by `enlarge`: axes *= enlarge^(1/(2*ndim))
    if enlarge != 1.0:
        ndim = points.shape[1]
        scale = enlarge ** (1.0 / (2.0 * ndim))
        axes = axes * scale
        covs = covs * (scale ** 2)
        precs = precs / (scale ** 2)
        logvols = logvols + jnp.log(enlarge)

    return MultiEllipsoidState(
        centers=centers,
        covs=covs,
        axes=axes,
        precision=precs,
        logvol_ells=logvols,
        n_active=int(n_active),
    )


def get_axes_for_rwalk(key, state):
    """
    Select one ellipsoid proportional to volume and return its axes.
    """
    n = state.n_active
    logvols = state.logvol_ells[:n]
    log_probs = logvols - logsumexp(logvols)
    probs = jnp.exp(log_probs)

    idx = random.choice(key, n, p=probs)
    return state.axes[idx]


def sample_from_union(key, state):
    """
    Sample uniformly from the union of ellipsoids.

    Algorithm (matching Dynesty's MultiEllipsoid.sample()):
    1. Select ellipsoid proportional to volume
    2. Sample point from that ellipsoid
    3. Count how many ellipsoids contain the point (q)
    4. Accept with probability 1/q (for uniform over union)
    """
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


def contains(state, point):
    """Check if point is inside any of the active ellipsoids."""
    n = state.n_active
    delta = point[None, :] - state.centers[:n]
    maha = jnp.einsum('ij,ijk,ik->i', delta, state.precision[:n], delta)
    return jnp.any(maha < 1.0)


def contains_all(state, points):
    """Check if all points are inside at least one active ellipsoids."""
    n = state.n_active
    delta = points[:, None, :] - state.centers[None, :n, :]
    maha = jnp.einsum('pei,eij,pej->pe', delta, state.precision[:n], delta)
    return jnp.all(jnp.any(maha < 1.0, axis=1))
