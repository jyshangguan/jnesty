"""JAX implementation of the two-component Gaussian mixture used to split bounds.

The reference implementation uses ``sklearn.mixture.GaussianMixture`` with
``n_components=2``, ``n_init=10`` and full covariances.  The port keeps the same
model and model-selection rule (best of ten initialisations by the Gaussian
mixture lower bound) but runs k-means initialisation and EM on the device.
"""

import jax
import jax.numpy as jnp

__all__ = ['gmm2_log_prob', 'gmm2_lower_bound', 'gmm2_fit']

_REG_COVAR = 1e-6
_N_INIT = 10
_KMEANS_ITER = 20
_EM_ITER = 100


def _sq_dist(points, centers):
    return jnp.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)


def _kmeans(points, key, n_clusters=2, n_iter=_KMEANS_ITER):
    """k-means++ initialisation followed by Lloyd iterations."""
    n_points = points.shape[0]
    first = jax.random.randint(key, (), 0, n_points)
    centers = jnp.zeros((n_clusters, points.shape[1]))
    centers = centers.at[0].set(points[first])
    d2 = _sq_dist(points, centers[0:1])[:, 0]
    for k in range(1, n_clusters):
        probs = d2 / jnp.sum(d2)
        idx = jax.random.choice(jax.random.fold_in(key, k), n_points, (), p=probs)
        centers = centers.at[k].set(points[idx])
        d2 = jnp.minimum(d2, _sq_dist(points, centers[k:k + 1])[:, 0])

    def body(_, cen):
        dist = _sq_dist(points, cen)
        labels = jnp.argmin(dist, axis=1)
        new_centers = jnp.stack([
            jnp.where(jnp.sum(labels == k) > 0,
                      jnp.sum(jnp.where(labels[:, None] == k, points, 0.0), axis=0)
                      / jnp.maximum(jnp.sum(labels == k), 1),
                      cen[k])
            for k in range(n_clusters)])
        return new_centers

    return jax.lax.fori_loop(0, n_iter, body, centers)


def _initial_parameters(points, key):
    dtype = points.dtype
    centers = _kmeans(points, key)
    dist = _sq_dist(points, centers)
    labels = jnp.argmin(dist, axis=1)
    n_dim = points.shape[1]
    weights = jnp.asarray(
        [jnp.mean((labels == k).astype(dtype)) for k in range(2)], dtype)
    means = centers.astype(dtype)
    covs = jnp.stack([
        jnp.atleast_2d(jnp.cov(points.T, bias=True,
                               aweights=(labels == k).astype(dtype)))
        for k in range(2)]).astype(dtype)
    covs = covs + _REG_COVAR * jnp.eye(n_dim, dtype=dtype)
    return weights, means, covs


def _log_gaussian(points, mean, cov):
    n_dim = points.shape[1]
    # Use ``jax.scipy.stats`` style explicit computation for numerical control.
    chol = jnp.linalg.cholesky(cov)
    diff = points - mean
    solved = jax.scipy.linalg.solve_triangular(chol, diff.T, lower=True).T
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5 * (jnp.sum(solved ** 2, axis=1) + log_det
                   + n_dim * jnp.log(2.0 * jnp.pi))


def gmm2_log_prob(points, weights, means, covs):
    """Log probability of each point under each component, shape (n, 2)."""
    log_p = jnp.stack([
        _log_gaussian(points, means[k], covs[k]) + jnp.log(weights[k])
        for k in range(2)], axis=1)
    return log_p


def gmm2_lower_bound(points, weights, means, covs):
    """Gaussian mixture lower bound (log likelihood + entropy)."""
    log_p = gmm2_log_prob(points, weights, means, covs)
    log_norm = jax.scipy.special.logsumexp(log_p, axis=1)
    resp = jnp.exp(log_p - log_norm[:, None])
    entropy = -jnp.sum(resp * jnp.where(resp > 0, jnp.log(resp), 0.0))
    return jnp.sum(log_norm) + entropy


def _em(points, params, n_iter=_EM_ITER):
    weights, means, covs = params
    n_dim = points.shape[1]
    dtype = points.dtype

    def body(_, carry):
        weights, means, covs = carry
        log_p = gmm2_log_prob(points, weights, means, covs)
        log_norm = jax.scipy.special.logsumexp(log_p, axis=1)
        resp = jnp.exp(log_p - log_norm[:, None])

        nk = jnp.sum(resp, axis=0) + jnp.asarray(1e-10, dtype)
        new_weights = (nk / jnp.sum(nk)).astype(dtype)
        new_means = ((resp.T @ points) / nk[:, None]).astype(dtype)
        new_covs = []
        for k in range(2):
            diff = points - new_means[k]
            cov = jnp.einsum('n,ni,nj->ij', resp[:, k], diff, diff) / nk[k]
            new_covs.append(
                cov + _REG_COVAR * jnp.eye(n_dim, dtype=dtype))
        return new_weights, new_means, jnp.stack(new_covs).astype(dtype)

    return jax.lax.fori_loop(0, n_iter, body, (weights, means, covs))


def gmm2_fit(points, key, n_init=_N_INIT):
    """Fit a two-component full-covariance Gaussian mixture.

    Parameters
    ----------
    points : jax.numpy.ndarray, shape (n_points, n_dim)
    key : PRNG key
    n_init : int, optional
        Number of initialisations, best is kept. Default is 10.

    Returns
    -------
    weights : array (2,)
    means : array (2, n_dim)
    covs : array (2, n_dim, n_dim)
    lower_bound : scalar
    """
    keys = jax.random.split(key, n_init)

    def run(k):
        init = _initial_parameters(points, k)
        fitted = _em(points, init)
        return (*fitted, gmm2_lower_bound(points, *fitted))

    all_params = jax.vmap(run)(keys)
    best = jnp.argmax(all_params[3])
    return (all_params[0][best], all_params[1][best],
            all_params[2][best], all_params[3][best])