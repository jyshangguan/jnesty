"""
Whitening transforms for efficient random walk sampling.

Implements shrinkage covariance estimation and whitening transforms
to handle correlated parameters and improve sampling efficiency.
"""

import jax
import jax.numpy as jnp
from jax import random


def compute_shrinkage_covariance(data, shrinkage=0.1, epsilon=1e-6):
    """
    Compute shrinkage covariance estimator.

    Combines sample covariance with diagonal covariance for robustness:
    Cov_shrink = (1-λ) * Cov_sample + λ * diag(Cov_sample) + ε * I

    Args:
        data: Data points [N, D]
        shrinkage: Shrinkage parameter λ (0-1)
        epsilon: Small constant for numerical stability

    Returns:
        Covariance matrix [D, D]
    """
    # Sample covariance
    data_centered = data - jnp.mean(data, axis=0)
    cov_sample = jnp.cov(data_centered, rowvar=False, bias=True)

    # Shrinkage: blend with diagonal
    cov_diag = jnp.diag(jnp.diag(cov_sample))
    cov_shrink = (1 - shrinkage) * cov_sample + shrinkage * cov_diag

    # Add small regularization
    D = cov_shrink.shape[0]
    cov_regularized = cov_shrink + epsilon * jnp.eye(D)

    return cov_regularized


def get_whitening_transform(cov):
    """
    Get whitening transform from covariance matrix.

    Uses Cholesky decomposition: x_white = L^{-1} @ x
    where cov = L @ L^T

    Args:
        cov: Covariance matrix [D, D]

    Returns:
        Cholesky factor L [D, D] such that cov = L @ L^T
    """
    try:
        L = jnp.linalg.cholesky(cov)
        return L
    except:
        # Fallback: use diagonal if Cholesky fails
        D = cov.shape[0]
        L = jnp.sqrt(jnp.diag(jnp.diag(cov))) + 1e-6
        return jnp.diag(L)


def whiten_data(data, chol):
    """
    Whiten data using Cholesky factor.

    x_white = L^{-1} @ x

    Args:
        data: Data to whiten [N, D]
        chol: Cholesky factor [D, D]

    Returns:
        Whitened data [N, D]
    """
    # Solve L @ x_white = data for x_white
    return jnp.linalg.solve(chol, data.T).T


def unwhiten_data(data_whitened, chol):
    """
    Unwhiten data using Cholesky factor.

    x = L @ x_white

    Args:
        data_whitened: Whitened data [N, D]
        chol: Cholesky factor [D, D]

    Returns:
        Unwhitened data [N, D]
    """
    return (chol @ data_whitened.T).T