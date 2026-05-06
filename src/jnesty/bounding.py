"""
Ellipsoid bounding for nested sampling.

This module provides functions to fit ellipsoids to point clouds
and sample uniformly from within them.
"""

import jax
import jax.numpy as jnp
from jax import random
import jax.scipy as jsp


def fit_ellipsoid(points):
    """
    Fit ellipsoid to points using covariance matrix.

    Args:
        points: (n, ndim) array of points

    Returns:
        center: (ndim,) center of ellipsoid
        axes: (ndim, ndim) transformation matrix
        radii: (ndim,) eigenvalues (squared radii)
    """
    ndim = points.shape[1]

    # Compute center
    center = jnp.mean(points, axis=0)

    # Compute covariance matrix
    centered = points - center
    cov = jnp.cov(centered.T)

    # Eigenvalue decomposition
    eigenvals, eigenvecs = jnp.linalg.eigh(cov)

    # Ensure positive eigenvalues
    eigenvals = jnp.maximum(eigenvals, 1e-10)

    # Transform matrix: eigenvecs @ diag(sqrt(eigenvals))
    axes = eigenvecs @ jnp.diag(jnp.sqrt(eigenvals))

    return center, axes, eigenvals


def sample_ellipsoid(key, center, axes, scale):
    """
    Sample uniformly from n-dimensional ellipsoid.

    Strategy:
    1. Sample point on unit n-sphere
    2. Sample radius^2 ~ Uniform(0, 1)
    3. Transform by axes and scale

    Args:
        key: JAX random key
        center: (ndim,) center of ellipsoid
        axes: (ndim, ndim) transformation matrix
        scale: scale factor for ellipsoid

    Returns:
        point: (ndim,) sampled point within ellipsoid
    """
    ndim = len(center)
    key1, key2 = random.split(key)

    # Sample direction on unit sphere
    direction = random.normal(key1, shape=(ndim,))
    direction /= jnp.linalg.norm(direction)

    # Sample radius for uniform distribution within ball
    # radius^2 ~ Uniform(0, 1) => radius ~ U^(1/ndim)
    radius = jnp.power(random.uniform(key2), 1.0 / ndim)

    # Transform: center + scale * radius * (axes @ direction)
    point = center + scale * radius * (axes @ direction)

    return point


def sample_ellipsoid_batch(key, center, axes, scale, n_samples):
    """
    Sample multiple points from ellipsoid (vectorized).

    Args:
        key: JAX random key
        center: (ndim,) center of ellipsoid
        axes: (ndim, ndim) transformation matrix
        scale: scale factor for ellipsoid
        n_samples: number of samples to generate

    Returns:
        points: (n_samples, ndim) sampled points
    """
    keys = random.split(key, n_samples)
    samples = jnp.stack([sample_ellipsoid(k, center, axes, scale) for k in keys])
    return samples


# Test function
def _test_ellipsoid():
    """Test ellipsoid fitting and sampling."""
    import matplotlib.pyplot as plt

    # Create test data: 2D Gaussian cloud
    key = random.PRNGKey(42)
    points = random.normal(key, shape=(100, 2)) * 0.5 + jnp.array([1.0, 2.0])

    # Fit ellipsoid
    center, axes, radii = fit_ellipsoid(points)

    print(f"Center: {center}")
    print(f"Radii: {radii}")

    # Sample from ellipsoid
    key, subkey = random.split(key)
    samples = sample_ellipsoid_batch(subkey, center, axes, 1.0, 1000)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, label='Original points')
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, label='Ellipsoid samples')
    ax.scatter(center[0], center[1], s=100, c='red', marker='x', label='Center')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Ellipsoid Fitting and Sampling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/test_ellipsoid.png')
    print("Saved: /tmp/test_ellipsoid.png")


if __name__ == "__main__":
    _test_ellipsoid()
