"""
Shared utility functions for JNesty.

Contains low-level helpers used across multiple modules:
random ball sampling, log-arithmetic, numpy conversion.
"""

import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.special import gammaln


def randsphere(key, n):
    """
    Draw a point uniformly within an n-dimensional unit ball.

    Matches Dynesty's randsphere() implementation exactly.
    Formula: xhat = z * (U^(1/n) / ||z||)
    where z ~ N(0, I) and U ~ Uniform(0,1)
    """
    key1, key2 = random.split(key, 2)
    z = random.normal(key1, shape=(n,))
    z_norm = jnp.linalg.norm(z)
    z_norm_safe = jnp.where(z_norm > 0, z_norm, 1.0)
    u = random.uniform(key2)
    radius = u ** (1.0 / n)
    xhat = z * (radius / z_norm_safe)
    return xhat


def logsubexp(a, b):
    """Compute log(exp(a) - exp(b)) safely, assuming a > b."""
    return a + jnp.log1p(-jnp.exp(b - a))


def logvol_prefactor(n):
    """Log of volume constant for n-dimensional unit sphere."""
    return n * jnp.log(2.0) + n * gammaln(1.0 / 2.0 + 1.0) - gammaln(n / 2.0 + 1.0)


def _convert_to_numpy(arr):
    """Convert JAX arrays or other array-like to numpy."""
    if hasattr(arr, '__array__'):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        return np.asarray(arr)
