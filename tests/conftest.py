import pytest
import numpy as np
import jax.numpy as jnp
from jax import random


@pytest.fixture
def key():
    return random.PRNGKey(42)


@pytest.fixture
def unit_points_2d():
    return random.uniform(random.PRNGKey(7), shape=(50, 2))


@pytest.fixture
def unit_points_3d():
    return random.uniform(random.PRNGKey(7), shape=(100, 3))


@pytest.fixture
def gaussian_points_2d():
    k = random.PRNGKey(99)
    pts = 0.5 + 0.1 * random.normal(k, shape=(100, 2))
    return jnp.clip(pts, 0.0, 1.0)


@pytest.fixture
def gaussian_loglikelihood():
    def _loglik(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2)
    return _loglik


@pytest.fixture
def simple_gaussian_loglikelihood():
    def _loglik(x):
        return -0.5 * jnp.sum(x ** 2)
    return _loglik


@pytest.fixture
def simple_prior_transform():
    def _pt(u):
        return (u - 0.5) * 10.0
    return _pt
