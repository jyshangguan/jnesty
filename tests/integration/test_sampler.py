import numpy as np
import jax.numpy as jnp
import pytest
from jax import random
from math import erf, sqrt, log, pi

from jnesty.sampler import run_nested_sampling, WhileLoopNSConfig


def _logZ_analytical(ndim):
    Z_1d = (1.0 / 10.0) * sqrt(2.0 * pi) * erf(5.0 / sqrt(2.0))
    return ndim * log(Z_1d)


def _run_gaussian(ndim, nlive, **config_overrides):
    def loglik(x):
        return -0.5 * jnp.sum(x ** 2)

    def prior_sample(key):
        return random.uniform(key, shape=(ndim,))

    def prior_transform(u):
        return (u - 0.5) * 10.0

    config = WhileLoopNSConfig(
        nlive=nlive,
        max_iterations=10000,
        delta_logZ_threshold=0.1,
        rwalk_K=20,
        verbose=False,
        print_progress=False,
        **config_overrides,
    )
    return run_nested_sampling(
        loglik, prior_sample, ndim, config,
        key=random.PRNGKey(42), prior_transform_fn=prior_transform,
    )


@pytest.mark.slow
class TestFullSampler:
    def test_2d_gaussian_converges(self):
        r = _run_gaussian(ndim=2, nlive=100)
        assert bool(r.delta_logZ < 0.1)

    def test_2d_gaussian_logz(self):
        r = _run_gaussian(ndim=2, nlive=100)
        true_logZ = _logZ_analytical(2)
        assert abs(float(r.logZ) - true_logZ) < 1.0

    def test_3d_gaussian_logz(self):
        r = _run_gaussian(ndim=3, nlive=150)
        true_logZ = _logZ_analytical(3)
        assert abs(float(r.logZ) - true_logZ) < 1.0

    def test_single_ellipsoid(self):
        r = _run_gaussian(ndim=2, nlive=100, bound='single')
        assert bool(r.delta_logZ < 0.2)

    def test_multi_ellipsoid(self):
        r = _run_gaussian(ndim=2, nlive=100, bound='multi',
                          bound_update_interval=50)
        assert bool(r.delta_logZ < 0.2)
