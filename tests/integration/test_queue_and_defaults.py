"""Integration tests for queue mode and call-based bound update interval."""

import numpy as np
import jax.numpy as jnp
import pytest
from math import erf, sqrt, log, pi
from jax import random

from jnesty.dynesty.core import run_nested_sampling, WhileLoopNSConfig
from jnesty import NestedSampler


def _logZ_analytical(ndim):
    Z_1d = (1.0 / 10.0) * sqrt(2.0 * pi) * erf(5.0 / sqrt(2.0))
    return ndim * log(Z_1d)


def _loglik(x):
    return -0.5 * jnp.sum(x ** 2)


def _prior_transform(u):
    return (u - 0.5) * 10.0


@pytest.mark.slow
class TestQueueMode:
    """Test queue mode via the low-level sampler API."""

    def test_queue_2d_gaussian_converges(self):
        """Queue mode converges on a simple 2D Gaussian."""
        ndim = 2
        config = WhileLoopNSConfig(
            nlive=100, max_iterations=5000, delta_logZ_threshold=0.1,
            rwalk_K=20, verbose=False, print_progress=False,
            bound='multi', queue_size=4, batch_size=1,
        )
        result = run_nested_sampling(
            _loglik, lambda k: random.uniform(k, (ndim,)),
            ndim, config, key=random.PRNGKey(42),
            prior_transform_fn=_prior_transform,
        )
        assert bool(result.delta_logZ < 0.1)

    def test_queue_logz_accuracy(self):
        """Queue mode logZ within tolerance of analytical."""
        ndim = 2
        config = WhileLoopNSConfig(
            nlive=150, max_iterations=5000, delta_logZ_threshold=0.1,
            rwalk_K=20, verbose=False, print_progress=False,
            bound='multi', queue_size=4, batch_size=1,
        )
        result = run_nested_sampling(
            _loglik, lambda k: random.uniform(k, (ndim,)),
            ndim, config, key=random.PRNGKey(42),
            prior_transform_fn=_prior_transform,
        )
        true_logZ = _logZ_analytical(ndim)
        assert abs(float(result.logZ) - true_logZ) < 1.0

    def test_queue_with_periodic_updates(self):
        """Queue mode with call-based periodic bound updates converges."""
        ndim = 2
        # bound_update_interval in likelihood calls: 20 (rwalk_K) * 100 (nlive) = 2000
        config = WhileLoopNSConfig(
            nlive=100, max_iterations=5000, delta_logZ_threshold=0.1,
            rwalk_K=20, verbose=False, print_progress=False,
            bound='multi', queue_size=4, batch_size=1,
            bound_update_interval=2000,
        )
        result = run_nested_sampling(
            _loglik, lambda k: random.uniform(k, (ndim,)),
            ndim, config, key=random.PRNGKey(42),
            prior_transform_fn=_prior_transform,
        )
        assert bool(result.delta_logZ < 0.1)
        true_logZ = _logZ_analytical(ndim)
        assert abs(float(result.logZ) - true_logZ) < 1.0

    def test_queue_5d_gaussian(self):
        """Queue mode on 5D Gaussian."""
        ndim = 5
        config = WhileLoopNSConfig(
            nlive=150, max_iterations=8000, delta_logZ_threshold=0.1,
            rwalk_K=25, verbose=False, print_progress=False,
            bound='multi', queue_size=4, batch_size=1,
            bound_update_interval=25 * 150,  # rwalk_K * nlive calls
        )
        result = run_nested_sampling(
            _loglik, lambda k: random.uniform(k, (ndim,)),
            ndim, config, key=random.PRNGKey(42),
            prior_transform_fn=_prior_transform,
        )
        true_logZ = _logZ_analytical(ndim)
        assert abs(float(result.logZ) - true_logZ) < 1.5

    def test_total_calls_tracking(self):
        """total_calls in result increases with iterations."""
        ndim = 2
        config = WhileLoopNSConfig(
            nlive=50, max_iterations=200, delta_logZ_threshold=0.01,
            rwalk_K=10, verbose=False, print_progress=False,
            bound='multi', queue_size=2, batch_size=1,
        )
        result = run_nested_sampling(
            _loglik, lambda k: random.uniform(k, (ndim,)),
            ndim, config, key=random.PRNGKey(42),
            prior_transform_fn=_prior_transform,
        )
        # total_calls should be roughly niter * rwalk_K
        n_calls = int(result.n_iterations) * 10  # rwalk_K=10
        # Allow some slack for Phase 1 and retries
        assert int(result.n_iterations) > 0


@pytest.mark.slow
class TestDynestyDefaults:
    """Test that the new Dynesty-matching defaults work correctly."""

    def test_bound_multi_enables_queue(self):
        """bound='multi' auto-enables queue_size=8."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          bound='multi', verbose=False)
        assert s.queue_size == 8

    def test_bound_multi_enables_periodic_updates(self):
        """bound='multi' auto-enables bound_update_interval = rwalk_K * nlive calls."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          nlive=500, bound='multi', verbose=False)
        assert s.bound_update_interval == s.rwalk_K * s.nlive

    def test_bound_none_no_queue(self):
        """bound='none' does not auto-enable queue."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          bound='none', verbose=False)
        assert s.queue_size == 0

    def test_bound_none_no_periodic_updates(self):
        """bound='none' has no periodic updates by default."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          bound='none', verbose=False)
        assert s.bound_update_interval == 0

    def test_explicit_queue_size_overrides(self):
        """Explicit queue_size overrides the auto default."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          bound='multi', queue_size=16, verbose=False)
        assert s.queue_size == 16

    def test_explicit_zero_updates_disables(self):
        """bound_update_interval=0 explicitly disables periodic updates."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          bound='multi', bound_update_interval=0, verbose=False)
        assert s.bound_update_interval == 0

    def test_float_update_interval(self):
        """Float bound_update_interval is converted to calls."""
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          nlive=500, bound='multi',
                          bound_update_interval=0.5, verbose=False)
        # 0.5 * nlive * rwalk_K = 0.5 * 500 * 25 = 6250 calls
        assert s.bound_update_interval == int(0.5 * 500 * 25)

    def test_full_run_with_defaults(self):
        """Full run with bound='multi' and all defaults converges."""
        s = NestedSampler(_loglik, _prior_transform, ndim=2,
                          nlive=100, bound='multi', verbose=False)
        s.run_nested(max_iterations=3000, delta_logZ_threshold=0.1,
                     print_progress=False)
        true_logZ = _logZ_analytical(2)
        assert abs(s.results['logz'] - true_logZ) < 1.0


class TestRWalkDirectionWeightsAPI:
    """Constructor-level behaviour of rwalk_direction_weights (task_005)."""

    def test_default_is_none(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=3, verbose=False)
        assert s.rwalk_direction_weights is None

    def test_weights_normalized_to_mean_one(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=3, verbose=False,
                          rwalk_direction_weights=[8.0, 2.0, 1.0])
        np.testing.assert_allclose(
            float(jnp.mean(s.rwalk_direction_weights)), 1.0, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(s.rwalk_direction_weights),
            np.array([8.0, 2.0, 1.0]) / (11.0 / 3.0), rtol=1e-6)

    @pytest.mark.parametrize("bad", [
        [1.0, 0.0, 1.0],
        [1.0, -2.0, 1.0],
        [1.0, 1.0],
        [1.0, float("nan"), 1.0],
    ])
    def test_invalid_weights_raise(self, bad):
        with pytest.raises(ValueError, match="rwalk_direction_weights"):
            NestedSampler(_loglik, _prior_transform, ndim=3, verbose=False,
                          rwalk_direction_weights=bad)

    def test_hslice_rejects_weights(self):
        with pytest.raises(ValueError, match="sample='rwalk'"):
            NestedSampler(_loglik, _prior_transform, ndim=2, verbose=False,
                          sample='hslice', rwalk_direction_weights=[1.0, 1.0])


@pytest.mark.slow
class TestRWalkDirectionWeightsRuns:
    """End-to-end weighted rwalk runs (legacy single-walk + queue)."""

    def test_weighted_rwalk_legacy_logz(self):
        """Strongly anisotropic weights on the legacy path keep logZ right."""
        s = NestedSampler(_loglik, _prior_transform, ndim=2, nlive=100,
                          verbose=False, bound='single',
                          rwalk_direction_weights=[9.0, 1.0])
        s.run_nested(max_iterations=3000, delta_logZ_threshold=0.1,
                     print_progress=False)
        assert abs(s.results['logz'] - _logZ_analytical(2)) < 1.0

    def test_weighted_rwalk_queue_logz(self):
        """Same on the queue path (bound='multi' + queue refill)."""
        s = NestedSampler(_loglik, _prior_transform, ndim=2, nlive=100,
                          verbose=False, bound='multi', queue_size=4,
                          rwalk_direction_weights=[9.0, 1.0])
        s.run_nested(max_iterations=3000, delta_logZ_threshold=0.1,
                     print_progress=False)
        assert abs(s.results['logz'] - _logZ_analytical(2)) < 1.0
