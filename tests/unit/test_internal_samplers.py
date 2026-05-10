import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jnesty.internal_samplers import (
    _propose_one, _single_walk, RWalkSampler, get_sampler,
)


class TestProposeOne:
    def test_shape(self, key):
        x = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(3)
        result = _propose_one(key, x, axes, 0.1, 3, None, 0)
        assert result.shape == (3,)

    def test_no_axes(self, key):
        x = jnp.array([0.5, 0.5])
        result = _propose_one(key, x, None, 0.1, 2, None, 0)
        assert result.shape == (2,)

    def test_walk_schedule(self, key):
        x = jnp.array([0.5, 0.5])
        axes = jnp.stack([jnp.eye(2), 2 * jnp.eye(2)])
        sched = jnp.array([0, 1, 0])
        # step_idx=1 should select axes[1] = 2*I
        result = _propose_one(key, x, axes, 0.1, 2, sched, 1)
        assert result.shape == (2,)


class TestSingleWalk:
    def test_accepts_high_likelihood(self, key, gaussian_loglikelihood):
        x = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(3)
        x_new, n_acc = _single_walk(
            key, x, -jnp.inf, gaussian_loglikelihood,
            axes, 0.1, 20, 3, None, None,
        )
        assert x_new.shape == (3,)
        assert jnp.all(x_new >= -1.0)
        assert int(n_acc) >= 0

    def test_respects_constraint(self, key, gaussian_loglikelihood):
        x = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(3)
        # Unreachable constraint
        x_new, n_acc = _single_walk(
            key, x, 1000.0, gaussian_loglikelihood,
            axes, 0.1, 20, 3, None, None,
        )
        np.testing.assert_array_equal(x_new, x)
        assert int(n_acc) == 0

    def test_prior_bounds(self, key, gaussian_loglikelihood):
        x = jnp.array([0.5, 0.5])
        bounds = jnp.array([[0.45, 0.45], [0.55, 0.55]])
        x_new, _ = _single_walk(
            key, x, -jnp.inf, gaussian_loglikelihood,
            jnp.eye(2), 0.01, 20, 2, bounds, None,
        )
        assert jnp.all(x_new >= 0.45)
        assert jnp.all(x_new <= 0.55)


class TestRWalkSampler:
    def test_sample_returns_tuple(self, key, gaussian_loglikelihood):
        s = RWalkSampler(ndim=3, batch_size=1)
        x = jnp.array([0.5, 0.5, 0.5])
        result = s.sample(key, x, -jnp.inf, gaussian_loglikelihood,
                          jnp.eye(3), 0.1, 20)
        assert len(result) == 3
        x_new, logL_new, n_acc = result
        assert x_new.shape == (3,)

    def test_sample_batch_mode(self, key, gaussian_loglikelihood):
        s = RWalkSampler(ndim=2, batch_size=4)
        x = jnp.array([0.5, 0.5])
        x_new, logL_new, n_acc = s.sample(
            key, x, -jnp.inf, gaussian_loglikelihood,
            jnp.eye(2), 0.1, 20,
        )
        assert x_new.shape == (2,)

    def test_tune_no_adapt_at_iteration_0(self):
        s = RWalkSampler(ndim=3)
        result = float(s.tune(1.0, 0.2, 3, 0))
        assert result == 1.0

    def test_tune_increases_scale_for_high_acceptance(self):
        s = RWalkSampler(ndim=3)
        result = float(s.tune(1.0, 0.8, 3, 5))
        assert result > 1.0

    def test_tune_decreases_scale_for_low_acceptance(self):
        s = RWalkSampler(ndim=3)
        result = float(s.tune(1.0, 0.2, 3, 5))
        assert result < 1.0


class TestFactory:
    def test_get_sampler_rwalk(self):
        s = get_sampler('rwalk', ndim=5)
        assert isinstance(s, RWalkSampler)
        assert s.ndim == 5

    def test_get_sampler_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            get_sampler('unknown', ndim=3)
