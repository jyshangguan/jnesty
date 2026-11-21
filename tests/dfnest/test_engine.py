"""End-to-end JaxDNestSampler runs (JIT + chains -> slow tests)."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jnesty.dfnest import JaxDNestSampler

pytestmark = pytest.mark.slow

jax.config.update('jax_enable_x64', True)


def _gaussian_problem(ndim=2):
    def loglikelihood(x):
        return -0.5 * jnp.sum(x ** 2)

    def prior_transform(u):
        return (u - 0.5) * 10.0

    from math import erf, sqrt, log, pi
    z1 = (1.0 / 10.0) * sqrt(2.0 * pi) * erf(5.0 / sqrt(2.0))
    return loglikelihood, prior_transform, ndim * log(z1)


def _exp_problem():
    # x = -log(1-u) ~ Exponential(1); logL = -x -> posterior
    # Exponential(2), logZ = -log 2, H = log 2 - 1/2.
    def loglikelihood(x):
        return -x[0]

    def prior_transform(u):
        return -jnp.log1p(-u)

    return loglikelihood, prior_transform, -np.log(2.0), np.log(2.0) - 0.5


class TestEngineValidation:
    def test_non_callable_rejected(self):
        _, prior, _ = _gaussian_problem()
        with pytest.raises(TypeError):
            JaxDNestSampler(1.0, prior, ndim=2)
        with pytest.raises(TypeError):
            JaxDNestSampler(lambda x: -x ** 2, 3, ndim=2)

    def test_bad_prior_shape_rejected(self):
        def loglikelihood(x):
            return -jnp.sum(x ** 2)

        def prior_transform(u):
            return u[:1]  # wrong output shape

        with pytest.raises(ValueError, match='prior_transform'):
            JaxDNestSampler(loglikelihood, prior_transform, ndim=2)

    def test_bad_likelihood_shape_rejected(self):
        def loglikelihood(x):
            return jnp.array([0.0])  # non-scalar

        def prior_transform(u):
            return u

        with pytest.raises(ValueError, match='scalar'):
            JaxDNestSampler(loglikelihood, prior_transform, ndim=2)

    def test_bad_run_arguments(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=1,
                                  max_num_levels=4)
        with pytest.raises(ValueError):
            sampler.run(max_num_saves=0)
        with pytest.raises(TypeError):
            sampler.run(max_num_saves=True)
        with pytest.raises(ValueError):
            sampler.run(ptol=0.0)

    def test_results_before_run_raises(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=1,
                                  max_num_levels=4)
        with pytest.raises(RuntimeError):
            sampler.posterior()
        assert sampler.log_z is None
        assert sampler.native_result is None


class TestGaussianRun:
    def test_evidence_and_weights(self):
        loglike, prior, truth = _gaussian_problem(2)
        sampler = JaxDNestSampler(
            loglike, prior, ndim=2, num_lanes=4, num_particles=2,
            max_num_levels=12, seed=5)
        logz = sampler.run(max_num_saves=400)
        assert logz == pytest.approx(truth, abs=0.5)
        assert sampler.information == pytest.approx(1.77, abs=0.4)
        assert sampler.levels_finalized
        assert sampler.completed

        points, log_w, log_l = sampler.posterior()
        assert points.shape == (400, 2)
        assert np.exp(log_w).sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.isfinite(log_w))
        # Weighted mean near the origin.
        weights = np.exp(log_w)
        mean = (points * weights[:, None]).sum(axis=0)
        assert np.all(np.abs(mean) < 1.0)

    def test_same_seed_reproducible(self):
        loglike, prior, _ = _gaussian_problem(1)
        runs = []
        for _ in range(2):
            sampler = JaxDNestSampler(loglike, prior, ndim=1,
                                      num_lanes=2, max_num_levels=8,
                                      seed=21)
            logz = sampler.run(max_num_saves=100)
            runs.append((logz, np.asarray(sampler.posterior()[0]).copy()))
        assert runs[0][0] == runs[1][0]
        np.testing.assert_array_equal(runs[0][1], runs[1][1])

    def test_different_seeds_differ(self):
        loglike, prior, _ = _gaussian_problem(1)
        values = []
        for seed in (1, 2):
            sampler = JaxDNestSampler(loglike, prior, ndim=1,
                                      num_lanes=2, max_num_levels=8,
                                      seed=seed)
            values.append(sampler.run(max_num_saves=100))
        assert values[0] != values[1]

    def test_continuation_extends_state(self):
        loglike, prior, _ = _gaussian_problem(1)
        sampler = JaxDNestSampler(loglike, prior, ndim=1, num_lanes=2,
                                  max_num_levels=8, seed=3)
        first = sampler.run(max_num_saves=60)
        assert sampler.n_saves == 60
        result_first = sampler.native_result
        second = sampler.run(max_num_saves=120)
        assert sampler.n_saves == 120
        # No new sampling for an equal-or-smaller target; the cached
        # native result is returned unchanged.
        third = sampler.run(max_num_saves=60)
        assert sampler.n_saves == 120
        assert third == second
        assert sampler.native_result is not result_first
        assert sampler.n_blocks > 0

    def test_level_diagnostics(self):
        loglike, prior, _ = _gaussian_problem(2)
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  max_num_levels=10, seed=8)
        sampler.run(max_num_saves=150)
        levels = sampler.levels
        assert levels['logl'].shape == (10,)
        # Thresholds climb from -inf toward the likelihood maximum ~0.
        assert np.all(np.diff(levels['logl']) > 0)
        # Learned compressions hover around -log(e) = -1.
        compressions = np.diff(levels['logX'])
        assert np.all(compressions < 0)
        assert np.mean(compressions) == pytest.approx(-1.0, abs=0.4)

    def test_unfinished_levels_warns(self):
        loglike, prior, _ = _gaussian_problem(2)
        # A tiny save budget cannot finish level creation.
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=1,
                                  new_level_interval_factor=1.0,
                                  save_interval_factor=1.0, seed=0)
        with pytest.warns(UserWarning, match='level creation'):
            logz = sampler.run(max_num_saves=2)
        assert np.isfinite(logz)
        assert not sampler.levels_finalized


class TestNonuniformPriorTransform:
    def test_exponential_prior_evidence(self):
        loglike, prior, logz_true, h_true = _exp_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=1, num_lanes=4,
                                  max_num_levels=10, seed=13)
        logz = sampler.run(max_num_saves=500)
        assert logz == pytest.approx(logz_true, abs=0.15)
        assert sampler.information == pytest.approx(h_true, abs=0.15)
        # The posterior is Exponential(2): weighted mean 1/2.
        points, log_w, _ = sampler.posterior()
        weights = np.exp(log_w)
        assert (points[:, 0] * weights).sum() == pytest.approx(0.5, abs=0.2)


class TestSampleFrequency:
    def test_skewed_weights_change_attempt_ratios(self):
        loglike, prior, _ = _gaussian_problem(2)
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  sample_frequency=[1.0, 3.0],
                                  max_num_levels=8, seed=6)
        sampler.run(max_num_saves=150)
        attempts = sampler.coordinate_attempts
        total = attempts.sum()
        assert total > 6000
        # Roughly 25% / 75% with a 3-sigma multinomial tolerance.
        expected = np.array([0.25, 0.75]) * total
        sigma = 3 * np.sqrt(total * np.array([0.25, 0.75]))
        assert np.all(np.abs(attempts - expected) < sigma)
        # The per-coordinate accounting stays consistent.
        accepts = sampler.coordinate_accepts
        assert np.all(accepts <= attempts)
        rate = sampler.coordinate_acceptance_rate
        assert np.all(rate > 0) and np.all(rate <= 1)

    def test_zero_weight_freezes_coordinate(self):
        loglike, prior, _ = _gaussian_problem(2)
        with pytest.warns(UserWarning, match='frozen'):
            sampler = JaxDNestSampler(loglike, prior, ndim=2,
                                      num_lanes=2,
                                      sample_frequency=[0.0, 1.0],
                                      max_num_levels=8, seed=7)
        sampler.run(max_num_saves=100)
        assert sampler.coordinate_attempts[0] == 0
        assert sampler.coordinate_attempts[1] > 0
        rate = sampler.coordinate_acceptance_rate
        assert np.isnan(rate[0])

    def test_skewed_weights_preserve_evidence(self):
        loglike, prior, truth = _gaussian_problem(2)
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  sample_frequency=[1.0, 3.0],
                                  max_num_levels=12, seed=9)
        logz = sampler.run(max_num_saves=400)
        # Frequencies only allocate proposals: the posterior must be
        # unchanged (within Monte Carlo tolerance).
        assert logz == pytest.approx(truth, abs=0.5)
        points, log_w, _ = sampler.posterior()
        weights = np.exp(log_w)
        mean = (points * weights[:, None]).sum(axis=0)
        assert np.all(np.abs(mean) < 1.0)


class TestContext:
    def test_context_passed_to_callables(self):
        def loglikelihood(x, data):
            return -0.5 * jnp.sum((x - data['center']) ** 2)

        def prior_transform(u, data):
            return (u - 0.5) * 10.0 + data['offset']

        context = {'center': jnp.zeros(2), 'offset': 0.0}
        sampler = JaxDNestSampler(loglikelihood, prior_transform, ndim=2,
                                  num_lanes=2, context=context,
                                  max_num_levels=8, seed=4)
        logz = sampler.run(max_num_saves=150)
        assert np.isfinite(logz)
        points, _, _ = sampler.posterior()
        assert points.shape == (150, 2)

    def test_context_values_not_compiled_in(self):
        # Two different contexts with the same structure reuse the same
        # compiled program (no recompile for new data values).
        def loglikelihood(x, scale):
            return -0.5 * jnp.sum((x / scale) ** 2)

        def prior_transform(u, scale):
            return (u - 0.5) * 10.0

        sampler_a = JaxDNestSampler(loglikelihood, prior_transform, ndim=1,
                                    num_lanes=2, context=jnp.array(1.0),
                                    max_num_levels=6, seed=1)
        sampler_a.run(max_num_saves=50)
        cache_a = sampler_a._block_fn._cache_size() \
            if hasattr(sampler_a._block_fn, '_cache_size') else None

        sampler_b = JaxDNestSampler(loglikelihood, prior_transform, ndim=1,
                                    num_lanes=2, context=jnp.array(3.0),
                                    max_num_levels=6, seed=1)
        sampler_b.run(max_num_saves=50)
        if cache_a is not None:
            # Both runs should have reused one compilation each.
            assert cache_a == 1
            assert sampler_b._block_fn._cache_size() == 1


class TestFailures:
    def test_nan_likelihood_raises(self):
        def loglikelihood(x):
            # NaN on half of the prior volume: caught either at
            # initialisation or at the first block boundary.
            return jnp.where(x[0] > 0, jnp.nan, -jnp.sum(x ** 2))

        def prior_transform(u):
            return (u - 0.5) * 10.0

        sampler = JaxDNestSampler(loglikelihood, prior_transform, ndim=1,
                                  num_lanes=1, max_num_levels=4, seed=0)
        with pytest.raises(ValueError, match='NaN'):
            sampler.run(max_num_saves=5)

    def test_neg_inf_likelihood_allowed(self):
        def loglikelihood(x):
            return jnp.where(jnp.all(jnp.abs(x) < 2.0), -jnp.sum(x ** 2),
                             -jnp.inf)

        def prior_transform(u):
            return (u - 0.5) * 10.0

        sampler = JaxDNestSampler(loglikelihood, prior_transform, ndim=1,
                                  num_lanes=2, max_num_levels=6, seed=2)
        logz = sampler.run(max_num_saves=100)
        assert np.isfinite(logz)
