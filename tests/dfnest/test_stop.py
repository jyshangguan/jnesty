"""Opt-in delta-logZ stopping: estimator, validation, stop semantics.

Fast estimator/validation tests run without compiling any JAX program;
the engine integration tests at the bottom are slow-marked (JIT + chains),
matching ``test_engine.py`` conventions.
"""

import math
import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jnesty.dfnest import JaxDNestSampler
from jnesty.dfnest.engine import DELTA_LOGZ_MIN_SAVES
from jnesty.dfnest.levels import delta_log_z_estimate

jax.config.update('jax_enable_x64', True)


def _gaussian_problem(ndim=2):
    def loglikelihood(x):
        return -0.5 * jnp.sum(x ** 2)

    def prior_transform(u):
        return (u - 0.5) * 10.0

    return loglikelihood, prior_transform, ndim


# ---------------------------------------------------------------------
# estimator: synthetic level books
# ---------------------------------------------------------------------

class TestDeltaLogZEstimate:
    def test_shell_accounting_matches_closed_form(self):
        # logX = [0, -1, -2], thresholds [-inf, -3, -5]:
        # shell 2 volume log(e^-1 - e^-2) counted at logl = -3, tail
        # volume e^-2 counted at logl = -5.
        logX = np.array([0.0, -1.0, -2.0])
        logl = np.array([-math.inf, -3.0, -5.0])
        delta, logz = delta_log_z_estimate(logX, logl, logl_max=-2.0)
        expected = math.log(math.exp(-1.0) - math.exp(-2.0)) - 3.0
        expected = np.logaddexp(expected, -2.0 + -5.0)
        assert logz == pytest.approx(float(expected), rel=1e-12)
        # delta = log(1 + Z_tail_max / Z_running) with
        # Z_tail_max = exp(logl_max + logX_tail) = exp(-4).
        assert delta == pytest.approx(
            float(np.logaddexp(0.0, -2.0 + -2.0 - expected)), rel=1e-12)

    def test_first_shell_with_inf_threshold_drops_out(self):
        logX = np.array([0.0, -1.0])
        logl = np.array([-math.inf, -3.0])
        _, logz = delta_log_z_estimate(logX, logl, logl_max=-2.0)
        # Only the tail term e^-1 * e^-3 survives.
        assert logz == pytest.approx(-4.0, rel=1e-12)

    def test_delta_decreases_as_logz_grows(self):
        logX = np.array([0.0, -1.0, -2.0])
        deltas = []
        for shift in (0.0, 2.0, 5.0, 10.0):
            logl = np.array([-math.inf, -3.0 + shift, -5.0 + shift])
            delta, _ = delta_log_z_estimate(logX, logl, logl_max=-2.0)
            deltas.append(delta)
        assert all(d1 > d2 for d1, d2 in zip(deltas, deltas[1:]))

    def test_delta_decreases_as_tail_volume_shrinks(self):
        logl = np.array([-math.inf, -3.0, -5.0])
        deltas = []
        for logx_tail in (-2.0, -5.0, -20.0, -50.0):
            logX = np.array([0.0, -1.0, logx_tail])
            delta, _ = delta_log_z_estimate(logX, logl, logl_max=-2.0)
            deltas.append(delta)
        assert all(d1 > d2 for d1, d2 in zip(deltas, deltas[1:]))

    def test_delta_near_zero_when_tail_evidence_negligible(self):
        # logl_max + logX_tail = -60 while logZ_running ~ -4.
        logX = np.array([0.0, -1.0, -60.0])
        logl = np.array([-math.inf, -3.0, -5.0])
        delta, logz = delta_log_z_estimate(logX, logl, logl_max=0.0)
        assert logz == pytest.approx(-4.0, rel=1e-12)
        assert delta < 1e-20
        assert delta >= 0.0

    def test_single_level_book_never_stops(self):
        # Degenerate book: no usable shell, logZ_running = -inf -> +inf.
        delta, logz = delta_log_z_estimate(
            np.array([0.0]), np.array([-math.inf]), logl_max=-1.0)
        assert delta == math.inf
        assert logz == -math.inf

    def test_all_inf_likelihood_never_stops(self):
        logX = np.array([0.0, -1.0])
        logl = np.array([-math.inf, -math.inf])
        delta, _ = delta_log_z_estimate(logX, logl, logl_max=-math.inf)
        assert delta == math.inf

    def test_zero_volume_ties_are_tolerated(self):
        # Tied logX values give zero-volume shells (log1p(-1) = -inf
        # terms) which must simply drop out of the logaddexp reduction.
        logX = np.array([0.0, -1.0, -1.0])
        logl = np.array([-math.inf, -3.0, -3.5])
        delta, logz = delta_log_z_estimate(logX, logl, logl_max=-2.0)
        assert math.isfinite(logz)
        assert math.isfinite(delta)

    def test_input_validation(self):
        with pytest.raises(ValueError, match='1-D'):
            delta_log_z_estimate(np.zeros((2, 2)), np.zeros(2), 0.0)
        with pytest.raises(ValueError, match='equal length'):
            delta_log_z_estimate(np.zeros(3), np.zeros(2), 0.0)
        with pytest.raises(ValueError, match='at least one level'):
            delta_log_z_estimate(np.zeros(0), np.zeros(0), 0.0)


# ---------------------------------------------------------------------
# engine: run() argument validation (no sampling, no JIT)
# ---------------------------------------------------------------------

class TestRunValidation:
    def test_delta_logz_zero_and_negative_rejected(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2)
        for bad in (0.0, -0.05):
            with pytest.raises(ValueError, match='delta_logz'):
                sampler.run(max_num_saves=10, delta_logz=bad)

    def test_delta_logz_non_finite_rejected(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2)
        for bad in (math.inf, -math.inf, math.nan):
            with pytest.raises(ValueError, match='delta_logz'):
                sampler.run(max_num_saves=10, delta_logz=bad)

    def test_delta_logz_non_numeric_rejected(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2)
        with pytest.raises(TypeError, match='delta_logz'):
            sampler.run(max_num_saves=10, delta_logz='0.05')
        with pytest.raises(TypeError, match='delta_logz'):
            sampler.run(max_num_saves=10, delta_logz=object())
        # bool is a numeric type in Python but not a valid threshold.
        with pytest.raises(TypeError, match='delta_logz'):
            sampler.run(max_num_saves=10, delta_logz=True)

    def test_min_saves_floor_constant(self):
        # Raised from the plan's initial 100 after validation showed the
        # postprocessed logZ is still noisy at a few hundred saves (see
        # task_004_dnest_stop/validation.json).
        assert DELTA_LOGZ_MIN_SAVES == 800

    def test_bad_delta_logz_leaves_state_untouched(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2)
        with pytest.raises(ValueError):
            sampler.run(max_num_saves=10, delta_logz=0.0)
        # No sampling happened and the default path is still armed.
        assert sampler.count_saves == 0
        assert sampler.delta_logz_stopped is False
        assert sampler.last_delta_logz is None


# ---------------------------------------------------------------------
# engine: stop semantics (JIT + chains -> slow)
# ---------------------------------------------------------------------

@pytest.mark.slow
class TestStopSemantics:
    @pytest.fixture()
    def stopped_sampler(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  num_particles=1, seed=42)
        sampler.run(max_num_saves=1000, delta_logz=0.05)
        return sampler

    def test_stop_before_budget(self, stopped_sampler):
        s = stopped_sampler
        assert s.delta_logz_stopped is True
        assert s.completed is False
        assert s.count_saves < 1000
        # The floor, not the criterion, controls the earliest stop.
        assert s.count_saves >= DELTA_LOGZ_MIN_SAVES
        assert s.last_delta_logz is not None
        assert s.last_delta_logz < 0.05

    def test_stop_logz_close_to_full_budget(self, stopped_sampler):
        # 2-D Gaussian, prior [-5, 5]^2: analytic logZ = -2.7673.
        logz = stopped_sampler.native_result.logz
        assert abs(logz - (-2.767294266185362)) < 0.2

    def test_continuation_after_delta_stop(self, stopped_sampler):
        s = stopped_sampler
        saves_at_stop = s.count_saves
        logz = s.run(max_num_saves=2000)
        # The chain continued past the stop point and completed.
        assert s.count_saves > saves_at_stop
        assert s.delta_logz_stopped is False
        assert s.completed is True
        assert abs(logz - (-2.767294266185362)) < 0.2

    def test_tiny_budget_stops_at_budget_with_warning(self):
        # Levels can never finish here (PTol 1e-9 is unmeetable within
        # the budget), so the criterion is gated off and the budget cap
        # applies -- with CDNest's existing warning.  No hang.
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  num_particles=1, seed=42)
        with pytest.warns(UserWarning, match='level creation never'):
            logz = sampler.run(max_num_saves=150, ptol=1e-9,
                               delta_logz=0.05)
        assert sampler.delta_logz_stopped is False
        assert sampler.completed is True
        assert sampler.count_saves == 150
        assert math.isfinite(logz)

    def test_floor_blocks_stop_below_min_saves(self):
        # Levels finish around 66 saves for this problem, but the 100
        # save floor keeps the criterion from firing before the budget.
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  num_particles=1, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            logz = sampler.run(max_num_saves=80, delta_logz=0.05)
        assert sampler.delta_logz_stopped is False
        assert sampler.completed is True
        assert sampler.count_saves == 80
        assert math.isfinite(logz)

    def test_default_run_has_no_delta_state(self):
        loglike, prior, _ = _gaussian_problem()
        sampler = JaxDNestSampler(loglike, prior, ndim=2, num_lanes=4,
                                  num_particles=1, seed=42)
        sampler.run(max_num_saves=120)
        assert sampler.delta_logz_stopped is False
        assert sampler.last_delta_logz is None
        assert sampler.completed is True
