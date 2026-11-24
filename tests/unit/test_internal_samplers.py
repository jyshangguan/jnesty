import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import jax

from jnesty.dynesty.internal_samplers import (
    _propose_one, _single_walk, HSliceSampler, RWalkSampler, get_sampler,
)


class TestProposeOne:
    def test_shape(self, key):
        x = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(3)
        result = _propose_one(key, x, axes, 0.1, 3, 3, None, 0)
        assert result[0].shape == (3,)

    def test_no_axes(self, key):
        x = jnp.array([0.5, 0.5])
        result = _propose_one(key, x, None, 0.1, 2, 2, None, 0)
        assert result[0].shape == (2,)

    def test_walk_schedule(self, key):
        x = jnp.array([0.5, 0.5])
        axes = jnp.stack([jnp.eye(2), 2 * jnp.eye(2)])
        sched = jnp.array([0, 1, 0])
        # step_idx=1 should select axes[1] = 2*I
        result = _propose_one(key, x, axes, 0.1, 2, 2, sched, 1)
        assert result[0].shape == (2,)


class TestSingleWalk:
    def test_accepts_high_likelihood(self, key, gaussian_loglikelihood):
        x = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(3)
        x_new, n_acc, n_total = _single_walk(
            key, x, -jnp.inf, gaussian_loglikelihood,
            axes, 0.1, 20, 3, 3, None, None,
        )
        assert x_new.shape == (3,)
        assert jnp.all(x_new >= -1.0)
        assert int(n_acc) >= 0

    def test_respects_constraint(self, key, gaussian_loglikelihood):
        x = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(3)
        # Unreachable constraint
        x_new, n_acc, n_total = _single_walk(
            key, x, 1000.0, gaussian_loglikelihood,
            axes, 0.1, 20, 3, 3, None, None,
        )
        np.testing.assert_array_equal(x_new, x)
        assert int(n_acc) == 0

    def test_prior_bounds(self, key, gaussian_loglikelihood):
        x = jnp.array([0.5, 0.5])
        bounds = jnp.array([[0.45, 0.45], [0.55, 0.55]])
        x_new, n_acc, n_total = _single_walk(
            key, x, -jnp.inf, gaussian_loglikelihood,
            jnp.eye(2), 0.01, 20, 2, 2, bounds, None,
        )
        assert jnp.all(x_new >= 0.45)
        assert jnp.all(x_new <= 0.55)


class TestRWalkSampler:
    def test_sample_returns_tuple(self, key, gaussian_loglikelihood):
        s = RWalkSampler(ndim=3, batch_size=1)
        x = jnp.array([0.5, 0.5, 0.5])
        result = s.sample(key, x, -jnp.inf, gaussian_loglikelihood,
                          jnp.eye(3), 0.1, 20)
        assert len(result) == 4
        x_new, logL_new, n_acc, n_total = result
        assert x_new.shape == (3,)

    def test_sample_batch_mode(self, key, gaussian_loglikelihood):
        s = RWalkSampler(ndim=2, batch_size=4)
        x = jnp.full((4, 2), 0.5)
        axes = jnp.broadcast_to(jnp.eye(2), (4, 2, 2))
        x_new, logL_new, n_acc, n_total = s.sample(
            key, x, -jnp.inf, gaussian_loglikelihood,
            axes, 0.1, 20,
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


class TestRWalkDirectionWeights:
    """Opt-in per-coordinate sampling weights for the rwalk kernel."""

    def test_constructor_normalizes_to_mean_one(self):
        s = RWalkSampler(ndim=3, rwalk_direction_weights=[8.0, 2.0, 1.0])
        w = s.rwalk_direction_weights
        assert w.shape == (3,)
        np.testing.assert_allclose(float(jnp.mean(w)), 1.0, rtol=1e-6)
        expected = np.array([8.0, 2.0, 1.0]) / (11.0 / 3.0)
        np.testing.assert_allclose(np.asarray(w), expected, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(s.rwalk_direction_weights_sqrt),
                                   np.sqrt(expected), rtol=1e-6)

    def test_constructor_none_default(self):
        s = RWalkSampler(ndim=3)
        assert s.rwalk_direction_weights is None
        assert s.rwalk_direction_weights_sqrt is None

    @pytest.mark.parametrize("bad", [
        [0.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, float("nan"), 1.0],
        [1.0, float("inf"), 1.0],
        [1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    def test_constructor_rejects_invalid_weights(self, bad):
        with pytest.raises(ValueError, match="rwalk_direction_weights"):
            RWalkSampler(ndim=3, rwalk_direction_weights=bad)

    def test_get_sampler_accepts_weights(self):
        s = get_sampler('rwalk', ndim=3,
                        rwalk_direction_weights=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(float(jnp.mean(s.rwalk_direction_weights)),
                                   1.0, rtol=1e-6)

    def test_proposal_variance_scales_with_weight(self):
        # Displacement variance along coordinate i is w_i * Var(ball draw);
        # the ball draw has Var(dr_i) = 1/(ndim+2) in every coordinate.
        ndim = 3
        w_sq = jnp.sqrt(jnp.array([4.0, 1.0, 1.0]))
        x = jnp.full((ndim,), 0.5)
        keys = random.split(random.PRNGKey(123), 4000)
        props = jax.vmap(lambda k: _propose_one(
            k, x, jnp.eye(ndim), 1.0, ndim, ndim, None, 0, w_sq)[0])(keys)
        disp = np.asarray(props) - 0.5
        var = disp.var(axis=0)
        np.testing.assert_allclose(var[0] / var[1], 4.0, rtol=0.15)
        np.testing.assert_allclose(var[1] / var[2], 1.0, rtol=0.15)
        # zero-mean (symmetric) proposal in every coordinate
        assert np.all(np.abs(disp.mean(axis=0)) < 0.05)

    def test_proposal_weights_compose_after_axes(self):
        ndim = 3
        w_sq = jnp.sqrt(jnp.array([4.0, 1.0, 1.0]))
        x = jnp.full((ndim,), 0.5)
        axes = 2.0 * jnp.eye(ndim)
        k = random.PRNGKey(7)
        u_ref, _ = _propose_one(k, x, axes, 1.0, ndim, ndim, None, 0)
        u_w, _ = _propose_one(k, x, axes, 1.0, ndim, ndim, None, 0, w_sq)
        np.testing.assert_allclose(
            np.asarray(u_w - x),
            np.asarray(u_ref - x) * np.sqrt([4.0, 1.0, 1.0]),
            rtol=1e-5)

    def test_non_clustered_dims_ignore_weights(self):
        ndim = 3
        n_cluster = 2
        w_sq = jnp.sqrt(jnp.array([9.0, 1.0, 1.0]))
        x = jnp.full((ndim,), 0.5)
        k = random.PRNGKey(11)
        u_ref, _ = _propose_one(k, x, None, 1.0, n_cluster, ndim, None, 0)
        u_w, _ = _propose_one(k, x, None, 1.0, n_cluster, ndim, None, 0,
                              w_sq)
        # Non-clustered coordinate: same subkey, same shape -> bit-identical
        np.testing.assert_array_equal(np.asarray(u_w[2]), np.asarray(u_ref[2]))
        # Clustered coordinate 0 moved 3x further (sqrt(9))
        np.testing.assert_allclose(
            float(u_w[0] - x[0]), 3.0 * float(u_ref[0] - x[0]), rtol=1e-5)

    def test_uniform_sqrt_weights_are_bit_identical(self, key,
                                                     gaussian_loglikelihood):
        ndim = 3
        x = jnp.full((ndim,), 0.5)
        args = (key, x, -jnp.inf, gaussian_loglikelihood, jnp.eye(ndim),
                0.1, 25, ndim, ndim, None, None)
        x_none, _, _ = _single_walk(*args)
        x_ones, _, _ = _single_walk(*args,
                                    direction_weights_sqrt=jnp.ones(ndim))
        np.testing.assert_array_equal(np.asarray(x_none), np.asarray(x_ones))

    def test_single_walk_forwards_weights(self, key, gaussian_loglikelihood):
        ndim = 3
        x = jnp.full((ndim,), 0.5)
        w_sq = jnp.sqrt(jnp.array([9.0, 1.0, 1.0]))
        args = (key, x, -jnp.inf, gaussian_loglikelihood, jnp.eye(ndim),
                0.1, 25, ndim, ndim, None, None)
        x_none, _, n_tot_none = _single_walk(*args)
        x_w, _, n_tot_w = _single_walk(*args, direction_weights_sqrt=w_sq)
        assert not np.allclose(np.asarray(x_none), np.asarray(x_w))
        # The step budget is unchanged: weights redistribute the walk, they
        # do not add or remove proposals.
        assert int(n_tot_w) == int(n_tot_none)

    def test_sample_batch_mode_forwards_weights(self, key,
                                                gaussian_loglikelihood):
        # batch_size > 1 path: weights must reach the vmapped walks too.
        ndim = 2
        x = jnp.full((4, ndim), 0.5)
        axes = jnp.broadcast_to(jnp.eye(ndim), (4, ndim, ndim))
        s = RWalkSampler(ndim=ndim, batch_size=4,
                         rwalk_direction_weights=[9.0, 1.0])
        expected_sqrt = np.sqrt(np.array([9.0, 1.0]) / 5.0)
        assert s.rwalk_direction_weights_sqrt is not None
        np.testing.assert_allclose(
            np.asarray(s.rwalk_direction_weights_sqrt), expected_sqrt,
            rtol=1e-6)
        x_new, logL_new, n_acc, n_tot = s.sample(
            key, x, -jnp.inf, gaussian_loglikelihood, axes, 0.1, 20)
        assert x_new.shape == (ndim,)

    def test_stationary_distribution_is_weight_independent(self):
        # Correctness proof at the kernel level: the per-dim scaled ball is
        # still centrally symmetric, so the stay-put Metropolis walk has the
        # SAME invariant distribution (the likelihood-constraint region)
        # whatever the weights.  Long walks under a fixed constraint must
        # agree between weighted and unweighted kernels.
        ndim = 2

        def loglik(u):
            return -0.5 * jnp.sum(((u - 0.5) / 0.1) ** 2)

        keys = random.split(random.PRNGKey(0), 2048)
        x0 = jnp.full((ndim,), 0.5)

        def stationary(w_sq):
            xs, _, _ = jax.vmap(lambda k: _single_walk(
                k, x0, jnp.array(-2.0), loglik, None, 1.0, 2000, ndim, ndim,
                None, None, None, w_sq))(keys)
            return np.asarray(xs)

        base = stationary(None)
        weighted = stationary(jnp.sqrt(jnp.array([25.0, 1.0])))
        # Invariant region: disk of radius 0.2 around 0.5 -> marginal
        # mean 0.5, var 0.01 for BOTH weight settings.
        for xs in (base, weighted):
            np.testing.assert_allclose(xs.mean(axis=0), 0.5, atol=0.01)
            np.testing.assert_allclose(xs.var(axis=0), 0.01, rtol=0.15)
        # And the two settings agree with each other (the target is the
        # same; only the mixing speed differs).
        np.testing.assert_allclose(weighted.mean(axis=0), base.mean(axis=0),
                                   atol=0.01)
        np.testing.assert_allclose(weighted.var(axis=0), base.var(axis=0),
                                   rtol=0.15)


def _wrapped_gaussian_loglik(prior_transform):
    """Unit-cube loglikelihood: logL(prior_transform(u))."""
    def _loglik(u):
        x = prior_transform(u)
        return -0.5 * jnp.sum((x - 0.5) ** 2)
    return _loglik


class TestHSliceSampler:
    """Hamiltonian slice sampler (dynesty's 'hslice')."""

    def test_get_sampler_hslice(self):
        s = get_sampler('hslice', ndim=4)
        assert isinstance(s, HSliceSampler)
        assert s.ndim == 4
        assert s.batch_size == 1
        assert s.ncdim == 4

    def test_accepts_hslice_kwargs_and_ignores_unknown(self):
        s = get_sampler('hslice', ndim=3, target_acceptance=0.5,
                        batch_size=1, ncdim=2, max_move=17,
                        max_reflects=4, max_chords=32, max_contract=9,
                        grad_loglikelihood=None, some_unknown_kwarg=1)
        assert s.max_move == 17
        assert s.max_reflects == 4
        assert s.max_chords == 32
        assert s.max_contract == 9
        assert s.ncdim == 2

    def test_batch_size_must_be_one(self):
        with pytest.raises(ValueError, match="batch_size"):
            HSliceSampler(ndim=3, batch_size=2)

    def test_sample_returns_four_tuple(self, key, gaussian_loglikelihood):
        ndim = 3
        s = HSliceSampler(ndim=ndim)
        x0 = jnp.array([0.5, 0.5, 0.5])
        x_new, logL_new, n_acc, n_total = s.sample(
            key, x0, jnp.array(-1.0), gaussian_loglikelihood,
            jnp.eye(ndim) * 0.1, 5.0, 3)
        assert x_new.shape == (ndim,)
        assert jnp.ndim(logL_new) == 0
        assert jnp.issubdtype(jnp.asarray(x_new).dtype, jnp.floating)
        assert jnp.issubdtype(jnp.asarray(logL_new).dtype, jnp.floating)
        assert jnp.issubdtype(jnp.asarray(n_acc).dtype, jnp.integer)
        assert jnp.issubdtype(jnp.asarray(n_total).dtype, jnp.integer)

    def test_counters_nonnegative(self, key, gaussian_loglikelihood):
        ndim = 2
        s = HSliceSampler(ndim=ndim)
        x_new, logL_new, n_acc, n_total = s.sample(
            key, jnp.full(ndim, 0.5), jnp.array(-1.0), gaussian_loglikelihood,
            jnp.eye(ndim) * 0.1, 10.0, 2)
        assert int(n_total) > 0
        assert 0 <= int(n_acc) <= int(n_total)

    def test_unreachable_constraint_returns_start(self, key,
                                                 gaussian_loglikelihood):
        ndim = 3
        s = HSliceSampler(ndim=ndim)
        x0 = jnp.array([0.4, 0.5, 0.6])
        x_new, logL_new, n_acc, n_total = s.sample(
            key, x0, jnp.array(jnp.inf), gaussian_loglikelihood,
            jnp.eye(ndim) * 0.1, 5.0, 3)
        np.testing.assert_array_equal(np.asarray(x_new), np.asarray(x0))
        assert int(n_acc) == 0
        assert int(n_total) > 0
        assert bool(jnp.isfinite(logL_new))

    def test_jit_compiles_and_matches_eager(self, key, gaussian_loglikelihood):
        ndim = 3
        s = HSliceSampler(ndim=ndim)
        x0 = jnp.array([0.5, 0.5, 0.5])
        axes = jnp.eye(ndim) * 0.1
        eager = s.sample(key, x0, jnp.array(-1.0), gaussian_loglikelihood,
                         axes, 5.0, 3)
        # loglikelihood_fn and n_steps are static (Python) arguments.
        jitted = jax.jit(s.sample, static_argnums=(3, 6))
        out = jitted(key, x0, jnp.array(-1.0), gaussian_loglikelihood,
                     axes, 5.0, 3)
        for a, b in zip(eager, out):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                       rtol=0, atol=0)

    def test_reachable_constraint_and_unit_cube(self, key,
                                               gaussian_loglikelihood,
                                               simple_prior_transform):
        # Correlated 5-D Gaussian in physical space (simple_prior_transform
        # maps u -> (u - 0.5) * 10).
        ndim = 5
        loglik = _wrapped_gaussian_loglik(simple_prior_transform)
        s = HSliceSampler(ndim=ndim)
        x0 = jnp.full(ndim, 0.5)
        axes = jnp.eye(ndim) * 0.05
        logl_start = float(loglik(x0))
        constraint = logl_start - 0.5 * ndim  # comfortably reachable
        x_new, logL_new, n_acc, n_total = s.sample(
            key, x0, jnp.array(constraint), loglik, axes, 5.0, 5)
        assert float(logL_new) > constraint
        assert bool(jnp.all(x_new > 0.0)) and bool(jnp.all(x_new < 1.0))
        assert int(n_total) > 0
        assert int(n_acc) > 0

    def test_prior_bounds_respected(self, key, gaussian_loglikelihood):
        ndim = 2
        s = HSliceSampler(ndim=ndim)
        bounds = jnp.array([[0.4, 0.4], [0.6, 0.6]])
        x_new, logL_new, n_acc, n_total = s.sample(
            key, jnp.full(ndim, 0.5), jnp.array(-1.0), gaussian_loglikelihood,
            jnp.eye(ndim) * 0.05, 5.0, 3, prior_bounds=bounds)
        def inside(u):
            return np.all((np.asarray(u) >= 0.4) & (np.asarray(u) <= 0.6))
        # Either the point moved inside the prior bounds, or it stayed put.
        assert inside(x_new)

    def test_empty_slice_set_keeps_point(self, key, gaussian_loglikelihood):
        # n_steps=0 -> no slice at all, point unchanged, only the final eval.
        ndim = 2
        s = HSliceSampler(ndim=ndim)
        x0 = jnp.array([0.3, 0.7])
        x_new, logL_new, n_acc, n_total = s.sample(
            key, x0, jnp.array(-1.0), gaussian_loglikelihood,
            jnp.eye(ndim), 1.0, 0)
        np.testing.assert_array_equal(np.asarray(x_new), np.asarray(x0))
        assert int(n_acc) == 0
        assert int(n_total) == 1

    def test_tune_no_adapt_at_iteration_0(self):
        s = HSliceSampler(ndim=3)
        assert float(s.tune(1.0, 0.9, 3, 0)) == 1.0

    def test_tune_increases_scale_for_high_acceptance(self):
        s = HSliceSampler(ndim=3)
        assert float(s.tune(1.0, 0.9, 3, 5)) > 1.0

    def test_tune_decreases_scale_for_low_acceptance(self):
        s = HSliceSampler(ndim=3)
        assert float(s.tune(1.0, 0.1, 3, 5)) < 1.0

    def test_tune_uses_ncdim(self):
        s3 = HSliceSampler(ndim=3, ncdim=3)
        s9 = HSliceSampler(ndim=9, ncdim=9)
        r3 = float(s3.tune(1.0, 0.9, 3, 5))
        r9 = float(s9.tune(1.0, 0.9, 9, 5))
        assert 1.0 < r9 < r3  # smaller adaptation for more dimensions
