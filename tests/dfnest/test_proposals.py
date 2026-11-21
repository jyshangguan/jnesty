"""Device-side proposal kernels: exactness against CDNest reference loops."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import random

from jnesty.dfnest import proposals

jax.config.update('jax_enable_x64', True)


class TestWrapUnit:
    def test_wraps_into_unit_interval(self):
        values = jnp.array([0.5, 1.0, 1.5, -0.5, -1.25, 12.75, -7.0])
        wrapped = jax.vmap(proposals.wrap_unit)(values)
        for value, result in zip(np.asarray(values), np.asarray(wrapped)):
            expected = value - np.floor(value)
            assert result == pytest.approx(expected)
            assert 0.0 <= result < 1.0


class TestRandh:
    def test_finite_and_mostly_small(self):
        key = random.PRNGKey(0)
        draws = jax.vmap(proposals.randh)(random.split(key, 5000))
        draws = np.asarray(draws)
        assert np.all(np.isfinite(draws))
        # Median |h| is small (the typical scale is ~10^-1.5 .. 1), but
        # the tails reach both huge and tiny values.
        assert np.median(np.abs(draws)) < 1.0
        assert np.quantile(np.abs(draws), 0.99) > 1e-3
        assert np.any(np.abs(draws) > 1.0)
        assert np.any(np.abs(draws) < 1e-6)

    def test_symmetric_sign(self):
        key = random.PRNGKey(3)
        draws = np.asarray(jax.vmap(proposals.randh)(random.split(key, 4000)))
        # The normal factor makes the sign symmetric.
        assert abs(np.mean(np.sign(draws))) < 0.1


class TestSelectCoordinate:
    def _draw(self, weights, n, seed):
        key = random.PRNGKey(seed)
        keys = random.split(key, n)
        weights = jnp.asarray(weights, dtype=jnp.float64)
        indices = jax.vmap(proposals.select_coordinate,
                           in_axes=(0, None))(keys, weights)
        return np.asarray(indices)

    def test_uniform_default(self):
        counts = np.bincount(self._draw([0.5, 0.5], 6000, 0), minlength=2)
        assert counts.sum() == 6000
        # A loose 3-sigma multinomial check.
        expected = 3000.0
        sigma = np.sqrt(6000 * 0.25)
        assert abs(counts[0] - expected) < 3 * sigma

    def test_weighted_selection_matches_weights(self):
        counts = np.bincount(self._draw([0.25, 0.75], 6000, 1), minlength=2)
        p = counts / counts.sum()
        np.testing.assert_allclose(p, [0.25, 0.75], atol=0.03)

    def test_zero_weight_never_selected(self):
        counts = np.bincount(self._draw([0.0, 1.0, 0.0], 2000, 2),
                             minlength=3)
        assert counts[0] == 0
        assert counts[2] == 0
        assert counts[1] == 2000

    def test_scale_equivalent_weights_identical(self):
        a = self._draw([1.0, 3.0], 500, 5)
        b = self._draw([2.0, 6.0], 500, 5)
        assert np.array_equal(a, b)

    def test_all_coordinates_reachable(self):
        counts = np.bincount(self._draw(np.full(5, 0.2), 5000, 7),
                             minlength=5)
        assert np.all(counts > 0)


class TestCountThresholdsBelow:
    def test_matches_numpy_count(self):
        rng = np.random.default_rng(0)
        capacity = 32
        for _ in range(50):
            n_levels = int(rng.integers(1, capacity + 1))
            thresholds = np.sort(rng.uniform(-5, 5, n_levels))
            thresholds[0] = -np.inf
            full = np.concatenate([thresholds,
                                   np.full(capacity - n_levels, np.nan)])
            for _ in range(5):
                logl = rng.uniform(-6, 6)
                m = proposals.count_thresholds_below(
                    jnp.asarray(full), jnp.asarray(logl),
                    jnp.asarray(n_levels, dtype=jnp.int32))
                expected = np.count_nonzero(
                    thresholds[:n_levels] < logl)
                assert int(m) == expected

    def test_inf_likelihood_counts_all(self):
        thresholds = jnp.array([0.0, 1.0, 2.0, np.nan, np.nan])
        m = proposals.count_thresholds_below(thresholds, jnp.asarray(10.0),
                                             jnp.int32(3))
        assert int(m) == 3

    def test_neg_inf_likelihood_counts_none(self):
        thresholds = jnp.array([-np.inf, 1.0, 2.0])
        m = proposals.count_thresholds_below(thresholds,
                                             jnp.asarray(-np.inf),
                                             jnp.int32(3))
        assert int(m) == 0


def _reference_visits_exceeds(level, logl, thresholds, n_levels,
                              capacity=16):
    """A direct transcription of CDNest's early-break loop."""
    visits = np.zeros(capacity, dtype=np.int64)
    exceeds = np.zeros(capacity, dtype=np.int64)
    j = int(level)
    while j < n_levels - 1:
        visits[j] += 1
        if thresholds[j + 1] < logl:
            exceeds[j] += 1
        else:
            break
        j += 1
    return visits, exceeds


class TestVisitsExceedsIncrements:
    """Fuzz the range-mask formulation against the C loop."""

    def test_matches_reference_loop(self):
        rng = np.random.default_rng(42)
        capacity = 16
        for _ in range(400):
            n_levels = int(rng.integers(1, capacity + 1))
            thresholds = np.sort(rng.uniform(-4, 4, n_levels))
            thresholds[0] = -np.inf
            full = np.concatenate([thresholds,
                                   np.full(capacity - n_levels, np.nan)])
            level = int(rng.integers(0, n_levels))
            logl = rng.uniform(-5, 5)
            # The particle invariant: its likelihood exceeds its level's
            # threshold (levels[level] < logl).
            logl = max(logl, thresholds[level] + 1e-9) \
                if np.isfinite(thresholds[level]) else logl

            v_inc, e_inc = proposals.visits_exceeds_increments(
                jnp.asarray(level), jnp.asarray(logl),
                jnp.asarray(full), jnp.asarray(n_levels, dtype=jnp.int32))
            ref_v, ref_e = _reference_visits_exceeds(
                level, logl, thresholds, n_levels, capacity)
            np.testing.assert_array_equal(np.asarray(v_inc), ref_v)
            np.testing.assert_array_equal(np.asarray(e_inc), ref_e)

    def test_top_level_particle_no_increments(self):
        thresholds = jnp.array([0.0, 1.0, 2.0, np.nan])
        v_inc, e_inc = proposals.visits_exceeds_increments(
            jnp.asarray(2), jnp.asarray(10.0), thresholds, jnp.int32(3))
        assert not np.any(np.asarray(v_inc))
        assert not np.any(np.asarray(e_inc))

    def test_single_level_no_increments(self):
        thresholds = jnp.array([-np.inf] + [np.nan] * 15)
        v_inc, e_inc = proposals.visits_exceeds_increments(
            jnp.asarray(0), jnp.asarray(1.0), thresholds, jnp.int32(1))
        assert not np.any(np.asarray(v_inc))
        assert not np.any(np.asarray(e_inc))

    def test_full_exceeds_chain(self):
        # The likelihood clears every threshold: the loop runs from level 1
        # up to n_levels - 2 inclusive.
        thresholds = jnp.array([0.0, 1.0, 2.0, 3.0] + [np.nan] * 12)
        v_inc, e_inc = proposals.visits_exceeds_increments(
            jnp.asarray(1), jnp.asarray(10.0), thresholds, jnp.int32(4))
        v = np.asarray(v_inc)
        e = np.asarray(e_inc)
        assert list(v[:4]) == [0, 1, 1, 0]
        assert list(e[:4]) == [0, 1, 1, 0]
        assert not np.any(v[4:]) and not np.any(e[4:])


def _make_state(logl, level, capacity=8, ndim=1, num_particles=1):
    num_particles = max(num_particles, len(logl))
    u = np.arange(num_particles * ndim, dtype=np.float64).reshape(
        num_particles, ndim) / (num_particles * ndim + 1.0)
    from jnesty.dfnest.state import LaneState
    return LaneState(
        u=jnp.asarray(u),
        logl=jnp.asarray(np.asarray(logl, dtype=float)[:num_particles]),
        tie=jnp.full((num_particles,), 0.5),
        level=jnp.asarray(np.asarray(level, dtype=int)[:num_particles],
                          dtype=jnp.int32),
        accepts=jnp.zeros(capacity, dtype=jnp.int64),
        tries=jnp.zeros(capacity, dtype=jnp.int64),
        visits=jnp.zeros(capacity, dtype=jnp.int64),
        exceeds=jnp.zeros(capacity, dtype=jnp.int64),
        coord_attempts=jnp.zeros(ndim, dtype=jnp.int64),
        coord_accepts=jnp.zeros(ndim, dtype=jnp.int64))


class TestUpdateParticle:
    def _run(self, state, which, thresholds, n_levels, logl_prop, seed=0,
             coord=0, row_new=None):
        key = random.PRNGKey(seed)
        row_new = (jnp.asarray(state.u[which]) if row_new is None
                   else jnp.asarray(row_new))
        return proposals.update_particle(
            state, jnp.asarray(which), jnp.asarray(coord), row_new,
            jnp.asarray(logl_prop), jnp.asarray(0.25),
            jnp.asarray(thresholds), jnp.asarray(n_levels, jnp.int32),
            key)

    def test_accept_above_threshold_updates_everything(self):
        thresholds = jnp.array([0.0, 1.0, 2.0] + [np.nan] * 5)
        state = _make_state([5.0], [1])
        out = self._run(state, 0, thresholds, 3, logl_prop=2.5, coord=0,
                        row_new=[0.7])
        # Accepted: logl and u updated, counters at level 1, and the
        # visits/exceeds chain from level 1 (next threshold 2.0 < 2.5).
        assert float(out.logl[0]) == 2.5
        assert float(out.u[0, 0]) == 0.7
        assert int(out.accepts[1]) == 1
        assert int(out.tries[1]) == 1
        assert int(out.coord_attempts[0]) == 1
        assert int(out.coord_accepts[0]) == 1
        # Visits/exceeds chain from level 1: threshold[2]=2.0 < 2.5, but
        # the loop bound j < n_levels-1 = 2 stops it there.
        assert list(np.asarray(out.visits)[:3]) == [0, 1, 0]
        assert list(np.asarray(out.exceeds)[:3]) == [0, 1, 0]

    def test_reject_below_threshold(self):
        thresholds = jnp.array([0.0, 1.0, 2.0] + [np.nan] * 5)
        state = _make_state([1.5], [1])
        out = self._run(state, 0, thresholds, 3, logl_prop=0.5,
                        row_new=[0.7])
        # Rejected: state unchanged, tries counted, accepts not.
        assert float(out.logl[0]) == 1.5
        assert float(out.u[0, 0]) != 0.7
        assert int(out.accepts[1]) == 0
        assert int(out.tries[1]) == 1
        # Visits still run with the final (old) likelihood: from level 1,
        # next threshold 2.0 is not exceeded -> one visit, no exceeds.
        assert list(np.asarray(out.visits)[:3]) == [0, 1, 0]
        assert list(np.asarray(out.exceeds)[:3]) == [0, 0, 0]

    def test_nonselected_particles_untouched(self):
        thresholds = jnp.array([0.0, 1.0] + [np.nan] * 6)
        state = _make_state([5.0, 5.0], [0, 1])
        out = self._run(state, 1, thresholds, 2, logl_prop=1.5,
                        row_new=[0.9])
        assert float(out.logl[0]) == 5.0
        assert int(out.accepts[0]) == 0


class TestUpdateLevel:
    def test_proposal_formula_reproduced(self):
        # With flat logX, flat push, levels done and equal tries, the
        # acceptance ratio is 0 and every eligible move is accepted, so
        # the final level must equal the CDNest proposal formula.
        capacity = 16
        n_levels = 4
        thresholds = jnp.array([-np.inf, -2.0, -1.0, 0.0]
                               + [np.nan] * (capacity - 4))
        log_x = jnp.zeros(capacity)
        push = jnp.zeros(capacity)
        tries = jnp.full(capacity, 7, dtype=jnp.int64)
        state = _make_state([0.5], [2])._replace(tries=tries)

        key = random.PRNGKey(123)
        keys = random.split(key, 4)
        current = 2
        scale = jnp.power(10.0, 2.0 * random.uniform(keys[0]))
        jump = jnp.int32(jnp.trunc(scale * random.normal(keys[1])))
        prop_raw = current + jump
        bump = jnp.where(random.uniform(keys[2]) < 0.5, -1, 1)
        prop_raw = jnp.where(prop_raw == current, current + bump, prop_raw)
        expected = jnp.mod(prop_raw, n_levels)

        out = proposals.update_level(
            state, jnp.asarray(0), thresholds, log_x, push,
            jnp.asarray(n_levels, jnp.int32), jnp.asarray(True), 100.0,
            keys)
        assert int(out.level[0]) == int(expected)

    def test_rejects_ineligible_levels(self):
        # beta -> -inf makes logA -> -inf, so acceptance is decided by the
        # eligibility comparison alone: an ineligible target level never
        # wins. Use a huge tries ratio instead (beta * log ratio).
        capacity = 8
        thresholds = jnp.array([-np.inf, -2.0, -1.0, 0.0, np.nan, np.nan,
                               np.nan, np.nan])
        # Particle at level 0 with logl -1.5: eligible levels are 0 and 1
        # (threshold < -1.5 only for -inf and -2.0).
        state = _make_state([-1.5], [0])
        out = proposals.update_level(
            state, jnp.asarray(0), thresholds, jnp.zeros(capacity),
            jnp.zeros(capacity), jnp.asarray(4, jnp.int32),
            jnp.asarray(True), 100.0, random.split(random.PRNGKey(7), 4))
        assert int(out.level[0]) in (0, 1)
