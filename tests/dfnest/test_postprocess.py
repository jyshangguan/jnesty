"""CDNest C-postprocess port: golden fixtures and edge cases."""

import math

import numpy as np
import pytest
from scipy.special import logsumexp

from jnesty.dfnest.postprocess import DNestResult, dnest_postprocess

GOLDEN_LEVELS_LOGX = np.array([0.0, -math.log(2.0), -math.log(4.0)])
GOLDEN_LEVELS_LOGL = np.array([-np.inf, -2.0, 0.0])
GOLDEN_LEVELS_TIE = np.array([0.42, 0.17, 0.91])

GOLDEN_ASSIGNMENT = np.array([0, 0, 0, 1, 0, 2, 99])
GOLDEN_LOGL = np.array([-3.0, -2.0, -1.0, -1.0, 1.0, 3.0, 2.0])
GOLDEN_TIE = np.array([0.1, 0.2, 0.8, 0.2, 0.7, 0.1, 0.5])


def _golden():
    return dnest_postprocess(GOLDEN_LEVELS_LOGX, GOLDEN_LEVELS_LOGL,
                             GOLDEN_LEVELS_TIE, GOLDEN_LOGL, GOLDEN_TIE,
                             GOLDEN_ASSIGNMENT)


class TestGoldenFixture:
    """A hand-checkable three-level fixture (values verified by hand)."""

    def test_sandwich_levels(self):
        result = _golden()
        # s0, s1 stay in level 0 (s1 equals the level-1 threshold: strict
        # comparison keeps it below); s2, s3 sandwich level 1; s4-s6 land
        # in level 2 (s6's assignment 99 clamps to 2).
        assert np.array_equal(result.sandwich, [0, 0, 1, 1, 2, 2, 2])

    def test_normalised_prior_masses_exact(self):
        result = _golden()
        expected = np.log(np.array([8, 8, 4, 4, 3, 3, 3]) / 33.0)
        np.testing.assert_allclose(result.logp, expected, atol=1e-12)
        assert abs(logsumexp(result.logp)) < 1e-12

    def test_logx_exact(self):
        result = _golden()
        expected = np.log([5 / 6, 2 / 3, 1 / 3, 5 / 12, 3 / 16, 1 / 16,
                           1 / 8])
        np.testing.assert_allclose(result.logx, expected, atol=1e-12)

    def test_evidence_information_ess(self):
        result = _golden()
        assert result.logz == pytest.approx(1.0573971075792052, abs=1e-11)
        assert result.information == pytest.approx(1.3315441610842516,
                                                   abs=1e-11)
        assert result.n_eff == pytest.approx(2.83530409375113, abs=1e-11)

    def test_posterior_weights_normalised(self):
        result = _golden()
        assert logsumexp(result.logwt) == pytest.approx(0.0, abs=1e-12)
        weights = np.exp(result.logwt)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    def test_unnormalised_identity(self):
        # logsumexp(logp + logl) == logz by construction.
        result = _golden()
        assert logsumexp(result.logp + GOLDEN_LOGL) == pytest.approx(
            result.logz, abs=1e-14)

    def test_relative_weights_cdnest_convention(self):
        result = _golden()
        relative = result.relative_weights
        assert relative.max() == pytest.approx(1.0)
        expected = np.exp(result.logwt - result.logwt.max())
        np.testing.assert_allclose(relative, expected)


class TestOrderStatisticEquivalence:
    """The simplified U formula must equal CDNest's literal expression."""

    def test_matches_literal_c_formula_for_n_above_one(self):
        rng = np.random.default_rng(0)
        for n_level in (2, 3, 7, 41):
            u_min = rng.uniform(0.0, 0.8)
            # Literal C: U = Umin + (1-Umin)*(1/(N+1) + (1-2/(N+1))*(N-1-j)/(N-1))
            literal = u_min + (1 - u_min) * (
                1.0 / (n_level + 1)
                + (1 - 2.0 / (n_level + 1)) * (n_level - 1
                                               - np.arange(n_level))
                / (n_level - 1.0))
            simplified = u_min + (1 - u_min) * (n_level
                                                - np.arange(n_level)) \
                / (n_level + 1.0)
            np.testing.assert_allclose(literal, simplified, rtol=0,
                                       atol=1e-15)


class TestEdgeCases:
    def test_singleton_level_midpoint(self):
        # The literal C formula evaluates 0*(0/0) here; the port uses the
        # algebraic limit, so logX stays finite while evidence/H are
        # unaffected.
        result = dnest_postprocess(
            np.array([0.0]), np.array([-np.inf]), np.array([0.5]),
            np.array([2.0]), np.array([0.3]), np.array([0]))
        assert result.logx[0] == pytest.approx(math.log(0.5))
        assert result.logz == pytest.approx(2.0)
        assert result.information == pytest.approx(0.0)
        assert result.n_eff == pytest.approx(1.0)

    def test_empty_intermediate_level(self):
        # No samples sandwich level 1; the global normalisation must not
        # be disturbed.
        result = dnest_postprocess(
            np.array([0.0, -math.log(2.0), -math.log(4.0)]),
            np.array([-np.inf, -2.0, 0.0]), np.array([0.1, 0.2, 0.3]),
            np.array([-1.0, 1.0]), np.array([0.5, 0.5]),
            np.array([0, 2]))
        assert np.isfinite(result.logz)
        assert logsumexp(result.logp) == pytest.approx(0.0, abs=1e-12)
        assert not np.any(np.isnan(result.logx))

    def test_ties_broken_by_tiebreaker(self):
        # Two samples with equal logL in one level: the smaller tiebreaker
        # sorts first and receives the larger volume.
        result = dnest_postprocess(
            np.array([0.0]), np.array([-np.inf]), np.array([0.5]),
            np.array([1.0, 1.0]), np.array([0.9, 0.1]),
            np.array([0, 0]))
        # Original order: sample 0 has tie 0.9, sample 1 tie 0.1.
        # Sorted ascending: (1.0, 0.1) then (1.0, 0.9).
        assert result.logx[1] > result.logx[0]
        assert result.logx[0] == pytest.approx(math.log(1 / 3))
        assert result.logx[1] == pytest.approx(math.log(2 / 3))

    def test_level_tiebreakers_ignored_for_sandwich(self):
        # Equal logL at the level boundary: the strict comparison keeps
        # the sample in the lower level regardless of level ties.
        a = dnest_postprocess(
            np.array([0.0, -math.log(2.0)]), np.array([-np.inf, -2.0]),
            np.array([0.9, 0.9]), np.array([-2.0]), np.array([0.5]),
            np.array([0]))
        b = dnest_postprocess(
            np.array([0.0, -math.log(2.0)]), np.array([-np.inf, -2.0]),
            np.array([0.1, 0.1]), np.array([-2.0]), np.array([0.5]),
            np.array([0]))
        assert a.sandwich[0] == b.sandwich[0] == 0

    def test_high_assignment_clamped(self):
        result = dnest_postprocess(
            np.array([0.0, -math.log(2.0)]), np.array([-np.inf, -5.0]),
            np.array([0.3, 0.4]), np.array([-4.0]), np.array([0.5]),
            np.array([17]))
        # Clamped to level 1; logl=-4 > -5 keeps it there.
        assert result.sandwich[0] == 1

    def test_temperature_divides_likelihoods(self):
        cold = _golden()
        hot = dnest_postprocess(
            GOLDEN_LEVELS_LOGX, GOLDEN_LEVELS_LOGL, GOLDEN_LEVELS_TIE,
            GOLDEN_LOGL, GOLDEN_TIE, GOLDEN_ASSIGNMENT, temperature=2.0)
        # Sandwiching uses untempered logL, so the prior masses agree.
        np.testing.assert_allclose(cold.logp, hot.logp, atol=1e-12)
        expected = logsumexp(cold.logp + GOLDEN_LOGL / 2.0)
        assert hot.logz == pytest.approx(expected, abs=1e-12)

    def test_inf_likelihood_allowed_for_zero_weight(self):
        # A -inf sample gets zero posterior weight without NaN poisoning.
        result = dnest_postprocess(
            np.array([0.0]), np.array([-np.inf]), np.array([0.5]),
            np.array([2.0, -np.inf]), np.array([0.2, 0.6]),
            np.array([0, 0]))
        assert np.isfinite(result.logz)
        # The -inf sample still carries half the normalised prior mass,
        # so Z = exp(2)/2.
        assert result.logz == pytest.approx(2.0 - math.log(2.0), abs=1e-12)
        weights = np.exp(result.logwt)
        assert weights[1] == pytest.approx(0.0, abs=1e-300)
        assert np.isfinite(result.information)
        assert np.isfinite(result.n_eff)

    def test_zero_width_level_zero_weights(self):
        # logX[0] == logX[1] -> Umin == 1 -> all samples in level 0 get
        # zero prior mass, carried by level 1.
        result = dnest_postprocess(
            np.array([0.0, 0.0, -math.log(4.0)]),
            np.array([-np.inf, -2.0, 0.0]), np.array([0.1, 0.2, 0.3]),
            np.array([-3.0, 1.0]), np.array([0.5, 0.5]),
            np.array([0, 1]))
        # logl=-3 sandwiches inside the zero-width level 0.
        assert result.sandwich[0] == 0
        assert result.logp[0] == -np.inf
        assert np.exp(result.logp[1]) == pytest.approx(1.0)


class TestInvalidInput:
    def test_nan_likelihood_rejected(self):
        with pytest.raises(ValueError, match='NaN'):
            dnest_postprocess(np.array([0.0]), np.array([-np.inf]),
                              np.array([0.5]), np.array([float('nan')]),
                              np.array([0.5]), np.array([0]))

    def test_posinf_likelihood_rejected(self):
        with pytest.raises(ValueError, match=r'\+inf'):
            dnest_postprocess(np.array([0.0]), np.array([-np.inf]),
                              np.array([0.5]), np.array([float('inf')]),
                              np.array([0.5]), np.array([0]))

    def test_negative_assignment_rejected(self):
        with pytest.raises(ValueError, match='non-negative'):
            dnest_postprocess(np.array([0.0]), np.array([-np.inf]),
                              np.array([0.5]), np.array([1.0]),
                              np.array([0.5]), np.array([-1]))

    def test_bad_temperature_rejected(self):
        with pytest.raises(ValueError, match='temperature'):
            dnest_postprocess(np.array([0.0]), np.array([-np.inf]),
                              np.array([0.5]), np.array([1.0]),
                              np.array([0.5]), np.array([0]),
                              temperature=0.5)

    def test_no_levels_rejected(self):
        with pytest.raises(ValueError, match='one level'):
            dnest_postprocess(np.zeros(0), np.zeros(0), np.zeros(0),
                              np.array([1.0]), np.array([0.5]),
                              np.array([0]))

    def test_no_samples_rejected(self):
        with pytest.raises(ValueError, match='one saved sample'):
            dnest_postprocess(np.array([0.0]), np.array([-np.inf]),
                              np.array([0.5]), np.zeros(0), np.zeros(0),
                              np.zeros(0))

    def test_unordered_thresholds_rejected(self):
        with pytest.raises(ValueError, match='non-decreasing'):
            dnest_postprocess(np.array([0.0, -1.0]),
                              np.array([-2.0, -5.0]),
                              np.array([0.1, 0.2]), np.array([1.0]),
                              np.array([0.5]), np.array([1]))

    def test_increasing_logx_rejected(self):
        with pytest.raises(ValueError, match='non-increasing'):
            dnest_postprocess(np.array([0.0, 0.5]),
                              np.array([-np.inf, -5.0]),
                              np.array([0.1, 0.2]), np.array([1.0]),
                              np.array([0.5]), np.array([1]))

    def test_all_zero_mass_rejected(self):
        # Every level has zero width -> no finite prior mass anywhere.
        with pytest.raises(ValueError, match='no finite prior mass'):
            dnest_postprocess(np.array([0.0, 0.0, 0.0]),
                              np.array([-np.inf, -4.0, -2.0]),
                              np.array([0.1, 0.2, 0.3]), np.array([-3.0]),
                              np.array([0.5]), np.array([0]))


def test_result_dataclass_defaults():
    result = DNestResult(level_logX=np.zeros(1),
                         level_logl=np.zeros(1), level_tie=np.zeros(1))
    assert result.logz != result.logz  # NaN sentinel
    with pytest.raises(RuntimeError):
        _ = result.relative_weights
