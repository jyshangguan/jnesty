"""Configuration resolution and validation for the DNest engine."""

import math

import numpy as np
import pytest

from jnesty.dfnest.state import (LEVEL_NUM_MAX, resolve_config,
                                 validate_sample_frequency)


class TestDerivedIntervals:
    """The derived controls must match CDNest's formulas exactly."""

    def test_default_single_worker(self):
        cfg = resolve_config(ndim=2)
        # thread_steps = int(ndim * tsf * K) = 2*10*1 = 20
        assert cfg.thread_steps == 20
        assert cfg.block_steps == 20
        # I_new = int(1 * 20 * 2) = 40; I_save inherits
        assert cfg.new_level_interval == 40
        assert cfg.save_interval == 40
        # R = int(40 * sqrt(10)) = 126
        assert cfg.regularisation == 126.0
        assert cfg.level_capacity == LEVEL_NUM_MAX == 1000
        assert cfg.total_particles == 1

    def test_multi_lane_particles(self):
        # CDNest: 5 params, 8 ranks, 2 particles, tsf 10 -> S = 100, G = 800
        cfg = resolve_config(ndim=5, num_lanes=8, num_particles=2)
        assert cfg.thread_steps == 100
        assert cfg.block_steps == 800
        assert cfg.new_level_interval == 1600
        assert cfg.save_interval == 1600
        assert cfg.regularisation == 1600 * math.sqrt(10) // 1
        assert cfg.total_particles == 16

    def test_truncation_matches_c_casts(self):
        # Non-integral products truncate toward zero like C's unsigned
        # assignment.
        cfg = resolve_config(ndim=3, thread_steps_factor=2.5,
                             new_level_interval_factor=1.7)
        assert cfg.thread_steps == int(3 * 2.5 * 1)
        assert cfg.new_level_interval == int(cfg.thread_steps * 1.7)
        assert cfg.regularisation == int(cfg.new_level_interval
                                         * math.sqrt(10.0))

    def test_save_interval_inherits_new_level_factor(self):
        cfg = resolve_config(ndim=2, new_level_interval_factor=5.0)
        # S = 20, G = 20, so I_new = int(20 * 5) = 100.
        assert cfg.save_interval == cfg.new_level_interval == 100
        cfg2 = resolve_config(ndim=2, new_level_interval_factor=5.0,
                              save_interval_factor=3.0)
        assert cfg2.save_interval == int(20 * 3.0)

    def test_fixed_levels_capacity(self):
        cfg = resolve_config(ndim=1, max_num_levels=37)
        assert cfg.level_capacity == 37
        assert cfg.max_num_levels == 37


class TestValidation:
    def test_bad_ndim(self):
        with pytest.raises(ValueError):
            resolve_config(ndim=0)
        with pytest.raises(TypeError):
            resolve_config(ndim=True)
        with pytest.raises(TypeError):
            resolve_config(ndim=2.0)

    def test_bad_lane_particles(self):
        with pytest.raises(ValueError):
            resolve_config(ndim=1, num_lanes=0)
        with pytest.raises(TypeError):
            resolve_config(ndim=1, num_particles=True)

    def test_new_level_interval_must_reach_block(self):
        # CDNest: new level interval >= totaltask * thread steps.
        with pytest.raises(ValueError):
            resolve_config(ndim=1, new_level_interval_factor=0.5)

    def test_compression_bounds(self):
        with pytest.raises(ValueError):
            resolve_config(ndim=1, compression=1.0)
        with pytest.raises(ValueError):
            resolve_config(ndim=1, compression=math.e * 1.01)
        with pytest.raises(ValueError):
            resolve_config(ndim=1, compression=0.5)

    def test_bad_ptol_lam_beta(self):
        with pytest.raises(ValueError):
            resolve_config(ndim=1, ptol=0.0)
        with pytest.raises(ValueError):
            resolve_config(ndim=1, lam=-1.0)
        with pytest.raises(ValueError):
            resolve_config(ndim=1, beta=float('inf'))

    def test_bad_save_interval(self):
        with pytest.raises(ValueError):
            resolve_config(ndim=1, save_interval_factor=0.0)


class TestSampleFrequency:
    def test_default_uniform(self):
        weights, raw = validate_sample_frequency(1.0, 3)
        assert np.allclose(weights, 1.0 / 3.0)
        assert np.allclose(raw, 1.0)

    def test_none_is_uniform(self):
        cfg = resolve_config(ndim=4)
        assert np.allclose(cfg.sample_frequency, 0.25)

    def test_scalar_broadcast(self):
        weights, _ = validate_sample_frequency(5, 2)
        assert np.allclose(weights, 0.5)

    def test_vector_normalised(self):
        weights, _ = validate_sample_frequency([1.0, 3.0], 2)
        assert np.allclose(weights, [0.25, 0.75])

    def test_scale_equivalence(self):
        w1, _ = validate_sample_frequency([2.0, 4.0], 2)
        w2, _ = validate_sample_frequency([1.0, 2.0], 2)
        assert np.array_equal(w1, w2)
        w3, _ = validate_sample_frequency(2, 3)
        w4, _ = validate_sample_frequency(7, 3)
        assert np.array_equal(w3, w4)

    def test_zero_weight_warns_but_allowed(self):
        with pytest.warns(UserWarning, match='frozen'):
            weights, _ = validate_sample_frequency([0.0, 1.0], 2)
        assert np.array_equal(weights, [0.0, 1.0])

    def test_all_zero_rejected(self):
        with pytest.raises(ValueError, match='positive sum'):
            validate_sample_frequency([0.0, 0.0], 2)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match='non-negative'):
            validate_sample_frequency([1.0, -0.1], 2)

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError, match='finite'):
            validate_sample_frequency([1.0, float('nan')], 2)
        with pytest.raises(ValueError, match='finite'):
            validate_sample_frequency([1.0, float('inf')], 2)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match='length'):
            validate_sample_frequency([1.0, 1.0], 3)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match='shape'):
            validate_sample_frequency([[1.0], [2.0]], 2)

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            resolve_config(ndim=1, sample_frequency=True)

    def test_integer_vector_accepted(self):
        weights, _ = validate_sample_frequency(np.array([1, 2, 3]), 3)
        assert np.allclose(weights, [1 / 6, 2 / 6, 3 / 6])
