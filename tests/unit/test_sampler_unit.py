import pytest
from jnesty.dynesty.core import WhileLoopNSConfig, WhileLoopNSResult


class TestWhileLoopNSConfig:
    def test_defaults(self):
        c = WhileLoopNSConfig()
        assert c.nlive == 500
        assert c.max_iterations == 10000
        assert c.delta_logZ_threshold == 0.01
        assert c.memory_frac == 0.9


class TestWhileLoopNSResult:
    def test_field_names(self):
        expected = [
            'logZ', 'logZ_error', 'H', 'delta_logZ', 'n_iterations',
            'runtime', 'samples', 'logL_samples', 'delta_logZ_trajectory',
            'scale_trajectory', 'acceptance_rate', 'live_x', 'live_logL',
            'samples_u', 'init_logX', 'init_iter_offset', 'ncall',
        ]
        assert list(WhileLoopNSResult._fields) == expected
