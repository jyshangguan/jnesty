import numpy as np
import jax.numpy as jnp
import pytest
from math import erf, sqrt, log, pi

from jnesty import NestedSampler


def _logZ_analytical(ndim):
    Z_1d = (1.0 / 10.0) * sqrt(2.0 * pi) * erf(5.0 / sqrt(2.0))
    return ndim * log(Z_1d)


def _loglik(x):
    return -0.5 * jnp.sum(x ** 2)


def _prior_transform(u):
    return (u - 0.5) * 10.0


@pytest.mark.slow
class TestNestedSamplerAPI:
    def test_init_defaults(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=3, verbose=False)
        assert s.rwalk_K == 25  # max(25, 3+20)
        assert s.rwalk_step_scale == 1.0
        assert s.batch_size > 0

    def test_init_custom(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=3,
                          nlive=100, rwalk_K=30, bound='single', verbose=False)
        assert s.nlive == 100
        assert s.rwalk_K == 30
        assert s.bound == 'single'

    def test_results_before_run_raises(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=2, verbose=False)
        with pytest.raises(RuntimeError):
            _ = s.results

    def test_run_and_results(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=2,
                          nlive=100, rwalk_K=20, verbose=False)
        s.run_nested(max_iterations=5000, delta_logZ_threshold=0.1,
                     print_progress=False)
        r = s.results
        assert 'logz' in r
        assert 'samples' in r
        assert r['samples'].shape[1] == 2

    def test_logz_accuracy(self):
        s = NestedSampler(_loglik, _prior_transform, ndim=2,
                          nlive=100, rwalk_K=20, verbose=False)
        s.run_nested(max_iterations=5000, delta_logZ_threshold=0.1,
                     print_progress=False)
        true_logZ = _logZ_analytical(2)
        assert abs(s.results['logz'] - true_logZ) < 1.0

    def test_print_summary(self, capsys):
        s = NestedSampler(_loglik, _prior_transform, ndim=2,
                          nlive=100, rwalk_K=20, verbose=False)
        s.run_nested(max_iterations=5000, delta_logZ_threshold=0.1,
                     print_progress=False)
        s.print_summary()
        captured = capsys.readouterr()
        assert 'Log evidence' in captured.out
