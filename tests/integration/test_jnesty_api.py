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


@pytest.mark.slow
class TestNautilusSamplerAPI:
    """Real-engine coverage of the NautilusSampler adapter (JAX nautilus)."""

    @pytest.fixture(autouse=True, scope='class')
    @classmethod
    def _engine_available(cls):
        pytest.importorskip('jnesty.nautilus')

    @staticmethod
    def _make_sampler():
        from jnesty import NautilusSampler
        return NautilusSampler(
            _loglik, _prior_transform, ndim=2, nlive=100, n_batch=50,
            n_networks=0, seed=0, device='cpu')

    def test_init_maps_to_native_engine(self):
        sampler = self._make_sampler()
        native = sampler.native_sampler
        assert native.n_dim == 2
        assert native.n_live == 100
        assert native.prior is _prior_transform
        assert native.likelihood is _loglik

    def test_run_and_accuracy(self):
        sampler = self._make_sampler()
        assert sampler.run(f_live=0.2, n_eff=0) is True
        assert abs(sampler.log_z - _logZ_analytical(2)) < 0.1

        points, log_w, log_l = sampler.posterior()
        assert len(points) == len(log_w) == len(log_l)
        assert np.exp(log_w).sum() == pytest.approx(1.0, abs=1e-9)
        assert sampler.n_like > 0
        assert sampler.n_eff > 0
