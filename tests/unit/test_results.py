import numpy as np
import jax.numpy as jnp
import pytest
from jnesty.results import Results, format_results, save_results, load_results, _SCALAR_KEYS
from jnesty.sampler import WhileLoopNSResult


def _make_raw_result(n_dead=100, ndim=2, nlive=50, converged=True):
    samples = np.random.randn(n_dead, ndim)
    logL = -np.cumsum(np.random.rand(n_dead))
    delta_logZ_val = 0.001 if converged else 1.0
    return WhileLoopNSResult(
        logZ=-5.0,
        logZ_error=0.1,
        H=2.0,
        delta_logZ=delta_logZ_val,
        n_iterations=n_dead,
        runtime=1.0,
        samples=jnp.array(samples),
        logL_samples=jnp.array(logL),
        delta_logZ_trajectory=jnp.full(n_dead, delta_logZ_val),
        scale_trajectory=jnp.ones(n_dead),
        acceptance_rate=0.5,
        live_x=jnp.array(np.random.rand(nlive, ndim)),
        live_logL=jnp.array(-np.cumsum(np.random.rand(nlive))),
    )


class TestResultsClass:
    def test_dict_access(self):
        r = Results({'logz': -5.0, 'logzerr': 0.1})
        assert r['logz'] == -5.0

    def test_attr_access(self):
        r = Results({'logz': -5.0, 'logzerr': 0.1})
        assert r.logz == -5.0

    def test_missing_key_raises(self):
        r = Results({'logz': -5.0})
        with pytest.raises(KeyError):
            _ = r['nonexistent']

    def test_missing_attr_raises(self):
        r = Results({'logz': -5.0})
        with pytest.raises(AttributeError):
            _ = r.nonexistent

    def test_contains(self):
        r = Results({'logz': -5.0})
        assert 'logz' in r
        assert 'missing' not in r

    def test_keys_values_items(self):
        r = Results({'a': 1, 'b': 2})
        assert set(r.keys()) == {'a', 'b'}
        assert set(r.values()) == {1, 2}

    def test_get_default(self):
        r = Results({'a': 1})
        assert r.get('missing', 42) == 42
        assert r.get('a') == 1


class TestFormatResults:
    def test_required_keys(self):
        raw = _make_raw_result()
        prior_transform = lambda u: u
        r = format_results(raw, prior_transform, ndim=2, nlive=50, rwalk_K=25,
                           delta_logZ_threshold=0.01)
        required = ['logz', 'logzerr', 'logwt', 'logl', 'samples', 'samples_u',
                     'logvol', 'niter', 'nlive', 'converged', 'delta_logz',
                     'information', 'eff', 'acceptance_rate']
        for key in required:
            assert key in r, f"Missing key: {key}"

    def test_sample_shapes(self):
        n_dead, ndim, nlive = 100, 2, 50
        raw = _make_raw_result(n_dead=n_dead, ndim=ndim, nlive=nlive)
        r = format_results(raw, lambda u: u, ndim=ndim, nlive=nlive,
                           rwalk_K=25, delta_logZ_threshold=0.01)
        assert r['samples'].shape == (n_dead + nlive, ndim)
        assert r['logl'].shape == (n_dead + nlive,)

    def test_converged_flag(self):
        raw = _make_raw_result(converged=True)
        r = format_results(raw, lambda u: u, ndim=2, nlive=50,
                           rwalk_K=25, delta_logZ_threshold=0.01)
        assert r['converged'] is True

        raw2 = _make_raw_result(converged=False)
        r2 = format_results(raw2, lambda u: u, ndim=2, nlive=50,
                            rwalk_K=25, delta_logZ_threshold=0.01)
        assert r2['converged'] is False

    def test_efficiency(self):
        n_dead = 100
        raw = _make_raw_result(n_dead=n_dead)
        r = format_results(raw, lambda u: u, ndim=2, nlive=50,
                           rwalk_K=25, delta_logZ_threshold=0.01)
        expected_eff = 100.0 * n_dead / (n_dead * 25)
        assert abs(r['eff'] - expected_eff) < 1e-10

    def test_logvol_monotone(self):
        raw = _make_raw_result()
        r = format_results(raw, lambda u: u, ndim=2, nlive=50,
                           rwalk_K=25, delta_logZ_threshold=0.01)
        logvol = r['logvol']
        for i in range(len(logvol) - 1):
            assert logvol[i + 1] <= logvol[i]


class TestFITSIO:
    def test_roundtrip(self, tmp_path):
        raw = _make_raw_result()
        r = format_results(raw, lambda u: u, ndim=2, nlive=50,
                           rwalk_K=25, delta_logZ_threshold=0.01)

        path = str(tmp_path / 'test.fits')
        save_results(r, path)
        loaded = load_results(path)

        assert abs(loaded['logz'] - r['logz']) < 1e-10
        assert abs(loaded['logzerr'] - r['logzerr']) < 1e-10
        np.testing.assert_allclose(loaded['logl'], r['logl'], atol=1e-10)

    def test_loaded_has_required_keys(self, tmp_path):
        raw = _make_raw_result()
        r = format_results(raw, lambda u: u, ndim=2, nlive=50,
                           rwalk_K=25, delta_logZ_threshold=0.01)

        path = str(tmp_path / 'test.fits')
        save_results(r, path)
        loaded = load_results(path)

        for key, _, _ in _SCALAR_KEYS:
            assert key in loaded, f"Missing scalar key after load: {key}"

        assert 'logl' in loaded
        assert 'logwt' in loaded
        assert 'logvol' in loaded
        assert 'samples' in loaded
