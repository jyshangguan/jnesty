import numpy as np
import jax.numpy as jnp
import pytest
import matplotlib
matplotlib.use('Agg')

from jnesty.plotting import _convert_to_numpy, _get_weights, _downsample, _get_plot_data


class TestConvertToNumpy:
    def test_jax(self):
        result = _convert_to_numpy(jnp.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_list(self):
        result = _convert_to_numpy([3, 4, 5])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [3, 4, 5])


class TestGetWeights:
    def test_normalizes(self):
        r = {'logwt': np.array([0.0, -1.0, -2.0])}
        w = _get_weights(r)
        assert abs(w.sum() - 1.0) < 1e-10
        assert np.all(w > 0)

    def test_missing_returns_none(self):
        assert _get_weights({}) is None

    def test_empty_returns_none(self):
        assert _get_weights({'logwt': np.array([])}) is None


class TestDownsample:
    def test_noop(self):
        samples = np.random.randn(100, 2)
        out, w = _downsample(samples, max_samples=200)
        assert out is samples

    def test_reduces_count(self):
        samples = np.random.randn(5000, 2)
        out, _ = _downsample(samples, max_samples=100)
        assert len(out) <= 100

    def test_weighted(self):
        samples = np.random.randn(5000, 2)
        weights = np.random.rand(5000)
        weights /= weights.sum()
        out, w = _downsample(samples, weights, max_samples=100)
        assert abs(w.sum() - 1.0) < 1e-10


class TestGetPlotData:
    def test_extracts_core(self):
        r = {
            'samples': np.random.randn(50, 3),
            'logl': -np.random.rand(50),
            'logwt': -np.random.rand(50),
            'logz': -5.0,
            'nlive': 50,
            'niter': 50,
        }
        data = _get_plot_data(r)
        assert 'samples' in data
        assert 'logl' in data
        assert isinstance(data['samples'], np.ndarray)
        assert data['weights'] is not None
