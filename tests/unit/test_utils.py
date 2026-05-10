import numpy as np
import jax.numpy as jnp
from jax import random
from jnesty.utils import randsphere, logsubexp, logvol_prefactor, _convert_to_numpy


class TestRandsphere:
    def test_shape(self, key):
        for n in [1, 2, 5, 10]:
            result = randsphere(random.PRNGKey(n), n)
            assert result.shape == (n,)

    def test_in_unit_ball(self):
        k = random.PRNGKey(0)
        for _ in range(500):
            k, subkey = random.split(k)
            pt = randsphere(subkey, 3)
            assert float(jnp.linalg.norm(pt)) <= 1.0 + 1e-6

    def test_reproducible(self):
        k = random.PRNGKey(123)
        a = randsphere(k, 5)
        b = randsphere(k, 5)
        np.testing.assert_array_equal(a, b)
        c = randsphere(random.PRNGKey(456), 5)
        assert not jnp.array_equal(a, c)


class TestLogsubexp:
    def test_basic(self):
        result = float(logsubexp(5.0, 3.0))
        expected = np.log(np.exp(5.0) - np.exp(3.0))
        assert abs(result - expected) < 1e-10

    def test_close_args(self):
        result = float(logsubexp(5.0, 4.999))
        assert np.isfinite(result)
        assert not np.isnan(result)

    def test_identical_returns_neginf(self):
        result = float(logsubexp(5.0, 5.0))
        assert result == -np.inf


class TestLogvolPrefactor:
    def test_known_values(self):
        assert abs(float(logvol_prefactor(1)) - np.log(2.0)) < 1e-10
        assert abs(float(logvol_prefactor(2)) - np.log(np.pi)) < 1e-10
        assert abs(float(logvol_prefactor(3)) - np.log(4 * np.pi / 3)) < 1e-10

    def test_finite_for_all_dims(self):
        for n in range(1, 50):
            val = float(logvol_prefactor(n))
            assert np.isfinite(val), f"Non-finite at n={n}"


class TestConvertToNumpy:
    def test_jax_array(self):
        arr = jnp.array([1.0, 2.0])
        result = _convert_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_numpy_array(self):
        arr = np.array([3.0, 4.0])
        result = _convert_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [3.0, 4.0])

    def test_list(self):
        result = _convert_to_numpy([5, 6, 7])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5, 6, 7])
