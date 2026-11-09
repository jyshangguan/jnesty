"""Bucketed emulator-training preprocessing matches the eager formula."""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from jnesty.nautilus import neural


@pytest.mark.parametrize('n_real', [5, 63, 64, 65, 200, 257])
def test_prepare_matches_eager_normalization_and_padding(n_real):
    rng = np.random.default_rng(n_real)
    x = rng.normal(size=(n_real, 7))
    y = rng.normal(size=n_real)

    x_pad, y_pad, mean, scale = neural._prepare_training_arrays(
        x, y, jnp.float32)

    x_j = jnp.asarray(x, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.float32)
    mean_ref = jnp.mean(x_j, axis=0)
    scale_ref = jnp.std(x_j, axis=0)
    scale_ref = jnp.where(scale_ref > 0, scale_ref, 1.0)
    x_norm = (x_j - mean_ref) / scale_ref
    n_target = neural._bucket_size(n_real)
    x_ref, y_ref = neural._pad_to(x_norm, y_j, n_target)

    assert x_pad.shape == (n_target, 7)
    assert y_pad.shape == (n_target,)
    np.testing.assert_allclose(mean, mean_ref, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(scale, scale_ref, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(x_pad, x_ref, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(y_pad, y_ref, rtol=0, atol=0)


def test_prepare_reuses_program_for_different_real_counts_in_one_bucket():
    key = (256, 3, 'float32')
    neural._COMPILED_PREPARE.pop(key, None)
    x1 = np.random.default_rng(0).normal(size=(130, 3))
    y1 = np.random.default_rng(1).normal(size=130)
    neural._prepare_training_arrays(x1, y1, jnp.float32)
    program = neural._COMPILED_PREPARE[key]

    x2 = np.random.default_rng(2).normal(size=(201, 3))
    y2 = np.random.default_rng(3).normal(size=201)
    neural._prepare_training_arrays(x2, y2, jnp.float32)
    assert neural._COMPILED_PREPARE[key] is program
