"""Tests for the emulator's training-set handling."""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from jnesty.nautilus.neural import train_emulator


@pytest.mark.parametrize('n_points', [64, 100, 128])
def test_training_set_smaller_than_one_batch(n_points):
    """Fewer points than one mini-batch must still train on those points.

    The batch index used to be built from ``max(1, n // batch) * batch``,
    which exceeds the array length when ``n < batch``; JAX clamps the
    out-of-range gathers to the last row, so the final point was repeated in
    most of every batch.  A single extreme point exposes that immediately: if
    it is duplicated, the fit is dragged towards it.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_points, 4))
    y = np.tanh(2.0 * x[:, 0])
    y[-1] = 50.0                       # one wild target at the very last row

    emulator = train_emulator(jnp.asarray(x), jnp.asarray(y),
                              jax.random.PRNGKey(0), n_networks=1,
                              max_epochs=300, patience=10)
    prediction = np.asarray(emulator.predict(jnp.asarray(x)))

    # Without duplication the wild point is one of n targets and cannot drag
    # the whole fit; with duplication it dominates the gradient.
    interior = np.abs(prediction[:-1]).max()
    assert interior < 5.0, f'fit dragged by the last row: {interior:.2f}'
