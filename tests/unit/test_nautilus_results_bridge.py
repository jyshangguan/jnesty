"""The Nautilus -> Results plotting bridge.

``NautilusSampler.to_results()`` / ``.results`` turn the native engine's
posterior into the Dynesty-style arrays that :mod:`jnesty.plotting` consumes,
so both sampler families share one set of plotting helpers.

The bridge tests fake the native engine (same pattern as
``tests/nautilus/test_nautilus_adapter.py``) so they run fast and without the
optional ``jnesty[nautilus]`` dependencies; one end-to-end smoke test uses the
real engine and is marked slow.
"""

import sys
import types

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import jnesty
from jnesty import plotting
from jnesty.results import Results

N_SAMPLES = 24


class FakeJaxSampler:
    """Constructor-recording stand-in for jnesty.nautilus.JaxSampler."""

    def __init__(self, prior, likelihood, n_dim=None, n_live=None, seed=None,
                 device=None, context=None, **kwargs):
        self.prior = prior
        self.likelihood = likelihood
        self.n_dim = n_dim
        self.n_live = n_live
        self.seed = seed
        self.device = device
        self.context = context
        self.kwargs = kwargs
        self.log_z = None
        self.n_eff = 0.0
        self.n_like = 0


class FakeNative:
    """A finished native sampler, exposing only the public bridge surface."""

    def __init__(self, n=N_SAMPLES, log_z=-1.5, n_live=20, n_like=480,
                 n_eff=321.5):
        rng = np.random.default_rng(7)
        self.points = 0.5 + 0.1 * rng.normal(size=(n, 2))
        self.log_l = -0.5 * np.sum((self.points - 0.5) ** 2 / 0.1 ** 2,
                                   axis=1)
        # Log shell volumes of an exponential-shrinking live set.
        log_v = -0.05 * np.arange(1, n + 1)
        self.log_w = log_v + self.log_l
        self.log_w -= np.logaddexp.reduce(self.log_w)
        self.log_z = log_z
        self.n_live = n_live
        self.n_like = n_like
        self.n_eff = n_eff
        self.posterior_calls = 0

    def posterior(self):
        self.posterior_calls += 1
        return self.points, self.log_w, self.log_l


@pytest.fixture
def fake_engine(monkeypatch):
    """Install a fake jnesty.nautilus module (no optional deps needed)."""
    module = types.ModuleType('jnesty.nautilus')
    module.JaxSampler = FakeJaxSampler
    monkeypatch.setitem(sys.modules, 'jnesty.nautilus', module)
    monkeypatch.setattr(jnesty, 'nautilus', module, raising=False)
    return module


def _problem():
    def loglikelihood(x):
        return x

    def prior_transform(u):
        return u

    return loglikelihood, prior_transform


def _bridged_sampler():
    """Construct the adapter as usual, then swap in a finished fake native."""
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.NautilusSampler(loglikelihood, prior_transform, ndim=2,
                                     nlive=20)
    native = FakeNative()
    sampler._native = native
    return sampler, native


def test_to_results_provides_plotting_keys(fake_engine):
    sampler, native = _bridged_sampler()
    result = sampler.to_results()

    assert isinstance(result, Results)
    for key in ('samples', 'logl', 'logwt', 'logz', 'nlive', 'niter',
                'logvol', 'logz_trajectory', 'logzerr', 'n_eff'):
        assert key in result, f'missing {key!r}'

    # Every array _get_plot_data consumes is numpy with consistent shapes.
    data = plotting._get_plot_data(result)
    for key in ('samples', 'logl', 'logwt', 'logz', 'nlive', 'niter',
                'logvol'):
        assert key in data, f'_get_plot_data dropped {key!r}'

    n = len(native.points)
    assert isinstance(data['samples'], np.ndarray)
    assert data['samples'].shape == (n, 2)
    assert data['samples'].dtype.kind == 'f'
    for key in ('logl', 'logwt', 'logvol'):
        assert isinstance(data[key], np.ndarray)
        assert data[key].shape == (n,)
    assert isinstance(data['niter'], (int, np.integer))
    assert data['niter'] == native.n_like
    assert int(data['nlive']) == native.n_live

    # A finished Nautilus run has evidence but no evidence error.
    assert data['logz'] == pytest.approx(native.log_z)
    assert np.isnan(result['logzerr'])
    assert np.isnan(result['information'])
    assert np.isnan(result['eff'])
    assert np.isnan(result['acceptance_rate'])
    assert result['n_eff'] == pytest.approx(native.n_eff)


def test_per_point_arrays_are_likelihood_ordered(fake_engine):
    sampler, native = _bridged_sampler()
    result = sampler.to_results()

    logl = result['logl']
    logvol = result['logvol']
    # Sorted by ascending logl so runplot's -ln X axis is well defined.
    assert np.all(np.diff(logl) >= 0)
    # Monotonised derived volume: -ln X must not decrease.
    assert np.all(np.diff(logvol) <= 0)

    trajectory = result['logz_trajectory']
    assert trajectory.shape == (len(native.points),)
    assert np.all(np.diff(trajectory) >= 0)
    # Weights from posterior() are normalised, so the running log-sum ends
    # at log(sum w) = 0.
    assert trajectory[-1] == pytest.approx(0.0, abs=1e-9)


def test_results_property_is_cached_and_rebuilds_after_a_run(fake_engine):
    sampler, native = _bridged_sampler()

    first = sampler.results
    assert isinstance(first, Results)
    assert native.posterior_calls == 1
    assert sampler.results is first
    assert native.posterior_calls == 1

    # A further run changes the likelihood count and invalidates the cache.
    native.n_like += 50
    second = sampler.results
    assert second is not first
    assert native.posterior_calls == 2
    assert sampler.results is second


def test_summary_does_not_keyerror(fake_engine, capsys):
    sampler, _ = _bridged_sampler()
    sampler.results.summary()
    out = capsys.readouterr().out
    assert 'logZ' in out


def test_shared_plotting_works_on_bridged_result(fake_engine):
    sampler, _ = _bridged_sampler()
    result = sampler.results

    fig, axes = plotting.cornerplot(result)
    try:
        assert fig is not None
        assert np.asarray(axes).shape == (2, 2)
    finally:
        plt.close(fig)

    fig, axes = plotting.cornerpoints(result)
    try:
        assert np.asarray(axes).shape == (2, 2)
    finally:
        plt.close(fig)


def test_shared_trajectory_plots_work_on_bridged_result(fake_engine):
    sampler, _ = _bridged_sampler()
    result = sampler.results

    fig, axes = plotting.runplot(result)
    try:
        assert len(axes) == 4
    finally:
        plt.close(fig)

    fig, axes = plotting.traceplot(result)
    try:
        assert np.asarray(axes).shape == (2, 2)
    finally:
        plt.close(fig)

    fig, axes = plotting.diagnostics(result)
    try:
        assert np.asarray(axes).shape == (2, 2)
    finally:
        plt.close(fig)


@pytest.mark.slow
def test_real_engine_bridge_smoke():
    """End-to-end bridge check on a tiny 2D Gaussian (CPU, no network)."""
    pytest.importorskip('jnesty.nautilus')
    import jax.numpy as jnp
    from jnesty import NautilusSampler

    def loglikelihood(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2) / 0.1 ** 2

    def prior_transform(u):
        return u

    sampler = NautilusSampler(loglikelihood, prior_transform, ndim=2,
                              nlive=150, n_batch=50, n_networks=0, seed=0,
                              device='cpu')
    assert sampler.run(f_live=0.3, n_eff=0)

    result = sampler.to_results()
    samples, log_w, log_l = sampler.posterior()
    order = np.argsort(log_l)
    assert len(result['samples']) == len(samples)
    assert result['samples'].shape == samples.shape
    assert np.allclose(result['samples'], np.asarray(samples)[order])
    assert np.allclose(result['logwt'], np.asarray(log_w)[order])
    assert np.allclose(result['logl'], np.asarray(log_l)[order])
    assert result['logz'] == pytest.approx(float(sampler.log_z))
    assert np.isnan(result['logzerr'])
    assert result['niter'] == int(sampler.n_like)

    fig, axes = plotting.cornerplot(result)
    plt.close(fig)
    assert np.asarray(axes).shape == (2, 2)