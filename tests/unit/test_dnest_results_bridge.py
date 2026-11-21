"""The DNest -> Results plotting bridge (fake native, fast)."""

import sys
import types

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import logsumexp

import jnesty
from jnesty import plotting
from jnesty.results import Results

N_SAMPLES = 24


class FakeDNestResult:
    """A finished native postprocess result (bridge input surface)."""

    def __init__(self, n=N_SAMPLES):
        rng = np.random.default_rng(7)
        # Model a real postprocess output: likelihoods saved in random
        # order, but logx assigned so that the (logl, tie)-sorted order
        # has monotonically decreasing volumes.
        order = np.argsort(rng.random(n))
        self.logl = np.sort(-rng.random(n) * 6.0 - 1.0)[order]
        self.tie = rng.random(n)
        self.assignment = rng.integers(0, 3, n)
        # Two levels with volumes 0 and -log(2), plus a last level at
        # -log(4); samples sandwich between them.
        self.level_logX = np.array([0.0, -np.log(2.0), -np.log(4.0)])
        self.level_logl = np.array([-np.inf, -5.0, -3.0])
        self.level_tie = np.array([0.11, 0.22, 0.33])
        self.samples_u = 0.5 + 0.1 * rng.normal(size=(n, 2))
        self.samples = 10.0 * self.samples_u - 5.0
        # logx decreasing in ascending-logl order, then scattered into
        # saved order (all likelihoods distinct, so the bridge's
        # (logl, tie) sort restores it).
        self.logx = -np.linspace(0.1, 4.0, n)[order]
        self.logp = np.log(np.full(n, 1.0 / n))
        self.logz = float(logsumexp(self.logp + self.logl))
        self.logwt = self.logp + self.logl - self.logz
        self.information = float(
            np.sum(np.exp(self.logwt) * self.logl) - self.logz)
        self.n_eff = float(np.exp(-np.sum(np.exp(self.logwt) * self.logwt)))
        self.level_visits = np.arange(3)
        self.level_exceeds = np.arange(3)
        self.level_accepts = np.arange(3)
        self.level_tries = np.arange(3) + 2


class FakeEngine:
    """A finished engine exposing only the bridge surface."""

    def __init__(self, n=N_SAMPLES, revision=1):
        self.native_result = FakeDNestResult(n)
        self.revision = revision
        self.n_steps = 4800
        self.n_like = 4816
        self.n_saves = n
        self.n_levels = 3
        self.num_lanes = 4
        self.num_particles = 2
        self.total_particles = 8
        self.acceptance_rate = 0.37
        self.levels_finalized = True
        self.completed = True
        self.max_num_saves = 24
        self.ptol = 0.1

    def posterior(self):
        result = self.native_result
        return (result.samples.copy(), result.logwt.copy(),
                result.logl.copy())


class _StubEngine:
    """Constructor no-op stand-in (the real engine is swapped in after)."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def fake_engine(monkeypatch):
    module = types.ModuleType('jnesty.dfnest')
    module.JaxDNestSampler = _StubEngine
    monkeypatch.setitem(sys.modules, 'jnesty.dfnest', module)
    monkeypatch.setattr(jnesty, 'dfnest', module, raising=False)
    return module


def _bridged_sampler(fake_engine, engine=None):
    loglikelihood = lambda x: x  # noqa: E731
    prior_transform = lambda u: u  # noqa: E731
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    sampler._native = engine or FakeEngine()
    return sampler


def test_to_results_provides_plotting_keys(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    result = sampler.to_results()

    assert isinstance(result, Results)
    for key in ('samples', 'samples_u', 'logl', 'logwt', 'logz', 'logzerr',
                'logvol', 'logz_trajectory', 'sample_levels', 'niter',
                'n_steps', 'n_like', 'n_saves', 'n_levels', 'n_eff',
                'sampler', 'num_lanes', 'num_particles', 'information'):
        assert key in result, f'missing {key!r}'

    data = plotting._get_plot_data(result)
    n = len(sampler._native.native_result.logl)
    assert isinstance(data['samples'], np.ndarray)
    assert data['samples'].shape == (n, 2)
    for key in ('logl', 'logwt', 'logvol', 'sample_levels'):
        assert data[key].shape == (n,)

    # A finished DNest run has evidence but no per-run evidence error.
    assert data['logz'] == pytest.approx(
        sampler._native.native_result.logz)
    assert np.isnan(result['logzerr'])
    assert np.isnan(result['eff'])
    assert np.isnan(result['delta_logz'])
    assert result['sampler'] == 'dnest'
    assert result['n_levels'] == 3
    assert result['niter'] == 4800
    assert result['n_like'] == 4816


def test_unnormalised_logwt_sums_to_logz(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    result = sampler.to_results()
    native = sampler._native.native_result
    assert logsumexp(result['logwt']) == pytest.approx(result['logz'],
                                                       abs=1e-12)
    # logwt is the *unnormalised* evidence weight: logp + logl.
    order = np.lexsort((native.tie, native.logl))
    np.testing.assert_allclose(result['logwt'],
                               native.logp[order] + native.logl[order])


def test_arrays_likelihood_ordered_and_logvol_monotone(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    result = sampler.to_results()
    native = sampler._native.native_result

    logl = result['logl']
    assert np.all(np.diff(logl) >= 0)
    assert np.all(np.diff(result['logvol']) <= 1e-12)
    trajectory = result['logz_trajectory']
    assert trajectory.shape == logl.shape
    assert trajectory[-1] == pytest.approx(result['logz'], abs=1e-12)
    np.testing.assert_allclose(result['sample_levels'],
                               native.assignment[np.lexsort(
                                   (native.tie, native.logl))])


def test_results_property_cached_and_revision_invalidated(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    native = sampler._native

    first = sampler.results
    assert isinstance(first, Results)
    assert sampler.results is first

    # A continued run bumps the revision and invalidates the cache.
    native.revision += 1
    second = sampler.results
    assert second is not first
    assert sampler.results is second


def test_summary_does_not_keyerror(fake_engine, capsys):
    sampler = _bridged_sampler(fake_engine)
    sampler.results.summary()
    out = capsys.readouterr().out
    assert 'logZ' in out
    assert 'Information' in out


def test_to_results_before_run_raises(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    sampler._native.native_result = None
    with pytest.raises(RuntimeError, match='no results'):
        sampler.to_results()


def test_shared_plotting_works_on_bridged_result(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    result = sampler.results

    fig, axes = plotting.cornerplot(result)
    try:
        assert fig is not None
    finally:
        plt.close(fig)

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


def test_posterior_normalised(fake_engine):
    sampler = _bridged_sampler(fake_engine)
    points, log_w, log_l = sampler.posterior()
    assert logsumexp(log_w) == pytest.approx(0.0, abs=1e-12)
    assert np.exp(log_w).sum() == pytest.approx(1.0, abs=1e-12)


def test_fits_roundtrip(fake_engine, tmp_path):
    from jnesty import load_results, save_results

    sampler = _bridged_sampler(fake_engine)
    result = sampler.results
    path = str(tmp_path / 'dnest.fits')
    save_results(result, path)
    loaded = load_results(path)

    assert loaded['logz'] == pytest.approx(result['logz'])
    assert np.isnan(loaded['logzerr'])
    assert loaded['sampler'] == 'dnest'
    assert loaded['n_levels'] == 3
    assert loaded['n_saves'] == N_SAMPLES
    assert loaded['n_like'] == 4816
    assert loaded['max_num_saves'] == 24
    assert loaded['ptol'] == pytest.approx(0.1)
    np.testing.assert_allclose(loaded['samples'], result['samples'])
    np.testing.assert_allclose(loaded['logvol'], result['logvol'])
    np.testing.assert_array_equal(loaded['sample_levels'],
                                  result['sample_levels'])
    assert loaded['converged'] is True
