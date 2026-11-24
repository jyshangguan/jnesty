"""Tests of the public DNestSampler adapter.

The API tests fake the engine module (mirroring
``tests/nautilus/test_nautilus_adapter.py``) so they run quickly without
compiling any JAX program.
"""

import sys
import types

import numpy as np
import pytest

import jnesty


class FakeJaxDNestSampler:
    """Stand-in for jnesty.dfnest.JaxDNestSampler recording constructor args."""

    def __init__(self, loglikelihood, prior_transform, ndim, num_lanes=1,
                 num_particles=1, sample_frequency=None,
                 thread_steps_factor=10.0, new_level_interval_factor=2.0,
                 save_interval_factor=None, max_num_levels=0, lam=10.0,
                 beta=100.0, compression=2.718281828459045, ptol=0.1,
                 seed=None, device=None, context=None):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.num_lanes = num_lanes
        self.num_particles = num_particles
        self.sample_frequency = sample_frequency
        self.thread_steps_factor = thread_steps_factor
        self.new_level_interval_factor = new_level_interval_factor
        self.save_interval_factor = save_interval_factor
        self.max_num_levels = max_num_levels
        self.lam = lam
        self.beta = beta
        self.compression = compression
        self.ptol = ptol
        self.seed = seed
        self.device = device
        self.context = context
        self.sample_frequency_input = sample_frequency
        self.run_calls = []
        self.posterior_calls = 0
        # diagnostics surface
        self.revision = 3
        self.n_steps = 1000
        self.n_like = 1004
        self.n_saves = 50
        self.n_levels = 7
        self.num_lanes = num_lanes
        self.total_particles = num_lanes * num_particles
        self.acceptance_rate = 0.42
        self.levels_finalized = True
        self.completed = True
        self.max_num_saves = 50
        self.ptol = ptol
        self.sample_frequency = np.ones(ndim) / ndim
        self.coordinate_attempts = np.arange(ndim) * 10
        self.coordinate_accepts = np.arange(ndim) * 4
        self.delta_logz_stopped = False
        self.last_delta_logz = None
        self.native_result = None
        self.config = types.SimpleNamespace(ndim=ndim)

    def run(self, max_num_saves=10000, ptol=None, verbose=False,
            delta_logz=None):
        self.run_calls.append(
            {'max_num_saves': max_num_saves, 'ptol': ptol,
             'verbose': verbose, 'delta_logz': delta_logz})
        if self.max_num_saves is not None:
            self.max_num_saves = max(max_num_saves, self.max_num_saves)
        return -1.23

    def posterior(self):
        self.posterior_calls += 1
        return 'points', 'log_w', 'log_l'


@pytest.fixture
def fake_engine(monkeypatch):
    module = types.ModuleType('jnesty.dfnest')
    module.JaxDNestSampler = FakeJaxDNestSampler
    monkeypatch.setitem(sys.modules, 'jnesty.dfnest', module)
    monkeypatch.setattr(jnesty, 'dfnest', module, raising=False)
    return module


def _problem():
    def loglikelihood(x):
        return x

    def prior_transform(u):
        return u

    return loglikelihood, prior_transform


def test_constructor_translation(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(
        loglikelihood, prior_transform, ndim=3, num_lanes=5,
        num_particles=2, sample_frequency=[1, 2, 3], seed=9,
        device='cpu', context=None, max_num_levels=11, lam=3.0, beta=7.0,
        compression=2.5, ptol=0.2, thread_steps_factor=4.0,
        new_level_interval_factor=3.0, save_interval_factor=6.0)
    native = sampler.native_sampler
    assert isinstance(native, FakeJaxDNestSampler)
    # JNesty argument order: likelihood first, prior transform second.
    assert native.loglikelihood is loglikelihood
    assert native.prior_transform is prior_transform
    assert native.ndim == 3
    assert native.num_lanes == 5
    assert native.num_particles == 2
    assert native.sample_frequency_input == [1, 2, 3]
    assert native.seed == 9
    assert native.device == 'cpu'
    assert native.max_num_levels == 11
    assert native.lam == 3.0
    assert native.beta == 7.0
    assert native.compression == 2.5
    assert native.ptol == 0.2
    assert native.thread_steps_factor == 4.0
    assert native.new_level_interval_factor == 3.0
    assert native.save_interval_factor == 6.0


def test_constructor_defaults(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    native = sampler.native_sampler
    assert native.num_lanes == 1
    assert native.num_particles == 1
    assert native.sample_frequency_input is None
    assert native.max_num_levels == 0
    assert native.lam == 10.0
    assert native.beta == 100.0
    assert native.save_interval_factor is None
    assert native.ptol == 0.1


def test_run_delegation(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    native = sampler.native_sampler
    assert sampler.run(max_num_saves=77, ptol=0.05, verbose=True) == -1.23
    (call,) = native.run_calls
    assert call == {'max_num_saves': 77, 'ptol': 0.05, 'verbose': True,
                    'delta_logz': None}
    # Defaults match the documented CDNest values.
    sampler.run()
    assert native.run_calls[1] == {
        'max_num_saves': 10000, 'ptol': None, 'verbose': False,
        'delta_logz': None}


def test_run_delta_logz_delegation(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    native = sampler.native_sampler
    assert sampler.run(max_num_saves=77, delta_logZ=0.05) == -1.23
    (call,) = native.run_calls
    # The adapter exposes the dynesty-style spelling; the native engine
    # receives the snake_case one.
    assert call == {'max_num_saves': 77, 'ptol': None, 'verbose': False,
                    'delta_logz': 0.05}
    assert sampler.run(delta_logZ=0.5) == -1.23
    assert native.run_calls[1]['delta_logz'] == 0.5


def test_posterior_delegation(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    native = sampler.native_sampler
    assert sampler.posterior() == ('points', 'log_w', 'log_l')
    assert native.posterior_calls == 1


def test_diagnostics_forwarded(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=4)
    assert sampler.n_steps == 1000
    assert sampler.n_like == 1004
    assert sampler.n_saves == 50
    assert sampler.n_levels == 7
    assert sampler.acceptance_rate == 0.42
    assert sampler.levels_finalized is True
    assert sampler.completed is True
    np.testing.assert_array_equal(sampler.coordinate_attempts,
                                  np.arange(4) * 10)
    np.testing.assert_array_equal(sampler.coordinate_accepts,
                                  np.arange(4) * 4)


def test_getattr_forwarding_and_missing(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    native = sampler.native_sampler
    native.custom_attribute = 'value'
    assert sampler.custom_attribute == 'value'
    with pytest.raises(AttributeError):
        _ = sampler.does_not_exist


def test_repr(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    text = repr(sampler)
    assert 'DNestSampler' in text
    assert 'num_lanes=1' in text
    assert 'num_particles=1' in text
    assert 'n_saves=50' in text
    # No live-point language leaks into the DNest repr.
    assert 'nlive' not in text


def test_print_summary_before_run(fake_engine, capsys):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    sampler.print_summary()
    out = capsys.readouterr().out
    assert 'no results yet' in out


def test_print_summary_delta_logz_stop_line(fake_engine, capsys):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.DNestSampler(loglikelihood, prior_transform, ndim=2)
    native = sampler.native_sampler
    native.native_result = types.SimpleNamespace(
        logz=-2.5, n_eff=100.0, information=1.5)
    native.coordinate_attempts = np.zeros(2)  # skip the coordinate block
    native.completed = False  # a delta-stopped run misses its save target
    native.delta_logz_stopped = True
    native.last_delta_logz = 0.031
    sampler.print_summary()
    out = capsys.readouterr().out
    assert 'Save target reached: False' in out
    assert 'Delta-logZ stop: yes (delta logZ = 0.031 at 50 saves)' in out


def test_top_level_export():
    assert 'DNestSampler' in jnesty.__all__
    assert jnesty.DNestSampler.__name__ == 'DNestSampler'
