"""Tests of the public NautilusSampler adapter.

The API tests fake the engine module so they run quickly and without the
optional 'jnesty[nautilus]' dependencies; one end-to-end smoke test uses
the real engine and is marked slow.
"""

import sys
import types

import numpy as np
import pytest

import jnesty


class FakeJaxSampler:
    """Stand-in for jnesty.nautilus.JaxSampler recording constructor args."""

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
        self.run_calls = []
        self.written = []
        self.posterior_kwargs = None
        self._discard = False
        self.log_z = -1.0
        self.n_eff = 10.0
        self.n_like = 99
        self.eta = 0.5
        self.forwarded = 'attribute'

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        return True

    def posterior(self, **kwargs):
        self.posterior_kwargs = kwargs
        return ('points', 'log_w', 'log_l')

    def write(self, filepath, overwrite=False):
        self.written.append((filepath, overwrite))

    @property
    def discard_exploration(self):
        return self._discard

    @discard_exploration.setter
    def discard_exploration(self, value):
        self._discard = value


@pytest.fixture
def fake_engine(monkeypatch):
    """Install a fake jnesty.nautilus module for one test.

    Both registries are patched because the adapter's lazy
    ``from . import nautilus`` can find the engine through sys.modules or
    through the attribute the real import left on the jnesty package (the
    slow engine tests may have imported it earlier in the same run).
    """
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


def test_constructor_translation(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.NautilusSampler(
        loglikelihood, prior_transform, ndim=3, nlive=123, seed=5,
        device='cpu', n_networks=2)
    native = sampler.native_sampler
    assert isinstance(native, FakeJaxSampler)
    # JNesty argument order: the likelihood comes first, the prior
    # transform second; the native engine takes them the other way round.
    assert native.likelihood is loglikelihood
    assert native.prior is prior_transform
    assert native.n_dim == 3
    assert native.n_live == 123
    assert native.seed == 5
    assert native.device == 'cpu'
    assert native.context is None
    assert native.kwargs == {'n_networks': 2}


def test_run_delegation(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.NautilusSampler(loglikelihood, prior_transform, ndim=2,
                                    nlive=10)
    native = sampler.native_sampler
    assert sampler.run(f_live=0.3, n_shell=2) is True
    (call,) = native.run_calls
    assert call['f_live'] == 0.3
    assert call['n_shell'] == 2
    assert call['n_eff'] == 10000
    assert call['verbose'] is False


def test_posterior_and_write_delegation(fake_engine, tmp_path):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.NautilusSampler(loglikelihood, prior_transform, ndim=2,
                                    nlive=10)
    native = sampler.native_sampler
    assert sampler.posterior(equal_weight=True) == ('points', 'log_w', 'log_l')
    # The adapter fills in the native defaults and forwards everything.
    kwargs = native.posterior_kwargs
    assert kwargs['equal_weight'] is True
    assert kwargs['equal_weight_boost'] == 1.0
    assert kwargs['return_as_dict'] is None
    assert kwargs['return_blobs'] is False
    path = tmp_path / 'checkpoint.h5'
    sampler.write(path)
    assert native.written == [(path, False)]
    sampler.write(path, overwrite=True)
    assert native.written == [(path, False), (path, True)]


def test_attribute_forwarding(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.NautilusSampler(loglikelihood, prior_transform, ndim=2,
                                    nlive=10)
    native = sampler.native_sampler
    # Explicitly wrapped diagnostics.
    assert sampler.log_z == -1.0
    assert sampler.n_eff == 10.0
    assert sampler.n_like == 99
    assert sampler.eta == 0.5
    # __getattr__ forwarding of native-only attributes.
    assert sampler.forwarded == 'attribute'
    with pytest.raises(AttributeError):
        _ = sampler.does_not_exist
    # The discard flag forwards including the setter.
    assert sampler.discard_exploration is False
    sampler.discard_exploration = True
    assert native.discard_exploration is True
    assert sampler.discard_exploration is True


def test_repr(fake_engine):
    loglikelihood, prior_transform = _problem()
    sampler = jnesty.NautilusSampler(loglikelihood, prior_transform, ndim=2,
                                    nlive=10)
    text = repr(sampler)
    assert 'NautilusSampler' in text
    assert 'n_dim=2' in text
    assert 'n_live=10' in text


@pytest.mark.slow
def test_real_engine_smoke():
    pytest.importorskip('jnesty.nautilus')
    import jax
    import jax.numpy as jnp
    from jnesty import NautilusSampler

    def loglikelihood(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2) / 0.1 ** 2

    def prior_transform(u):
        return u

    sampler = NautilusSampler(loglikelihood, prior_transform, ndim=2,
                              nlive=150, n_batch=50, n_networks=0, seed=0,
                              device='cpu')
    assert jax.config.jax_enable_x64 is True
    assert sampler.run(f_live=0.3, n_eff=0)
    assert sampler.log_z == pytest.approx(np.log(2 * np.pi * 0.01), abs=0.1)
    points, log_w, _ = sampler.posterior()
    assert len(points) == len(log_w)
    assert np.exp(log_w).sum() == pytest.approx(1.0, abs=1e-9)
    assert sampler.n_like > 0
