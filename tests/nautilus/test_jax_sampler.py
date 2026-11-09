"""End-to-end tests of :class:`jnesty.nautilus.JaxSampler`."""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from jnesty.nautilus import JaxSampler

N_LIVE = 200
N_BATCH = 100


def gaussian_problem(n_dim=2, sigma=0.1, mean=0.5):
    def prior(u):
        return u

    def likelihood(x):
        return -0.5 * jnp.sum((x - mean) ** 2) / sigma ** 2

    return prior, likelihood, n_dim


def run_gaussian(n_networks=0, seed=0, f_live=0.2, **kwargs):
    prior, likelihood, n_dim = gaussian_problem()
    sampler = JaxSampler(prior, likelihood, n_dim=n_dim, n_live=N_LIVE,
                         n_batch=N_BATCH, n_networks=n_networks,
                         seed=seed, **kwargs)
    sampler.run(f_live=f_live, n_eff=0)
    return sampler


@pytest.mark.parametrize('n_networks', [0, 1])
def test_gaussian_evidence_and_moments(n_networks):
    sampler = run_gaussian(n_networks=n_networks)
    # Normalised 2-D Gaussian inside the unit cube: log Z = log(2 pi sigma^2).
    log_z_true = np.log(2 * np.pi * 0.1 ** 2)
    assert sampler.log_z == pytest.approx(log_z_true, abs=0.1)

    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    assert weights.sum() == pytest.approx(1.0)
    mean = np.average(points, weights=weights, axis=0)
    cov = np.cov(points, aweights=weights, rowvar=False)
    assert np.allclose(mean, 0.5, atol=0.02)
    assert np.allclose(cov, np.eye(2) * 0.01, atol=0.004)
    assert sampler.n_eff > 0
    assert 0 < sampler.eta < 1


def test_constant_likelihood():
    def prior(u):
        return u

    def likelihood(x):
        return jnp.array(0.0)

    sampler = JaxSampler(prior, likelihood, n_dim=2, n_live=200, n_batch=100,
                         n_networks=0, seed=0)
    sampler.run(f_live=0.1, n_eff=0)
    assert sampler.log_z == pytest.approx(0.0, abs=1e-6)
    assert len(sampler.bounds) == 0 or len(sampler.bounds) == 1


def test_multimodal_sampling():
    def prior(u):
        return u

    def likelihood(x):
        d1 = jnp.sum((x - jnp.array([0.25, 0.5])) ** 2)
        d2 = jnp.sum((x - jnp.array([0.75, 0.5])) ** 2)
        return jnp.logaddexp(jnp.log(0.5) - 0.5 * d1 / 0.01 ** 2,
                             jnp.log(0.5) - 0.5 * d2 / 0.01 ** 2)

    sampler = JaxSampler(prior, likelihood, n_dim=2, n_live=300, n_batch=100,
                         n_networks=2, seed=0)
    sampler.run(f_live=0.2, n_eff=0)
    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    left = weights[points[:, 0] < 0.5].sum()
    right = weights[points[:, 0] > 0.5].sum()
    # Both modes must be represented.  The split is not 50/50 even in the ideal
    # limit: the importance weights retain exploration points drawn from the
    # wide prior, which pulls mass towards the lower half (the pseudo-importance
    # bias), so this checks coverage, not a precise share.
    assert 0.25 < left < 0.75
    assert 0.25 < right < 0.75


def test_reproducible_with_seed():
    a = run_gaussian(seed=7)
    b = run_gaussian(seed=7)
    assert a.log_z == b.log_z
    assert a.n_like == b.n_like


def test_discard_exploration_changes_weights():
    sampler = run_gaussian(seed=3)
    n_before = len(sampler.posterior()[0])
    log_z_before = sampler.log_z
    sampler.discard_exploration = True
    n_after = len(sampler.posterior()[0])
    assert n_after <= n_before
    assert sampler.log_z != log_z_before or n_after == n_before


def test_blobs():
    def prior(u):
        return u

    def likelihood(x):
        return (-0.5 * jnp.sum((x - 0.5) ** 2) / 0.1 ** 2, x[0])

    sampler = JaxSampler(prior, likelihood, n_dim=2, n_live=200, n_batch=100,
                         n_networks=0, seed=0)
    sampler.run(f_live=0.2, n_eff=0)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)
    assert np.allclose(points[:, 0], blobs)


def test_checkpoint_round_trip(tmp_path):
    prior, likelihood, n_dim = gaussian_problem()
    path = tmp_path / 'checkpoint.h5'

    sampler = JaxSampler(prior, likelihood, n_dim=n_dim, n_live=N_LIVE,
                         n_batch=N_BATCH, n_networks=0, seed=11,
                         filepath=path)
    sampler.run(f_live=0.2, n_eff=0, n_like_max=800)
    log_z_partial = sampler.log_z
    n_like_partial = sampler.n_like

    resumed = JaxSampler(prior, likelihood, n_dim=n_dim, n_live=N_LIVE,
                         n_batch=N_BATCH, n_networks=0, seed=11,
                         filepath=path, resume=True)
    assert resumed.n_like == n_like_partial
    assert resumed.log_z == pytest.approx(log_z_partial, rel=0, abs=1e-12)

    resumed.run(f_live=0.2, n_eff=0)
    reference = run_gaussian(seed=11)
    assert resumed.log_z == pytest.approx(reference.log_z, rel=0, abs=1e-10)
    assert resumed.n_like == reference.n_like