"""Comparison against the reference CPU implementation and calling conventions."""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from scipy.special import logsumexp

from nautilus import Prior, Sampler
from jnesty.nautilus import JaxSampler
from jnesty.nautilus.eval import JaxEvaluator


def _upstream(prior, likelihood, n_dim, n_live, n_networks, seed, **kwargs):
    sampler = Sampler(prior, likelihood, n_dim=n_dim, n_live=n_live,
                      n_batch=100, n_networks=n_networks, seed=seed, **kwargs)
    sampler.run(f_live=0.2, n_eff=0)
    return sampler


def _jax(prior, likelihood, n_dim, n_live, n_networks, seed):
    sampler = JaxSampler(prior, likelihood, n_dim=n_dim, n_live=n_live,
                         n_batch=100, n_networks=n_networks, seed=seed)
    sampler.run(f_live=0.2, n_eff=0)
    return sampler


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_gaussian_agrees_with_upstream(seed):
    def prior(u):
        return u

    def likelihood(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2) / 0.1 ** 2

    n_live = 300
    ref = _upstream(prior, likelihood, 2, n_live, 1, seed)
    got = _jax(prior, likelihood, 2, n_live, 1, seed)

    # Both estimates carry Monte Carlo error of order 1/sqrt(n_live); require
    # agreement well inside that scale rather than bitwise equality.
    # 0.1 in log Z covers the Monte Carlo scatter at 300 live points.  The two
    # implementations use independent random streams, and the reference itself
    # varies by ~0.05 between seeds.
    assert got.log_z == pytest.approx(ref.log_z, abs=0.1)

    ref_points, ref_w, _ = ref.posterior()
    got_points, got_w, _ = got.posterior()
    ref_mean = np.average(ref_points, weights=np.exp(ref_w), axis=0)
    got_mean = np.average(got_points, weights=np.exp(got_w), axis=0)
    # Both estimates are compared against each other, not against the truth; at
    # 300 live points with a loose stop the posterior mean scatters by ~0.01-0.02
    # between random streams.
    assert np.allclose(got_mean, ref_mean, atol=0.02)


def test_ins_weights_match_reference_formula():
    """The result formulas must be identical to the reference implementation."""
    def prior(u):
        return u

    def likelihood(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2) / 0.2 ** 2

    sampler = JaxSampler(prior, likelihood, n_dim=2, n_live=100, n_batch=100,
                         n_networks=0, seed=4)
    sampler.run(f_live=0.3, n_eff=0)

    points, log_w, log_l = sampler.posterior()
    log_v = np.repeat(sampler.shell_log_v
                      - np.log(np.maximum(sampler.shell_n, 1)),
                      sampler.shell_n)
    expected_log_w = log_v + log_l
    expected_log_w -= logsumexp(expected_log_w)
    assert np.allclose(log_w, expected_log_w, atol=1e-12)

    shell_log_z = sampler.shell_log_l + sampler.shell_log_v
    assert sampler.log_z == pytest.approx(logsumexp(shell_log_z), abs=1e-12)


def test_evaluator_vectorized_matches_vmap():
    def prior(u):
        return 10.0 * u - 5.0

    def likelihood_scalar(x):
        return -0.5 * jnp.sum(x ** 2)

    def likelihood_batch(x):
        return -0.5 * jnp.sum(x ** 2, axis=-1)

    points = np.random.default_rng(0).uniform(size=(37, 3))
    scalar = JaxEvaluator(prior, likelihood_scalar, 3, vectorized=False)
    batch = JaxEvaluator(prior, likelihood_batch, 3, vectorized=True)
    a, _ = scalar.evaluate(points)
    b, _ = batch.evaluate(points)
    assert np.allclose(a, b, atol=1e-12)


def test_evaluator_prior_object_and_dict():
    prior = Prior()
    prior.add_parameter('a', dist=(0, 2))
    prior.add_parameter('b', dist=(0, 2))

    def likelihood(params):
        return -0.5 * (params['a'] ** 2 + params['b'] ** 2)

    evaluator = JaxEvaluator(prior, likelihood, 2, pass_dict=True,
                             vectorized=False)
    points = np.random.default_rng(1).uniform(size=(11, 2))
    log_l, _ = evaluator.evaluate(points)
    assert log_l.shape == (11,)
    assert np.all(np.isfinite(log_l))


def test_evaluator_rejects_nan():
    def prior(u):
        return u

    def likelihood(x):
        return jnp.nan

    evaluator = JaxEvaluator(prior, likelihood, 2)
    with pytest.raises(ValueError, match='NaN'):
        evaluator.evaluate(np.array([[0.5, 0.5]]))


def _multimodal_stats(sampler, discard):
    points, log_w, _ = sampler.posterior()
    w = np.exp(log_w)
    w /= w.sum()
    return {
        'logz': sampler.log_z,
        'mass_left': w[points[:, 0] < -1.0].sum(),
        'mass_right': w[points[:, 0] > 2.0].sum(),
        'mean_x0': np.average(points[:, 0], weights=w),
    }


def test_multimodal_posterior_matches_upstream():
    """Ensemble-averaged posterior statistics must match the reference.

    The multimodal problem puts two modes inside a wide prior, so it is a
    sensitive check: missing a mode, or mis-assigning shell volumes to points,
    changes these numbers immediately.  A single seed carries noticeable Monte
    Carlo scatter (the exploration points dominate the tails), so the test
    compares the average over three seeds, which is what a bias shows up in.
    """
    def prior_np(u):
        return (u - 0.5) * 10.0

    def likelihood_np(x):
        d1 = np.sum(x ** 2, axis=-1)
        d2 = np.sum((x - 3.0) ** 2, axis=-1)
        return np.logaddexp(np.log(0.6) - 0.5 * d1, np.log(0.4) - 0.5 * d2)

    def prior_jax(u):
        return (u - 0.5) * 10.0

    def likelihood_jax(x):
        d1 = jnp.sum(x ** 2)
        d2 = jnp.sum((x - 3.0) ** 2)
        return jnp.logaddexp(jnp.log(0.6) - 0.5 * d1, jnp.log(0.4) - 0.5 * d2)

    reference, port = [], []
    for seed in (0, 1, 2):
        ref = Sampler(prior_np, likelihood_np, n_dim=2, n_live=300,
                      n_batch=100, n_networks=4, seed=seed)
        ref.run(f_live=0.05, n_eff=0, discard_exploration=True)
        reference.append(_multimodal_stats(ref, True))

        got = JaxSampler(prior_jax, likelihood_jax, n_dim=2, n_live=300,
                         n_batch=100, n_networks=4, seed=seed)
        got.run(f_live=0.05, n_eff=0, discard_exploration=True)
        port.append(_multimodal_stats(got, True))

    # Tolerances reflect the Monte Carlo scatter of this problem rather than
    # tight numerical agreement: the wide prior means exploration points, which
    # dominate the tails, differ run to run, and the two implementations use
    # independent random streams.  The analytic posterior for this problem has
    # mean(x0) ~ 1.145 and log Z ~ -2.785, so both implementations are required
    # to sit near the same answer, not to match each other's noise.
    for key, tolerance in [('logz', 0.1), ('mean_x0', 0.12),
                           ('mass_left', 0.05), ('mass_right', 0.06)]:
        ref_value = float(np.mean([r[key] for r in reference]))
        got_value = float(np.mean([r[key] for r in port]))
        assert got_value == pytest.approx(ref_value, abs=tolerance), (
            f'{key}: port {got_value:.4f} vs reference {ref_value:.4f}')
