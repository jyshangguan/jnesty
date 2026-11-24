"""Opt-in temperature post-processing for the DNest engine/adapter.

Unit A of the dnest enhancements: the ``temperature`` parameter threaded
from :meth:`DNestSampler.results_at_temperature` /
:meth:`DNestSampler.posterior_at_temperature` through the engine's
``_compute_native_result`` into the CDNest postprocessor, plus the
dynesty-plotting compatibility fields (``nlive``,
``delta_logZ_trajectory``, ``rwalk_K``) in the results bridge.

Real-engine tests (JAX JIT + chains), hence slow-marked like
``test_engine.py``; the fast fake-engine coverage of the formatting
itself lives in ``tests/unit/test_dnest_results_bridge.py``.  The
postprocessor's own temperature arithmetic is covered in
``test_postprocess.py``.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import logsumexp

import jax.numpy as jnp

from jnesty import DNestSampler, plotting

pytestmark = pytest.mark.slow


def _gaussian_problem(ndim=2):
    """Standard Gaussian in a [-5, 5]^ndim box (logl <= 0, so logZ(T)
    rises towards the prior total mass 0 as T grows)."""
    def loglikelihood(x):
        return -0.5 * jnp.sum(x ** 2)

    def prior_transform(u):
        return (u - 0.5) * 10.0

    return loglikelihood, prior_transform, ndim


@pytest.fixture(scope='module')
def sampler():
    loglike, prior, _ = _gaussian_problem()
    s = DNestSampler(loglike, prior, ndim=2, seed=42)
    s.run(max_num_saves=400)
    return s


@pytest.fixture(scope='module')
def ensemble_sampler():
    """A non-trivial ensemble (2 lanes x 3 particles) for the nlive
    proxy semantics."""
    loglike, prior, _ = _gaussian_problem()
    s = DNestSampler(loglike, prior, ndim=2, num_lanes=2,
                     num_particles=3, seed=7)
    s.run(max_num_saves=60)
    return s


def _sorted_logp(sampler):
    """The bridge's per-sample prior masses, in (logl, tie) order."""
    native = sampler.native_sampler.native_result
    order = np.lexsort((native.tie, native.logl))
    return np.asarray(native.logp)[order]


# ---------------------------------------------------------------------------
# T = 1 identity
# ---------------------------------------------------------------------------
def test_results_at_temperature_1_bit_identical(sampler):
    """T = 1 must reproduce to_results() field-for-field, bit-exactly."""
    ref = sampler.to_results()
    hot = sampler.results_at_temperature(1.0)
    assert set(hot.keys()) == set(ref.keys())

    def same(a, b):
        if isinstance(a, np.ndarray):
            return a.tobytes() == b.tobytes()
        if isinstance(a, float) and np.isnan(a):
            return bool(np.isnan(b))
        return a == b

    for key in ref.keys():
        assert same(ref[key], hot[key]), f'field {key!r} differs'


def test_results_at_temperature_integer_one(sampler):
    """An integer temperature is a valid real number (float(2) == 2.0)."""
    ref = sampler.results_at_temperature(1.0)
    hot = sampler.results_at_temperature(1)
    assert hot['logz'] == ref['logz']
    assert hot['logwt'].tobytes() == ref['logwt'].tobytes()


def test_posterior_at_temperature_1_matches_posterior(sampler):
    points, log_w, log_l = sampler.posterior_at_temperature(1.0)
    pts0, log_w0, log_l0 = sampler.posterior()
    np.testing.assert_array_equal(points, pts0)
    np.testing.assert_array_equal(log_w, log_w0)
    np.testing.assert_array_equal(log_l, log_l0)
    assert logsumexp(log_w) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# plotting-compatibility gap fields
# ---------------------------------------------------------------------------
def test_gap_fields_present_and_semantic(sampler, ensemble_sampler):
    for s, expected_nlive in ((sampler, 1), (ensemble_sampler, 6)):
        res = s.to_results()
        # nlive is the ensemble size (num_lanes * num_particles), a proxy
        # for the static-NS live-point count, not a static-NS nlive.
        assert res['nlive'] == expected_nlive
        assert res['nlive'] == (s.num_lanes * s.num_particles)
        # No per-iteration convergence trajectory exists in diffusive NS.
        traj = res['delta_logZ_trajectory']
        assert isinstance(traj, np.ndarray)
        assert traj.shape == (res['n_saves'],)
        assert np.all(np.isnan(traj))
        # Random-walk steps are not a DNest concept.
        assert np.isnan(res['rwalk_K'])


def test_shared_plotting_runs_without_keyerror(sampler):
    """runplot/cornerplot/traceplot/diagnostics on a dnest Results."""
    res = sampler.to_results()

    fig, axes = plotting.runplot(res)
    try:
        assert np.asarray(axes).shape == (4,)
    finally:
        plt.close(fig)

    fig, axes = plotting.cornerplot(res)
    try:
        assert fig is not None
    finally:
        plt.close(fig)

    fig, axes = plotting.traceplot(res)
    try:
        assert np.asarray(axes).shape == (2, 2)
    finally:
        plt.close(fig)

    fig, axes = plotting.diagnostics(res)
    try:
        assert np.asarray(axes).shape == (2, 2)
    finally:
        plt.close(fig)


def test_shared_plotting_runs_on_tempered_result(sampler):
    res = sampler.results_at_temperature(2.0)
    fig, axes = plotting.runplot(res)
    try:
        assert np.asarray(axes).shape == (4,)
    finally:
        plt.close(fig)
    fig, axes = plotting.cornerplot(res)
    try:
        assert fig is not None
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# temperature ladder
# ---------------------------------------------------------------------------
def test_logz_changes_and_is_finite(sampler):
    """T = 2 changes the evidence (direction documented below)."""
    res1 = sampler.to_results()
    res2 = sampler.results_at_temperature(2.0)
    for res in (res1, res2):
        assert np.isfinite(res['logz'])
        # Unnormalised bridge weights sum to the (tempered) evidence.
        assert logsumexp(res['logwt']) == pytest.approx(res['logz'],
                                                        abs=1e-10)
        # The trajectory is the cumulative sum in likelihood order.
        assert res['logz_trajectory'][-1] == pytest.approx(res['logz'],
                                                           abs=1e-10)
    assert abs(res2['logz'] - res1['logz']) > 0.1
    # This problem has logl <= 0 everywhere, so tempering flattens the
    # integrand towards the prior masses and logZ(T) rises towards the
    # prior total mass logZ(inf) = log sum(p) = 0.
    assert res1['logz'] < res2['logz'] < 0.0


def test_large_temperature_approaches_the_prior(sampler):
    """T -> infinity: weights -> prior masses, likelihood factor -> 1.

    Note: the *literal* dynesty-style expectation "max/min weight ratio
    -> 1" does not hold for diffusive NS, because the limiting weights
    are the prior masses themselves and those span many nats
    (order-statistic volumes across levels).  What does become uniform
    is the likelihood factor ``exp(logl / T)``; we assert both facts.
    """
    logp = _sorted_logp(sampler)
    logl = sampler.to_results()['logl']
    span = float(logl.max() - logl.min())
    # The prior masses are far from uniform (this is why the full weight
    # ratio cannot tend to 1 for a dnest run).
    assert float(logp.max() - logp.min()) > 10.0

    prev_distance = np.inf
    prev_factor_ratio = np.inf
    prev_logz_abs = np.inf
    for T in (10.0, 100.0, 1000.0):
        res = sampler.results_at_temperature(T)
        # (a) tempered weights converge to the prior masses:
        #     logwt(T) - logp = logl / T elementwise.
        distance = float(np.max(np.abs(res['logwt'] - logp)))
        assert distance < prev_distance
        assert distance == pytest.approx(span / T, rel=0.05)
        # (b) the likelihood factor's max/min ratio -> 1.
        factor_ratio = float(np.exp(span / T))
        assert factor_ratio < prev_factor_ratio
        if T >= 100.0:
            assert factor_ratio < 1.3
        # (c) the evidence approaches the prior total mass 0.
        assert abs(res['logz']) < 1.0
        assert abs(res['logz']) < prev_logz_abs
        prev_distance = distance
        prev_factor_ratio = factor_ratio
        prev_logz_abs = abs(res['logz'])

    assert distance < 0.05
    assert factor_ratio < 1.05


def test_posterior_at_temperature_weights(sampler):
    pts, log_w, log_l = sampler.posterior_at_temperature(2.0)
    pts0, log_w0, log_l0 = sampler.posterior()
    # Sample coordinates and saved likelihoods are temperature-independent.
    np.testing.assert_array_equal(pts, pts0)
    np.testing.assert_array_equal(log_l, log_l0)
    # Weights are tempered (CDNest: logl / T) and normalised.
    assert logsumexp(log_w) == pytest.approx(0.0, abs=1e-12)
    assert float(np.max(np.abs(log_w - log_w0))) > 0.01
    # Consistent with the Results bridge: logwt (unnormalised, in
    # (logl, tie) order) minus the normalised posterior weights (native
    # save order, re-sorted here) is the tempered evidence.
    native = sampler.native_sampler.native_result
    order = np.lexsort((native.tie, native.logl))
    res2 = sampler.results_at_temperature(2.0)
    np.testing.assert_allclose(res2['logwt'] - log_w[order], res2['logz'],
                               atol=1e-10)


def test_tempered_information_and_neff(sampler):
    """H(T) -> 0 and the entropy ESS settles at the prior-mass ESS."""
    res1 = sampler.to_results()
    res1000 = sampler.results_at_temperature(1000.0)
    assert res1['information'] > res1000['information'] >= 0.0
    assert res1000['information'] < 0.01
    logp = _sorted_logp(sampler)
    w_prior = np.exp(logp - logsumexp(logp))
    ess_prior = float(np.exp(-np.sum(w_prior * np.log(w_prior))))
    assert res1000['n_eff'] == pytest.approx(ess_prior, rel=0.01)


# ---------------------------------------------------------------------------
# engine caching / invalidation
# ---------------------------------------------------------------------------
def test_engine_caches_per_temperature(sampler):
    eng = sampler.native_sampler
    r2a = eng.native_result_at_temperature(2.0)
    r2b = eng.native_result_at_temperature(2.0)
    assert r2a is r2b
    assert eng.native_result_at_temperature(1.0) is eng.native_result
    assert eng.native_result_at_temperature(5.0) is not r2a
    # Formatting is deterministic: repeated adapter calls give the same
    # field values (fresh Results objects).
    a = sampler.results_at_temperature(2.0)
    b = sampler.results_at_temperature(2.0)
    assert a is not b
    assert a['logz'] == b['logz']
    assert a['logwt'].tobytes() == b['logwt'].tobytes()


def test_tempered_calls_do_not_disturb_t1_cache(sampler):
    cached = sampler.results
    sampler.results_at_temperature(2.0)
    sampler.posterior_at_temperature(5.0)
    assert sampler.results is cached


def test_continued_run_invalidates_tempered_cache():
    loglike, prior, _ = _gaussian_problem()
    s = DNestSampler(loglike, prior, ndim=2, seed=5)
    s.run(max_num_saves=120)
    eng = s.native_sampler
    hot_a = eng.native_result_at_temperature(2.0)
    assert len(hot_a.logl) == 120

    s.run(max_num_saves=240)
    hot_b = eng.native_result_at_temperature(2.0)
    assert hot_b is not hot_a
    assert len(hot_b.logl) == 240
    assert len(eng.native_result.logl) == 240
    assert hot_a.logz != hot_b.logz


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _bare_sampler():
    loglike, prior, _ = _gaussian_problem()
    return DNestSampler(loglike, prior, ndim=2, seed=0)


def test_bad_temperatures_rejected():
    sampler = _bare_sampler()
    for bad in (0.5, -1.0, float('nan'), float('inf'), 0):
        with pytest.raises(ValueError, match='temperature'):
            sampler.results_at_temperature(bad)
        with pytest.raises(ValueError, match='temperature'):
            sampler.posterior_at_temperature(bad)
    for bad in ('hot', True, None, [2.0]):
        with pytest.raises(TypeError, match='temperature'):
            sampler.results_at_temperature(bad)
        with pytest.raises(TypeError, match='temperature'):
            sampler.posterior_at_temperature(bad)


def test_temperature_before_run_raises():
    sampler = _bare_sampler()
    with pytest.raises(RuntimeError, match='no particles were saved'):
        sampler.results_at_temperature(2.0)
    with pytest.raises(RuntimeError, match='no particles were saved'):
        sampler.posterior_at_temperature(2.0)
