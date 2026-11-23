"""
Integration tests for the Hamiltonian slice sampler (``sample='hslice'``).

The problem is a 5-D correlated Gaussian inside a [-10, 10]^5 box prior, for
which the evidence is analytic:

    Sigma = diag([1, .5, .2, .1, .05])  with 0.5 correlation between
            adjacent dimensions
    logZ_truth = 0.5 * logdet(2 pi Sigma) - ndim * log(20)

The hslice tests are deterministic: ``run_nested`` seeds the run from
``seed`` (default 42), so the same sampler configuration always produces the
same evidence.  The measured seed-to-seed spread of the evidence at
nlive=150 is ~0.2 for hslice and ~0.6 for rwalk (the rwalk control run uses
seed 42, which sits at -0.1), which is why the tolerances below are of order
the intrinsic scatter rather than arbitrarily tight.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jnesty import NestedSampler


NDIM = 5
BOX = 20.0  # prior maps u in [0, 1] to [-10, 10]


@pytest.fixture(scope='module', autouse=True)
def _float32_precision():
    """Pin float32 for this module, then restore the caller's setting.

    ``jax_enable_x64`` is process-global and another test module enables x64
    at import time, which changes every acceptance decision (and hence the
    evidence) of these chaotic sampling runs.  Pinning the mode here keeps
    the assertions below reproducible regardless of test ordering; the values
    quoted in the comments were measured in both modes.
    """
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', old)


def _make_problem():
    """5-D correlated Gaussian: (loglik, prior_transform, logZ_truth)."""
    diag = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
    sigma = np.diag(diag)
    for i in range(NDIM - 1):
        s = 0.5 * np.sqrt(diag[i] * diag[i + 1])
        sigma[i, i + 1] = sigma[i + 1, i] = s
    # Sanity: SPD.
    np.linalg.cholesky(sigma)
    inv = np.linalg.inv(sigma)
    logdet = float(np.log(np.linalg.det(2.0 * np.pi * sigma)))
    logz_truth = 0.5 * logdet - NDIM * np.log(BOX)

    def loglik(x):
        x = jnp.asarray(x)
        # Cast inside the trace so the constants follow the active precision
        # mode (jax_enable_x64 is process-global).
        a = jnp.asarray(inv, dtype=x.dtype)
        return -0.5 * (x @ a @ x) - 0.5 * logdet

    def prior_transform(u):
        return (jnp.asarray(u) - 0.5) * BOX

    return loglik, prior_transform, logz_truth


LOGLIK, PRIOR_TRANSFORM, LOGZ_TRUTH = _make_problem()


def _run(sample, rwalk_K, seed=42, nlive=150, **kwargs):
    sampler = NestedSampler(
        LOGLIK, PRIOR_TRANSFORM, NDIM, nlive=nlive, sample=sample,
        bound='single', queue_size=0, rwalk_K=rwalk_K, verbose=False,
        **kwargs)
    sampler.run_nested(max_iterations=6000, delta_logZ_threshold=0.05,
                       print_progress=False, seed=seed)
    return sampler, sampler.results


class TestHSliceEvidence:
    def test_5d_correlated_gaussian_logz(self):
        """hslice recovers the analytic evidence of a 5-D Gaussian."""
        # target_acceptance=0.4: the step size at which the trajectory spans
        # the constrained region with max_move=100.  Measured over 7 seeds:
        # mean evidence error -0.06 +/- 0.19 at 0.4 versus -0.25 +/- 0.23 at
        # the shared default 0.5 (see HSliceSampler's docstring).
        sampler, results = _run('hslice', rwalk_K=5, target_acceptance=0.4)
        assert abs(results['logz'] - LOGZ_TRUTH) < 0.35, (
            f"hslice logz={results['logz']:.3f} vs truth {LOGZ_TRUTH:.3f}")
        # The run must have converged and be a real run, not a lucky stop.
        assert results['niter'] > 3 * 150
        assert results['delta_logz'] < 0.05
        assert results['ncall'] >= results['niter']

    def test_rwalk_control_same_problem(self):
        """The same problem definition is recovered by the rwalk sampler.

        This is a control on the *problem*, not a precision test of rwalk:
        at nlive=150 its intrinsic seed-to-seed evidence spread is +-0.6
        (measured over 6 seeds), ten times larger than hslice's, so the
        tolerance is the larger of the hslice band and 2.5 sigma.
        """
        _, results = _run('rwalk', rwalk_K=25)
        tolerance = max(0.35, 2.5 * results['logzerr'])
        assert abs(results['logz'] - LOGZ_TRUTH) < tolerance, (
            f"rwalk logz={results['logz']:.3f} vs truth {LOGZ_TRUTH:.3f} "
            f"(tolerance {tolerance:.2f})")

    def test_hslice_ncall_exceeds_niter(self):
        _, results = _run('hslice', rwalk_K=5, target_acceptance=0.4,
                          nlive=100)
        assert results['ncall'] >= results['niter']
        assert results['ncall'] > 0


class TestHSlicePlumbing:
    def test_queue_guard_forces_queue_off(self):
        """queue mode is rwalk-only: it must be disabled for hslice."""
        sampler = NestedSampler(
            LOGLIK, PRIOR_TRANSFORM, NDIM, nlive=50, sample='hslice',
            bound='multi', queue_size=4, verbose=False)
        assert sampler.queue_size == 0
        assert sampler.batch_size == 1

    def test_default_sample_is_rwalk(self):
        sampler = NestedSampler(
            LOGLIK, PRIOR_TRANSFORM, NDIM, nlive=50, verbose=False)
        assert sampler.sample == 'rwalk'
        assert sampler.grad_loglikelihood is None
        assert sampler.hslice_kwargs is None

    def test_invalid_sample_raises(self):
        with pytest.raises(ValueError, match="sample must be"):
            NestedSampler(LOGLIK, PRIOR_TRANSFORM, NDIM, nlive=20,
                          sample='slice', verbose=False)

    def test_hslice_kwargs_stored(self):
        sampler = NestedSampler(
            LOGLIK, PRIOR_TRANSFORM, NDIM, nlive=50, sample='hslice',
            hslice_kwargs=dict(max_move=7, max_reflects=2), verbose=False)
        assert sampler.hslice_kwargs == dict(max_move=7, max_reflects=2)
        assert sampler.queue_size == 0
        assert sampler.batch_size == 1

    def test_explicit_gradient_is_used(self):
        """A user-supplied unit-cube gradient must be honoured."""
        sigma = np.diag([1.0, 0.5, 0.2, 0.1, 0.05])
        for i in range(NDIM - 1):
            s = 0.5 * np.sqrt(sigma[i, i] * sigma[i + 1, i + 1])
            sigma[i, i + 1] = sigma[i + 1, i] = s
        inv = np.linalg.inv(sigma)

        def grad_loglik_u(u):
            # d/du logL(prior_transform(u)) = BOX * (-Sigma^-1 x)
            u = jnp.asarray(u)
            x = (u - 0.5) * BOX
            return -BOX * (jnp.asarray(inv, dtype=u.dtype) @ x)

        sampler, results = _run(
            'hslice', rwalk_K=5, target_acceptance=0.4, nlive=100,
            grad_loglikelihood=grad_loglik_u)
        assert abs(results['logz'] - LOGZ_TRUTH) < 0.6
        assert results['ncall'] >= results['niter']


class TestCoreQueueGuard:
    def test_config_level_queue_guard(self):
        """The engine itself drops queue mode for hslice (belt and braces)."""
        from jax import random as jrandom
        from jnesty.dynesty.core import WhileLoopNSConfig, run_nested_sampling

        config = WhileLoopNSConfig(
            nlive=20, max_iterations=200, delta_logZ_threshold=10.0,
            rwalk_K=2, sample='hslice', queue_size=4, bound='single',
            verbose=False, print_progress=False)
        result = run_nested_sampling(
            LOGLIK, lambda key: jrandom.uniform(key, shape=(NDIM,)), NDIM,
            config, key=jrandom.PRNGKey(0),
            prior_transform_fn=PRIOR_TRANSFORM)
        assert isinstance(result.ncall, int)
        assert result.ncall > 0


class TestSeedReproducibility:
    def _tiny_run(self, seed):
        sampler = NestedSampler(
            LOGLIK, PRIOR_TRANSFORM, NDIM, nlive=50, sample='hslice',
            bound='single', queue_size=0, rwalk_K=3, verbose=False)
        sampler.run_nested(max_iterations=300, delta_logZ_threshold=10.0,
                           print_progress=False, seed=seed)
        return sampler.results['logz']

    def test_same_seed_identical(self):
        assert self._tiny_run(7) == self._tiny_run(7)

    def test_different_seed_differs(self):
        assert self._tiny_run(7) != self._tiny_run(8)

    def test_default_seed_is_42(self):
        logz_default = self._tiny_run(42)
        assert logz_default == self._tiny_run(42)


class TestAxesRouting:
    """The core must route IDENTITY axes to hslice (upstream parity).

    Upstream dynesty (nestedsamplers.py ``propose_live``) gives ellipsoid
    axes only to 'rwalk'/'rslice'/'slice'; hslice receives ``np.identity``.
    jnesty's core historically passed the bound axes to every sampler,
    which collapsed the hslice evidence under ``bound='multi'``
    (T1 2026-11-23: corr5 logZ err -4.29/-4.77, gaussian20 +3.80).  These
    tests pin the corrected routing: hslice -> identity (both bounds),
    rwalk -> bound axes (both bounds).  Deterministic: no statistical
    tolerance is involved.
    """

    NDIM2 = 2

    @staticmethod
    def _loglik2(x):
        return -0.5 * jnp.sum(jnp.asarray(x) ** 2)

    @staticmethod
    def _ptform2(u):
        return (jnp.asarray(u) - 0.5) * 10.0

    def _run_with_spy(self, monkeypatch, sample, bound):
        """Run a tiny engine loop with a spy internal sampler.

        Returns the list of ``is_identity`` flags recorded for the axes
        the core passed to ``sampler_obj.sample``.  The spy accepts every
        proposal (returns the start point with logL above the constraint),
        so the loop itself is trivial; only the routing is under test.
        """
        import jnesty.dynesty.internal_samplers as _isamplers
        from jax import random as jrandom
        from jnesty.dynesty.core import (WhileLoopNSConfig,
                                         clear_compiled_units,
                                         run_nested_sampling)

        recorded = []

        class _SpySampler:
            def __init__(self, ndim, target_acceptance=0.5, batch_size=1,
                         ncdim=None, **kwargs):
                self.ndim = ndim
                self.batch_size = 1
                self.ncdim = ncdim if ncdim is not None else ndim

            def sample(self, key, x_starts, logL_constraint,
                       loglikelihood_fn, axes, scale, n_steps,
                       prior_bounds=None, walk_schedule=None, context=None):
                # ``axes`` is traced inside the jitted loop (even the
                # identity constant), so materialize it at runtime with a
                # debug callback and record whether it is the identity.
                def _record(a, ndim=self.ndim):
                    arr = np.asarray(a)
                    recorded.append(bool(
                        arr.shape == (ndim, ndim)
                        and np.allclose(arr, np.eye(ndim, dtype=arr.dtype))))
                jax.debug.callback(_record, axes)
                x = jnp.asarray(x_starts)
                logl = jnp.asarray(logL_constraint) + 1.0
                one = jnp.array(1, dtype=jnp.int32)
                return x, logl, one, one

            def tune(self, scale, acceptance_rate, ndim, iteration):
                return scale

        def _spy_get_sampler(name, ndim, **kwargs):
            return _SpySampler(ndim, **kwargs)

        # The engine caches compiled kernels; make sure this run compiles
        # against the spy rather than reusing a previously cached program.
        clear_compiled_units()
        monkeypatch.setattr(_isamplers, 'get_sampler', _spy_get_sampler)

        config = WhileLoopNSConfig(
            nlive=8, max_iterations=3, delta_logZ_threshold=10.0,
            rwalk_K=2, sample=sample, bound=bound, queue_size=0,
            batch_size=1, verbose=False, print_progress=False)
        run_nested_sampling(
            self._loglik2,
            lambda key: jrandom.uniform(key, shape=(self.NDIM2,)),
            self.NDIM2, config, key=jrandom.PRNGKey(0),
            prior_transform_fn=self._ptform2)
        monkeypatch.undo()
        clear_compiled_units()
        return recorded

    def test_hslice_gets_identity_axes_single(self, monkeypatch):
        recorded = self._run_with_spy(monkeypatch, 'hslice', 'single')
        assert recorded and all(recorded)

    def test_hslice_gets_identity_axes_multi(self, monkeypatch):
        recorded = self._run_with_spy(monkeypatch, 'hslice', 'multi')
        assert recorded and all(recorded)

    def test_rwalk_gets_bound_axes_single(self, monkeypatch):
        recorded = self._run_with_spy(monkeypatch, 'rwalk', 'single')
        assert recorded and not any(recorded)

    def test_rwalk_gets_bound_axes_multi(self, monkeypatch):
        recorded = self._run_with_spy(monkeypatch, 'rwalk', 'multi')
        assert recorded and not any(recorded)