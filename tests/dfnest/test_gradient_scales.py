"""Opt-in gradient-informed per-coordinate proposal scales (task_007 Unit B).

Covers the four layers of the feature:

* the pure state helpers (:func:`~jnesty.dfnest.state.
  update_gradient_magnitude_average` and
  :func:`~jnesty.dfnest.state.derive_scale_multipliers`) — fast tests;
* constructor validation of ``gradient_scales`` / ``gradient_interval``
  and adapter forwarding — fast tests;
* default bit-identity: ``gradient_scales=False`` (the default) must
  reproduce the pre-feature sampling path byte-for-byte.  The reference
  digest below was produced by the *pre-change* code (git-stash red-green
  at implementation time): every array of a seeded 2-D run, SHA-256 over
  ``sorted(name) + name + array.tobytes()``;
* engine integration: scales adapt in the expected direction on an
  anisotropic target, survive forbidden (``-inf``) likelihood regions,
  and keep the evidence correct on the 128-d GMRF "G1" problem of the
  T1c Phase-1 evidence study.  G1 is recreated *verbatim* from
  ``task01c_rwalk_evidence_robustness/common_g1.py`` (which copies
  ``task03_continuity_priors/diag_evidence.py``); the reconstruction was
  verified bit-identical to the blr_mc ``gmrf_precision`` output, and the
  off-arm reproduces the recorded T1c dfnest reference
  (logZ = -139.4596, err = -0.91 at seed 42).

Real-engine tests (JAX JIT + chains), hence slow-marked like
``test_engine.py``; the fast tests above run without compiling anything.
"""

import hashlib
import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import jax.scipy.special as jsp

from jnesty import DNestSampler
from jnesty.dfnest.state import (
    GRADIENT_SCALES_CAP,
    derive_scale_multipliers,
    update_gradient_magnitude_average,
    validate_gradient_options,
)

# Enable x64 at *module import*, before any problem is built (the same
# convention as test_engine.py/test_stop.py).  This matters for the G1
# fixture: its jnp constants (the whitening matrix, the m vector) must be
# float64, as in the T1c study (whose import chain enables x64 through
# blr_mc.fitting.nested).  Without this line the G1 numbers depend on
# which test modules pytest happens to collect first, because with x64
# off ``jnp.asarray`` silently rounds those constants to float32.
jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# problem builders
# ---------------------------------------------------------------------------
def _isotropic_problem(ndim=2):
    """Standard Gaussian in a [-5, 5]^ndim box."""
    def loglikelihood(x):
        return -0.5 * jnp.sum(x ** 2)

    def prior_transform(u):
        return (u - 0.5) * 10.0

    return loglikelihood, prior_transform


def _anisotropic_problem(sigma=(1.0, 0.1)):
    """Anisotropic Gaussian: coordinate 1 is the *sensitive* (narrow)
    one, so it must end up with the smaller scale multiplier."""
    sig = jnp.asarray(sigma, dtype=jnp.float64)

    def loglikelihood(x):
        return -0.5 * jnp.sum((x / sig) ** 2)

    def prior_transform(u):
        return (u - 0.5) * 10.0

    # Z = (2 pi sig0 sig1) / 100 for a prior uniform on [-5, 5]^2
    truth = math.log(2.0 * math.pi * sigma[0] * sigma[1] / 100.0)
    return loglikelihood, prior_transform, truth


# ---------------------------------------------------------------------------
# pure helpers (fast)
# ---------------------------------------------------------------------------
class TestUpdateGradientMagnitudeAverage:
    def test_first_observation_is_the_average(self):
        avg, count = update_gradient_magnitude_average(
            np.zeros(3), np.array([1.0, 2.0, 4.0]), 0)
        assert list(avg) == [1.0, 2.0, 4.0]
        assert count == 1

    def test_incremental_mean_matches_direct_mean(self):
        rng = np.random.default_rng(3)
        samples = np.abs(rng.normal(size=(17, 4))) * 10.0 ** rng.uniform(
            -3, 3, size=(17, 4))
        avg = np.zeros(4)
        count = 0
        for mags in samples:
            avg, count = update_gradient_magnitude_average(avg, mags, count)
        assert count == 17
        np.testing.assert_allclose(avg, samples.mean(axis=0), rtol=1e-12)

    def test_shape_and_count_validation(self):
        with pytest.raises(ValueError):
            update_gradient_magnitude_average(np.zeros(3), np.zeros(4), 0)
        with pytest.raises(ValueError):
            update_gradient_magnitude_average(np.zeros(3), np.zeros(3), -1)


class TestDeriveScaleMultipliers:
    def test_direction_and_geometric_mean(self):
        # sigma = [1, 0.1]  =>  E|grad| ~ [1, 10]  =>  scales ~ [sqrt(10),
        # 1/sqrt(10)]: the sensitive coordinate gets the SMALLER multiplier
        # and the geometric mean over live coordinates is exactly one.
        scales = derive_scale_multipliers(np.array([1.0, 10.0]))
        np.testing.assert_allclose(scales, [math.sqrt(10.0),
                                            1.0 / math.sqrt(10.0)],
                                   rtol=1e-12)
        assert scales[1] < scales[0]
        np.testing.assert_allclose(
            np.exp(np.mean(np.log(scales))), 1.0, rtol=1e-12)

    def test_all_zeros_return_ones(self):
        np.testing.assert_array_equal(
            derive_scale_multipliers(np.zeros(5)), np.ones(5))

    def test_dead_coordinate_gets_cap_others_normalised(self):
        avg = np.array([0.0, 3.0, 3.0])
        scales = derive_scale_multipliers(avg)
        assert scales[0] == GRADIENT_SCALES_CAP
        # the two live coordinates average out to 1 exactly
        np.testing.assert_allclose(
            np.exp(np.mean(np.log(scales[1:]))), 1.0, rtol=1e-12)
        assert np.all(scales == np.asarray([100.0, 1.0, 1.0]))

    def test_extreme_ratios_are_clipped(self):
        scales = derive_scale_multipliers(
            np.array([1e-12, 1.0, 1.0, 1.0]))
        assert scales.max() <= GRADIENT_SCALES_CAP + 1e-9
        assert scales.min() >= 1.0 / GRADIENT_SCALES_CAP - 1e-9

    def test_validation(self):
        with pytest.raises(ValueError):
            derive_scale_multipliers(np.array([1.0, -0.5]))
        with pytest.raises(ValueError):
            derive_scale_multipliers(np.array([1.0, np.nan]))
        with pytest.raises(ValueError):
            derive_scale_multipliers(np.zeros((2, 2)))


class TestOptionValidation:
    def test_valid_options_pass_through(self):
        assert validate_gradient_options(True, 100) == (True, 100)
        assert validate_gradient_options(False, 7) == (False, 7)
        assert validate_gradient_options(np.bool_(True), 5) == (True, 5)

    def test_flag_must_be_boolean(self):
        for bad in ('yes', 1, 0, 1.0, None, [True]):
            with pytest.raises(TypeError):
                validate_gradient_options(bad, 100)

    def test_interval_must_be_positive_integer(self):
        for bad in (0, -5):
            with pytest.raises(ValueError):
                validate_gradient_options(True, bad)
        for bad in (1.5, 'x', True, None):
            with pytest.raises(TypeError):
                validate_gradient_options(True, bad)

    def test_adapter_forwards_and_defaults(self):
        loglike, prior = _isotropic_problem()
        s = DNestSampler(loglike, prior, ndim=2, gradient_scales=True,
                         gradient_interval=7)
        native = s.native_sampler
        assert native.gradient_scales is True
        assert native.gradient_interval == 7
        assert native._grad_of_u is not None

        s2 = DNestSampler(loglike, prior, ndim=2)
        native2 = s2.native_sampler
        assert native2.gradient_scales is False
        assert native2.gradient_interval == 100
        assert native2._grad_of_u is None

    def test_adapter_rejects_bad_options(self):
        loglike, prior = _isotropic_problem()
        with pytest.raises(TypeError):
            DNestSampler(loglike, prior, ndim=2, gradient_scales='yes')
        with pytest.raises(ValueError):
            DNestSampler(loglike, prior, ndim=2, gradient_scales=True,
                         gradient_interval=0)


# ---------------------------------------------------------------------------
# engine integration (slow: JIT + chains)
# ---------------------------------------------------------------------------
def _reference_arrays(sampler, logz):
    """Every output array of a completed run (name -> array).

    The digest of this dict (SHA-256 over ``sorted(name) + name +
    array.tobytes()``) is the bit-identity reference: it was computed by
    the *pre-feature* code, so any byte-level drift in the default path
    fails the test.
    """
    nat = sampler.native_sampler
    res = nat.native_result
    levels = nat.levels
    return {
        'logz': np.float64(logz),
        'n_like': np.int64(nat.n_like),
        'n_saves': np.int64(nat.n_saves),
        'n_levels': np.int64(nat.n_levels),
        'n_steps': np.int64(nat.n_steps),
        'n_blocks': np.int64(nat.n_blocks),
        'acceptance': np.float64(nat.acceptance_rate),
        'logl': np.asarray(res.logl),
        'logwt': np.asarray(res.logwt),
        'logp': np.asarray(res.logp),
        'samples_u': np.asarray(res.samples_u),
        'samples': np.asarray(res.samples),
        'tie': np.asarray(res.tie),
        'level_logl': levels['logl'],
        'level_logX': levels['logX'],
        'level_visits': levels['visits'],
        'level_exceeds': levels['exceeds'],
        'level_accepts': levels['accepts'],
        'level_tries': levels['tries'],
        'coord_att': nat.coordinate_attempts,
        'coord_acc': nat.coordinate_accepts,
        'info': np.float64(nat.information),
        'neff': np.float64(nat.n_eff),
    }


def _digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.asarray(arrays[name]).tobytes())
    return h.hexdigest()


@pytest.mark.slow
class TestDefaultBitIdentity:
    """``gradient_scales=False`` must be byte-identical to pre-feature code.

    Reference produced by the git-stash red-green at implementation time
    (2-D gaussian, 2 lanes x 2 particles, seed 42, 150 saves, CPU): the
    digest covers all 23 output arrays, and the scalar spots below make
    failures readable.
    """

    REFERENCE_DIGEST = ('0254afab2e450788ab9c2fcc9c4b25b5'
                        'dc7de0c843b67507736839006606bcdf')
    REFERENCE_LOGZ_HEX = '-0x1.59ebfca574025p+1'
    REFERENCE_COUNTS = (24004, 150, 26)  # (n_like, n_saves, n_levels)

    def _run(self, **kwargs):
        loglike, prior = _isotropic_problem()
        s = DNestSampler(loglike, prior, ndim=2, num_lanes=2,
                         num_particles=2, seed=42, **kwargs)
        logz = s.run(max_num_saves=150)
        return s, logz

    def test_prechange_digest(self):
        s, logz = self._run()
        assert _digest(_reference_arrays(s, logz)) == self.REFERENCE_DIGEST
        assert float(logz).hex() == self.REFERENCE_LOGZ_HEX
        nat = s.native_sampler
        assert (nat.n_like, nat.n_saves, nat.n_levels) == \
            self.REFERENCE_COUNTS
        # the opt-in field is absent (None), the event stream is inert
        assert nat._particles.grad_scales is None
        assert nat.n_gradient_events == 0
        assert nat.gradient_interval == 100

    def test_explicit_false_is_the_default(self):
        _, logz_default = self._run()
        _, logz_false = self._run(gradient_scales=False,
                                  gradient_interval=25)
        assert float(logz_default).hex() == float(logz_false).hex()


@pytest.fixture(scope='module')
def anisotropic_pair():
    """(off, off_z, on, on_z, truth) on the sigma=[1, 0.1] Gaussian."""
    loglike, prior, truth = _anisotropic_problem()
    off = DNestSampler(loglike, prior, ndim=2, num_lanes=2,
                       num_particles=2, seed=42)
    off_z = off.run(max_num_saves=200)
    on = DNestSampler(loglike, prior, ndim=2, num_lanes=2,
                      num_particles=2, seed=42, gradient_scales=True,
                      gradient_interval=8)
    on_z = on.run(max_num_saves=200)
    return off, off_z, on, on_z, truth


@pytest.mark.slow
class TestScalesAdaptAnisotropic:
    """The scales must adapt in the expected *direction* (task brief)."""

    def test_sensitive_coordinate_gets_smaller_multiplier(
            self, anisotropic_pair):
        off, off_z, on, on_z, truth = anisotropic_pair
        nat = on.native_sampler
        assert nat.n_gradient_events > 0
        mags = nat.gradient_magnitudes
        mult = nat.gradient_scale_multipliers
        # coordinate 1 (sigma=0.1) is the sensitive one: larger average
        # |d logL / du|, smaller multiplier.
        assert mags[1] > mags[0]
        assert mult[1] < mult[0]
        # geometric mean of the multipliers is one (step budget unchanged)
        np.testing.assert_allclose(
            np.exp(np.mean(np.log(mult))), 1.0, rtol=1e-12)

    def test_gradient_branch_is_live_and_counted(self, anisotropic_pair):
        off, off_z, on, on_z, truth = anisotropic_pair
        # the modulated proposal is a different (still valid) chain
        assert float(off_z).hex() != float(on_z).hex()
        # gradient forward passes show up in the likelihood accounting
        assert on.n_like > off.n_like
        assert on.native_sampler.n_gradient_events == on.n_like - off.n_like

    def test_evidence_stays_correct(self, anisotropic_pair):
        off, off_z, on, on_z, truth = anisotropic_pair
        # symmetry of the scaled proposal => the evidence is unbiased:
        # both arms within 1 nat of the analytic value at this budget.
        assert abs(off_z - truth) <= 1.0
        assert abs(on_z - truth) <= 1.0


@pytest.mark.slow
class TestForbiddenRegions:
    """``-inf`` likelihood regions must be skipped, not fatal.

    The BLR production problem has forbidden regions; a gradient event
    landing on such a particle (or on a non-differentiable point) must
    leave the running average untouched.
    """

    def test_inf_particles_are_skipped_not_fatal(self):
        from scipy.special import ndtr

        def loglikelihood(x):
            ok = x[0] < 2.0
            return jnp.where(
                ok, -0.5 * jnp.sum((x / jnp.array([1.0, 0.1])) ** 2),
                -jnp.inf)

        def prior_transform(u):
            return (u - 0.5) * 10.0

        s = DNestSampler(loglikelihood, prior_transform, ndim=2,
                         num_lanes=2, num_particles=2, seed=11,
                         gradient_scales=True, gradient_interval=8)
        logz = s.run(max_num_saves=80)
        nat = s.native_sampler
        assert nat.n_gradient_events > 0
        mult = nat.gradient_scale_multipliers
        assert np.all(np.isfinite(mult)) and np.all(mult > 0)
        # truncated analytic truth: x0 < 2 removes the upper tail
        truth = math.log(2.0 * math.pi * 1.0 * 0.1 * float(ndtr(2.0))
                         / 100.0)
        assert abs(logz - truth) <= 1.0


# ---------------------------------------------------------------------------
# G1: the 128-d GMRF evidence problem of the T1c Phase-1 study (slow)
# ---------------------------------------------------------------------------
#: Analytic truth of the frozen G1 construction (T1c common_g1.py).
G1_TRUTH = -138.55200433888717
#: T1c's recorded dfnest reference at this exact config (seed 42):
#: logZ = -139.4596, err = -0.91 — reproduced exactly here once the
#: problem is built with x64 enabled (as the T1c import chain does via
#: blr_mc); with x64 off, jnp.asarray rounds the whitening matrix to
#: float32 and the chain diverges (measured -139.7191).  The assertions
#: use the environment-independent evidence band of the T1c verdict.
G1_T1C_DFNEST_ERR = -0.91


def _build_g1():
    """Recreate the frozen G1 problem (see module docstring for provenance).

    Returns ``(loglikelihood, prior_transform, truth)`` with
    ``z ~ N(0, Q^-1)`` via GMRF whitening of iid standard normals and
    ``logL = N(z; m, 0.5^2 I)``; ``Q`` is the (4, 8, 4) second-difference
    precision with the 0.3/0.3/0.3 smoothness and the mean-marginal
    rescale to ``sigma_z = 1``.
    """
    shape = (4, 8, 4)
    lam_r, lam_phi, lam_eta = 0.3, 0.3, 0.3

    def stencil_scales():
        n_r, n_phi, n_eta = shape
        s_r = lam_r * n_r ** 2 / np.sqrt(max(1, n_r - 2))
        s_phi = lam_phi * n_phi ** 2 / np.sqrt(max(1, n_phi))
        s_eta = lam_eta * n_eta ** 2 / np.sqrt(max(1, n_eta - 2))
        return s_r, s_phi, s_eta

    n_r, n_phi, n_eta = shape
    n_cells = n_r * n_phi * n_eta
    s_r, s_phi, s_eta = stencil_scales()
    n_rows = ((n_r - 2) * n_phi * n_eta + (n_eta - 2) * n_r * n_phi
              + n_phi * n_r * n_eta)
    D = np.zeros((n_rows, n_cells))
    cell = lambda i, p, e: (i * n_phi + p) * n_eta + e  # noqa: E731
    row = 0
    for i in range(1, n_r - 1):
        for p in range(n_phi):
            for e in range(n_eta):
                D[row, cell(i - 1, p, e)] += s_r
                D[row, cell(i, p, e)] += -2.0 * s_r
                D[row, cell(i + 1, p, e)] += s_r
                row += 1
    for e in range(1, n_eta - 1):
        for i in range(n_r):
            for p in range(n_phi):
                D[row, cell(i, p, e - 1)] += s_eta
                D[row, cell(i, p, e)] += -2.0 * s_eta
                D[row, cell(i, p, e + 1)] += s_eta
                row += 1
    for k in range(n_phi):
        for i in range(n_r):
            for e in range(n_eta):
                D[row, cell(i, (k - 1) % n_phi, e)] += s_phi
                D[row, cell(i, k, e)] += -2.0 * s_phi
                D[row, cell(i, (k + 1) % n_phi, e)] += s_phi
                row += 1
    DtD = D.T @ D
    Q0 = DtD + 1e-2 * float(np.mean(np.diag(DtD))) * np.eye(n_cells)
    c = float(np.mean(np.diag(np.linalg.inv(Q0)))) / 1.0 ** 2
    Q = c * Q0
    L = np.linalg.cholesky(Q)
    Lt = jnp.asarray(np.ascontiguousarray(L.T))
    m = 0.7 * np.random.default_rng(5).standard_normal(n_cells)

    def prior_transform(u):
        u = jnp.clip(jnp.asarray(u), 1e-12, 1. - 1e-12)
        return solve_triangular(Lt, jsp.ndtri(u), lower=False)

    def loglikelihood(x):
        r = (jnp.asarray(x) - jnp.asarray(m)) / 0.5
        return (-0.5 * jnp.sum(r ** 2)
                - 0.5 * n_cells * np.log(2. * np.pi * 0.5 ** 2))

    C = np.linalg.inv(Q)
    tot = C + 0.25 * np.eye(n_cells)
    sign, logdet = np.linalg.slogdet(2. * np.pi * tot)
    truth = -0.5 * (logdet + m @ np.linalg.solve(tot, m))
    return loglikelihood, prior_transform, truth


@pytest.fixture(scope='module')
def g1_pair():
    """(off, off_z, on, on_z, truth) at the T1c dfnest configuration.

    Exact T1c Phase-1 arm: ``num_lanes=8, num_particles=1,
    thread_steps_factor=1.0, new_level_interval_factor=2.0,
    save_interval_factor=2.0, max_num_saves=1500, seed=42``.
    """
    loglike, prior, truth = _build_g1()
    assert truth == pytest.approx(G1_TRUTH, abs=1e-9)  # construction guard

    def build(**kwargs):
        return DNestSampler(loglike, prior, ndim=128, num_lanes=8,
                            num_particles=1, thread_steps_factor=1.0,
                            new_level_interval_factor=2.0,
                            save_interval_factor=2.0, seed=42, **kwargs)

    off = build()
    off_z = off.run(max_num_saves=1500)
    on = build(gradient_scales=True)
    on_z = on.run(max_num_saves=1500)
    return off, off_z, on, on_z, truth


@pytest.mark.slow
class TestG1Evidence:
    def test_evidence_within_band(self, g1_pair):
        """|logZ_err| <= 2 with the scales on (T1c band; dfnest measured
        -0.91 there, -1.17 in the implementing environment)."""
        off, off_z, on, on_z, truth = g1_pair
        assert abs(on_z - truth) <= 2.0
        assert abs(off_z - truth) <= 2.0

    def test_same_budget_and_gradient_accounting(self, g1_pair):
        off, off_z, on, on_z, truth = g1_pair
        offn, onn = off.native_sampler, on.native_sampler
        assert on.n_saves == off.n_saves == 1500
        assert onn.n_gradient_events > 0
        # exactly one counted forward pass per gradient event
        assert on.n_like - off.n_like == onn.n_gradient_events
        mult = onn.gradient_scale_multipliers
        assert np.all(np.isfinite(mult)) and np.all(mult > 0)
        np.testing.assert_allclose(
            np.exp(np.mean(np.log(mult))), 1.0, rtol=1e-9)

    def test_efficiency_diagnostic(self, g1_pair):
        """Mixing comparison (reported, not asserted): n_eff and
        acceptance at the same budget, on vs off."""
        off, off_z, on, on_z, truth = g1_pair
        offn, onn = off.native_sampler, on.native_sampler
        m = onn.gradient_scale_multipliers
        print('\nG1 efficiency (1500 saves, seed 42):'
              '\n  off: logZ %.4f  n_eff %.1f  acc %.3f  n_like %d'
              '\n  on : logZ %.4f  n_eff %.1f  acc %.3f  n_like %d'
              '  (events %d, multiplier range [%.3g, %.3g])'
              % (off_z, offn.n_eff, offn.acceptance_rate, offn.n_like,
                 on_z, onn.n_eff, onn.acceptance_rate, onn.n_like,
                 onn.n_gradient_events, float(m.min()), float(m.max())))
        # sanity only: both arms must produce a usable posterior
        assert 10 < onn.n_eff and 10 < offn.n_eff
