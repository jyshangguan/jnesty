"""CDNest-compatible postprocessing for the JAX DNest engine.

This module is a float64 port of the C postprocessor that CDNest's
``sampler.run()`` executes (``src/dnestpostprocess.c`` at commit
``3aaa039f9a0a76919c0ed65b355a50ecb497e6a2``).  The sibling
``python/cydnest/postprocess.py`` is *not* the contract: it randomises the
level compressions and its posterior resampling is disabled.

The algorithm, per level ``i`` with samples sandwiched between thresholds:

1. Stored level assignments are clamped to the current number of levels
   (defensive; assignments are always valid in this engine).
2. Each sample's *sandwich* level is the highest level whose threshold is
   strictly below its (untempered) likelihood.  Level tiebreakers are not
   consulted, exactly as in C.
3. Within a level, samples are sorted ascending by ``(logl, tiebreaker)``.
4. Sorted position ``j`` receives the deterministic order statistic
   ``U_j = Umin + (1 - Umin) * (N - j) / (N + 1)`` between the level's
   volume bounds, hence ``logX_j = logX_max + log(U_j)``.  For ``N >= 2``
   this is algebraically identical to the C expression; for ``N == 1`` it
   is the well-defined limit (the literal C formula evaluates ``0*(0/0)``
   and yields NaN) so singleton levels keep a finite, meaningful ``logX``
   without affecting evidence or information.
5. Each sample's prior mass is the trapezoid between its neighbours'
   volumes, normalised globally.  (C includes a constant ``log(0.5)`` that
   cancels in the normalisation.)
6. ``logZ = logsumexp(logp + logl/T)``; posterior weights are
   ``logwt = logp + logl/T - logZ``; ``H = sum(w * logl/T) - logZ``; and
   the effective sample size is the *entropy* ESS ``exp(-sum(w log w))``.

CDNest reports no evidence uncertainty for a single run, so ``logzerr`` is
deliberately absent/NaN downstream; the entropy ESS measures posterior
weight concentration, not chain autocorrelation.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import logsumexp

__all__ = ['DNestResult', 'dnest_postprocess']


def _logdiffexp(a, b):
    """``log(exp(a) - exp(b))`` for ``a >= b`` (elementwise, float64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return a + np.log1p(-np.exp(b - a))


@dataclass
class DNestResult:
    """Postprocessed outputs of one DNest run, in native save order.

    Attributes
    ----------
    level_logX, level_logl, level_tie : ndarray (n_levels,)
        Level prior volumes, log-likelihood thresholds and threshold
        tiebreakers.
    level_visits, level_exceeds, level_accepts, level_tries : ndarray
        Level statistics (from the engine, for diagnostics).
    samples_u : ndarray (n_samples, ndim) or None
        Saved unit-cube coordinates, in the order they were saved.
    samples : ndarray (n_samples, ndim) or None
        Physical coordinates (filled in by the engine via the prior
        transform).
    logl, tie : ndarray (n_samples,)
        Saved log-likelihoods and tiebreakers.
    assignment : ndarray (n_samples,)
        Stored level assignments (clamped to the number of levels).
    sandwich : ndarray (n_samples,)
        Postprocessed sandwiching levels; group the logX/logp statistics.
    logx : ndarray (n_samples,)
        Postprocessed per-sample log prior volumes.
    logp : ndarray (n_samples,)
        Normalised log prior masses; ``logsumexp(logp) == 0``.
    logwt : ndarray (n_samples,)
        Normalised log posterior weights; ``logsumexp(logwt) == 0``.
    logz, information, n_eff : float
        Evidence, information (nats) and entropy effective sample size.

    Notes
    -----
    ``logwt``, ``logz``, ``information`` and ``n_eff`` reflect the
    ``temperature`` the postprocessor was called with (likelihoods
    divided by ``T``); ``logx``/``logp``/``sandwich``/``assignment`` and
    the sample coordinates are temperature-independent.
    """

    level_logX: np.ndarray
    level_logl: np.ndarray
    level_tie: np.ndarray
    level_visits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    level_exceeds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    level_accepts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    level_tries: np.ndarray = field(default_factory=lambda: np.zeros(0))
    samples_u: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
    logl: Optional[np.ndarray] = None
    tie: Optional[np.ndarray] = None
    assignment: Optional[np.ndarray] = None
    sandwich: Optional[np.ndarray] = None
    logx: Optional[np.ndarray] = None
    logp: Optional[np.ndarray] = None
    logwt: Optional[np.ndarray] = None
    logz: float = float('nan')
    information: float = float('nan')
    n_eff: float = float('nan')

    @property
    def relative_weights(self):
        """CDNest-style max-normalised posterior weights (maximum is 1)."""
        if self.logwt is None:
            raise RuntimeError('postprocessing has not been run')
        return np.exp(self.logwt - np.max(self.logwt))


def dnest_postprocess(level_logX, level_logl, level_tie, sample_logl,
                      sample_tie, sample_assignment, temperature=1.0,
                      level_counters=None):
    """Run the CDNest C postprocessor on saved samples.

    Parameters
    ----------
    level_logX, level_logl, level_tie : array-like (n_levels,)
        Level prior volumes, thresholds and threshold tiebreakers.  Volumes
        must be non-increasing and thresholds non-decreasing.
    sample_logl, sample_tie : array-like (n_samples,)
        Saved log-likelihoods and tiebreakers.
    sample_assignment : array-like (n_samples,)
        Stored level assignments; values above ``n_levels - 1`` are clamped
        (CDNest's defensive behaviour).
    temperature : float, optional
        Posterior temperature ``T >= 1``; likelihoods are divided by ``T``
        for the evidence and weights, while level placement always uses the
        untempered values (as in C).
    level_counters : dict, optional
        Optional per-level ``visits``/``exceeds``/``accepts``/``tries``
        arrays copied into the result for diagnostics.

    Returns
    -------
    DNestResult
    """
    level_logX = np.asarray(level_logX, dtype=np.float64)
    level_logl = np.asarray(level_logl, dtype=np.float64)
    level_tie = np.asarray(level_tie, dtype=np.float64)
    logl = np.asarray(sample_logl, dtype=np.float64)
    tie = np.asarray(sample_tie, dtype=np.float64)
    assignment = np.asarray(sample_assignment)

    n_levels = level_logX.shape[0]
    n_samples = logl.shape[0]

    if n_levels < 1:
        raise ValueError('dnest_postprocess needs at least one level')
    if n_samples < 1:
        raise ValueError('dnest_postprocess needs at least one saved sample')
    if tie.shape != (n_samples,) or assignment.shape != (n_samples,):
        raise ValueError(
            'sample_tie and sample_assignment must match sample_logl')
    if not (math.isfinite(temperature) and temperature >= 1.0):
        raise ValueError(
            f'temperature must be finite and >= 1, got {temperature}')
    if np.any(np.isnan(logl)) or np.any(np.isposinf(logl)):
        raise ValueError('sample log-likelihoods must be finite or -inf, '
                         'never NaN or +inf')
    if np.any(assignment < 0):
        raise ValueError('sample level assignments must be non-negative')

    # Threshold ordering / volume monotonicity sanity checks.
    if np.any(np.diff(level_logl[:n_levels]) < 0):
        raise ValueError('level thresholds must be non-decreasing')
    if np.any(np.diff(level_logX[:n_levels]) > 0):
        raise ValueError('level logX values must be non-increasing')

    assignment = np.minimum(assignment, n_levels - 1).astype(np.int64)

    # Sandwich level: highest level whose threshold is strictly below the
    # (untempered) likelihood, starting from the stored assignment.  With
    # ordered thresholds, #{thresholds < logl} = m makes this max(a, m - 1).
    m = np.array([np.count_nonzero(level_logl[:n_levels] < li) for li in logl],
                 dtype=np.int64)
    sandwich = np.maximum(assignment, m - 1)

    logx = np.full(n_samples, np.nan)
    logp = np.full(n_samples, -np.inf)

    for i in range(n_levels):
        sel = np.flatnonzero(sandwich == i)
        n_level = sel.shape[0]
        if n_level == 0:
            continue
        # Ascending (logl, tiebreaker); fully identical rows fall back to
        # sample order through lexsort's stability (C's qsort order for
        # identical records is unspecified).
        order = np.lexsort((tie[sel], logl[sel]))
        sel_sorted = sel[order]

        x_max = level_logX[i]
        if i == n_levels - 1:
            u_min = 0.0
            x_min = -np.inf
        else:
            x_min = level_logX[i + 1]
            u_min = math.exp(x_min - x_max)

        j = np.arange(n_level)
        # Singleton-safe form of CDNest's order statistic.
        u_vals = u_min + (1.0 - u_min) * (n_level - j) / (n_level + 1.0)
        logx_level = x_max + np.log(u_vals)
        logx[sel_sorted] = logx_level

        # Trapezoidal prior mass between neighbouring volumes.
        right = np.empty(n_level, dtype=np.float64)
        left = np.empty(n_level, dtype=np.float64)
        right[0] = x_max
        left[n_level - 1] = x_min
        if n_level > 1:
            right[1:] = logx_level[:-1]
            left[:-1] = logx_level[1:]

        if np.any(left > right):
            raise ValueError(f'non-monotone logX within level {i}; '
                             'this indicates a bookkeeping bug')
        logp[sel_sorted] = _logdiffexp(right, left)

    norm = logsumexp(logp)
    if not np.isfinite(norm):
        raise ValueError('postprocessing produced no finite prior mass; '
                         'all saved samples have zero weight')
    logp = logp - norm

    logl_t = logl / temperature
    logwt_unnormalised = logp + logl_t
    logz = float(logsumexp(logwt_unnormalised))
    logwt = logwt_unnormalised - logz
    w = np.exp(logwt)

    # 0 * log(0) guards: zero-weight samples must not poison H or the ESS.
    wlogl = np.where(w > 0, w * np.where(np.isfinite(logl_t), logl_t, 0.0),
                     0.0)
    information = float(np.sum(wlogl) - logz)
    with np.errstate(invalid='ignore'):
        wlogw = np.where(w > 0, w * logwt, 0.0)
    n_eff = float(math.exp(-np.sum(wlogw)))

    counters = level_counters or {}
    return DNestResult(
        level_logX=level_logX[:n_levels].copy(),
        level_logl=level_logl[:n_levels].copy(),
        level_tie=level_tie[:n_levels].copy(),
        level_visits=np.asarray(counters.get('visits', np.zeros(0))),
        level_exceeds=np.asarray(counters.get('exceeds', np.zeros(0))),
        level_accepts=np.asarray(counters.get('accepts', np.zeros(0))),
        level_tries=np.asarray(counters.get('tries', np.zeros(0))),
        logl=logl,
        tie=tie,
        assignment=assignment,
        sandwich=sandwich,
        logx=logx,
        logp=logp,
        logwt=logwt,
        logz=logz,
        information=information,
        n_eff=n_eff,
    )
