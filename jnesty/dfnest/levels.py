"""Host-side level bookkeeping (CDNest ``do_bookkeeping`` and friends).

Every function here is a float64 transcription of the corresponding part
of CDNest's ``src/dnest.c`` (commit ``3aaa039f9a0a...``):

* :func:`enough_levels`      -- the PTol / max-levels stop test for level
  creation (including the capacity fallback at ``LEVEL_NUM_MAX``).
* :func:`create_level`       -- sort the above-threshold reservoir in
  descending ``(logL, tiebreaker)`` order, place the new threshold at
  ``int(N / compression)`` and retain only the records above it.
* :func:`recalculate_log_X`  -- learn each level's prior volume from the
  regularised exceed/visit statistics.
* :func:`renormalise_visits` -- cap the counter histories once level
  creation finishes (with CDNest's ``+1`` smoothing and truncation).
* :func:`kill_probabilities` -- the lagging-particle kill probabilities.
* :func:`delta_log_z_estimate` -- the opt-in dynesty-style stopping
  statistic (remaining evidence ratio from the level structure alone).
"""

import math
import warnings

import numpy as np

from .state import LEVEL_NUM_MAX

__all__ = ['enough_levels', 'create_level', 'recalculate_log_X',
           'renormalise_visits', 'kill_probabilities', 'log_push_of',
           'delta_log_z_estimate']


def enough_levels(level_logl, n_levels, max_num_levels, ptol):
    """CDNest's ``enough_levels``: has level creation finished?

    With ``max_num_levels == 0`` the criterion is the PTol test over the
    most recent level gaps: at least 10 levels, a mean gap below ``ptol``
    and a maximum gap below ``1.1 * ptol``.  The gap window is the last 20
    gaps (``int(sqrt(20) * sqrt(0.25 * n))`` once ``n > 80``), truncated at
    the first gap because level 0's threshold is ``-inf``.  The hard
    capacity ``LEVEL_NUM_MAX`` also stops creation.  With a positive
    ``max_num_levels`` the criterion is simply ``n_levels >= max_num_levels``.
    """
    level_logl = np.asarray(level_logl, dtype=np.float64)
    n_levels = int(n_levels)
    ptol = float(ptol)

    if max_num_levels != 0:
        return n_levels >= int(max_num_levels)

    if n_levels >= LEVEL_NUM_MAX:
        return True
    if n_levels < 10:
        return False

    num_to_check = 20
    if n_levels > 80:
        num_to_check = int(math.sqrt(20.0) * math.sqrt(0.25 * n_levels))

    k = n_levels - 1
    total = 0.0
    biggest = -np.inf
    count = 0
    for _ in range(num_to_check):
        if k < 1:
            break
        diff = level_logl[k] - level_logl[k - 1]
        total += diff
        if diff > biggest:
            biggest = diff
        k -= 1
        count += 1
    return (total / count < ptol) and (biggest < ptol * 1.1)


def create_level(book, above_vals, above_ties, compression):
    """Create one new level from the above-threshold reservoir.

    Implements CDNest's ``do_bookkeeping`` creation branch: sort descending
    by ``(logL, tiebreaker)``, pick the record at ``int((1/compression) *
    N)`` as the new threshold, and retain only the top ``index`` records
    for future level creation.

    Parameters
    ----------
    book : LevelBook
        The level array to append to (must have spare capacity).
    above_vals, above_ties : ndarray (n,)
        The accumulated reservoir.
    compression : float
        Level compression ratio (``> 1``).

    Returns
    -------
    created : bool
        Whether a level was appended.
    above_vals, above_ties : ndarray
        The truncated reservoir.
    threshold : float or None
        The new level's threshold (diagnostics).
    """
    n = above_vals.shape[0]
    if book.n_levels >= book.capacity:
        warnings.warn(
            f'Level array is at capacity ({book.capacity}); CDNest stops '
            'creating levels here. Increase max_num_levels or the '
            'compression if this fires early.', stacklevel=2)
        return False, above_vals, above_ties, None

    order = np.lexsort((-above_ties, -above_vals))
    vals = above_vals[order]
    ties = above_ties[order]
    index = int((1.0 / compression) * n)

    i = book.n_levels
    book.logl[i] = vals[index]
    book.tie[i] = ties[index]
    book.logX[i] = 0.0
    book.visits[i] = 0
    book.exceeds[i] = 0
    book.accepts[i] = 0
    book.tries[i] = 0
    book.n_levels = i + 1

    # Retain exactly the records above the new threshold.
    return True, vals[:index].copy(), ties[:index].copy(), float(vals[index])


def recalculate_log_X(book, regularisation, compression):
    """CDNest's ``recalculate_log_X``.

    ``logX[0] = 0`` and
    ``logX[i] = logX[i-1] + log((exceeds[i-1] + R/compression) /
    (visits[i-1] + R))`` with ``R`` the regularisation.  The counters are
    non-negative and ``exceeds <= visits`` always holds, so every ratio lies
    in ``(0, 1)`` and the volumes strictly decrease.
    """
    n = book.n_levels
    book.logX[0] = 0.0
    r_over_c = (1.0 / compression) * regularisation
    for i in range(1, n):
        numerator = book.exceeds[i - 1] + r_over_c
        denominator = book.visits[i - 1] + regularisation
        book.logX[i] = book.logX[i - 1] + math.log(numerator / denominator)


def renormalise_visits(book, regularisation):
    """CDNest's ``renormalise_visits``: cap the counter histories.

    Once level creation finishes, every level whose counters exceed the
    regularisation is rescaled to the regularisation with the ``+1``
    smoothing ``accepts = (accepts + 1) / (tries + 1) * R`` (truncated to
    integer, as the C assignment to ``unsigned long long`` does), and
    likewise for exceeds/visits.  This keeps old statistics from drowning
    out new ones while the sampler diffuses across the finished level set.
    """
    r = regularisation
    for i in range(book.n_levels):
        if book.tries[i] >= r:
            book.accepts[i] = int(
                (book.accepts[i] + 1.0) / (book.tries[i] + 1.0) * r)
            book.tries[i] = int(r)
        if book.visits[i] >= r:
            book.exceeds[i] = int(
                (book.exceeds[i] + 1.0) / (book.visits[i] + 1.0) * r)
            book.visits[i] = int(r)


def log_push_of(level_index, n_levels, lam, levels_done):
    """CDNest's ``log_push`` for one level index."""
    if levels_done:
        return 0.0
    return (int(level_index) - (int(n_levels) - 1)) / float(lam)


def kill_probabilities(push):
    """CDNest's lagging-particle kill probability ``(1 - 1/(1+e^{-push-4}))**3``.

    Deeply lagging particles (very negative push) are killed with
    probability near one; particles at the highest level are killed with
    probability ~6e-6.  ``exp`` may overflow to ``inf`` for extreme push,
    which correctly yields a kill probability of one.
    """
    push = np.asarray(push, dtype=np.float64)
    with np.errstate(over='ignore'):
        rate = 1.0 - 1.0 / (1.0 + np.exp(-push - 4.0))
    return rate ** 3


def delta_log_z_estimate(level_logX, level_logl, logl_max):
    """Remaining-evidence ratio for the opt-in delta-logZ stopping rule.

    Implements the dynesty-style stopping statistic adapted to a diffusive
    nested sampling level book::

        delta = logaddexp(0, logl_max + logX_tail - logz_running)

    where ``logX_tail = level_logX[-1]`` is the *learned prior volume of the
    last (highest-threshold) level*: every point with likelihood above the
    highest threshold lives inside prior volume ``exp(logX_tail)``, so this
    is the level-book quantity that bounds the unexplored volume, and the
    maximum conceivable evidence contribution of that region is
    ``exp(logl_max + logX_tail)``.

    ``logz_running`` is a cheap, deliberately *conservative* (low-biased)
    estimate of the current evidence derived from the level structure alone
    -- the CDNest postprocessor is O(n_saves) and must not run per block:

    * the shell between thresholds ``i-1`` and ``i`` has volume
      ``exp(logX[i-1]) - exp(logX[i])`` and likelihood within
      ``[logl[i-1], logl[i]]``; it is counted at its *lower* bound
      ``logl[i-1]`` (the first shell, whose lower bound is level 0's
      ``-inf`` threshold, drops out entirely);
    * the last level's region is counted at its threshold likelihood,
      ``logX[-1] + logl[-1]``.

    A level spans a factor-``compression`` volume slice, so the per-shell
    likelihood curvature biases ``logz_running`` low by O(1) nat.  That is
    the safe direction for a stop decision: an underestimated current
    evidence inflates ``delta`` and delays stopping.  The estimator does
    not see the order-statistic noise of the saved particles, which is why
    the engine also enforces a minimum save floor before stopping.

    Parameters
    ----------
    level_logX, level_logl : array-like (n_levels,)
        Learned level prior volumes (non-increasing) and log-likelihood
        thresholds (non-decreasing), as maintained by the
        :class:`~jnesty.dfnest.state.LevelBook`.
    logl_max : float
        The maximum log-likelihood seen so far (running max over the live
        particles); may be ``-inf`` for a degenerate all-rejected run.

    Returns
    -------
    delta : float
        ``log(1 + Z_tail_max / Z_running)``; ``+inf`` when the running
        evidence cannot be bounded (degenerate books, e.g. a single level
        or an all-``-inf`` likelihood), meaning "do not stop".
    logz_running : float
        The conservative running evidence estimate described above.

    Notes
    -----
    All computations are O(n_levels) float64 NumPy with no random numbers,
    so calling this once per bookkeeping cycle never perturbs the sampling
    trajectory.
    """
    logX = np.asarray(level_logX, dtype=np.float64)
    logl = np.asarray(level_logl, dtype=np.float64)
    logl_max = float(logl_max)
    if logX.ndim != 1 or logl.ndim != 1:
        raise ValueError('level_logX and level_logl must be 1-D arrays')
    if logX.shape[0] != logl.shape[0]:
        raise ValueError('level_logX and level_logl must have equal length')
    n = logX.shape[0]
    if n < 1:
        raise ValueError('delta_log_z_estimate needs at least one level')

    # Shell volumes between consecutive thresholds (logX is strictly
    # decreasing in a live book; ties give a zero-volume shell).
    if n > 1:
        with np.errstate(divide='ignore'):
            shell_logv = logX[:-1] + np.log1p(-np.exp(logX[1:] - logX[:-1]))
        terms = shell_logv + logl[:-1]
    else:
        terms = np.zeros(0, dtype=np.float64)

    # Tail region counted at its threshold likelihood (lower bound).
    tail_lower = float(logX[n - 1]) + float(logl[n - 1])
    logz_running = float(np.logaddexp.reduce(
        np.concatenate([terms, np.array([tail_lower])])))

    logx_tail = float(logX[n - 1])
    if (not math.isfinite(logz_running)) or (logl_max == -math.inf):
        # logz_running == -inf (no usable shells) or no finite likelihood
        # yet: the remaining evidence cannot be bounded; never stop.
        return math.inf, logz_running
    delta = float(np.logaddexp(0.0, logl_max + logx_tail - logz_running))
    if math.isnan(delta):
        return math.inf, logz_running
    return delta, logz_running
