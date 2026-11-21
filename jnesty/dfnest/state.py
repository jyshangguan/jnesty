"""Configuration, validation and state containers for the JAX DNest engine.

The run controls mirror CDNest (commit ``3aaa039f9a0a76919c0ed65b355a50ecb497e6a2``,
``src/dnest.c``) so that a CDNest OPTIONS file maps one-to-one:

======================  ====================================================
CDNest option           meaning here
======================  ====================================================
NumberParticles         ``num_particles`` (per lane; CDNest: per MPI rank)
(NumberCores)           ``num_lanes`` (independent JAX workers)
ThreadStepsFactor       ``thread_steps_factor``
NewLevelIntervalFactor  ``new_level_interval_factor``
SaveIntervalFactor      ``save_interval_factor`` (None: inherit new-level)
MaxNumberLevels         ``max_num_levels`` (0: PTol-controlled)
BacktrackingLength      ``lam`` (log_push scale while building levels)
StrengthEqualPush       ``beta`` (equal-push strength after building)
PTol                    ``ptol`` (mean level-spacing stop test)
MaxNumberSaves          ``max_num_saves`` (run length, not convergence)
======================  ====================================================

Derived quantities follow the C formulas exactly
(``options_load`` in ``src/dnest.c``)::

    thread_steps        = int(ndim * thread_steps_factor * num_particles)
    new_level_interval  = int(num_lanes * thread_steps * new_level_interval_factor)
    save_interval       = int(num_lanes * thread_steps * save_interval_factor)
    regularisation      = int(new_level_interval * sqrt(lam))

``thread_steps`` sequential updates are run per lane per block, so one block
performs ``num_lanes * thread_steps`` likelihood evaluations.
"""

import math
import warnings
from typing import NamedTuple

import numpy as np

#: CDNest's LEVEL_NUM_MAX (``src/dnest.h``): hard cap on the level array.
LEVEL_NUM_MAX = 1000

__all__ = ['DNestConfig', 'LaneState', 'LevelBook', 'resolve_config',
           'LEVEL_NUM_MAX', 'validate_sample_frequency']


# ----------------------------------------------------------------------------
# validation helpers
# ----------------------------------------------------------------------------

def _require_int(name, value, minimum=0):
    """Validate an integer option; bools are rejected (they are ints in Python)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer, got {type(value).__name__}')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value}')
    return value


def _require_float(name, value, minimum=None, exclusive=False):
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f'{name} must be a real number, got {type(value).__name__}')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite, got {value}')
    if minimum is not None:
        ok = value > minimum if exclusive else value >= minimum
        if not ok:
            kind = '>' if exclusive else '>='
            raise ValueError(f'{name} must be {kind} {minimum}, got {value}')
    return value


def validate_sample_frequency(sample_frequency, ndim):
    """Validate and normalise the per-coordinate proposal weights.

    ``sample_frequency`` selects *which coordinate* a perturbation proposes
    to move.  It accepts a positive scalar (broadcast over all dimensions)
    or a length-``ndim`` sequence of non-negative relative weights; the
    returned vector is normalised to sum to one.  Entries may be zero (the
    coordinate is then never proposed and is effectively frozen), but the
    total must be positive.  Scale-equivalent inputs (``2`` and ``1``,
    ``[2, 4]`` and ``[1, 2]``) yield identical normalised weights.

    Returns
    -------
    weights : ndarray, shape (ndim,)
        The *normalised* coordinate-selection probabilities.
    raw : ndarray, shape (ndim,)
        The broadcast input, for diagnostics.
    """
    if isinstance(sample_frequency, (bool, np.bool_)):
        raise TypeError('sample_frequency must be a real number or array, '
                        'not a boolean')
    if (np.asarray(sample_frequency).dtype
            if not isinstance(sample_frequency, (list, tuple))
            else False) is np.dtype(bool):
        raise TypeError('sample_frequency must not contain booleans')
    raw = np.asarray(sample_frequency, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.full(ndim, float(raw))
    elif raw.ndim == 1:
        if raw.shape[0] != ndim:
            raise ValueError(
                f'sample_frequency must be a scalar or have length ndim={ndim}, '
                f'got shape {raw.shape}')
    else:
        raise ValueError(
            'sample_frequency must be a scalar or 1-D array, got shape '
            f'{raw.shape}')
    if not np.all(np.isfinite(raw)):
        raise ValueError('sample_frequency entries must be finite')
    if np.any(raw < 0):
        raise ValueError('sample_frequency entries must be non-negative')
    total = raw.sum()
    if not (total > 0):
        raise ValueError('sample_frequency must have a positive sum')
    weights = raw / total
    if np.any(weights == 0):
        warnings.warn(
            'sample_frequency contains zero weights: those coordinates are '
            'never proposed by the built-in perturbation and remain frozen '
            'at their initial values.', stacklevel=2)
    return weights, raw


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

class DNestConfig:
    """Fully resolved, immutable run configuration for the DNest engine.

    Construct this through :func:`resolve_config` (which performs all
    validation and derives the CDNest interval formulas) rather than
    directly.
    """

    __slots__ = (
        'ndim', 'num_lanes', 'num_particles', 'total_particles',
        'thread_steps', 'block_steps', 'new_level_interval', 'save_interval',
        'regularisation', 'level_capacity', 'max_num_levels',
        'thread_steps_factor', 'new_level_interval_factor',
        'save_interval_factor', 'lam', 'beta', 'compression', 'ptol',
        'sample_frequency', 'sample_frequency_raw',
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return (f'DNestConfig(ndim={self.ndim}, num_lanes={self.num_lanes}, '
                f'num_particles={self.num_particles}, '
                f'thread_steps={self.thread_steps})')

    def options_summary(self):
        """Dict of the CDNest-equivalent options (for logging and demos)."""
        return {
            'NumberParticles': self.num_particles,
            'NumberLanes': self.num_lanes,
            'ThreadStepsFactor': self.thread_steps_factor,
            'NewLevelIntervalFactor': self.new_level_interval_factor,
            'SaveIntervalFactor': self.save_interval_factor,
            'ThreadSteps': self.thread_steps,
            'NewLevelInterval': self.new_level_interval,
            'SaveInterval': self.save_interval,
            'Regularisation': self.regularisation,
            'MaxNumberLevels': self.max_num_levels,
            'BacktrackingLength': self.lam,
            'StrengthEqualPush': self.beta,
            'Compression': self.compression,
            'PTol': self.ptol,
            'LevelCapacity': self.level_capacity,
        }


def resolve_config(
    ndim,
    num_lanes=1,
    num_particles=1,
    thread_steps_factor=10.0,
    new_level_interval_factor=2.0,
    save_interval_factor=None,
    max_num_levels=0,
    lam=10.0,
    beta=100.0,
    compression=math.e,
    ptol=0.1,
    sample_frequency=None,
):
    """Validate options and derive the CDNest interval formulas.

    Parameters
    ----------
    ndim : int
        Number of parameters (at least 1).
    num_lanes, num_particles : int
        Independent workers (JAX analogue of MPI ranks) and local particles
        per worker, exactly as CDNest counts them.  The total ensemble is
        ``num_lanes * num_particles`` diffusive particles.
    thread_steps_factor, new_level_interval_factor, save_interval_factor : float
        CDNest interval factors.  ``save_interval_factor=None`` inherits
        ``new_level_interval_factor`` (CDNest's default behaviour).
    max_num_levels : int
        0 builds levels until the PTol criterion fires (capped at
        :data:`LEVEL_NUM_MAX`); a positive value fixes the number of levels.
    lam, beta : float
        Backtracking length (log_push scale) and equal-push strength.
    compression : float
        Level compression ratio; must lie in ``(1, e]`` as CDNest requires.
    ptol : float
        Mean log-likelihood spacing that stops level creation.
    sample_frequency : scalar or sequence
        Per-coordinate relative proposal weights; see
        :func:`validate_sample_frequency`.

    Returns
    -------
    DNestConfig
    """
    ndim = _require_int('ndim', ndim, minimum=1)
    num_lanes = _require_int('num_lanes', num_lanes, minimum=1)
    num_particles = _require_int('num_particles', num_particles, minimum=1)
    max_num_levels = _require_int('max_num_levels', max_num_levels, minimum=0)

    thread_steps_factor = _require_float(
        'thread_steps_factor', thread_steps_factor, minimum=0.0, exclusive=True)
    new_level_interval_factor = _require_float(
        'new_level_interval_factor', new_level_interval_factor, minimum=1.0)
    if save_interval_factor is None:
        save_interval_factor = new_level_interval_factor
    else:
        save_interval_factor = _require_float(
            'save_interval_factor', save_interval_factor, minimum=0.0,
            exclusive=True)
    lam = _require_float('lam', lam, minimum=0.0, exclusive=True)
    beta = _require_float('beta', beta, minimum=0.0)
    compression = _require_float('compression', compression)
    if not (1.0 < compression <= math.e):
        raise ValueError(
            f'compression must lie in (1, e] as CDNest requires, '
            f'got {compression}')
    ptol = _require_float('ptol', ptol, minimum=0.0, exclusive=True)

    if sample_frequency is None:
        sample_frequency = 1.0
    weights, raw = validate_sample_frequency(sample_frequency, ndim)

    # Derived quantities -- exactly the C expressions from options_load().
    thread_steps = int(ndim * thread_steps_factor * num_particles)
    block_steps = num_lanes * thread_steps
    new_level_interval = int(block_steps * new_level_interval_factor)
    save_interval = int(block_steps * save_interval_factor)
    regularisation = int(new_level_interval * math.sqrt(lam))

    if thread_steps < 1:
        raise ValueError(
            f'derived thread_steps is {thread_steps}; increase '
            f'thread_steps_factor or num_particles')
    if new_level_interval < block_steps:
        # CDNest: "new level interval should be equal to or larger than
        # totaltask * thread step".
        raise ValueError(
            f'derived new_level_interval ({new_level_interval}) must be >= '
            f'num_lanes * thread_steps ({block_steps}); increase '
            f'new_level_interval_factor')
    if save_interval < 1:
        raise ValueError(
            f'derived save_interval is {save_interval}; increase '
            f'save_interval_factor')
    if regularisation < 1:
        raise ValueError(
            f'derived regularisation is {regularisation}; increase '
            f'new_level_interval_factor or lam')

    level_capacity = max_num_levels if max_num_levels > 0 else LEVEL_NUM_MAX

    return DNestConfig(
        ndim=ndim,
        num_lanes=num_lanes,
        num_particles=num_particles,
        total_particles=num_lanes * num_particles,
        thread_steps=thread_steps,
        block_steps=block_steps,
        new_level_interval=new_level_interval,
        save_interval=save_interval,
        regularisation=float(regularisation),
        level_capacity=level_capacity,
        max_num_levels=max_num_levels,
        thread_steps_factor=thread_steps_factor,
        new_level_interval_factor=new_level_interval_factor,
        save_interval_factor=save_interval_factor,
        lam=lam,
        beta=beta,
        compression=compression,
        ptol=ptol,
        sample_frequency=weights,
        sample_frequency_raw=raw,
    )


# ----------------------------------------------------------------------------
# device-side lane state (a pytree carried through lax.scan)
# ----------------------------------------------------------------------------

class LaneState(NamedTuple):
    """State of one worker (lane): ``num_particles`` local particles.

    All arrays have no leading lane axis; ``jax.vmap`` supplies it.

    ``accepts``/``tries``/``visits``/``exceeds`` are lane-local replicas of
    the global level counters, reset to the broadcast global values at each
    block boundary (exactly how CDNest workers keep their private copy of
    ``levels`` between MPI broadcasts).  ``coord_attempts`` /
    ``coord_accepts`` are cumulative totals that are never reset.
    """

    u: np.ndarray        # (K, D) unit-cube coordinates
    logl: np.ndarray     # (K,) log-likelihoods
    tie: np.ndarray      # (K,) tiebreakers in [0, 1)
    level: np.ndarray    # (K,) int32 level assignments
    accepts: np.ndarray  # (C,) int64 per-level MH accepts
    tries: np.ndarray    # (C,) int64 per-level MH tries
    visits: np.ndarray   # (C,) int64 per-level visits
    exceeds: np.ndarray  # (C,) int64 per-level exceeds
    coord_attempts: np.ndarray  # (D,) int64 cumulative proposal counts
    coord_accepts: np.ndarray   # (D,) int64 cumulative accepted counts


# ----------------------------------------------------------------------------
# host-side level bookkeeping
# ----------------------------------------------------------------------------

class LevelBook:
    """Host-side level array with fixed capacity (CDNest's ``levels_combine``).

    Thresholds are strictly ordered after the first level is created; the
    first level uses ``-inf`` (CDNest uses ``-DBL_MAX``, which compares
    identically for every finite likelihood).
    """

    def __init__(self, capacity: int, rng: np.random.Generator, lam: float):
        self.capacity = int(capacity)
        self.lam = float(lam)
        self.logl = np.full(self.capacity, -np.inf, dtype=np.float64)
        self.tie = np.zeros(self.capacity, dtype=np.float64)
        self.logX = np.zeros(self.capacity, dtype=np.float64)
        self.visits = np.zeros(self.capacity, dtype=np.int64)
        self.exceeds = np.zeros(self.capacity, dtype=np.int64)
        self.accepts = np.zeros(self.capacity, dtype=np.int64)
        self.tries = np.zeros(self.capacity, dtype=np.int64)
        self.n_levels = 1
        # CDNest: LikelihoodType like_tmp = {-DBL_MAX, gsl_rng_uniform(...)}
        self.logl[0] = -np.inf
        self.tie[0] = rng.random()
        self.logX[0] = 0.0
        self.levels_done = False

    def log_push_array(self):
        """CDNest log_push per level: (level-(n-1))/lam while building, else 0."""
        if self.levels_done:
            return np.zeros(self.capacity, dtype=np.float64)
        return (np.arange(self.capacity) - (self.n_levels - 1)) / self.lam

    def push(self, level_index):
        """log_push of one level index (used by the lagging-particle logic)."""
        if self.levels_done:
            return 0.0
        return (int(level_index) - (self.n_levels - 1)) / self.lam
