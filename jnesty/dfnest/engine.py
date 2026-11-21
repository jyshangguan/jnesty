"""Host orchestration for the JAX DNest engine: :class:`JaxDNestSampler`.

The engine owns everything that changes shape or requires decisions and
keeps it on the host, exactly like CDNest's root worker:

* the :class:`~jnesty.dfnest.state.LevelBook` (thresholds, learned
  ``logX`` volumes, visit/exceed/accept/try counters);
* the above-threshold reservoir from which new levels are cut;
* the saved particle stream, the run/likelihood counters and the
  root-lane lagging-particle replacements.

One compiled block program (see :mod:`jnesty.dfnest.core`) advances all
lanes between the host's bookkeeping passes, mirroring the structure of
CDNest's MPI loop: ``thread_steps`` sequential updates per worker with a
frozen copy of the levels, then gather, bookkeep, broadcast, save.

Notes on parity-faithful details
--------------------------------
* Levels are frozen during a block; creation happens at most once per
  block and only while the PTol criterion is unmet.
* ``kill_lagging_particles`` runs on the *root lane only* over its local
  particles, exactly as CDNest's MPI root does; with one particle per
  lane an all-lagging root simply keeps its particle.
* Saves pick one (lane, particle) uniformly over the whole ensemble and
  happen at most once per block (CDNest's single ``if``, not a catch-up
  loop).
* The run length is ``max_num_saves``; ``ptol`` only stops level creation.
  There is no delta-logZ stopping rule and no per-run evidence error.
"""

import math
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from . import levels as level_ops
from .core import (make_block_fn, make_init_fn, make_logl_of_u,
                   make_to_physical, probe_callables)
from .postprocess import DNestResult, dnest_postprocess
from .state import DNestConfig, LaneState, LevelBook, resolve_config

__all__ = ['JaxDNestSampler']


class JaxDNestSampler:
    """Diffusive nested sampling with JAX/GPU-accelerated blocks.

    Parameters
    ----------
    loglikelihood : callable
        JAX-compatible log-likelihood of one parameter vector; may return
        ``-inf`` for forbidden regions but must be scalar and finite
        otherwise.  With ``context``: ``loglikelihood(x, context)``.
    prior_transform : callable
        JAX-compatible map from the unit hypercube to physical parameters;
        with ``context``: ``prior_transform(u, context)``.
    ndim : int
        Number of parameters (at least 1).
    num_lanes : int, optional
        Independent workers (the JAX analogue of CDNest's MPI ranks).
        Default 1, the single-worker CDNest configuration.
    num_particles : int, optional
        Local particles *per lane*, exactly as CDNest counts them.  The
        total ensemble is ``num_lanes * num_particles``.
    sample_frequency : scalar or sequence, optional
        Relative weights for which coordinate each proposal perturbs
        (larger = proposed more often).  Scalar broadcasts; default is
        uniform (the CDNest behaviour).  Entries may be zero (frozen
        coordinate); the sum must be positive.
    thread_steps_factor, new_level_interval_factor, save_interval_factor : float, optional
        CDNest interval factors (see
        :mod:`jnesty.dfnest.state`).  ``save_interval_factor=None``
        inherits ``new_level_interval_factor``.
    max_num_levels : int, optional
        0 (default) creates levels until the PTol criterion fires; a
        positive value fixes the number of levels.
    lam, beta : float, optional
        Backtracking length and equal-push strength (CDNest defaults 10
        and 100).
    compression : float, optional
        Level compression, in ``(1, e]``; default ``e``.
    ptol : float, optional
        Mean level log-likelihood spacing that stops level creation
        (CDNest default 0.1).
    seed : int or None, optional
        Seed for reproducible runs (host and device streams).
    device : str, jax.Device or None, optional
        Device for the compiled programs; None uses the default device.
        The global ``jax_platform_name`` is never modified.
    context : pytree or None, optional
        Passed as a second argument to both callables, keeping large data
        out of the compiled kernels.

    Notes
    -----
    Constructing a sampler enables JAX float64 mode process-wide (CDNest
    computes in double precision); the same caveat as the Nautilus engine
    applies.  The run stops at ``max_num_saves`` saved particles; this is
    a run-length control, not a convergence criterion.
    """

    def __init__(self, loglikelihood, prior_transform, ndim,
                 num_lanes=1, num_particles=1, sample_frequency=None,
                 thread_steps_factor=10.0, new_level_interval_factor=2.0,
                 save_interval_factor=None, max_num_levels=0, lam=10.0,
                 beta=100.0, compression=math.e, ptol=0.1, seed=None,
                 device=None, context=None):
        if not callable(loglikelihood):
            raise TypeError('loglikelihood must be callable')
        if not callable(prior_transform):
            raise TypeError('prior_transform must be callable')

        # CDNest works in double precision; enable x64 for this process.
        # (Same documented process-global side effect as Nautilus.)
        jax.config.update('jax_enable_x64', True)

        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.context = context

        self.config = resolve_config(
            ndim=ndim, num_lanes=num_lanes, num_particles=num_particles,
            sample_frequency=sample_frequency,
            thread_steps_factor=thread_steps_factor,
            new_level_interval_factor=new_level_interval_factor,
            save_interval_factor=save_interval_factor,
            max_num_levels=max_num_levels, lam=lam, beta=beta,
            compression=compression, ptol=ptol)
        self.ptol = self.config.ptol

        if device is None:
            self.device = None
        elif isinstance(device, jax.Device):
            self.device = device
        else:
            self.device = jax.devices(str(device))[0]

        # RNG: one host stream (saves, lagging kills) and one device key
        # stream (proposals), both derived from the same seed.
        if seed is None:
            self._rng = np.random.default_rng()
            self._key = random.PRNGKey(
                int(np.random.default_rng().integers(0, 2 ** 31 - 1)))
        else:
            if isinstance(seed, bool) or not isinstance(
                    seed, (int, np.integer)):
                raise TypeError('seed must be an integer or None')
            seed = int(seed)
            if not (0 <= seed < 2 ** 31):
                raise ValueError('seed must be in [0, 2**31)')
            self._rng = np.random.default_rng(seed)
            self._key = random.PRNGKey(seed)

        # Compiled programs (built after validation so bad callables fail
        # at construction, before any sampling).
        self._logl_of_u = make_logl_of_u(prior_transform, loglikelihood)
        probe_callables(prior_transform, loglikelihood, self.config.ndim,
                        context)
        self._block_fn = make_block_fn(self._logl_of_u, self.config,
                                       self.device)
        self._init_fn = make_init_fn(self._logl_of_u, self.config,
                                     self.device)
        self._to_physical = make_to_physical(prior_transform, self.device)

        # Run state ------------------------------------------------------
        self._levels = LevelBook(self.config.level_capacity, self._rng,
                                 self.config.lam)
        # max_num_levels == 1 makes levels "done" before any creation.
        if level_ops.enough_levels(self._levels.logl, self._levels.n_levels,
                                   self.config.max_num_levels, self.ptol):
            self._levels.levels_done = True

        self._above_vals = np.zeros(0, dtype=np.float64)
        self._above_ties = np.zeros(0, dtype=np.float64)
        self._saves_u = []
        self._saves_logl = []
        self._saves_tie = []
        self._saves_level = []
        self._saves_pid = []
        self._particles = None
        self.count_saves = 0
        self.count_mcmc_steps = 0
        self._n_like = 0
        self.n_blocks = 0
        self.n_kills = 0
        self._coord_attempts = np.zeros(self.config.ndim, dtype=np.int64)
        self._coord_accepts = np.zeros(self.config.ndim, dtype=np.int64)
        self._native_result = None
        self._revision = 0
        self._dirty = False
        self._completed = False
        self._last_target = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, max_num_saves=10000, ptol=None, verbose=False):
        """Run the sampler until ``max_num_saves`` particles are saved.

        A larger target than the current save count continues the native
        state (levels, particles, reservoir, counters); an equal or smaller
        target is a no-op that returns the existing result.

        Returns the natural-log evidence ``log_z`` (CDNest's ``run()``
        return convention).

        Parameters
        ----------
        max_num_saves : int, optional
            Total save target for this engine (default 10000).
        ptol : float or None, optional
            Overrides the level-creation PTol for subsequent bookkeeping.
        verbose : bool, optional
            Print level-creation and progress lines.
        """
        if isinstance(max_num_saves, bool) or not isinstance(
                max_num_saves, (int, np.integer)):
            raise TypeError(
                f'max_num_saves must be an integer, got '
                f'{type(max_num_saves).__name__}')
        max_num_saves = int(max_num_saves)
        if max_num_saves < 1:
            raise ValueError(f'max_num_saves must be >= 1, got {max_num_saves}')
        if ptol is not None:
            if isinstance(ptol, bool) or not isinstance(
                    ptol, (int, float, np.integer, np.floating)):
                raise TypeError('ptol must be a real number or None')
            ptol = float(ptol)
            if not (math.isfinite(ptol) and ptol > 0):
                raise ValueError(f'ptol must be finite and > 0, got {ptol}')
            self.ptol = ptol

        self._last_target = max_num_saves

        if self._particles is None:
            self._key, k_init = random.split(self._key)
            state = self._init_fn(k_init, self.context)
            logl0 = np.asarray(state.logl)
            if np.any(np.isnan(logl0)) or np.any(np.isposinf(logl0)):
                raise ValueError(
                    'the initial prior samples produced NaN or +inf '
                    'log-likelihoods; check loglikelihood/prior_transform')
            self._particles = state
            self._n_like += self.config.total_particles
            self._dirty = True

        target = max(self.count_saves, max_num_saves)
        if target > self.count_saves:
            self._completed = False
            progress_every = 50
            while self.count_saves < target:
                self._run_block()
                self._do_bookkeeping(verbose=verbose)
                if (self.count_mcmc_steps
                        >= (self.count_saves + 1) * self.config.save_interval):
                    self._save_particle()
                if verbose and self.n_blocks % progress_every == 0:
                    self.print_status()
            self._completed = True
            if verbose:
                self.print_status()
        elif verbose and self.count_saves:
            print('DNest: save target already reached; no new sampling.')

        self._compute_native_result()
        return self._native_result.logz

    def posterior(self):
        """Return ``(points, log_w, log_l)`` in native save order.

        ``points`` are physical NumPy arrays, ``log_w`` are posterior
        log-weights normalised to ``logsumexp(log_w) == 0`` and ``log_l``
        are the saved log-likelihoods.
        """
        if self._native_result is None:
            raise RuntimeError('run() must complete before posterior()')
        result = self._native_result
        return result.samples, result.logwt, result.logl

    def print_status(self, header=False):
        """One-line run status (level count, saves, evidence so far)."""
        logz = ('--' if self._native_result is None
                else f'{self._native_result.logz:.4f}')
        print(f'DNest: blocks={self.n_blocks} levels={self.n_levels} '
              f'saves={self.count_saves}/{self._last_target} '
              f'logZ={logz} likes={self._n_like}')

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    @property
    def native_result(self):
        """The :class:`~jnesty.dfnest.postprocess.DNestResult` (or None)."""
        return self._native_result

    @property
    def revision(self):
        """Monotonic counter bumped whenever the native result changes."""
        return self._revision

    @property
    def log_z(self):
        """Estimated log evidence (CDNest C-postprocess semantics)."""
        if self._native_result is None:
            return None
        return self._native_result.logz

    @property
    def information(self):
        """Information H in nats."""
        if self._native_result is None:
            return None
        return self._native_result.information

    @property
    def n_eff(self):
        """Entropy effective sample size ``exp(-sum(w log w))``."""
        if self._native_result is None:
            return 0.0
        return self._native_result.n_eff

    @property
    def n_like(self):
        """Total likelihood evaluations (initialisation plus blocks)."""
        return self._n_like

    @property
    def n_steps(self):
        """Total MCMC updates across all lanes."""
        return self.count_mcmc_steps

    @property
    def n_saves(self):
        """Number of saved particles."""
        return self.count_saves

    @property
    def n_levels(self):
        """Number of likelihood levels."""
        return self._levels.n_levels

    @property
    def levels_finalized(self):
        """Whether level creation has finished (PTol/max-levels/capacity)."""
        return self._levels.levels_done

    @property
    def completed(self):
        """Whether the last ``run()`` reached its save target."""
        return self._completed

    @property
    def max_num_saves(self):
        """The most recent save target."""
        return self._last_target

    @property
    def num_lanes(self):
        return self.config.num_lanes

    @property
    def num_particles(self):
        return self.config.num_particles

    @property
    def total_particles(self):
        return self.config.total_particles

    @property
    def acceptance_rate(self):
        """MH acceptance over all levels: accepts / tries."""
        n = self._levels.n_levels
        tries = int(self._levels.tries[:n].sum())
        if tries == 0:
            return float('nan')
        return float(self._levels.accepts[:n].sum() / tries)

    @property
    def sample_frequency(self):
        """Normalised per-coordinate proposal weights (ndim,)."""
        return self.config.sample_frequency.copy()

    @property
    def sample_frequency_raw(self):
        """The user-supplied (broadcast) sample_frequency input."""
        return self.config.sample_frequency_raw.copy()

    @property
    def coordinate_attempts(self):
        """Per-coordinate proposal counts (ndim,), cumulative."""
        return self._coord_attempts.copy()

    @property
    def coordinate_accepts(self):
        """Per-coordinate accepted-proposal counts (ndim,), cumulative."""
        return self._coord_accepts.copy()

    @property
    def coordinate_acceptance_rate(self):
        """Per-coordinate acceptance rates; NaN where never proposed."""
        attempts = self._coord_attempts
        accepts = self._coord_accepts
        return np.where(attempts > 0, accepts / np.maximum(attempts, 1),
                        np.nan)

    @property
    def levels(self):
        """Dict of the live level arrays (diagnostics view)."""
        book = self._levels
        n = book.n_levels
        return {
            'logX': book.logX[:n].copy(),
            'logl': book.logl[:n].copy(),
            'tie': book.tie[:n].copy(),
            'visits': book.visits[:n].copy(),
            'exceeds': book.exceeds[:n].copy(),
            'accepts': book.accepts[:n].copy(),
            'tries': book.tries[:n].copy(),
        }

    def __repr__(self):
        return (f'JaxDNestSampler(ndim={self.config.ndim}, '
                f'num_lanes={self.config.num_lanes}, '
                f'num_particles={self.config.num_particles}, '
                f'n_saves={self.count_saves}, n_levels={self.n_levels})')

    # ------------------------------------------------------------------
    # internals: block execution, bookkeeping, saves, kills, results
    # ------------------------------------------------------------------
    def _run_block(self):
        cfg = self.config
        book = self._levels

        self._key, block_key = random.split(self._key)
        keys = random.split(block_key,
                            (cfg.num_lanes, cfg.thread_steps))

        start_accepts = book.accepts.copy()
        start_tries = book.tries.copy()
        start_visits = book.visits.copy()
        start_exceeds = book.exceeds.copy()

        state, ys = self._block_fn(
            self._particles, keys,
            jnp.asarray(book.logl), jnp.asarray(book.logX),
            jnp.asarray(book.log_push_array()),
            jnp.asarray(book.n_levels, dtype=jnp.int32),
            jnp.asarray(book.levels_done, dtype=jnp.bool_),
            jnp.asarray(book.accepts), jnp.asarray(book.tries),
            jnp.asarray(book.visits), jnp.asarray(book.exceeds),
            self.context)
        self._particles = state

        above_val, above_tie, emit, invalid = ys
        invalid = np.asarray(invalid)
        if invalid.any():
            lane, step = np.argwhere(invalid)[0]
            raise ValueError(
                f'the likelihood returned NaN or +inf at lane {int(lane)}, '
                f'step {int(step)} of block {self.n_blocks + 1}; the '
                'proposal was rejected, but the likelihood is ill-defined '
                'at that point')

        # Aggregate lane-local counters into the global level array
        # (CDNest: levels_combine += sum of worker increments).
        num_lanes = cfg.num_lanes
        book.accepts += (np.asarray(state.accepts).sum(axis=0)
                         - num_lanes * start_accepts)
        book.tries += (np.asarray(state.tries).sum(axis=0)
                       - num_lanes * start_tries)
        book.visits += (np.asarray(state.visits).sum(axis=0)
                        - num_lanes * start_visits)
        book.exceeds += (np.asarray(state.exceeds).sum(axis=0)
                         - num_lanes * start_exceeds)

        # Cumulative coordinate diagnostics (never reset).
        self._coord_attempts = np.asarray(state.coord_attempts).sum(axis=0)
        self._coord_accepts = np.asarray(state.coord_accepts).sum(axis=0)

        emit = np.asarray(emit)
        if emit.any():
            self._above_vals = np.concatenate(
                [self._above_vals, np.asarray(above_val)[emit]])
            self._above_ties = np.concatenate(
                [self._above_ties, np.asarray(above_tie)[emit]])

        self.count_mcmc_steps += cfg.block_steps
        self._n_like += cfg.block_steps
        self.n_blocks += 1
        self._dirty = True

    def _do_bookkeeping(self, verbose=False):
        """CDNest's do_bookkeeping: level creation, finalisation, logX."""
        book = self._levels
        cfg = self.config

        created = False
        if ((not book.levels_done)
                and self._above_vals.shape[0] >= cfg.new_level_interval):
            created, self._above_vals, self._above_ties, threshold = \
                level_ops.create_level(
                    book, self._above_vals, self._above_ties,
                    cfg.compression)
            if created and verbose:
                print(f'DNest: created level {book.n_levels - 1} '
                      f'with logL = {threshold:.6g}')

        if created:
            if level_ops.enough_levels(book.logl, book.n_levels,
                                       cfg.max_num_levels, self.ptol):
                book.levels_done = True
                level_ops.renormalise_visits(book, cfg.regularisation)
                if verbose:
                    print(f'DNest: finished creating levels '
                          f'({book.n_levels} levels).')
            else:
                self._kill_lagging(verbose=verbose)

        level_ops.recalculate_log_X(book, cfg.regularisation,
                                    cfg.compression)
        self._dirty = True

    def _kill_lagging(self, verbose=False):
        """CDNest's kill_lagging_particles, on the root lane's particles.

        CDNest only runs this on the MPI root over its ``num_particles``
        local particles; lane 0 is the root here.  Each lagging particle is
        replaced by a rejection-sampled surviving source with acceptance
        ``exp(push_source - max_push)``; if every root particle is lagging,
        CDNest warns and keeps them.
        """
        cfg = self.config
        book = self._levels
        num_particles = cfg.num_particles

        level0 = np.asarray(self._particles.level[0])
        push = (level0.astype(np.float64)
                - (book.n_levels - 1)) / cfg.lam
        with np.errstate(over='ignore'):
            probabilities = level_ops.kill_probabilities(push)
        bad = self._rng.random(num_particles) <= probabilities
        if not bad.any():
            return
        if bad.all():
            warnings.warn(
                'DNest: all root-lane particles are lagging; CDNest keeps '
                'them unchanged.', stacklevel=2)
            return

        max_push = push.max()
        u = self._particles.u
        logl = self._particles.logl
        tie = self._particles.tie
        level = self._particles.level
        replaced = 0
        for i in np.flatnonzero(bad):
            while True:
                source = int(self._rng.integers(0, num_particles))
                if bad[source]:
                    continue
                if self._rng.random() < math.exp(push[source] - max_push):
                    break
            u = u.at[0, i].set(u[0, source])
            logl = logl.at[0, i].set(logl[0, source])
            tie = tie.at[0, i].set(tie[0, source])
            level = level.at[0, i].set(level[0, source])
            replaced += 1
        self._particles = self._particles._replace(
            u=u, logl=logl, tie=tie, level=level)
        self.n_kills += replaced
        self._dirty = True
        if verbose:
            print(f'DNest: replaced {replaced} lagging root-lane '
                  f'particle(s).')

    def _save_particle(self):
        """Save one uniformly chosen (lane, particle); CDNest save_particle."""
        cfg = self.config
        lane = int(self._rng.integers(0, cfg.num_lanes))
        particle = int(self._rng.integers(0, cfg.num_particles))
        state = self._particles
        self._saves_u.append(np.asarray(state.u[lane, particle]).copy())
        self._saves_logl.append(float(state.logl[lane, particle]))
        self._saves_tie.append(float(state.tie[lane, particle]))
        self._saves_level.append(int(state.level[lane, particle]))
        self._saves_pid.append(lane * cfg.num_particles + particle)
        self.count_saves += 1
        self._dirty = True

    def _compute_native_result(self):
        """Run the C-compatible postprocessor and cache the result."""
        if not self._dirty and self._native_result is not None:
            return
        if self.count_saves == 0:
            raise RuntimeError(
                'no particles were saved; run() must reach its save target '
                'before results are available')
        book = self._levels
        n = book.n_levels
        counters = {
            'visits': book.visits[:n], 'exceeds': book.exceeds[:n],
            'accepts': book.accepts[:n], 'tries': book.tries[:n],
        }
        if not book.levels_done:
            warnings.warn(
                'DNest: level creation never finished (PTol unmet within '
                'the save budget); results are computed but the level set '
                'is incomplete.', stacklevel=2)
        result = dnest_postprocess(
            book.logX[:n], book.logl[:n], book.tie[:n],
            np.asarray(self._saves_logl), np.asarray(self._saves_tie),
            np.asarray(self._saves_level), level_counters=counters)
        result.samples_u = np.stack(self._saves_u)
        result.samples = np.asarray(self._to_physical(
            jnp.asarray(result.samples_u), self.context))
        self._native_result = result
        self._dirty = False
        self._revision += 1
