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
  There is no per-run evidence error.  An *opt-in* dynesty-style delta-logZ
  stopping rule is available through ``run(delta_logz=...)``
  (:func:`jnesty.dfnest.levels.delta_log_z_estimate`); it is off unless
  requested and never changes the default sampling path.

An *opt-in* gradient-informed proposal is available through
``gradient_scales=True``: every ``gradient_interval`` MCMC steps the
engine computes ``jax.grad`` of the log-likelihood w.r.t. the unit-cube
row at one live particle, folds the per-coordinate magnitudes into a
running average, and modulates the per-coordinate ``randh`` proposal
scale by ``gm(avg)/avg`` (geometric mean one).  The proposal stays in the
symmetric one-coordinate class, so detailed balance is preserved; the
gradient is diagnostic only and draws from its own fold-in key stream.
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
from .state import (DNestConfig, LaneState, LevelBook, derive_scale_multipliers,
                    resolve_config, update_gradient_magnitude_average,
                    validate_gradient_options)

__all__ = ['JaxDNestSampler']

#: Minimum number of saved particles before the opt-in delta-logZ stop may
#: fire.  The stopping statistic bounds the *unexplored* evidence, not the
#: order-statistic noise of the saved stream, so a floor keeps small save
#: counts from tripping it.  Validation on the T2 analytic problems
#: (task_004_dnest_stop): the postprocessed logZ still swings by ~0.2 nat
#: at a few hundred saves (worst: gaussian20, |logZ(300) - logZ(8000)| =
#: 0.22) but is within 0.03 nat of its 8000-save value from ~800 saves on,
#: for every tested problem and dimension; 800 also stays well inside the
#: regime the T2 bench found evidence-tight (2000 saves).
DELTA_LOGZ_MIN_SAVES = 800

#: Fold-in tag deriving the gradient-event key stream from the seed.
#: ``random.fold_in`` (unlike ``random.split``) does not consume the base
#: key, so the proposal and host streams — and therefore the default
#: sampling path — are untouched whether or not the feature is on.
_GRADIENT_KEY_TAG = 0x5CA1E5

#: Upper bound on gradient events settled at one block boundary; keeps a
#: very small ``gradient_interval`` from degenerating into a per-step host
#: round-trip on large blocks.
_GRADIENT_EVENTS_PER_BLOCK_MAX = 32


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
    gradient_scales : bool, optional
        Opt-in gradient-informed per-coordinate proposal scales.  When
        ``True``, every ``gradient_interval`` MCMC steps the engine
        computes ``jax.grad`` of the log-likelihood with respect to the
        unit-cube row at one uniformly chosen live particle and folds
        ``|d logL / du_i|`` into a per-coordinate running average.  The
        proposal scale of coordinate ``i`` becomes
        ``gm(avg) / avg_i`` (normalised to geometric mean one): sensitive
        coordinates get smaller steps, insensitive ones larger steps.  The
        proposal stays in the symmetric one-coordinate class, so detailed
        balance is preserved (the gradient is diagnostic, not part of the
        proposal distribution).  Default ``False``: the gradient branch is
        never traced and the sampling path is bit-identical to a run
        without the argument.
    gradient_interval : int, optional
        MCMC steps between gradient computations (default 100).  Gradient
        events are settled at block boundaries (at most 32 per block);
        each costs one forward+backward likelihood pass at a single
        particle, i.e. an amortised ~``1 / gradient_interval`` backward
        pass per MCMC step.

    Notes
    -----
    Constructing a sampler enables JAX float64 mode process-wide (CDNest
    computes in double precision); the same caveat as the Nautilus engine
    applies.  The run stops at ``max_num_saves`` saved particles; this is
    a run-length control, not a convergence criterion.  A dynesty-style
    ``delta_logz`` convergence stop is available as an opt-in ``run()``
    argument (off by default; the default path is unchanged).

    Post-processing at a temperature ``T >= 1`` is available through
    :meth:`native_result_at_temperature` (CDNest convention: likelihoods
    divided by ``T`` for evidence and weights, level placement
    untempered); the default ``T = 1`` result and the sampling path are
    untouched by that opt-in API.

    The gradient-informed scales compose orthogonally with
    ``sample_frequency``: the frequency weights allocate *how often* each
    coordinate is proposed, the gradient scales *how far* the step
    reaches.  The gradient events draw from a key stream derived by
    folding a tag into the seed, never from the proposal stream, so the
    two features (and the default path) never interact through the RNG.
    """

    def __init__(self, loglikelihood, prior_transform, ndim,
                 num_lanes=1, num_particles=1, sample_frequency=None,
                 thread_steps_factor=10.0, new_level_interval_factor=2.0,
                 save_interval_factor=None, max_num_levels=0, lam=10.0,
                 beta=100.0, compression=math.e, ptol=0.1, seed=None,
                 device=None, context=None, gradient_scales=False,
                 gradient_interval=100):
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

        gradient_scales, gradient_interval = validate_gradient_options(
            gradient_scales, gradient_interval)
        self.gradient_scales = gradient_scales
        self.gradient_interval = gradient_interval

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

        # Gradient-event key stream for the opt-in gradient-informed
        # scales: derived by folding a tag into the seed key.  fold_in
        # does not consume ``self._key``, so the proposal and host streams
        # are byte-for-byte what they would be without this line; the key
        # is only ever split when gradient_scales is on.
        self._grad_key = random.fold_in(self._key, _GRADIENT_KEY_TAG)

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
        # Diagnostic gradient program (d logL / d u at one row); built
        # only when the feature is on so the default construction never
        # compiles it.
        if gradient_scales:
            self._grad_of_u = jax.jit(
                jax.grad(self._logl_of_u, argnums=0), device=self.device)
        else:
            self._grad_of_u = None

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
        # Opt-in gradient-informed scales: the per-coordinate running
        # average of |d logL / du| and its observation count.  Host-side
        # state only; the derived multipliers are what reach the device
        # (as LaneState.grad_scales, re-derived at each block update).
        self._grad_avg = np.zeros(self.config.ndim, dtype=np.float64)
        self._n_grad_events = 0
        self._steps_since_grad = 0
        self._native_result = None
        # Postprocess views at temperature T > 1, keyed by T; dropped
        # whenever the canonical (T = 1) result is recomputed.
        self._tempered_results = {}
        self._revision = 0
        self._dirty = False
        self._completed = False
        self._last_target = None
        # Opt-in delta-logZ stopping state (inert unless delta_logz is
        # passed to run(); the tracker is a read-only max over the live
        # particles and consumes no randomness).
        self._logl_max = -math.inf
        self._track_logl_max = False
        self._delta_logz_stopped = False
        self._last_delta_logz = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, max_num_saves=10000, ptol=None, verbose=False,
            delta_logz=None):
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
        delta_logz : float or None, optional
            Opt-in dynesty-style stopping criterion (default ``None``,
            i.e. run to the save budget exactly as CDNest does).  When set,
            the run may stop once (a) level creation has finished, (b) at
            least :data:`DELTA_LOGZ_MIN_SAVES` particles were saved, and
            (c) the estimated remaining-evidence ratio
            ``delta = logaddexp(0, logl_max + logX_tail - logZ_running)``
            dropped below ``delta_logz``; see
            :func:`jnesty.dfnest.levels.delta_log_z_estimate`.  The budget
            cap still applies if the criterion never fires.

        Notes
        -----
        ``delta_logz=None`` (the default) leaves the sampling path
        bit-identical to a run without this argument: the statistic is a
        read-only function of the level book and never consumes
        randomness.  A run stopped by the criterion reports
        ``delta_logz_stopped`` and keeps its state, so a later ``run()``
        with a larger budget continues the chain.
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
        if delta_logz is not None:
            if isinstance(delta_logz, bool) or not isinstance(
                    delta_logz, (int, float, np.integer, np.floating)):
                raise TypeError('delta_logz must be a real number or None')
            delta_logz = float(delta_logz)
            if not (math.isfinite(delta_logz) and delta_logz > 0):
                raise ValueError(
                    f'delta_logz must be finite and > 0, got {delta_logz}')
            self._track_logl_max = True

        self._last_target = max_num_saves

        if self._particles is None:
            self._key, k_init = random.split(self._key)
            state = self._init_fn(k_init, self.context)
            logl0 = np.asarray(state.logl)
            if np.any(np.isnan(logl0)) or np.any(np.isposinf(logl0)):
                raise ValueError(
                    'the initial prior samples produced NaN or +inf '
                    'log-likelihoods; check loglikelihood/prior_transform')
            if self.gradient_scales:
                # Attach unit multipliers; the first gradient event
                # (after gradient_interval steps) replaces them with the
                # data-derived ones.
                state = state._replace(
                    grad_scales=jnp.ones(
                        (self.config.num_lanes, self.config.ndim),
                        dtype=jnp.float64))
            self._particles = state
            self._n_like += self.config.total_particles
            if self._track_logl_max:
                self._logl_max = max(self._logl_max, float(logl0.max()))
            self._dirty = True

        target = max(self.count_saves, max_num_saves)
        if target > self.count_saves:
            self._completed = False
            self._delta_logz_stopped = False
            progress_every = 50
            while self.count_saves < target:
                self._run_block()
                self._do_bookkeeping(verbose=verbose)
                if (self.count_mcmc_steps
                        >= (self.count_saves + 1) * self.config.save_interval):
                    self._save_particle()
                if (delta_logz is not None
                        and self._delta_logz_converged(delta_logz)):
                    self._delta_logz_stopped = True
                    if verbose:
                        print(f'DNest: delta-logZ stop at saves='
                              f'{self.count_saves} (delta logZ = '
                              f'{self._last_delta_logz:.4g} < '
                              f'{delta_logz:g}); save target {target} '
                              'not reached.')
                    break
                if verbose and self.n_blocks % progress_every == 0:
                    self.print_status()
            if not self._delta_logz_stopped:
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

    def native_result_at_temperature(self, temperature):
        """Return the (cached) postprocess result at a temperature.

        The canonical ``T = 1`` view is exactly ``native_result``.  A
        temperature ``T > 1`` re-runs the CDNest postprocessor with
        likelihoods divided by ``T`` for the evidence and the weights
        (level placement and prior masses stay untempered, the CDNest
        convention), cached per temperature so repeated calls are cheap.
        Results are dropped automatically when the sampler continues
        (``run()``) and the saved stream changes.

        Parameters
        ----------
        temperature : float
            Posterior temperature ``T >= 1``.  ``T = 1`` is the standard
            posterior; ``T -> infinity`` approaches the prior (weights
            proportional to the prior masses).

        Returns
        -------
        DNestResult
            Tempered postprocess view; ``samples``/``samples_u`` are
            shared with the canonical result (they do not depend on
            ``T``), while ``logz``/``logwt``/``information``/``n_eff``
            are the tempered values.
        """
        return self._compute_native_result(temperature)

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
    def delta_logz_stopped(self):
        """Whether the last ``run()`` stopped via the delta-logZ criterion."""
        return self._delta_logz_stopped

    @property
    def last_delta_logz(self):
        """The stopping statistic at the most recent check (or None)."""
        return self._last_delta_logz

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
    def gradient_magnitudes(self):
        """Running average of ``|d logL / du_i|`` per coordinate (ndim,).

        Host-side diagnostic state; all zeros before the first finite
        gradient observation.  Drives :attr:`gradient_scale_multipliers`.
        """
        return self._grad_avg.copy()

    @property
    def gradient_scale_multipliers(self):
        """Per-coordinate proposal scale multipliers in force (ndim,).

        The multipliers the next block will apply (identical across
        lanes): geometric mean one over live coordinates, larger for
        insensitive coordinates, clipped to
        ``[1 / GRADIENT_SCALES_CAP, GRADIENT_SCALES_CAP]``.  All ones
        until the first gradient observation lands — i.e. exactly the
        un-modulated proposal.
        """
        return derive_scale_multipliers(self._grad_avg)

    @property
    def n_gradient_events(self):
        """Number of finite gradient observations folded in so far."""
        return self._n_grad_events

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
    def _delta_logz_converged(self, delta_logz):
        """Gate + estimate for the opt-in delta-logZ stop (run loop only).

        Returns True when the run may stop: level creation finished, the
        minimum save floor was reached and the remaining-evidence ratio
        dropped below ``delta_logz``.  Non-finite estimates (degenerate
        level books) never stop the run; the budget cap then applies.
        """
        if (self.count_saves < DELTA_LOGZ_MIN_SAVES
                or not self._levels.levels_done):
            return False
        book = self._levels
        n = book.n_levels
        delta, _ = level_ops.delta_log_z_estimate(
            book.logX[:n], book.logl[:n], self._logl_max)
        self._last_delta_logz = delta
        if not math.isfinite(delta):
            return False
        return delta < delta_logz

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

        # Running maximum log-likelihood for the opt-in delta-logZ stop
        # (read-only transfer; no RNG is consumed either way).
        if self._track_logl_max:
            live_max = float(np.asarray(state.logl).max())
            if live_max > self._logl_max:
                self._logl_max = live_max

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

        # Opt-in gradient-informed scales: settle the gradient events due
        # at this block boundary (see _run_gradient_events).  Inert when
        # gradient_scales is off; consumes only the fold-in key stream.
        if self.gradient_scales:
            self._run_gradient_events()

    def _run_gradient_events(self):
        """Settle gradient events due since the last settlement.

        One *event* computes ``jax.grad`` of the log-likelihood w.r.t. the
        unit-cube row at one uniformly chosen live particle (chosen with
        the fold-in key stream, never the proposal stream) and folds
        ``|d logL / du_i|`` into the running average.  Events are due
        every ``gradient_interval`` MCMC steps; because the engine only
        regains host control at block boundaries, they are settled there
        (catch-up, at most :data:`_GRADIENT_EVENTS_PER_BLOCK_MAX` per
        block).  Whenever at least one new observation landed, the scale
        multipliers are re-derived and pushed into ``LaneState.grad_scales``
        (identical across lanes), taking effect from the next block.
        """
        cfg = self.config
        self._steps_since_grad += cfg.block_steps
        n_events = min(int(self._steps_since_grad // self.gradient_interval),
                       _GRADIENT_EVENTS_PER_BLOCK_MAX)
        if n_events > 0:
            self._steps_since_grad -= n_events * self.gradient_interval
            for _ in range(n_events):
                self._gradient_event()
            if self._n_grad_events > 0:
                scales = jnp.asarray(np.broadcast_to(
                    derive_scale_multipliers(self._grad_avg),
                    (cfg.num_lanes, cfg.ndim)))
                self._particles = self._particles._replace(
                    grad_scales=scales)
                self._dirty = True

    def _gradient_event(self):
        """One gradient observation at a uniformly chosen live particle.

        A particle sitting on a ``-inf`` (forbidden-region) likelihood, or
        a position where the gradient is not finite, is skipped: a single
        bad observation must not poison the running average that drives
        the proposal scales.  The forward pass of the gradient counts as
        one likelihood evaluation in ``n_like`` (the reverse pass adds
        problem-dependent cost on top of that).
        """
        cfg = self.config
        self._grad_key, key = random.split(self._grad_key)
        # One flat draw over the (lane, particle) ensemble.
        flat = int(np.asarray(random.randint(
            key, (), 0, cfg.total_particles)))
        lane = flat // cfg.num_particles
        particle = flat % cfg.num_particles
        logl_here = float(self._particles.logl[lane, particle])
        if not math.isfinite(logl_here):
            return
        u_row = self._particles.u[lane, particle]
        grad = np.asarray(self._grad_of_u(u_row, self.context))
        if not np.all(np.isfinite(grad)):
            return
        self._n_like += 1
        self._grad_avg, self._n_grad_events = update_gradient_magnitude_average(
            self._grad_avg, np.abs(grad), self._n_grad_events)

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

    def _compute_native_result(self, temperature=1.0):
        """Run the C-compatible postprocessor and cache the result.

        With the default ``temperature=1.0`` this computes and caches the
        canonical CDNest result in ``native_result`` — bit-identical to
        the pre-temperature API, and what ``run()``/``posterior()``/
        ``native_result`` expose.  A temperature ``T > 1`` instead
        computes the tempered view (likelihoods divided by ``T`` for the
        evidence and weights; level placement and prior masses always
        untempered, the CDNest postprocessor convention) and caches it in
        ``_tempered_results`` keyed by ``T``, so repeated calls at the
        same temperature are cheap.  Recomputing the canonical result
        (the saved stream changed) drops every tempered view.

        Parameters
        ----------
        temperature : float, optional
            Posterior temperature ``T >= 1`` (default 1.0).  ``T = 1``
            is the standard posterior; larger values flatten it towards
            the prior.

        Returns
        -------
        DNestResult
            The (cached) postprocess result at the requested temperature.
        """
        if isinstance(temperature, bool) or not isinstance(
                temperature, (int, float, np.integer, np.floating)):
            raise TypeError(
                f'temperature must be a real number, got '
                f'{type(temperature).__name__}')
        temperature = float(temperature)
        if not (math.isfinite(temperature) and temperature >= 1.0):
            raise ValueError(
                f'temperature must be finite and >= 1, got {temperature}')

        if temperature == 1.0:
            if not self._dirty and self._native_result is not None:
                return self._native_result
            if self.count_saves == 0:
                raise RuntimeError(
                    'no particles were saved; run() must reach its save '
                    'target before results are available')
            # The saved stream changed since the last postprocess (or
            # this is the first one): all tempered views are stale.
            self._tempered_results.clear()
            book = self._levels
            n = book.n_levels
            counters = {
                'visits': book.visits[:n], 'exceeds': book.exceeds[:n],
                'accepts': book.accepts[:n], 'tries': book.tries[:n],
            }
            if not book.levels_done:
                warnings.warn(
                    'DNest: level creation never finished (PTol unmet '
                    'within the save budget); results are computed but '
                    'the level set is incomplete.', stacklevel=2)
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
            return self._native_result

        # Tempered view (T > 1): make sure the canonical snapshot exists
        # and is fresh first — this also raises the canonical
        # 'no results yet' error and drops stale tempered views.
        self._compute_native_result()
        cached = self._tempered_results.get(temperature)
        if cached is not None:
            return cached
        book = self._levels
        n = book.n_levels
        counters = {
            'visits': book.visits[:n], 'exceeds': book.exceeds[:n],
            'accepts': book.accepts[:n], 'tries': book.tries[:n],
        }
        result = dnest_postprocess(
            book.logX[:n], book.logl[:n], book.tie[:n],
            np.asarray(self._saves_logl), np.asarray(self._saves_tie),
            np.asarray(self._saves_level), temperature=temperature,
            level_counters=counters)
        # Sample coordinates are temperature-independent (level placement
        # uses the untempered likelihood): reuse the canonical arrays.
        result.samples_u = self._native_result.samples_u
        result.samples = self._native_result.samples
        self._tempered_results[temperature] = result
        return result
