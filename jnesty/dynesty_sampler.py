"""
JNesty public API — Dynesty-style nested samplers.

This module holds both user-facing Dynesty-style sampler classes:

* :class:`NestedSampler` — the static nested sampler; a thin wrapper that
  configures the engine, runs the NS loop, and formats results via the
  results module.
* :class:`DynamicNestedSampler` — Dynesty-style dynamic nested sampling on
  top of the static engine; the weight / stopping / combine math it uses
  lives in :mod:`jnesty.dynesty._dynamic`.

For the JAX Nautilus importance sampler see :class:`jnesty.NautilusSampler`
(:mod:`jnesty.nautilus_sampler`).
"""

from typing import Optional, Callable, Dict, Any, Union
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import warnings

from .dynesty.core import run_nested_sampling, WhileLoopNSConfig
from .results import Results, format_results
from .dynesty._dynamic import (combine_saved_and_new, compute_integrals,
                              estimate_fresh_batch_size_from_memory,
                              generate_fresh_batch_points, kish_ess,
                              seed_batch_from_saved, stopping_function,
                              weight_function)

# Default number of parallel random walks (lanes) of the queue mode.  The 
# walks of one refill run in parallel while their steps run in sequence, so 
# this is the only parallel axis available inside a walk.
DEFAULT_QUEUE_SIZE = 8
# Number of random walk steps per walk of the legacy (non-queue) batch mode.
LEGACY_STEPS_PER_WALK = 5


def auto_queue_size(loglikelihood, prior_transform, ndim,
                    candidates=(8, 32, 64, 128), memory_frac=0.9,
                    context=None, verbose=False):
    '''
    Choose the number of parallel random walks (lanes) from the measured 
    throughput of the likelihood.

    One queue refill runs 'queue_size' walks in parallel whose steps run in 
    sequence, so the cost of one nested sampling iteration is rwalk_K times 
    the time of one evaluation of 'queue_size' points.  The lanes are chosen 
    where the time per point is the smallest, i.e., where the GPU is 
    saturated, and where one batch still fits in the memory budget.

    Parameters
    ----------
    loglikelihood : callable
        The log-likelihood function of one point.
    prior_transform : callable
        The prior transform of one point.
    ndim : int
        The number of dimensions.
    candidates : tuple (default: (8, 32, 64, 128))
        The candidate lane numbers, tested in increasing order.
    memory_frac : float (default: 0.9)
        The fraction of the GPU memory that one batch may use.
    context : optional
        The context passed to the log-likelihood and the prior transform.
    verbose : bool (default: False)
        Print the measured time per point of every candidate if True.

    Returns
    -------
    queue_size : int
        The number of lanes with the smallest time per point.
    '''
    import time
    stats = jax.devices()[0].memory_stats()
    budget = int(stats['bytes_limit'] * memory_frac) if stats is not None else None
    if context is None:
        loglike, ptform = loglikelihood, prior_transform
    else:
        loglike = lambda x: loglikelihood(x, context)
        ptform = lambda u: prior_transform(u, context)
    # A batch of prior samples is enough: the cost only depends on the shapes
    pos = jax.vmap(ptform)(random.uniform(
        random.PRNGKey(0), (int(max(candidates)), ndim)))
    vmapped = jax.vmap(loglike)
    best, best_time = int(candidates[0]), np.inf
    for q in candidates:
        batch = lambda p, q=q: vmapped(p[:q])
        try:
            compiled = jax.jit(batch).lower(pos).compile()
            ma = compiled.memory_analysis() if hasattr(
                compiled, 'memory_analysis') else None
            peak = getattr(ma, 'peak_memory_in_bytes', None) if ma is not None else None
            if peak is None and ma is not None:
                peak = (getattr(ma, 'temp_size_in_bytes', 0)
                        + getattr(ma, 'argument_size_in_bytes', 0)
                        + getattr(ma, 'output_size_in_bytes', 0)) or None
            if (peak is not None) and (budget is not None) and (peak > budget):
                break  # The larger candidates would not fit either
            for _ in range(3):
                compiled(pos).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(5):
                compiled(pos).block_until_ready()
            dt = (time.perf_counter() - t0) / 5. / q
        except Exception as e:
            if verbose:
                print('  Lane probe: queue_size={0} failed ({1})'.format(q, e))
            break
        if verbose:
            print('  Lane probe: queue_size={0:4d}, {1:.3f} ms/point'.format(
                q, dt * 1e3))
        if dt < best_time:
            best, best_time = q, dt
    return int(best)


class NestedSampler:
    """
    Simple nested sampler following Dynesty's API design.

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood function. Signature: loglikelihood(x) -> float
    prior_transform : callable
        Prior transform. Signature: prior_transform(u) -> array
    ndim : int
        Number of dimensions.
    nlive : int, optional
        Number of live points. Default: 500
    rwalk_K : int or None, optional
        Random walk steps per iteration. Auto-tuned if None.
    rwalk_step_scale : float or None, optional
        Initial proposal scale. Auto-tuned if None.
    target_acceptance : float, optional
        Target acceptance rate for scale adaptation. Default: 0.5
    scale_adapt_interval : int, optional
        Iterations between scale adaptation. Default: 1
    device : str, optional
        Device to use ('gpu' or 'cpu'). Default: 'gpu'
    verbose : bool, optional
        Print progress. Default: True
    bound : str, optional
        Bounding method: 'none', 'single', 'multi'. Default: 'none'
    bound_update_interval : int, float, or None, optional
        Controls periodic bound refitting, measured in likelihood calls.
        None (default): automatic — rwalk_K * nlive calls for bound='multi'
        (matching Dynesty's walks * nlive), off otherwise.
        Set to 0 to disable, float in (0,1) for fraction of rwalk_K * nlive.
    max_ellipsoids : int, optional
        Maximum ellipsoids for multi-ellipsoid. Default: 20
    batch_size : int or None, optional
        Number of parallel walks for GPU-parallel likelihood evaluation.
        Only used when queue_size=0 (legacy mode). None (default): auto-tuned
        to ~5 steps/walk. Set to 1 to disable parallelism.
    queue_size : int, 'auto', or None, optional
        Dynesty-style queue mode. None (default): automatic — 8 for
        bound='multi' (matching Dynesty's multiprocessing), 0 otherwise.
        An integer sets the number of parallel walks directly. 'auto' picks
        the number of walks whose measured time per likelihood evaluation is
        the smallest (see auto_queue_size), which also keeps the batch inside
        the memory budget.  Each queue entry gets full rwalk_K steps; scale
        adapts at queue drain.
    memory_frac : float, optional
        Fraction of GPU memory to use for batch walks. Default: 0.9.
        Caps batch_size if it would exceed this fraction of GPU memory.
        Ignored on CPU.
    """

    def __init__(
        self,
        loglikelihood: Callable[[jnp.ndarray], float],
        prior_transform: Callable[[jnp.ndarray], jnp.ndarray],
        ndim: int,
        nlive: int = 500,
        rwalk_K: Optional[int] = None,
        rwalk_step_scale: Optional[float] = None,
        target_acceptance: float = 0.5,
        scale_adapt_interval: int = 1,
        device: str = 'gpu',
        verbose: bool = True,
        bound: str = 'none',
        bound_update_interval: Optional[Union[int, float]] = None,
        max_ellipsoids: int = 20,
        batch_size: Optional[int] = None,
        queue_size: Optional[Union[int, str]] = None,
        memory_frac: float = 0.9,
        unit_cube_batch_size: int = 200,
        min_eff: float = 10.0,
        min_ncall: Optional[int] = None,
        context=None,
    ):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        #: Optional pytree passed to every traced function as an argument.
        #: With it, ``loglikelihood(x, context)`` / ``prior_transform(u, context)``
        #: carry the dataset, so the compiled kernels do not embed it and are
        #: reused from one dataset to the next (see ``clear_compiled_units()``).
        self.context = context
        self.ndim = ndim
        self.nlive = nlive
        self.target_acceptance = target_acceptance
        self.scale_adapt_interval = scale_adapt_interval
        self.device = device
        self.verbose = verbose
        self.bound = bound
        self.max_ellipsoids = max_ellipsoids
        self.memory_frac = memory_frac
        self.unit_cube_batch_size = unit_cube_batch_size
        self.min_eff = min_eff
        self.min_ncall = min_ncall

        # Auto-tune parameters
        if rwalk_K is None:
            rwalk_K = max(25, ndim + 20)
            if verbose:
                print(f"Auto-tuned rwalk_K = {rwalk_K} (based on ndim={ndim})")

        if rwalk_step_scale is None:
            rwalk_step_scale = 1.0
            if verbose:
                print(f"Auto-tuned rwalk_step_scale = {rwalk_step_scale:.4f} (Dynesty default)")

        self.rwalk_K = rwalk_K
        self.rwalk_step_scale = rwalk_step_scale

        # Resolve bound_update_interval (in likelihood calls, matching Dynesty)
        # Dynesty default: walks * nlive likelihood calls ≈ nlive iterations.
        # JNesty: None means auto (rwalk_K * nlive calls for multi, 0 otherwise).
        if bound_update_interval is None:
            if bound == 'multi':
                bound_update_interval = rwalk_K * nlive  # Dynesty default (calls)
            else:
                bound_update_interval = 0
        elif isinstance(bound_update_interval, float) and bound_update_interval > 0:
            bound_update_interval = int(bound_update_interval * nlive * rwalk_K)
        self.bound_update_interval = bound_update_interval

        if verbose and bound == 'multi':
            if bound_update_interval == 0:
                print("Bound update: once at start (no periodic updates)")
            else:
                print(f"Bound update interval: {bound_update_interval} calls "
                      f"(~{bound_update_interval / rwalk_K:.0f} iters, "
                      f"{bound_update_interval / rwalk_K / nlive:.1f} x nlive)")

        # Auto-tune batch_size / queue_size
        # Queue mode: Dynesty-style batch queue where each walk gets full rwalk_K steps.
        # Legacy mode: split rwalk_K across batch_size walks (original JNesty behavior).
        # Dynesty default: always uses multiprocessing (queue) with multi-ellipsoid.
        if queue_size is None:
            if bound == 'multi':
                queue_size = DEFAULT_QUEUE_SIZE  # Dynesty-style GPU parallelism
            else:
                queue_size = 0  # legacy mode for non-multi bounds
        elif isinstance(queue_size, str):
            if queue_size != 'auto':
                raise ValueError("queue_size must be an integer, 'auto', or "
                                 "None (got '{0}')!".format(queue_size))
            if bound == 'multi':
                queue_size = auto_queue_size(
                    loglikelihood, prior_transform, ndim, 
                    memory_frac=memory_frac, context=context, verbose=verbose)
            else:
                queue_size = 0  # the queue mode needs the multi-ellipsoid bound
                if verbose:
                    print('Queue mode needs bound=\'multi\', so queue_size=0')

        if queue_size > 1:
            # Queue mode: scale adapts at queue drain (no per-iteration noise)
            batch_size = 1  # not used in queue mode, set for config compat
            if verbose:
                print(f"Queue mode: queue_size={queue_size}, "
                      f"each walk gets full {rwalk_K} steps")
        elif batch_size is None:
            steps_per_walk = LEGACY_STEPS_PER_WALK
            batch_size = max(1, rwalk_K // steps_per_walk)
            if verbose:
                print(f"Auto-tuned batch_size = {batch_size} "
                      f"(rwalk_K={rwalk_K}, ~{max(1, rwalk_K // batch_size)} steps/walk)")
        self.batch_size = batch_size
        self.queue_size = queue_size

        if device == 'cpu':
            jax.config.update('jax_platform_name', 'cpu')

        self._results = None
        self._raw_result = None

    def run_nested(
        self,
        max_iterations: int = 100000,
        delta_logZ_threshold: float = 0.01,
        print_progress: bool = True,
        logl_min: float = -float('inf'),
        logl_max: float = float('inf'),
        init_live_x=None,
        init_live_logL=None,
        init_logvol: float = 0.0,
        init_logZ_val: float = -float('inf'),
        init_logX: float = 0.0,
        init_ncall: int = 0,
        init_iter_offset: int = 0,
    ) -> None:
        """
        Run the nested sampling algorithm.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations. Default: 100000
        delta_logZ_threshold : float, optional
            Convergence threshold. Default: 0.01
        print_progress : bool, optional
            Show progress bar. Default: True
        logl_min, logl_max : float, optional
            Batch log-likelihood bounds (used by DynamicNestedSampler).
        init_live_x, init_live_logL : array-like, optional
            Pre-sampled seed live points (unit cube) and their logL values.
            If provided, Phase 1 uniform rejection is skipped.
        init_logvol : float, optional
            Starting log-volume (path-A fallback when -inf-logL points were rejected).
        """
        if self.verbose:
            print("=" * 70)
            print("JNesty Nested Sampling")
            print("=" * 70)
            print(f"Dimensions: {self.ndim}")
            print(f"Live points: {self.nlive}")
            print(f"Max iterations: {max_iterations}")
            print(f"Device: {self.device.upper()}")
            print(f"Walk steps per iteration: {self.rwalk_K}")
            print(f"Initial step scale: {self.rwalk_step_scale:.4f}")
            print(f"Convergence threshold: delta_logZ < {delta_logZ_threshold}")
            print("=" * 70)
            print()

        self._delta_logZ_threshold = delta_logZ_threshold

        def prior_sample(key):
            return random.uniform(key, shape=(self.ndim,), minval=0.0, maxval=1.0)

        config = WhileLoopNSConfig(
            nlive=self.nlive,
            max_iterations=max_iterations,
            delta_logZ_threshold=delta_logZ_threshold,
            rwalk_K=self.rwalk_K,
            rwalk_step_scale=self.rwalk_step_scale,
            target_acceptance=self.target_acceptance,
            verbose=self.verbose,
            print_progress=print_progress,
            bound=self.bound,
            bound_update_interval=self.bound_update_interval,
            max_ellipsoids=self.max_ellipsoids,
            batch_size=self.batch_size,
            queue_size=self.queue_size,
            memory_frac=self.memory_frac,
            unit_cube_batch_size=self.unit_cube_batch_size,
            min_eff=self.min_eff,
            min_ncall=self.min_ncall,
            logl_min=logl_min,
            logl_max=logl_max,
            init_live_x=init_live_x,
            init_live_logL=init_live_logL,
            init_logvol=init_logvol,
            init_logZ_val=init_logZ_val,
            init_logX=init_logX,
            init_ncall=init_ncall,
            init_iter_offset=init_iter_offset,
        )

        key = random.PRNGKey(42)
        self._raw_result = run_nested_sampling(
            loglikelihood_fn=self.loglikelihood,
            prior_sample_fn=prior_sample,
            ndim=self.ndim,
            config=config,
            key=key,
            prior_transform_fn=self.prior_transform,
            context=self.context,
        )

        if self.verbose:
            print()
            print("=" * 70)
            print("Sampling Complete!")
            print("=" * 70)

    @property
    def results(self) -> Results:
        """
        Results object following Dynesty's format.

        Raises RuntimeError if run_nested() has not been called.
        """
        if self._raw_result is None:
            raise RuntimeError("Must call run_nested() before accessing results")

        if self._results is None:
            self._results = format_results(
                self._raw_result, self.prior_transform, self.ndim,
                self.nlive, self.rwalk_K, self._delta_logZ_threshold,
                context=self.context,
            )

        return self._results

    def print_summary(self) -> None:
        """Print a formatted summary of the results."""
        if self._raw_result is None:
            print("No results yet. Call run_nested() first.")
            return
        self.results.summary()

    def get_samples(self, weights: bool = False) -> np.ndarray:
        """Get posterior samples in physical space."""
        if weights:
            print("Warning: Weighted sampling not yet implemented. "
                  "Returning unweighted samples.")
        return self.results['samples']

    def get_logz(self) -> tuple:
        """Get (logZ, logZ_error)."""
        r = self.results
        return r['logz'], r['logzerr']

    def to_dynesty_results(self):
        """
        Convert results to a Dynesty Results object for use with
        Dynesty's plotting functions.
        """
        try:
            from dynesty.utils import Results as DynestyResults
        except ImportError:
            raise ImportError("dynesty package required. Install with: pip install dynesty")

        r = self.results
        samples_id = np.arange(len(r['samples']))

        dynesty_dict = {
            'samples_u': r['samples_u'],
            'samples_id': samples_id,
            'logl': r['logl'],
            'samples': r['samples'],
            'nlive': r['nlive'],
            'niter': r['niter'],
            'logwt': r['logwt'],
            'logvol': r['logvol'],
            'logz': r['logz_trajectory'],
            'logzerr': r['logzerr_trajectory'],
        }

        return DynestyResults(dynesty_dict)

    # Plotting convenience methods (delegate to Dynesty's plotting)

    def plot_run(self, lnz_truth=None, **kwargs):
        """Run plot via Dynesty's plotting.runplot()."""
        from dynesty import plotting as dyplot
        return dyplot.runplot(self.to_dynesty_results(), lnz_truth=lnz_truth, **kwargs)

    def plot_trace(self, truths=None, dims=None, thin=1, **kwargs):
        """Trace plot via Dynesty's plotting.traceplot()."""
        from dynesty import plotting as dyplot
        return dyplot.traceplot(self.to_dynesty_results(), truths=truths,
                                dims=dims, thin=thin, **kwargs)

    def plot_corner(self, truths=None, **kwargs):
        """Corner plot via Dynesty's plotting.cornerplot()."""
        from dynesty import plotting as dyplot
        return dyplot.cornerplot(self.to_dynesty_results(), truths=truths, **kwargs)

    def plot_diagnostics(self, **kwargs):
        """Diagnostics via Dynesty's plotting.runplot()."""
        from dynesty import plotting as dyplot
        return dyplot.runplot(self.to_dynesty_results(), **kwargs)


# -----------------------------------------------------------------------------
# Dynamic nested sampling
# -----------------------------------------------------------------------------
def _build_static_sampler(loglikelihood, prior_transform, ndim,
                           nlive, bound, **kwargs):
    """Construct a NestedSampler with kwargs suitably defaulted."""
    accepted_kwargs = {k: v for k, v in kwargs.items() if k != 'max_iterations'}
    return NestedSampler(loglikelihood, prior_transform, ndim,
                          nlive=nlive, bound=bound, **accepted_kwargs)


class DynamicNestedSampler:
    """Dynesty-style dynamic nested sampler.

    Wraps JNesty's static engine for both the base run and each batch.

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood in physical parameter space.
    prior_transform : callable
        Maps a unit-cube sample to physical space.
    ndim : int
    nlive : int, default 500
    bound : {'none', 'single', 'multi'}, default 'none'
    **static_kwargs :
        Forwarded to NestedSampler.
    """

    def __init__(self, loglikelihood, prior_transform, ndim,
                 nlive=500, bound='none', **static_kwargs):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.nlive0 = nlive
        self.bound = bound
        self.static_kwargs = static_kwargs
        self._results = None
        self.saved_run = None
        self.batch_bounds_log = []
        self.batch_nlive_log = []

    def run_nested(self,
                   nlive_init=None,
                   dlogz_init=0.01,
                   nlive_batch=None,
                   maxbatch=100,
                   max_iterations=100000,
                   pfrac=0.8,
                   maxfrac=0.8,
                   pad=1,
                   n_effective=None,
                   evid_thresh=0.1,
                   n_mc=0,
                   use_stop=True,
                   print_progress=True,
                   seed=None):
        """Run the dynamic nested sampling loop.

        max_iterations is forwarded to every internal static sub-run (the base
        run and each batch) as the per-run hard iteration cap; convergence is
        controlled by dlogz_init.
        """
        import jax.numpy as jnp

        nlive_init = nlive_init or self.nlive0
        nlive_batch = nlive_batch or self.nlive0
        if n_effective is None:
            n_effective = max(self.ndim * self.ndim, 10000)

        rstate = np.random.default_rng(seed)

        # === Base run ===
        if print_progress:
            print(f"[dynamic] base run: nlive_init={nlive_init}, dlogz={dlogz_init}")
        base_sampler = _build_static_sampler(
            self.loglikelihood, self.prior_transform, self.ndim,
            nlive=nlive_init, bound=self.bound, **self.static_kwargs)
        # The queue size may have been resolved by the static sampler ('auto')
        self.queue_size = base_sampler.queue_size
        base_sampler.run_nested(
            max_iterations=max_iterations,
            delta_logZ_threshold=dlogz_init,
            print_progress=print_progress,
        )
        base_res = self._normalize_static_results(base_sampler.results, nlive_init, 0)
        self.saved_run = base_res
        self.batch_bounds_log = [(-np.inf, np.inf)]
        self.batch_nlive_log = [nlive_init]

        # Memory-adaptive fresh-point chunk size (identical strategy to the
        # static sampler's estimate_batch_size_from_memory).
        memory_frac = self.static_kwargs.get('memory_frac', 0.9)
        fresh_batch_size = estimate_fresh_batch_size_from_memory(
            self.loglikelihood, self.prior_transform, self.ndim,
            requested_batch_size=50, memory_frac=memory_frac,
            verbose=print_progress)

        # === Batch loop ===
        wt_kwargs = {'pfrac': pfrac, 'maxfrac': maxfrac, 'pad': pad}
        stop_kwargs = {'pfrac': pfrac, 'evid_thresh': evid_thresh,
                       'target_n_effective': n_effective, 'n_mc': n_mc}

        for n in range(maxbatch):
            res = self._build_results()
            if use_stop:
                stop, stop_vals = stopping_function(res, stop_kwargs,
                                                     rstate=rstate,
                                                     return_vals=True)
                stop_val = stop_vals[2]
            else:
                stop = False
                stop_val = float('nan')

            if stop:
                if print_progress:
                    print(f"[dynamic] stop after batch {n} (stop_val={stop_val:.3f})")
                break

            logl_bounds = weight_function(res, args=wt_kwargs)
            logl_min_b, logl_max_b = logl_bounds

            if print_progress:
                ess_now = kish_ess(res['logwt'])
                print(f"[dynamic] batch {n+1}/{maxbatch}: "
                      f"logl_bounds=({logl_min_b:.2f}, {logl_max_b:.2f}), "
                      f"ESS={ess_now:.0f}/{n_effective}, stop_val={stop_val:.3f}")

            # Seed batch live points
            if np.isneginf(logl_min_b):
                init_x = None
                init_logL = None
            else:
                # Path B: subset-select + generate FRESH points (matching dynesty)
                live_u, live_logl, scale, logl_min_eff = seed_batch_from_saved(
                    res, nlive_batch, logl_min_b, rstate)
                # Generate nlive_batch FRESH points above logl_min via
                # rejection sampling from the fitted bound.  This decorrelates
                # the batch's live points from the saved run, matching
                # dynesty's _new_point(logl_min) behavior.
                from jax import random as jrandom
                fresh_key = jrandom.PRNGKey(int(rstate.integers(0, 2**31 - 1)))
                fresh_u, fresh_logl, self._compiled_logl = generate_fresh_batch_points(
                    self.loglikelihood, self.prior_transform,
                    live_u, logl_min_eff, self.ndim, nlive_batch,
                    self.bound, scale, fresh_key,
                    fresh_batch_size=fresh_batch_size,
                    _compiled_logl=getattr(self, '_compiled_logl', None))
                init_x = jnp.asarray(fresh_u)
                init_logL = jnp.asarray(fresh_logl)
                logl_min_b = logl_min_eff

            batch_sampler = _build_static_sampler(
                self.loglikelihood, self.prior_transform, self.ndim,
                nlive=nlive_batch, bound=self.bound, **self.static_kwargs)
            # Pre-load saved history below logl_min for correct convergence
            init_logZ_val = -float('inf')
            init_logX_val = 0.0
            init_ncall_val = 0
            init_iter_off = 0
            if not np.isneginf(logl_min_b):
                saved_logl = np.asarray(self.saved_run['logl'], dtype=float)
                saved_logvol = np.asarray(self.saved_run['logvol'], dtype=float)
                saved_logz = np.asarray(self.saved_run['logz'], dtype=float)
                vol_idx = int(np.searchsorted(saved_logl, logl_min_b, side='right'))
                vol_idx = min(vol_idx, len(saved_logl) - 1)
                if vol_idx > 0:
                    init_logZ_val = float(saved_logz[vol_idx - 1])
                    init_logX_val = float(saved_logvol[vol_idx - 1])
                    init_ncall_val = int(vol_idx * 25)  # approximate ncall from pre-loaded history
                    init_iter_off = vol_idx

            # Use a smaller dlogz for batches to run more iterations,
            # compensating for JNesty's different convergence behavior vs dynesty.
            batch_sampler.run_nested(
                max_iterations=max_iterations,
                delta_logZ_threshold=dlogz_init,
                print_progress=print_progress,
                init_live_x=init_x,
                init_live_logL=init_logL,
                logl_min=logl_min_b,
                logl_max=logl_max_b,
                init_logZ_val=init_logZ_val,
                init_logX=init_logX_val,
                init_ncall=init_ncall_val,
                init_iter_offset=init_iter_off,
            )
            new_res = self._normalize_static_results(
                batch_sampler.results, nlive_batch, n + 1)

            self.saved_run = combine_saved_and_new(
                self.saved_run, new_res, logl_min_b, logl_max_b)
            self.batch_bounds_log.append((logl_min_b, logl_max_b))
            self.batch_nlive_log.append(nlive_batch)

        self._results = self._build_results()
        if print_progress:
            self.print_summary()

    @staticmethod
    def _normalize_static_results(res, nlive, batch_idx):
        """Convert a static-format Results dict into the dynamic-compatible form.

        Static format_results() appends the remaining live points to the dead
        points, producing arrays of length niter + nlive. For those tail points
        samples_n decreases from (nlive-1) down to 0, matching dynesty's
        add_live_points() semantics.
        """
        res = dict(res)
        n_total = len(res['logl'])
        niter = int(res.get('niter', n_total))
        n_live_tail = n_total - niter  # usually == nlive

        # Convert scalar logz to trajectory if needed
        if 'logz_trajectory' in res and not hasattr(res.get('logz'), '__len__'):
            res['logz'] = np.asarray(res['logz_trajectory'], dtype=float)
        elif not hasattr(res.get('logz', None), '__len__'):
            logwt = np.asarray(res['logwt'], dtype=float)
            res['logz'] = np.logaddexp.accumulate(logwt)
        res['logz'] = np.asarray(res['logz'], dtype=float)

        # samples_n: nlive for the main loop; dynesty uses n=nlive-it for the
        # tail (it=0..nlive-1) giving [nlive, nlive-1, ..., 1].
        if 'samples_n' not in res:
            samples_n = np.full(niter, nlive, dtype=float)
            if n_live_tail > 0:
                tail_len = n_live_tail
                start = nlive
                end = max(nlive - tail_len + 1, 1)
                tail = np.arange(start, end - 1, -1, dtype=float)
                if len(tail) < tail_len:
                    tail = np.concatenate([tail, np.ones(tail_len - len(tail))])
                samples_n = np.concatenate([samples_n, tail])
            res['samples_n'] = samples_n
        if 'samples_batch' not in res:
            res['samples_batch'] = np.full(n_total, batch_idx, dtype=int)
        # scale history (extend to n_total if needed)
        if 'scale' not in res:
            traj = res.get('scale_trajectory', None)
            if traj is None or len(traj) < n_total:
                traj = np.ones(n_total)
            res['scale'] = np.asarray(traj)
        # aliases used by combine_saved_and_new
        res['n'] = res['samples_n']
        res['u'] = np.asarray(res.get('samples_u', np.zeros((n_total, 1))))
        res['batch'] = res['samples_batch']
        # Sanitize -inf logl values (float32-safe replacement)
        if np.any(np.isneginf(np.asarray(res['logl']))):
            logl_arr = np.asarray(res['logl'], dtype=float)
            safe_min = np.finfo(logl_arr.dtype).min / 2
            logl_arr = np.where(np.isneginf(logl_arr), safe_min, logl_arr)
            res['logl'] = logl_arr
        # logzerr trajectory: dynesty stores per-iter variance
        if 'logzerr_trajectory' in res and len(res['logzerr_trajectory']) == n_total:
            logzerr_traj = np.asarray(res['logzerr_trajectory'], dtype=float)
            res['logzerr'] = logzerr_traj
            res['logzvar'] = logzerr_traj ** 2
        elif 'logzerr' not in res or not hasattr(res['logzerr'], '__len__'):
            try:
                _, _, logzvar_traj, _ = compute_integrals(
                    logl=res['logl'], logvol=res['logvol'])
                res['logzvar'] = logzvar_traj
                res['logzerr'] = np.sqrt(np.maximum(logzvar_traj, 0.0))
            except Exception:
                res['logzvar'] = np.zeros(n_total)
                res['logzerr'] = np.zeros(n_total)
        # also expose information as h for consistency with combine output
        h = np.asarray(res.get('information', 0.0), dtype=float)
        res.setdefault('h', np.full(n_total, float(h)) if h.ndim == 0 else h)
        return res

    def _build_results(self):
        if self.saved_run is None:
            return None
        res = dict(self.saved_run)
        res['batch_nlive'] = list(self.batch_nlive_log)
        res['batch_bounds'] = list(self.batch_bounds_log)
        return res

    @property
    def results(self):
        if self._results is None:
            self._results = self._build_results()
        return self._results

    def to_dynesty_results(self):
        """Convert the dynamic results dict to a dynesty Results object.

        Needed so dynesty.plotting.runplot/traceplot/cornerplot can be used
        directly on JNesty dynamic output.
        """
        try:
            from dynesty.utils import Results as DynestyResults
        except ImportError:
            raise ImportError("dynesty required. Install with: pip install dynesty")
        r = self.results
        if r is None:
            raise RuntimeError("run_nested() has not been called yet")
        n = len(r['logl'])
        samples_u = np.asarray(r.get('samples_u', r.get('u', np.zeros((n, 1)))))
        items = [
            ('samples_u', samples_u),
            ('samples_id', np.arange(n)),
            ('logl', np.asarray(r['logl'], dtype=float)),
            ('samples', samples_u),  # physical == unit cube for prior_ident; caller can override
            ('samples_n', np.asarray(r['samples_n'], dtype=float)),
            ('niter', n),
            ('logwt', np.asarray(r['logwt'], dtype=float)),
            ('logvol', np.asarray(r['logvol'], dtype=float)),
            ('logz', np.asarray(r['logz'], dtype=float)),
            ('logzerr', np.asarray(r['logzerr'], dtype=float)),
            ('information', np.asarray(r['h'], dtype=float)),
            ('batch_nlive', list(r.get('batch_nlive', []))),
            ('batch_logl_bounds', list(r.get('batch_bounds', []))),
            ('samples_batch', np.asarray(r['samples_batch'], dtype=int)),
        ]
        return DynestyResults(items)

    def print_summary(self):
        if self._results is None:
            print("[dynamic] no results yet")
            return
        r = self._results
        logz = float(np.atleast_1d(r['logz'])[-1])
        logzerr = float(np.atleast_1d(r['logzerr'])[-1])
        h = float(np.atleast_1d(r['h'])[-1])
        print(f"[dynamic] niter={len(r['logl'])}, logz={logz:.4f} "
              f"± {logzerr:.4f}, h={h:.2f}, "
              f"ESS={kish_ess(r['logwt']):.0f}, "
              f"nbatches={len(self.batch_nlive_log)-1}")
