"""
Core nested sampling loop.

Provides the NestedSamplingLoop class that manages the NS iteration:
live points, evidence accumulation, convergence checking.
Delegates point proposal to InternalSampler and bounding to Bound.

Returns raw WhileLoopNSResult for formatting by results.py.
"""

import time
from typing import Optional, NamedTuple

import jax
import jax.numpy as jnp
from jax import random, lax

from .internal_samplers import RWalkSampler, _single_walk, vmap_queue_refill
from .utils import logsubexp

# Import for progress tracking
try:
    from jax.experimental import io_callback
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class WhileLoopNSConfig(NamedTuple):
    """Configuration for the nested sampling run."""
    nlive: int = 500
    max_iterations: int = 10000
    delta_logZ_threshold: float = 0.01
    rwalk_K: int = 25
    rwalk_step_scale: float = 1.0
    target_acceptance: float = 0.5
    prior_bounds: Optional[jnp.ndarray] = None
    verbose: bool = True
    print_progress: bool = True
    bound: str = 'none'
    bound_update_interval: int = 0
    max_ellipsoids: int = 20
    batch_size: int = 1
    queue_size: int = 0  # 0 = legacy mode (batch), >1 = queue mode (Dynesty-style)
    memory_frac: float = 0.9
    ncdim: int = None  # number of clustered dimensions (defaults to ndim)
    unit_cube_batch_size: int = 200  # batch size for uniform rejection phase
    min_eff: float = 10.0           # efficiency threshold to switch to rwalk (%)
    min_ncall: int = None           # min calls before switch (default 2*nlive)


class WhileLoopNSResult(NamedTuple):
    """Raw results from the nested sampling loop."""
    logZ: float
    logZ_error: float
    H: float
    delta_logZ: float
    n_iterations: int
    runtime: float
    samples: jnp.ndarray
    logL_samples: jnp.ndarray
    delta_logZ_trajectory: jnp.ndarray
    scale_trajectory: jnp.ndarray
    acceptance_rate: float
    live_x: jnp.ndarray = None
    live_logL: jnp.ndarray = None


def estimate_batch_size_from_memory(
    loglikelihood_fn, live_x, bound_axes, ndim, rwalk_K,
    requested_batch_size, memory_frac=0.9, verbose=True,
):
    """
    Cap batch_size based on available GPU memory.

    Compiles a vmapped walk (batch_size=2) with the actual likelihood,
    uses XLA's memory_analysis() to get peak memory, then caps batch_size
    to fit within memory_frac of total GPU memory.

    Returns the capped (or unchanged) batch_size.
    """
    if requested_batch_size <= 1:
        return requested_batch_size

    device = jax.devices()[0]
    stats = device.memory_stats()
    if stats is None:
        return requested_batch_size

    trial_batch = 2
    trial_steps = max(1, rwalk_K // trial_batch)
    x_start = live_x[0]

    try:
        trial_keys = random.split(random.PRNGKey(0), trial_batch)
        trial_fn = jax.vmap(
            lambda wk: _single_walk(
                wk, x_start, jnp.array(-jnp.inf), loglikelihood_fn,
                bound_axes, jnp.array(1.0), trial_steps, ndim, ndim,
                None, None
            )
        )
        compiled = jax.jit(trial_fn).lower(trial_keys).compile()
    except Exception as e:
        if verbose:
            print(f"WARNING: Memory probe failed ({e}). Falling back to batch_size=1.")
        return 1

    ma = compiled.memory_analysis()
    if ma is None:
        return requested_batch_size

    peak = ma.peak_memory_in_bytes
    per_walk = max(1, peak // trial_batch)

    available = int(stats['bytes_limit'] * memory_frac)
    max_batch = max(1, available // per_walk)
    capped = min(requested_batch_size, max_batch)

    if verbose and capped < requested_batch_size:
        print(f"  Memory cap: batch_size {requested_batch_size} -> {capped} "
              f"(per-walk: {per_walk / 1024:.0f} KB, "
              f"budget: {available / (1024**2):.0f} MB, "
              f"GPU: {stats['bytes_limit'] / (1024**3):.1f} GB)")

    return capped


def _batch_logL_eval(loglikelihood_fn, live_x, batch_size=50):
    """Evaluate logL for all live points in batches to avoid OOM."""
    nlive = live_x.shape[0]
    if nlive <= batch_size:
        return jnp.vectorize(loglikelihood_fn, signature='(n)->()')(live_x)
    results = []
    for i in range(0, nlive, batch_size):
        batch = live_x[i:i + batch_size]
        results.append(jnp.vectorize(loglikelihood_fn, signature='(n)->()')(batch))
    return jnp.concatenate(results)


def _run_uniform_phase(loglikelihood_fn, live_x, live_logL, worst_x_buffer,
                       worst_logL_buffer, delta_logZ_buffer, scale_buffer,
                       logZ, delta_logZ, iteration_offset, key, nlive,
                       max_iterations, delta_logZ_threshold, unit_cube_batch_size,
                       min_eff, min_ncall, ndim):
    """
    Phase 1: Uniform rejection sampling from the unit cube.

    Draws batches of random points, evaluates logL via vmap, picks the first
    valid replacement. Switches to rwalk when efficiency drops below min_eff.

    Returns updated state tuple ready for Phase 2 (rwalk).
    """
    vmapped_loglike = jax.vmap(loglikelihood_fn)

    # Use consistent dtype matching the likelihood output
    dtype = live_logL.dtype

    def phase1_cond(state):
        # state: live_x, live_logL, wx_buf, wl_buf, dlz_buf, sc_buf,
        #        logZ, delta_logZ, it, key, scale, tot_calls
        dlz = state[7]
        it = state[8]
        tot_calls = state[11]
        # Stop if converged, hit max iterations, or efficiency below threshold
        converged = dlz < delta_logZ_threshold
        too_many = it >= max_iterations
        # Efficiency: (it + nlive) / tot_calls * 100
        # Switch when eff < min_eff AND tot_calls >= min_ncall
        eff = (it + nlive) * 100.0 / jnp.maximum(tot_calls, 1)
        switch = (eff < min_eff) & (tot_calls >= min_ncall)
        return (~converged) & (~too_many) & (~switch)

    def phase1_body(state):
        live_x, live_logL, wx_buf, wl_buf, dlz_buf, sc_buf, \
            logZ, dlz, it, key, scale, tot_calls = state

        # Find worst point
        worst_idx = jnp.argmin(live_logL)
        worst_logL = live_logL[worst_idx]

        # Draw batch of uniform random points
        key, subkey = random.split(key)
        u_batch = random.uniform(subkey, (unit_cube_batch_size, ndim))
        logL_batch = vmapped_loglike(u_batch)

        # Find first valid point (logL > worst_logL)
        valid = logL_batch > worst_logL
        first_valid = jnp.argmax(valid)
        any_valid = jnp.any(valid)
        x_new = jnp.where(any_valid, u_batch[first_valid], live_x[0])
        logL_new = jnp.where(any_valid, logL_batch[first_valid], -jnp.inf)
        # Count actual proposals used (matching Dynesty's nc per iteration)
        actual_calls = jnp.where(any_valid, first_valid.astype(jnp.int32) + 1,
                                 jnp.array(unit_cube_batch_size, dtype=jnp.int32))

        # Update live points
        live_x_new = live_x.at[worst_idx].set(x_new)
        live_logL_new = live_logL.at[worst_idx].set(logL_new)

        # Update evidence
        logX_old = -it / nlive
        logX_new_val = -(it + 1) / nlive
        log_dX = logsubexp(logX_old, logX_new_val)
        log_dZ = worst_logL + log_dX
        logZ_new = jnp.logaddexp(logZ, log_dZ)

        # Calculate delta_logZ
        max_live_logL = jnp.max(live_logL_new)
        delta_logZ_new = jax.scipy.special.logsumexp(
            jnp.array([jnp.zeros((), dtype=dtype), max_live_logL + logX_new_val - logZ_new])
        )

        # Store in buffers
        wx_buf_new = wx_buf.at[it].set(live_x[worst_idx])
        wl_buf_new = wl_buf.at[it].set(worst_logL)
        dlz_buf_new = dlz_buf.at[it].set(delta_logZ_new)
        sc_buf_new = sc_buf.at[it].set(scale)

        return (live_x_new, live_logL_new,
                wx_buf_new, wl_buf_new, dlz_buf_new, sc_buf_new,
                logZ_new, delta_logZ_new,
                it + 1, key, scale, tot_calls + actual_calls)

    init_total_calls = jnp.array(nlive, dtype=jnp.int32)
    phase1_init = (live_x, live_logL,
                   worst_x_buffer, worst_logL_buffer,
                   delta_logZ_buffer, scale_buffer,
                   jnp.array(logZ, dtype=dtype),
                   jnp.array(delta_logZ, dtype=dtype),
                   jnp.array(iteration_offset, dtype=jnp.int32), key,
                   jnp.array(1.0, dtype=dtype),  # scale placeholder
                   init_total_calls)

    phase1_fn = phase1_body

    compiled_phase1 = jax.jit(lambda s: lax.while_loop(phase1_cond, phase1_fn, s))
    phase1_result = compiled_phase1(phase1_init)

    return phase1_result


def run_nested_sampling(
    loglikelihood_fn,
    prior_sample_fn,
    ndim,
    config,
    key=None,
    prior_transform_fn=None,
):
    """
    Run nested sampling with iteration-granular adaptive termination.

    Two-phase approach matching Dynesty:
    1. Phase 1: Uniform rejection sampling until efficiency < min_eff
    2. Transition: Fit initial bound from live points
    3. Phase 2: Random walk sampling with adaptive bounding

    Parameters
    ----------
    loglikelihood_fn : callable
        Log-likelihood function (physical space).
    prior_sample_fn : callable
        Prior sampling function (returns unit cube samples).
    ndim : int
        Problem dimensionality.
    config : WhileLoopNSConfig
        Run configuration.
    key : jax.random.PRNGKey, optional
    prior_transform_fn : callable, optional

    Returns
    -------
    WhileLoopNSResult
    """
    if key is None:
        key = random.PRNGKey(42)

    nlive = config.nlive
    max_iterations = config.max_iterations
    delta_logZ_threshold = config.delta_logZ_threshold
    rwalk_K = config.rwalk_K
    min_ncall = config.min_ncall if config.min_ncall is not None else 2 * nlive

    # Wrap likelihood with prior_transform if needed
    if prior_transform_fn is not None:
        def loglikelihood_wrapped(x):
            return loglikelihood_fn(prior_transform_fn(x))
        loglikelihood_for_jit = loglikelihood_wrapped
    else:
        loglikelihood_for_jit = loglikelihood_fn

    # Create progress bar covering entire sampling run
    if config.print_progress and TQDM_AVAILABLE:
        pbar = tqdm(total=max_iterations, desc="JNesty")
    else:
        pbar = None

    start_time = time.time()

    # Initialize live points from prior
    keys = random.split(key, nlive + 1)
    live_x = jnp.stack([prior_sample_fn(k) for k in keys[:-1]])
    live_logL = _batch_logL_eval(loglikelihood_for_jit, live_x)

    # Pre-allocate buffers — use dtype matching the likelihood output
    buf_dtype = live_logL.dtype
    worst_x_buffer = jnp.zeros((max_iterations, ndim), dtype=live_x.dtype)
    worst_logL_buffer = jnp.full(max_iterations, -jnp.inf, dtype=buf_dtype)
    delta_logZ_buffer = jnp.full(max_iterations, jnp.inf, dtype=buf_dtype)
    scale_buffer = jnp.full(max_iterations, jnp.inf, dtype=buf_dtype)

    logZ = jnp.array(-jnp.inf, dtype=buf_dtype)
    max_live_logL = jnp.max(live_logL)
    delta_logZ = jax.scipy.special.logsumexp(
        jnp.array([0.0, max_live_logL - logZ], dtype=buf_dtype)
    )

    # === PHASE 1: Uniform rejection sampling ===
    phase1_result = _run_uniform_phase(
        loglikelihood_for_jit, live_x, live_logL,
        worst_x_buffer, worst_logL_buffer,
        delta_logZ_buffer, scale_buffer,
        logZ, delta_logZ,
        jnp.array(0),  # iteration offset
        keys[-1],
        nlive=nlive,
        max_iterations=max_iterations,
        delta_logZ_threshold=delta_logZ_threshold,
        unit_cube_batch_size=config.unit_cube_batch_size,
        min_eff=config.min_eff,
        min_ncall=min_ncall,
        ndim=ndim,
    )

    # Unpack Phase 1 results
    live_x = phase1_result[0]
    live_logL = phase1_result[1]
    worst_x_buffer = phase1_result[2]
    worst_logL_buffer = phase1_result[3]
    delta_logZ_buffer = phase1_result[4]
    scale_buffer = phase1_result[5]
    logZ = phase1_result[6]
    delta_logZ = phase1_result[7]
    phase1_iters = int(phase1_result[8])
    key = phase1_result[9]

    phase1_total_calls = int(phase1_result[11])
    phase1_eff = (phase1_iters + nlive) * 100.0 / max(phase1_total_calls, 1)

    converged_phase1 = float(delta_logZ) < delta_logZ_threshold

    # Update progress bar after Phase 1
    if pbar is not None:
        pbar.n = phase1_iters
        loglstar_p1 = float(jnp.min(live_logL))
        pbar.set_postfix_str(
            f'logZ: {float(logZ):.2f} | dlogZ: {float(delta_logZ):.3f} ({delta_logZ_threshold}) | '
            f'logl*: {loglstar_p1:.1f} | eff(%): {phase1_eff:.1f}')
        pbar.refresh()

    if converged_phase1:
        # Already converged during Phase 1
        if pbar is not None:
            pbar.close()
        iteration_final = phase1_iters
        final_hist_accept = 0
        final_hist_total = phase1_total_calls
    else:
        # === TRANSITION: Fit initial bound ===

        from .bounding import get_bound
        from .internal_samplers import get_sampler

        bound_obj = get_bound(config.bound, ndim,
                              max_ellipsoids=config.max_ellipsoids,
                              scale=config.rwalk_step_scale)

        if config.bound != 'none':
            # Fit in unit-cube space (matching Dynesty: self.live_u[:, :self.ncdim])
            bound_obj.fit(live_x)

        # Memory-cap batch_size
        bound_axes_for_probe = bound_obj.get_axes()

        effective_batch_size = estimate_batch_size_from_memory(
            loglikelihood_fn=loglikelihood_for_jit,
            live_x=live_x,
            bound_axes=bound_axes_for_probe,
            ndim=ndim,
            rwalk_K=rwalk_K,
            requested_batch_size=config.batch_size,
            memory_frac=config.memory_frac,
            verbose=config.verbose,
        )
        config = config._replace(batch_size=effective_batch_size)

        sampler_obj = get_sampler('rwalk', ndim,
                                  target_acceptance=config.target_acceptance,
                                  batch_size=effective_batch_size,
                                  ncdim=config.ncdim if config.ncdim else ndim)

        # The sampler may round batch_size down for GPU sharding (multiple of n_devices).
        # Use the actual batch_size from the sampler for all downstream code.
        effective_batch_size = sampler_obj.batch_size
        config = config._replace(batch_size=effective_batch_size)

        # === PHASE 2: Rwalk with adaptive bounding ===
        use_multi_ellipsoid = config.bound == 'multi'
        bound_update_interval = config.bound_update_interval
        use_chunked_loop = use_multi_ellipsoid and bound_update_interval > 0
        max_ellipsoids = config.max_ellipsoids

        current_scale = config.rwalk_step_scale
        acceptance_count = 0
        total_proposals = 0

        bound_axes = bound_obj.get_axes()

        if use_multi_ellipsoid and hasattr(bound_obj, 'state') and bound_obj.state is not None:
            me_axes = bound_obj.state.axes
            me_logvol_ells = bound_obj.state.logvol_ells
        else:
            me_axes = jnp.zeros((max_ellipsoids, ndim, ndim))
            me_logvol_ells = jnp.full(max_ellipsoids, -jnp.inf)

        has_prior_bounds = config.prior_bounds is not None

        # Queue mode setup: pre-fill queue with initial candidates
        if config.queue_size > 1:
            _ncdim_init = config.ncdim if config.ncdim else ndim
            key, _refill_key = random.split(key)
            _loglstar_init = jnp.min(live_logL)
            _init_qx, _init_qlogL, _init_qnacc, _init_qntot = vmap_queue_refill(
                _refill_key, live_x, live_logL, _loglstar_init, loglikelihood_for_jit,
                me_axes, me_logvol_ells, bound_axes, current_scale,
                rwalk_K, ndim, _ncdim_init, config.queue_size, use_multi_ellipsoid,
                config.prior_bounds if has_prior_bounds else None)

        # State for Phase 2 (continues from Phase 1)
        _base_state = (
            live_x,                  # 0
            live_logL,               # 1
            worst_x_buffer,          # 2
            worst_logL_buffer,       # 3
            delta_logZ_buffer,       # 4
            scale_buffer,            # 5
            logZ,                    # 6
            delta_logZ,              # 7
            jnp.array(phase1_iters, dtype=jnp.int32), # 8  iteration
            key,                     # 9  key
            current_scale,           # 10 scale
            jnp.array(0, dtype=jnp.int32),  # 11 hist_accept (accumulated, reset at bound update)
            jnp.array(0, dtype=jnp.int32),  # 12 hist_total (accumulated, reset at bound update)
            bound_axes,              # 13 bound_axes
            me_axes,                 # 14 me_axes
            me_logvol_ells,          # 15 me_logvol_ells
            jnp.array(phase1_total_calls, dtype=jnp.int32), # 16 total_calls at last bound update
            jnp.array(phase1_total_calls, dtype=jnp.int32),  # 17 total_calls (non-resetting)
        )
        if config.queue_size > 1:
            init_state = _base_state + (
                _init_qx, _init_qlogL, _init_qnacc, _init_qntot,
                jnp.array(0, dtype=jnp.int32),  # 22: queue_head
            )
        else:
            init_state = _base_state

        def cond_fn(state):
            dlz = state[7]
            it = state[8]
            return (dlz >= delta_logZ_threshold) & (it < max_iterations)

        def body_fn(state):
            live_x = state[0]
            live_logL = state[1]
            iteration = state[8]
            key = state[9]
            scale = state[10]
            me_axes_state = state[14]
            me_logvol_ells = state[15]

            # 1. Find worst live point
            worst_idx = jnp.argmin(live_logL)
            worst_logL = live_logL[worst_idx]

            # 2-4. Retry loop: select starting point, ellipsoid, run walk.
            # Retry with fresh randomness until a valid point (logL > loglstar)
            # is found, matching Dynesty's _new_point() while-True behavior.
            key, retry_key = random.split(key)

            _bound_axes = state[13]

            def _retry_cond(rstate):
                return ~rstate[0]  # while not valid

            def _retry_body(rstate):
                _, rkey, _, _, racc, rtot = rstate
                rkey, sk1, sk2, sk3, sk4, wk = random.split(rkey, 6)

                # Select starting point(s)
                above_mask = live_logL > worst_logL
                n_above = jnp.sum(above_mask)
                rand_vals = random.uniform(sk1, (live_x.shape[0],))
                rand_vals = jnp.where(above_mask, rand_vals, -1.0)
                idx_above = jnp.argmax(rand_vals)
                idx_fallback = random.randint(sk2, (), 0, live_x.shape[0])
                idx = jnp.where(n_above > 0, idx_above, idx_fallback)
                x_current = live_x[idx]

                if effective_batch_size > 1:
                    above_probs = above_mask.astype(jnp.float32)
                    above_probs = above_probs / jnp.maximum(above_probs.sum(), 1)
                    batch_idxs = random.choice(sk4, live_x.shape[0],
                                               shape=(effective_batch_size,),
                                               p=above_probs, replace=True)
                    x_starts = live_x[batch_idxs]
                else:
                    x_starts = x_current

                # Select axes
                if use_multi_ellipsoid:
                    logvol = me_axes_state.shape[0]
                    ell_log_probs = me_logvol_ells - jax.scipy.special.logsumexp(me_logvol_ells)
                    ell_probs = jnp.exp(ell_log_probs)
                    ell_probs = jnp.where(jnp.isnan(ell_probs) | (ell_probs < 0), 0.0, ell_probs)
                    ell_probs = ell_probs / jnp.maximum(ell_probs.sum(), 1e-30)
                    if effective_batch_size > 1:
                        ell_indices = random.choice(sk3, logvol,
                                                    shape=(effective_batch_size,),
                                                    p=ell_probs, replace=True)
                        ba = me_axes_state[ell_indices]
                    else:
                        ell_idx = random.choice(sk3, logvol, p=ell_probs)
                        ba = me_axes_state[ell_idx]
                else:
                    ba = _bound_axes

                x_cand, logL_cand, n_acc, n_tot = sampler_obj.sample(
                    wk, x_starts, worst_logL, loglikelihood_for_jit,
                    ba, scale, rwalk_K,
                    prior_bounds=config.prior_bounds if has_prior_bounds else None,
                    walk_schedule=None,
                )
                valid = logL_cand > worst_logL
                return (valid, rkey, x_cand, logL_cand, racc + n_acc, rtot + n_tot)

            _init_retry = (
                jnp.array(False),                          # valid
                retry_key,                                 # key
                jnp.zeros(ndim, dtype=live_x.dtype),      # x_new placeholder
                jnp.array(-jnp.inf, dtype=live_logL.dtype), # logL_new placeholder
                jnp.array(0, dtype=jnp.int32),             # acc
                jnp.array(0, dtype=jnp.int32),             # tot
            )
            _, _, x_new, logL_new, iter_accepted, iter_total = lax.while_loop(
                _retry_cond, _retry_body, _init_retry)

            # 5. Accumulate accept/total history and adapt scale
            hist_accept = state[11] + iter_accepted
            hist_total = state[12] + iter_total
            facc = hist_accept / jnp.maximum(hist_total, 1)
            new_scale = sampler_obj.tune(scale, facc, ndim, iteration)

            # 6. Update live points
            live_x_new = live_x.at[worst_idx].set(x_new)
            live_logL_new = live_logL.at[worst_idx].set(logL_new)

            # 7. Update evidence
            logX_old = -iteration / nlive
            logX_new_val = -(iteration + 1) / nlive
            log_dX = logsubexp(logX_old, logX_new_val)
            log_dZ = worst_logL + log_dX
            logZ_new = jnp.logaddexp(state[6], log_dZ)

            # 8. Calculate delta_logZ
            max_live_logL = jnp.max(live_logL_new)
            delta_logZ_new = jax.scipy.special.logsumexp(
                jnp.array([jnp.zeros((), dtype=buf_dtype), max_live_logL + logX_new_val - logZ_new])
            )

            # 9. Store samples in buffers
            worst_x_buf_new = state[2].at[iteration].set(live_x[worst_idx])
            worst_logL_buf_new = state[3].at[iteration].set(worst_logL)
            dlz_buf_new = state[4].at[iteration].set(delta_logZ_new)
            scale_buf_new = state[5].at[iteration].set(new_scale)

            total_calls = state[17] + iter_total

            return (
                live_x_new, live_logL_new,
                worst_x_buf_new, worst_logL_buf_new,
                dlz_buf_new, scale_buf_new,
                logZ_new, delta_logZ_new,
                iteration + 1, key, new_scale,
                hist_accept,
                hist_total,
                state[13],
                me_axes_state, state[15],
                state[16],
                total_calls,
            )

        # === Queue mode body_fn (Dynesty-style) ===
        # Each call tests ONE pre-computed candidate from the queue.
        # Iteration only advances when a valid candidate is found.
        # Scale/bound update only when queue drains.
        if config.queue_size > 1:
            _queue_size = config.queue_size
            _ncdim_q = config.ncdim if config.ncdim else ndim

            # cond_fn with safety limit on total calls
            _max_calls_limit = max_iterations * rwalk_K * 20

            def cond_fn_q(state):
                dlz = state[7]
                it = state[8]
                calls = state[17]
                return (dlz >= delta_logZ_threshold) & (it < max_iterations) & (calls < _max_calls_limit)

            def body_fn_q(state):
                live_x = state[0]
                live_logL = state[1]
                iteration = state[8]
                key = state[9]
                scale = state[10]
                me_axes_state = state[14]
                me_logvol_ells = state[15]
                queue_x_arr = state[18]
                queue_logL_arr = state[19]
                queue_nacc_arr = state[20]
                queue_ntot_arr = state[21]
                queue_head = state[22]

                # 1. Read current candidate from queue
                cand_x = queue_x_arr[queue_head]
                cand_logL = queue_logL_arr[queue_head]
                cand_nacc = queue_nacc_arr[queue_head]
                cand_ntot = queue_ntot_arr[queue_head]

                # 2. Find worst live point (= loglstar)
                worst_idx = jnp.argmin(live_logL)
                worst_logL = live_logL[worst_idx]

                # 3. Test candidate against loglstar
                valid = cand_logL > worst_logL

                # 4. Accumulate stats from this candidate (always)
                hist_accept = state[11] + cand_nacc
                hist_total = state[12] + cand_ntot

                # 5. Compute post-acceptance state (branchless)
                x_accept = jnp.where(valid, cand_x, live_x[worst_idx])
                logL_accept = jnp.where(valid, cand_logL, worst_logL)
                live_x_new = live_x.at[worst_idx].set(x_accept)
                live_logL_new = live_logL.at[worst_idx].set(logL_accept)

                # 6. Queue management
                new_head = queue_head + 1
                queue_drained = new_head >= _queue_size

                # 7. Queue drain: update scale + refill (lax.cond avoids wasted GPU compute)
                def _do_drain(drain_args):
                    sc, ky, lx, ll, it_d, ha, ht = drain_args
                    facc = ha.astype(jnp.float32) / jnp.maximum(ht, 1).astype(jnp.float32)
                    new_sc = sampler_obj.tune(sc, facc, ndim, it_d).astype(sc.dtype)
                    ky, rkey = random.split(ky)
                    loglstar_r = jnp.min(ll)
                    qx, ql, qna, qnt = vmap_queue_refill(
                        rkey, lx, ll, loglstar_r, loglikelihood_for_jit,
                        me_axes_state, me_logvol_ells, state[13], new_sc,
                        rwalk_K, ndim, _ncdim_q, _queue_size, use_multi_ellipsoid,
                        config.prior_bounds if has_prior_bounds else None)
                    return new_sc, qx, ql, qna, qnt, ky

                def _skip_drain(drain_args):
                    sc, ky, lx, ll, it_d, ha, ht = drain_args
                    return sc, queue_x_arr, queue_logL_arr, queue_nacc_arr, queue_ntot_arr, ky

                drain_args = (scale, key, live_x_new, live_logL_new, iteration, hist_accept, hist_total)
                d_sc, d_qx, d_ql, d_qna, d_qnt, d_key = lax.cond(
                    queue_drained, _do_drain, _skip_drain, drain_args)

                new_hist_accept = jnp.where(queue_drained, jnp.array(0, dtype=jnp.int32), hist_accept)
                new_hist_total = jnp.where(queue_drained, jnp.array(0, dtype=jnp.int32), hist_total)
                final_head = jnp.where(queue_drained, jnp.array(0, dtype=jnp.int32), new_head)

                # 8. Evidence update (only when valid)
                logX_old = -iteration / nlive
                logX_new_val = -(iteration + 1) / nlive
                log_dX = logsubexp(logX_old, logX_new_val)
                log_dZ = worst_logL + log_dX
                logZ_new = jnp.where(valid, jnp.logaddexp(state[6], log_dZ), state[6])

                max_live_logL = jnp.max(live_logL_new)
                dlz_new_val = jax.scipy.special.logsumexp(
                    jnp.array([jnp.zeros((), dtype=buf_dtype), max_live_logL + logX_new_val - logZ_new]))
                delta_logZ_new = jnp.where(valid, dlz_new_val, state[7])

                # 9. Buffer writes (only when valid)
                wx_buf_new = jnp.where(valid,
                    state[2].at[iteration].set(live_x[worst_idx]), state[2])
                wl_buf_new = jnp.where(valid,
                    state[3].at[iteration].set(worst_logL), state[3])
                dlz_buf_new = jnp.where(valid,
                    state[4].at[iteration].set(dlz_new_val), state[4])
                sc_buf_new = jnp.where(valid,
                    state[5].at[iteration].set(d_sc), state[5])

                # 10. Iteration advance (only when valid)
                new_iteration = jnp.where(valid, iteration + 1, iteration)

                total_calls = state[17] + cand_ntot

                return (
                    live_x_new, live_logL_new,
                    wx_buf_new, wl_buf_new,
                    dlz_buf_new, sc_buf_new,
                    logZ_new, delta_logZ_new,
                    new_iteration, d_key, d_sc,
                    new_hist_accept, new_hist_total,
                    state[13], me_axes_state, me_logvol_ells,
                    state[16], total_calls,
                    d_qx, d_ql, d_qna, d_qnt,
                    final_head,
                )

            # Override with queue versions
            cond_fn = cond_fn_q
            body_fn = body_fn_q

        # Execute Phase 2 loop
        if use_chunked_loop:
            from .multi_ellipsoid import fit_multi_ellipsoid

            current_state = init_state
            total_done = phase1_iters

            if pbar is not None:
                def _progress_cb(it, dlz, lz, loglstar, eff):
                    if int(it) % 100 == 0:
                        pbar.n = int(it)
                        loglstar_val = float(loglstar)
                        if loglstar_val <= -1e6:
                            loglstar_str = '-inf'
                        else:
                            loglstar_str = f'{loglstar_val:.1f}'
                        pbar.set_postfix_str(
                            f'logZ: {float(lz):.2f} | dlogZ: {float(dlz):.3f} ({delta_logZ_threshold}) | '
                            f'logl*: {loglstar_str} | eff(%): {float(eff):.1f}')
                        pbar.refresh()

            _body_fn = body_fn

            if config.print_progress and TQDM_AVAILABLE:
                def body_fn_progress(state):
                    new_state = _body_fn(state)
                    try:
                        loglstar = jnp.min(new_state[1])
                        total_calls_val = new_state[17]
                        eff = (new_state[8] + nlive) * 100.0 / jnp.maximum(total_calls_val, 1)
                        io_callback(_progress_cb, None, new_state[8], new_state[7],
                                     new_state[6], loglstar, eff)
                    except Exception:
                        pass
                    return new_state
                body_fn_for_chunk = body_fn_progress
            else:
                body_fn_for_chunk = body_fn

            def chunk_cond(state):
                converged = state[7] < delta_logZ_threshold
                too_many = state[8] >= max_iterations
                chunk_done = (state[17] - state[16]) >= bound_update_interval
                return (~converged) & (~too_many) & (~chunk_done)

            compiled_chunk = jax.jit(lambda s: lax.while_loop(chunk_cond, body_fn_for_chunk, s))

            while total_done < max_iterations:
                # Build chunk start state: set pos 16 to current total_calls
                calls_at_start = int(current_state[17])
                if config.queue_size > 1:
                    state_with_chunk = (
                        *current_state[:16], jnp.array(calls_at_start, dtype=jnp.int32),
                        current_state[17],
                        *current_state[18:],  # queue arrays + head
                    )
                else:
                    state_with_chunk = (*current_state[:16], jnp.array(calls_at_start, dtype=jnp.int32), current_state[17])
                current_state = compiled_chunk(state_with_chunk)

                iteration = int(current_state[8])
                delta_logZ_val = float(current_state[7])
                total_done = iteration

                if delta_logZ_val < delta_logZ_threshold or total_done >= max_iterations:
                    break

                # Refit multi-ellipsoid in unit-cube space (matching Dynesty)
                live_u_cur = current_state[0]
                me_state = fit_multi_ellipsoid(live_u_cur, max_ellipsoids=max_ellipsoids, enlarge=1.25)

                if config.verbose:
                    calls_now = int(current_state[17])
                    msg = (f"  Iter {total_done} ({calls_now} calls): refit multi-ellipsoid -> "
                           f"{me_state.n_active} ellipsoid(s), "
                           f"logZ={float(current_state[6]):.2f}, dlogZ={delta_logZ_val:.3f}")
                    if pbar is not None:
                        pbar.write(msg)
                    else:
                        print(msg)

                # Repack with updated multi-ellipsoid state
                # Reset scale history so scale adapts to new axes quickly
                # (Dynesty resets history after every walk in single-worker mode)
                if config.queue_size > 1:
                    # Queue mode: force queue drain by setting head = queue_size
                    current_state = (
                        current_state[0], current_state[1],
                        current_state[2], current_state[3],
                        current_state[4], current_state[5],
                        current_state[6], current_state[7],
                        current_state[8], current_state[9],
                        current_state[10],
                        jnp.array(0, dtype=jnp.int32),  # 11: hist_accept reset
                        jnp.array(0, dtype=jnp.int32),  # 12: hist_total reset
                        current_state[13],
                        me_state.axes, me_state.logvol_ells,
                        current_state[17], current_state[17],  # 16: reset calls_at_update to current total_calls
                        current_state[18], current_state[19],
                        current_state[20], current_state[21],
                        jnp.array(config.queue_size, dtype=jnp.int32),  # 22: force queue drain
                    )
                else:
                    current_state = (
                        current_state[0], current_state[1],
                        current_state[2], current_state[3],
                        current_state[4], current_state[5],
                        current_state[6], current_state[7],
                        current_state[8], current_state[9],
                        current_state[10],
                        jnp.array(0, dtype=jnp.int32),  # 11: hist_accept reset
                        jnp.array(0, dtype=jnp.int32),  # 12: hist_total reset
                        current_state[13],
                        me_state.axes, me_state.logvol_ells,
                        current_state[17],  # 16: reset calls_at_update to current total_calls
                        current_state[17],  # 17: total_calls preserved
                    )

            if pbar is not None:
                pbar.close()

            final_state = current_state

        elif pbar is not None:
            def progress_cb(it, dlz, lz, loglstar, eff):
                if int(it) % 100 == 0:
                    pbar.n = int(it)
                    loglstar_val = float(loglstar)
                    if loglstar_val <= -1e6:
                        loglstar_str = '-inf'
                    else:
                        loglstar_str = f'{loglstar_val:.1f}'
                    pbar.set_postfix_str(
                        f'logZ: {float(lz):.2f} | dlogZ: {float(dlz):.3f} ({delta_logZ_threshold}) | '
                        f'logl*: {loglstar_str} | eff(%): {float(eff):.1f}')
                    pbar.refresh()

            _orig_body = body_fn

            def body_fn_prog(state):
                new_state = _orig_body(state)
                try:
                    loglstar = jnp.min(new_state[1])
                    total_calls_val = new_state[17]
                    eff = (new_state[8] + nlive) * 100.0 / jnp.maximum(total_calls_val, 1)
                    io_callback(progress_cb, None, new_state[8], new_state[7],
                                 new_state[6], loglstar, eff)
                except Exception:
                    pass
                return new_state

            final_state = lax.while_loop(cond_fn, body_fn_prog, init_state)
            if pbar is not None:
                pbar.close()
        else:
            final_state = lax.while_loop(cond_fn, body_fn, init_state)

        # Unpack Phase 2 final state
        live_x = final_state[0]
        live_logL = final_state[1]
        worst_x_buffer = final_state[2]
        worst_logL_buffer = final_state[3]
        delta_logZ_buffer = final_state[4]
        scale_buffer = final_state[5]
        delta_logZ = final_state[7]
        iteration_final = int(final_state[8])
        final_hist_accept = float(final_state[11])
        final_hist_total = float(final_state[12]) + phase1_total_calls

    # === Final results processing ===
    live_x_final = live_x
    live_logL_final = live_logL
    delta_logZ_final = delta_logZ

    runtime = time.time() - start_time
    actual_iterations = iteration_final

    # Transform samples to physical space
    if prior_transform_fn is not None:
        samples = jnp.vectorize(prior_transform_fn, signature='(n)->(n)')(
            worst_x_buffer[:actual_iterations]
        )
    else:
        samples = worst_x_buffer[:actual_iterations]

    logL_samples = worst_logL_buffer[:actual_iterations]
    delta_logZ_trajectory = delta_logZ_buffer[:actual_iterations]
    scale_trajectory = scale_buffer[:actual_iterations]
    best_final_logL = jnp.max(live_logL_final)

    # Recalculate final logZ and H
    iterations_arr = jnp.arange(actual_iterations)
    logX_i = -iterations_arr / nlive
    logX_i_plus_1 = -(iterations_arr + 1) / nlive
    log_dX_dead = logsubexp(logX_i, logX_i_plus_1)
    log_dZ_dead = logL_samples + log_dX_dead

    logX_final = -actual_iterations / nlive
    log_dX_final = logX_final - jnp.log(nlive)
    log_dZ_final = best_final_logL + log_dX_final

    log_dZ_all = jnp.concatenate([log_dZ_dead, jnp.array([log_dZ_final])])
    logZ = float(jax.scipy.special.logsumexp(log_dZ_all))

    log_weights_all = jnp.concatenate([
        log_dZ_dead - logZ,
        jnp.array([log_dZ_final - logZ])
    ])
    weights_all = jnp.exp(log_weights_all)
    logL_all = jnp.concatenate([logL_samples, jnp.array([best_final_logL])])
    # Avoid 0 * (-inf) = NaN: replace -inf logL with 0 before weighting
    safe_logL = jnp.where(jnp.isfinite(logL_all), logL_all, jnp.array(0.0))
    H = float(jnp.sum(weights_all * safe_logL)) - logZ

    logZ_error = float(jnp.sqrt(jnp.abs(H) / nlive))

    total_acceptance = final_hist_accept
    total_prop = final_hist_total
    acceptance_rate = total_acceptance / total_prop if total_prop > 0 else 0.0

    if config.verbose:
        print()
        print(f"Sampling completed in {runtime:.2f}s")
        print(f"Total iterations: {actual_iterations} "
              f"(uniform: {phase1_iters}, rwalk: {actual_iterations - phase1_iters})")
        print(f"Final logZ: {logZ:.4f} +/- {logZ_error:.4f}")
        print(f"Information H: {H:.4f}")
        print(f"Final delta_logZ: {float(delta_logZ_final):.6f}")
        print(f"Converged: {float(delta_logZ_final) < delta_logZ_threshold}")
        print(f"Speed: {actual_iterations / runtime:.1f} iterations/sec")
        print(f"Acceptance rate: {acceptance_rate:.1%}")

    return WhileLoopNSResult(
        logZ=logZ,
        logZ_error=logZ_error,
        H=H,
        delta_logZ=float(delta_logZ_final),
        n_iterations=actual_iterations,
        runtime=runtime,
        samples=samples,
        logL_samples=logL_samples,
        delta_logZ_trajectory=delta_logZ_trajectory,
        scale_trajectory=scale_trajectory,
        acceptance_rate=acceptance_rate,
        live_x=live_x_final,
        live_logL=live_logL_final,
    )
