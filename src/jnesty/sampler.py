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

from .internal_samplers import RWalkSampler
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

    Uses Bound and InternalSampler objects for pluggable strategies.
    The core loop uses lax.while_loop for exact convergence stopping.

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

    # Wrap likelihood with prior_transform if needed
    if prior_transform_fn is not None:
        def loglikelihood_wrapped(x):
            return loglikelihood_fn(prior_transform_fn(x))
        loglikelihood_for_jit = loglikelihood_wrapped
    else:
        loglikelihood_for_jit = loglikelihood_fn

    if config.verbose:
        print("=" * 70)
        print("JNesty Nested Sampling")
        print("=" * 70)
        print(f"Live points: {nlive}")
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence threshold: delta_logZ < {delta_logZ_threshold}")
        print(f"Bound: {config.bound}")
        print(f"Walk steps: {rwalk_K}")
        if config.bound == 'multi':
            print(f"Max ellipsoids: {config.max_ellipsoids}")
            print(f"Bound update interval: {config.bound_update_interval}")
        print("=" * 70)
        print()

    start_time = time.time()

    # Initialize live points from prior
    keys = random.split(key, nlive + 1)
    live_x = jnp.stack([prior_sample_fn(k) for k in keys[:-1]])
    live_logL = jnp.vectorize(loglikelihood_for_jit, signature='(n)->()')(live_x)

    # Create bound and sampler via factories
    from .bounding import get_bound
    from .internal_samplers import get_sampler

    bound_obj = get_bound(config.bound, ndim,
                          max_ellipsoids=config.max_ellipsoids,
                          scale=config.rwalk_step_scale)
    sampler_obj = get_sampler('rwalk', ndim,
                              target_acceptance=config.target_acceptance,
                              batch_size=config.batch_size)

    # Fit initial bound
    if config.bound != 'none':
        if prior_transform_fn is not None:
            live_physical = jnp.stack([prior_transform_fn(x) for x in live_x])
            bound_obj.fit(live_physical)
        else:
            bound_obj.fit(live_x)

    # Compile-time flags
    use_multi_ellipsoid = config.bound == 'multi'
    bound_update_interval = config.bound_update_interval
    use_chunked_loop = use_multi_ellipsoid and bound_update_interval > 0
    max_ellipsoids = config.max_ellipsoids

    # Pre-allocate buffers
    worst_x_buffer = jnp.zeros((max_iterations, ndim))
    worst_logL_buffer = jnp.full(max_iterations, -jnp.inf)
    delta_logZ_buffer = jnp.full(max_iterations, jnp.inf)
    scale_buffer = jnp.full(max_iterations, jnp.inf)

    # Initial state
    logZ = -jnp.inf
    current_scale = config.rwalk_step_scale
    acceptance_count = 0
    total_proposals = 0

    max_live_logL = jnp.max(live_logL)
    delta_logZ = jax.scipy.special.logsumexp(
        jnp.array([0.0, max_live_logL - logZ])
    )

    # Get initial axes and schedule
    bound_axes = bound_obj.get_axes()
    if use_multi_ellipsoid and hasattr(bound_obj, 'get_walk_schedule'):
        walk_schedule = bound_obj.get_walk_schedule(rwalk_K)
    else:
        walk_schedule = jnp.zeros(rwalk_K, dtype=jnp.int32)

    # For multi-ellipsoid: store state arrays for chunked updates
    if use_multi_ellipsoid and hasattr(bound_obj, 'state') and bound_obj.state is not None:
        me_axes = bound_obj.state.axes
        me_logvol_ells = bound_obj.state.logvol_ells
    else:
        me_axes = jnp.zeros((max_ellipsoids, ndim, ndim))
        me_logvol_ells = jnp.full(max_ellipsoids, -jnp.inf)

    has_prior_bounds = config.prior_bounds is not None

    # Pack state as tuple for lax.while_loop
    init_state = (
        live_x,                  # 0
        live_logL,               # 1
        worst_x_buffer,          # 2
        worst_logL_buffer,       # 3
        delta_logZ_buffer,       # 4
        scale_buffer,            # 5
        logZ,                    # 6
        delta_logZ,              # 7
        jnp.array(0),            # 8  iteration
        keys[-1],                # 9  key
        current_scale,           # 10 scale
        acceptance_count,        # 11 acc_count
        total_proposals,         # 12 tot_proposals
        bound_axes,              # 13 bound_axes
        walk_schedule,           # 14 walk_schedule
        me_axes,                 # 15 me_axes
        me_logvol_ells,          # 16 me_logvol_ells
        jnp.array(0),            # 17 chunk_start
    )

    def cond_fn(state):
        dlz = state[7]
        it = state[8]
        return (dlz >= delta_logZ_threshold) & (it < max_iterations)

    def body_fn(state):
        live_x = state[0]
        live_logL = state[1]
        logZ = state[6]
        iteration = state[8]
        key = state[9]
        scale = state[10]
        bound_axes = state[13]
        schedule = state[14]
        me_axes_state = state[15]

        # 1. Find worst live point
        worst_idx = jnp.argmin(live_logL)
        worst_logL = live_logL[worst_idx]

        # 2. Pick random starting point
        key, subkey = random.split(key)
        idx = random.randint(subkey, (), 0, live_x.shape[0])
        x_current = live_x[idx]

        # 3. Run random walk via sampler
        key, walk_key = random.split(key)

        if use_multi_ellipsoid:
            ws = schedule
            ba = me_axes_state
        else:
            ws = None
            ba = bound_axes

        x_new, logL_new, iter_acceptance = sampler_obj.sample(
            walk_key, x_current, worst_logL, loglikelihood_for_jit,
            ba, scale, rwalk_K,
            prior_bounds=config.prior_bounds if has_prior_bounds else None,
            walk_schedule=ws,
        )

        # 4. Adaptive scale tuning
        if config.batch_size > 1:
            steps_per_walk = max(1, rwalk_K // config.batch_size)
        else:
            steps_per_walk = rwalk_K
        current_acceptance = iter_acceptance / steps_per_walk
        new_scale = sampler_obj.tune(scale, current_acceptance, ndim, iteration)

        # 5. Update live points
        live_x_new = live_x.at[worst_idx].set(x_new)
        live_logL_new = live_logL.at[worst_idx].set(logL_new)

        # 6. Update evidence
        logX_old = -iteration / nlive
        logX_new_val = -(iteration + 1) / nlive
        log_dX = logsubexp(logX_old, logX_new_val)
        log_dZ = worst_logL + log_dX
        logZ_new = jnp.logaddexp(logZ, log_dZ)

        # 7. Calculate delta_logZ
        max_live_logL = jnp.max(live_logL_new)
        delta_logZ_new = jax.scipy.special.logsumexp(
            jnp.array([0.0, max_live_logL + logX_new_val - logZ_new])
        )

        # 8. Store samples in buffers
        worst_x_buf_new = state[2].at[iteration].set(live_x[worst_idx])
        worst_logL_buf_new = state[3].at[iteration].set(worst_logL)
        dlz_buf_new = state[4].at[iteration].set(delta_logZ_new)
        scale_buf_new = state[5].at[iteration].set(new_scale)

        return (
            live_x_new, live_logL_new,
            worst_x_buf_new, worst_logL_buf_new,
            dlz_buf_new, scale_buf_new,
            logZ_new, delta_logZ_new,
            iteration + 1, key, new_scale,
            state[11] + iter_acceptance, state[12] + max(1, rwalk_K // config.batch_size),
            bound_axes, schedule,
            me_axes_state, state[16],
            state[17],
        )

    # Execute loop
    if use_chunked_loop:
        from .multi_ellipsoid import fit_multi_ellipsoid

        current_state = init_state
        total_done = 0

        if config.print_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=max_iterations, desc="Nested Sampling")

        def _progress_cb(it, dlz, lz):
            if int(it) % 100 == 0:
                pbar.n = int(it)
                pbar.set_postfix_str(f'logZ: {float(lz):.2f} | dlogZ: {float(dlz):.3f}')
                pbar.refresh()

        _body_fn = body_fn

        if config.print_progress and TQDM_AVAILABLE:
            def body_fn_progress(state):
                new_state = _body_fn(state)
                try:
                    io_callback(_progress_cb, None, new_state[8], new_state[7], new_state[6])
                except Exception:
                    pass
                return new_state
            body_fn_for_chunk = body_fn_progress
        else:
            body_fn_for_chunk = body_fn

        def chunk_cond(state):
            converged = state[7] < delta_logZ_threshold
            too_many = state[8] >= max_iterations
            chunk_done = (state[8] - state[17]) >= bound_update_interval
            return (~converged) & (~too_many) & (~chunk_done)

        compiled_chunk = jax.jit(lambda s: lax.while_loop(chunk_cond, body_fn_for_chunk, s))

        while total_done < max_iterations:
            state_with_chunk = (*current_state[:-1], jnp.array(total_done))
            current_state = compiled_chunk(state_with_chunk)

            iteration = int(current_state[8])
            delta_logZ_val = float(current_state[7])
            total_done = iteration

            if delta_logZ_val < delta_logZ_threshold or total_done >= max_iterations:
                break

            # Refit multi-ellipsoid
            live_x_cur = current_state[0]
            me_state = fit_multi_ellipsoid(live_x_cur, max_ellipsoids=max_ellipsoids)

            if config.verbose and total_done % (bound_update_interval * 5) < bound_update_interval:
                print(f"  Iter {total_done}: refit multi-ellipsoid -> {me_state.n_active} ellipsoid(s), "
                      f"logZ={float(current_state[6]):.2f}, dlogZ={delta_logZ_val:.3f}")

            # Recompute walk schedule
            _log_probs = me_state.logvol_ells - jax.scipy.special.logsumexp(me_state.logvol_ells)
            _probs = jnp.exp(_log_probs)
            def _sched_step(_acc, _):
                _acc = _acc + _probs
                _best = jnp.argmax(_acc)
                _acc = _acc.at[_best].add(-1.0)
                return _acc, _best
            _, new_schedule = lax.scan(
                _sched_step, jnp.zeros(max_ellipsoids), None, length=rwalk_K
            )

            # Repack with updated multi-ellipsoid state
            current_state = (
                current_state[0], current_state[1],
                current_state[2], current_state[3],
                current_state[4], current_state[5],
                current_state[6], current_state[7],
                current_state[8], current_state[9],
                current_state[10], current_state[11],
                current_state[12], current_state[13],
                new_schedule,
                me_state.axes, me_state.logvol_ells,
                current_state[17],
            )

        if config.print_progress and TQDM_AVAILABLE:
            pbar.close()

        final_state = current_state

    elif config.print_progress and TQDM_AVAILABLE:
        pbar = tqdm(total=max_iterations, desc="Nested Sampling")

        def progress_cb(it, dlz, lz):
            if int(it) % 100 == 0:
                pbar.n = int(it)
                pbar.set_postfix_str(f'logZ: {float(lz):.2f} | dlogZ: {float(dlz):.3f}')
                pbar.refresh()

        _orig_body = body_fn

        def body_fn_prog(state):
            new_state = _orig_body(state)
            try:
                io_callback(progress_cb, None, new_state[8], new_state[7], new_state[6])
            except Exception:
                pass
            return new_state

        final_state = lax.while_loop(cond_fn, body_fn_prog, init_state)
        pbar.close()
    else:
        final_state = lax.while_loop(cond_fn, body_fn, init_state)

    # Unpack final state
    live_x_final = final_state[0]
    live_logL_final = final_state[1]
    worst_x_buffer = final_state[2]
    worst_logL_buffer = final_state[3]
    delta_logZ_buffer = final_state[4]
    scale_buffer = final_state[5]
    delta_logZ_final = final_state[7]
    iteration_final = final_state[8]
    final_acceptance_count = final_state[11]
    final_total_proposals = final_state[12]

    if config.verbose:
        print()

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
    H = float(jnp.sum(weights_all * logL_all)) - logZ

    logZ_error = float(jnp.sqrt(jnp.abs(H) / nlive))

    total_acceptance = float(final_acceptance_count)
    total_prop = float(final_total_proposals)
    acceptance_rate = total_acceptance / total_prop if total_prop > 0 else 0.0

    if config.verbose:
        print(f"Nested sampling completed in {runtime:.2f}s")
        print(f"Total iterations: {actual_iterations}")
        print(f"Final logZ: {logZ:.4f} ± {logZ_error:.4f}")
        print(f"Information H: {H:.4f}")
        print(f"Final delta_logZ: {delta_logZ_final:.6f}")
        print(f"Converged: {delta_logZ_final < delta_logZ_threshold}")
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
