"""
True adaptive nested sampling using while_loop with progress tracking.

This provides iteration-granular convergence checking by using lax.while_loop
with io_callback + tqdm for real-time progress updates.

Key benefit: 100% efficiency (no wasted iterations) + real-time progress!
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Optional, NamedTuple
import time
from .bounding import fit_ellipsoid, sample_ellipsoid
from .multi_ellipsoid import fit_multi_ellipsoid

# Import for progress tracking
try:
    from jax.experimental import io_callback
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


def randsphere(key, n):
    """
    Draw a point uniformly within an n-dimensional unit ball.

    Matches Dynesty's randsphere() implementation exactly.
    Formula: xhat = z * (U^(1/n) / ||z||)
    where z ~ N(0, I) and U ~ Uniform(0,1)

    Args:
        key: JAX random key
        n: Dimensionality

    Returns:
        Point uniformly distributed in unit n-ball (||x|| <= 1)
    """
    key1, key2 = random.split(key, 2)

    # z ~ N(0, I) - Gaussian for direction
    z = random.normal(key1, shape=(n,))

    # Avoid division by zero (unlikely but possible)
    z_norm = jnp.linalg.norm(z)
    z_norm_safe = jnp.where(z_norm > 0, z_norm, 1.0)

    # U^(1/n) for uniform radius in ball
    # Volume of n-ball scales as r^n, so P(R < r) = r^n
    # Therefore r ~ U^(1/n) for uniform distribution
    u = random.uniform(key2)
    radius = u ** (1.0 / n)

    # Combine: z * (radius / ||z||)
    # z/||z|| gives random direction on unit sphere
    # Multiplying by radius gives uniform within ball
    xhat = z * (radius / z_norm_safe)

    return xhat


class WhileLoopNSConfig(NamedTuple):
    """Configuration for while_loop nested sampling run."""
    nlive: int = 500
    max_iterations: int = 10000
    delta_logZ_threshold: float = 0.01
    rwalk_K: int = 25  # Number of walk steps per iteration (Dynesty default)
    rwalk_L: int = 16  # (Not used in while_loop mode)
    rwalk_step_scale: float = 1.0  # Initial proposal scale
    target_acceptance: float = 0.5  # Target acceptance rate for adaptive tuning
    scale_adapt_interval: int = 1  # Adjust scale every walk (matches Dynesty)
    use_ellipsoid: bool = False  # Use ellipsoid-bounded proposals (default: False for multi-modal safety)
    ellipsoid_update_interval: int = 500  # Update ellipsoid every N iterations (0 = once at start)
    prior_bounds: Optional[jnp.ndarray] = None  # Prior bounds (2, ndim) array. Reject proposals outside these bounds.
    verbose: bool = True
    print_progress: bool = True  # Show tqdm progress bar (default: True)
    bound: str = 'none'  # 'none', 'single', 'multi'
    bound_update_interval: int = 0  # 0 = once at start, >0 = every N iterations
    max_ellipsoids: int = 20  # Max ellipsoids for multi-ellipsoid decomposition


class WhileLoopNSResult(NamedTuple):
    """Results from while_loop nested sampling run."""
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


def run_nested_sampling_while_loop(
    loglikelihood_fn,
    prior_sample_fn,
    ndim: int,
    config: WhileLoopNSConfig,
    key: Optional = None,
    prior_transform_fn: Optional[callable] = None
) -> WhileLoopNSResult:
    """
    Run nested sampling with iteration-granular adaptive termination.

    Uses lax.while_loop to enable stopping exactly when converged,
    not at batch boundaries.

    Multi-ellipsoid state (axes, logvol_ells) is passed as traced arrays
    through the while_loop state, avoiding re-JIT on bound updates.

    Args:
        loglikelihood_fn: Log-likelihood function (expects physical space)
        prior_sample_fn: Prior sampling function (returns unit cube or physical samples)
        ndim: Problem dimensionality
        config: WhileLoopNSConfig with parameters
        key: JAX random key
        prior_transform_fn: Transform from unit cube to physical space
                             If None, assume prior_sample_fn returns physical samples

    Returns:
        WhileLoopNSResult with all samples and convergence info
    """

    if key is None:
        key = random.PRNGKey(42)

    nlive = config.nlive
    max_iterations = config.max_iterations
    delta_logZ_threshold = config.delta_logZ_threshold
    rwalk_K = config.rwalk_K
    rwalk_L = config.rwalk_L
    rwalk_step_scale = config.rwalk_step_scale

    # Wrap likelihood function with prior_transform if needed
    # This avoids Python if statements inside JIT-compiled code
    if prior_transform_fn is not None:
        def loglikelihood_with_transform(x):
            x_physical = prior_transform_fn(x)
            return loglikelihood_fn(x_physical)
        loglikelihood_for_jit = loglikelihood_with_transform

        # Extract scale from transform for printing
        test_u = jnp.zeros(ndim)
        test_x = prior_transform_fn(test_u)
        test_u2 = jnp.ones(ndim)
        test_x2 = prior_transform_fn(test_u2)
        scale = (test_x2[0] - test_x[0]) / (test_u2[0] - test_u[0])
        prior_log_density = -ndim * jnp.log(scale)
    else:
        loglikelihood_for_jit = loglikelihood_fn
        scale = None
        prior_log_density = 0.0

    if config.verbose:
        print("="*70)
        print("While-Loop Nested Sampling with Iteration-Granular Termination")
        print("="*70)
        print(f"Live points: {nlive}")
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence threshold: {delta_logZ_threshold}")
        if config.bound != 'none':
            print(f"Bound: {config.bound}")
            print(f"Bound update interval: {config.bound_update_interval}")
            if config.bound == 'multi':
                print(f"Max ellipsoids: {config.max_ellipsoids}")
        if prior_transform_fn is not None:
            print(f"Using prior_transform: YES (scale={scale:.1f})")
            print(f"Prior log density: {prior_log_density:.4f}")
        print("="*70)
        print()

    # Initialize progress
    if config.verbose and config.print_progress and not TQDM_AVAILABLE:
        import sys
        print("Running nested sampling...", end='', flush=True)

    start_time = time.time()

    # Initialize live points from prior
    keys = random.split(key, nlive + 1)
    live_x = jnp.stack([prior_sample_fn(k) for k in keys[:-1]])

    # Transform to physical space for likelihood evaluation
    if prior_transform_fn is not None:
        live_x_physical = jnp.stack([prior_transform_fn(x) for x in live_x])
        live_logL = jnp.vectorize(loglikelihood_for_jit, signature='(n)->()')(live_x)
        ellipsoid_center, ellipsoid_axes, ellipsoid_radii = fit_ellipsoid(live_x_physical)
    else:
        live_logL = jnp.vectorize(loglikelihood_for_jit, signature='(n)->()')(live_x)
        ellipsoid_center, ellipsoid_axes, ellipsoid_radii = fit_ellipsoid(live_x)

    ellipsoid_scale = rwalk_step_scale

    # Pre-allocate arrays for samples (fixed buffer)
    worst_x_buffer = jnp.zeros((max_iterations, ndim))
    worst_logL_buffer = jnp.full(max_iterations, -jnp.inf)
    delta_logZ_buffer = jnp.full(max_iterations, jnp.inf)
    scale_buffer = jnp.full(max_iterations, jnp.inf)

    # Initial state
    iteration = 0
    logZ = -jnp.inf
    current_key = keys[-1]
    current_scale = rwalk_step_scale
    acceptance_count = 0
    total_proposals = 0
    ellipsoid_update_counter = 0

    # Compute initial delta_logZ (work in unit cube space)
    logX = 0.0
    max_live_logL = jnp.max(live_logL)
    delta_logZ = jax.scipy.special.logsumexp(
        jnp.array([0.0, max_live_logL + logX - logZ])
    )

    # Compile-time constants
    use_unit_cube = prior_transform_fn is not None
    use_multi_ellipsoid = config.bound == 'multi'
    max_ellipsoids = config.max_ellipsoids

    # Determine bound update interval
    # 0 = fit once at start (no periodic updates)
    # >0 = periodic updates every N iterations
    bound_update_interval = config.bound_update_interval
    use_chunked_loop = use_multi_ellipsoid and bound_update_interval > 0

    # Initialize multi-ellipsoid arrays
    if use_multi_ellipsoid:
        me_state = fit_multi_ellipsoid(live_x, max_ellipsoids=max_ellipsoids)
        me_axes_init = me_state.axes
        me_logvol_ells_init = me_state.logvol_ells
        if config.verbose:
            print(f"Initial multi-ellipsoid: {me_state.n_active} ellipsoid(s)")
    else:
        # Placeholder arrays — never used since use_multi_ellipsoid is False
        me_axes_init = jnp.zeros((max_ellipsoids, ndim, ndim))
        me_logvol_ells_init = jnp.full(max_ellipsoids, -jnp.inf)

    # Pack state for while_loop
    # Multi-ellipsoid arrays are traced through the loop (not closure-captured)
    # Last element is chunk_start (used only in chunked mode)
    init_state = (
        live_x,
        live_logL,
        worst_x_buffer,
        worst_logL_buffer,
        delta_logZ_buffer,
        scale_buffer,
        logZ,
        delta_logZ,
        iteration,
        current_key,
        current_scale,
        acceptance_count,
        total_proposals,
        ellipsoid_center,
        ellipsoid_axes,
        ellipsoid_scale,
        ellipsoid_update_counter,
        me_axes_init,
        me_logvol_ells_init,
        jnp.array(0),  # chunk_start (used in chunked mode only)
    )

    def cond_fn(state):
        """Check whether to continue iterating."""
        _, _, _, _, _, _, _, delta_logZ, iteration, *_ = state
        chunk_start = state[-1]
        return (delta_logZ >= delta_logZ_threshold) & (iteration < max_iterations)

    def body_fn(state):
        """Single NS iteration with multi-step random walk and adaptive scale."""
        (live_x, live_logL,
         worst_x_buffer, worst_logL_buffer, delta_logZ_buffer, scale_buffer,
         logZ, delta_logZ, iteration, key, scale, acceptance_count, total_proposals,
         ellipsoid_center, ellipsoid_axes, ellipsoid_scale, ellipsoid_update_counter,
         me_axes, me_logvol_ells, chunk_start) = state

        # 1. Find worst live point
        worst_idx = jnp.argmin(live_logL)
        worst_logL = live_logL[worst_idx]
        worst_x = live_x[worst_idx]

        # Pre-compute deterministic walk schedule (once per iteration, not per step)
        if use_multi_ellipsoid:
            _log_probs = me_logvol_ells - jax.scipy.special.logsumexp(me_logvol_ells)
            _probs = jnp.exp(_log_probs)
            def _sched_step(_acc, _):
                _acc = _acc + _probs
                _best = jnp.argmax(_acc)
                _acc = _acc.at[_best].add(-1.0)
                return _acc, _best
            _, walk_schedule = lax.scan(
                _sched_step, jnp.zeros(max_ellipsoids), None, length=rwalk_K
            )

        # 2. Multi-step random walk to generate replacement point
        key, subkey = random.split(key)
        idx = random.randint(subkey, (), 0, live_x.shape[0])
        x_current = live_x[idx]

        def walk_step(walk_state, step_idx):
            """Single random walk step with constraint check."""
            x, walk_key, walk_accepted = walk_state

            walk_key, subkey1 = random.split(walk_key, 2)

            # Propose new point
            if use_multi_ellipsoid:
                # Use pre-computed schedule (no per-step random.choice)
                sel_idx = walk_schedule[step_idx]
                selected_axes = me_axes[sel_idx]
                dr = randsphere(subkey1, ndim)
                du = selected_axes @ dr
                x_proposed = x + scale * du
            elif config.use_ellipsoid:
                x_proposed = sample_ellipsoid(
                    subkey1,
                    ellipsoid_center,
                    ellipsoid_axes,
                    ellipsoid_scale
                )
            else:
                dr = randsphere(subkey1, ndim)
                delta = scale * dr
                x_proposed = x + delta

            # Check if within unit cube bounds
            in_unit_cube = jnp.all((x_proposed >= 0.0) & (x_proposed <= 1.0))

            in_bounds = True
            if config.prior_bounds is not None:
                in_bounds = jnp.all(
                    (x_proposed >= config.prior_bounds[0]) &
                    (x_proposed <= config.prior_bounds[1])
                )

            logL_proposed = jnp.where(
                in_unit_cube,
                loglikelihood_for_jit(x_proposed),
                -jnp.inf
            )

            # Accept if in bounds AND satisfies constraint (no Metropolis, matching Dynesty)
            constraint_satisfied = logL_proposed > worst_logL
            accept = in_unit_cube & in_bounds & constraint_satisfied

            x_new = jnp.where(accept, x_proposed, x)
            walk_accepted_new = jnp.where(accept, walk_accepted + 1, walk_accepted)

            return (x_new, walk_key, walk_accepted_new), None

        init_walk_state = (x_current, key, 0)
        final_walk_state, _ = lax.scan(
            walk_step,
            init_walk_state,
            jnp.arange(rwalk_K)
        )
        x_new, final_key, iter_acceptance = final_walk_state

        # Evaluate likelihood of replacement point
        logL_new = loglikelihood_for_jit(x_new)

        # 3. Adaptive scale tuning (Robbins-Munro, matches Dynesty)
        should_adapt = iteration > 0
        current_acceptance = iter_acceptance / rwalk_K
        proposed_scale = scale * jnp.exp(
            (current_acceptance - config.target_acceptance) / ndim / config.target_acceptance
        )
        new_scale = jnp.where(should_adapt, proposed_scale, scale)

        new_acceptance_count = acceptance_count + iter_acceptance
        new_total_proposals = total_proposals + rwalk_K
        new_ellipsoid_scale = new_scale

        # 4. Update live points
        live_x_new = live_x.at[worst_idx].set(x_new)
        live_logL_new = live_logL.at[worst_idx].set(logL_new)

        # 5. Update evidence
        logX_old = -iteration / nlive
        logX_new = -(iteration + 1) / nlive

        log_dX = logX_old + jnp.log1p(-jnp.exp(logX_new - logX_old))
        log_dZ = worst_logL + log_dX

        logZ_new = jnp.logaddexp(logZ, log_dZ)

        # 6. Calculate delta_logZ
        max_live_logL = jnp.max(live_logL_new)
        delta_logZ_new = jax.scipy.special.logsumexp(
            jnp.array([0.0, max_live_logL + logX_new - logZ_new])
        )

        # 7. Store samples in buffers
        worst_x_buffer_new = worst_x_buffer.at[iteration].set(worst_x)
        worst_logL_buffer_new = worst_logL_buffer.at[iteration].set(worst_logL)
        delta_logZ_buffer_new = delta_logZ_buffer.at[iteration].set(delta_logZ_new)
        scale_buffer_new = scale_buffer.at[iteration].set(new_scale)

        iteration_new = iteration + 1

        new_state = (
            live_x_new,
            live_logL_new,
            worst_x_buffer_new,
            worst_logL_buffer_new,
            delta_logZ_buffer_new,
            scale_buffer_new,
            logZ_new,
            delta_logZ_new,
            iteration_new,
            final_key,
            new_scale,
            new_acceptance_count,
            new_total_proposals,
            ellipsoid_center,
            ellipsoid_axes,
            new_ellipsoid_scale,
            ellipsoid_update_counter,
            me_axes,
            me_logvol_ells,
            chunk_start,
        )

        return new_state

    # Chunked execution for multi-ellipsoid with periodic bound updates
    if use_chunked_loop:
        current_state = init_state
        total_done = 0

        if config.print_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=config.max_iterations, desc="Nested Sampling")

        if config.print_progress and TQDM_AVAILABLE:
            def _progress_cb(it, dlz, lz):
                if int(it) % 100 == 0:
                    pbar.n = int(it)
                    pbar.set_postfix_str(f'logZ: {float(lz):.2f} | dlogZ: {float(dlz):.3f}')
                    pbar.refresh()

            _body_fn = body_fn
            def body_fn_for_chunk(state):
                new_state = _body_fn(state)
                _, _, _, _, _, _, logZ_v, delta_logZ_v, iteration_v, *_ = new_state
                try:
                    io_callback(_progress_cb, None, iteration_v, delta_logZ_v, logZ_v)
                except Exception:
                    pass
                return new_state
        else:
            body_fn_for_chunk = body_fn

        # Pre-compile the chunked while_loop — same function for all chunks
        def chunk_cond(state):
            _, _, _, _, _, _, _, delta_logZ, iteration, _, _, _, _, _, _, _, _, _, _, chunk_start = state
            converged = delta_logZ < delta_logZ_threshold
            too_many = iteration >= max_iterations
            chunk_done = (iteration - chunk_start) >= bound_update_interval
            return (~converged) & (~too_many) & (~chunk_done)

        compiled_chunk = jax.jit(lambda s: lax.while_loop(chunk_cond, body_fn_for_chunk, s))

        while total_done < config.max_iterations:
            # Inject current chunk_start into state tuple (last element)
            state_with_chunk = (*current_state[:-1], jnp.array(total_done))
            current_state = compiled_chunk(state_with_chunk)

            # Unpack to check convergence and refit
            (live_x, live_logL, worst_x_buffer, worst_logL_buffer,
             delta_logZ_buffer, scale_buffer, logZ, delta_logZ,
             iteration, current_key, current_scale, acceptance_count,
             total_proposals, ellipsoid_center, ellipsoid_axes,
             ellipsoid_scale, ellipsoid_update_counter,
             me_axes, me_logvol_ells, _chunk_start) = current_state

            total_done = int(iteration)

            if float(delta_logZ) < delta_logZ_threshold or total_done >= config.max_iterations:
                break

            # Refit multi-ellipsoid from current live points
            me_state = fit_multi_ellipsoid(live_x, max_ellipsoids=max_ellipsoids)
            if config.verbose and total_done % (bound_update_interval * 5) < bound_update_interval:
                print(f"  Iter {total_done}: refit multi-ellipsoid -> {me_state.n_active} ellipsoid(s), "
                      f"logZ={float(logZ):.2f}, dlogZ={float(delta_logZ):.3f}")

            # Repack state with updated multi-ellipsoid arrays
            # Same compiled function handles the new arrays (no re-JIT)
            current_state = (
                live_x, live_logL, worst_x_buffer, worst_logL_buffer,
                delta_logZ_buffer, scale_buffer, logZ, delta_logZ,
                iteration, current_key, current_scale, acceptance_count,
                total_proposals, ellipsoid_center, ellipsoid_axes,
                ellipsoid_scale, ellipsoid_update_counter,
                me_state.axes, me_state.logvol_ells,
                _chunk_start,  # will be overwritten at next chunk start
            )

        if config.print_progress and TQDM_AVAILABLE:
            pbar.close()

        final_state = current_state

    elif config.print_progress and TQDM_AVAILABLE:
        # Single while_loop with progress tracking (no chunking)
        pbar = tqdm(total=config.max_iterations, desc="Nested Sampling")

        def progress_callback_throttled(iteration_val, delta_logZ_val, logZ_val):
            if int(iteration_val) % 100 == 0:
                pbar.n = int(iteration_val)
                pbar.set_postfix_str(f'logZ: {float(logZ_val):.2f} | dlogZ: {float(delta_logZ_val):.3f} > {delta_logZ_threshold:.3f}')
                pbar.refresh()

        original_body_fn = body_fn

        def body_fn_with_progress(state):
            new_state = original_body_fn(state)
            _, _, _, _, _, _, logZ_v, delta_logZ_v, iteration_v, *_ = new_state
            try:
                io_callback(progress_callback_throttled, None, iteration_v, delta_logZ_v, logZ_v)
            except Exception:
                pass
            return new_state

        final_state = lax.while_loop(cond_fn, body_fn_with_progress, init_state)
        pbar.close()

    else:
        # Single while_loop without progress tracking
        final_state = lax.while_loop(cond_fn, body_fn, init_state)

    # Unpack final state
    (live_x_final, live_logL_final,
     worst_x_buffer, worst_logL_buffer, delta_logZ_buffer, scale_buffer,
     logZ_final, delta_logZ_final, iteration_final, key_final,
     final_scale, final_acceptance_count, final_total_proposals,
     ellipsoid_center_final, ellipsoid_axes_final, ellipsoid_scale_final,
     ellipsoid_update_counter_final, _, _, _) = final_state

    if config.verbose and config.print_progress and not TQDM_AVAILABLE:
        total_calls = iteration_final * config.rwalk_K
        eff = 100.0 * iteration_final / total_calls if total_calls > 0 else 0.0
        print(f" Done {iteration_final} iterations | {total_calls} calls | {eff:.1f}% eff")
    elif config.verbose:
        print()

    runtime = time.time() - start_time

    # Extract only the samples that were actually used
    actual_iterations = iteration_final

    # Transform samples from unit cube to physical space if using prior_transform
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

    # Calculate final logZ and H
    iterations = jnp.arange(actual_iterations)

    logX_i = -iterations / nlive
    logX_i_plus_1 = -(iterations + 1) / nlive

    def logsubexp(a, b):
        return a + jnp.log1p(-jnp.exp(b - a))

    log_dX_dead = logsubexp(logX_i, logX_i_plus_1)
    log_dZ_dead = logL_samples + log_dX_dead

    # Volume shrinkage for final live points
    logX_final = -actual_iterations / nlive
    log_dX_final = logX_final - jnp.log(nlive)

    log_dZ_final = best_final_logL + log_dX_final

    log_dZ_all = jnp.concatenate([log_dZ_dead, jnp.array([log_dZ_final])])
    logZ = float(jax.scipy.special.logsumexp(log_dZ_all))

    # Calculate H
    log_weights_all = jnp.concatenate([
        log_dZ_dead - logZ,
        jnp.array([log_dZ_final - logZ])
    ])
    weights_all = jnp.exp(log_weights_all)
    logL_all = jnp.concatenate([logL_samples, jnp.array([best_final_logL])])
    H = float(jnp.sum(weights_all * logL_all)) - logZ

    # Standard error
    logZ_error = float(jnp.sqrt(jnp.abs(H) / nlive))

    # Calculate acceptance rate
    total_acceptance = float(final_acceptance_count)
    total_proposals = float(final_total_proposals)
    acceptance_rate = total_acceptance / total_proposals if total_proposals > 0 else 0.0

    if config.verbose:
        print(f"\nNested sampling completed in {runtime:.2f}s")
        print(f"Total iterations: {actual_iterations}")
        print(f"Final logZ: {logZ:.4f} ± {logZ_error:.4f}")
        print(f"Information H: {H:.4f}")
        print(f"Final delta_logZ: {delta_logZ_final:.6f}")
        print(f"Converged: {delta_logZ_final < delta_logZ_threshold}")
        print(f"Speed: {actual_iterations / runtime:.1f} iterations/sec")
        print(f"Acceptance rate: {acceptance_rate:.1%}")
        print(f"Final scale: {final_scale:.4f}")

        if delta_logZ_final > delta_logZ_threshold:
            import warnings
            warnings.warn(
                f"Run did NOT converge after {max_iterations} iterations. "
                f"Final delta_logZ ({delta_logZ_final:.6f}) > threshold ({delta_logZ_threshold:.6f}). "
                f"Increase max_iterations."
            )
        else:
            print(f"Converged at iteration {actual_iterations}")

    result = WhileLoopNSResult(
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

    return result
