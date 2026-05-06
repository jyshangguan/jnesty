# J-Nesty Technical Reference

## Architecture

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `api.py` | Public `NestedSampler` class. Three-argument init (loglikelihood, prior_transform, ndim), auto-tuned parameters, `run_nested()` method, results formatted as Dynesty-style dict, convenience plotting methods delegating to Dynesty. |
| `while_loop_sampler.py` | Core sampling engine. `randsphere()` n-ball proposal, `WhileLoopNSConfig`/`WhileLoopNSResult` NamedTuples, `run_nested_sampling_while_loop()` implementing the full NS loop inside `lax.while_loop` with per-iteration convergence, adaptive scale, and optional chunked execution for periodic bound refits. |
| `multi_ellipsoid.py` | JIT-compiled multi-ellipsoid decomposition. `MultiEllipsoidState` NamedTuple, recursive k-means splitting with BIC criterion via `_fit_multi_ellipsoid_core` (masked operations + `lax.while_loop`), union sampling, containment checks. |
| `bounding.py` | Single ellipsoid fitting via covariance eigenvalue decomposition, uniform ellipsoid sampling (`sample_ellipsoid`), batched sampling. |
| `whitening.py` | Shrinkage covariance estimation and Cholesky whitening transforms. Used by the legacy `constrained_sampler.py` for correlated parameter spaces. |
| `plotting.py` | Dynesty-style plotting: `runplot`, `traceplot`, `cornerplot`, `cornerpoints`, `diagnostics`. Uses matplotlib with optional `corner` and `scipy` dependencies. Converts JAX arrays to numpy for plotting. |
| `sampler.py` | Legacy NS implementation using `lax.scan` (fixed-length, no early stopping). Retained for reference; not used by the public API. |
| `constrained_sampler.py` | Legacy batched random walk sampler with Metropolis acceptance and whitening. Retained for reference; the active sampler in `while_loop_sampler.py` uses pure constraint enforcement instead. |

### Data Flow

```
User code
  |
  v
NestedSampler.__init__()          -- validate args, auto-tune rwalk_K / rwalk_step_scale
  |
  v
NestedSampler.run_nested()        -- build prior_sample fn (unit cube), create WhileLoopNSConfig
  |
  v
run_nested_sampling_while_loop()   -- initialize nlive points, fit initial bound, enter loop
  |
  +-- lax.while_loop (or chunked Python loop for multi-ellipsoid)
  |     |
  |     +-- Find worst live point (argmin logL)
  |     +-- Multi-step random walk (K steps, randsphere proposals)
  |     +-- Adaptive scale (Robbins-Munro)
  |     +-- Update live points, accumulate evidence
  |     +-- Check delta_logZ convergence
  |
  v
WhileLoopNSResult                 -- raw JAX arrays
  |
  v
NestedSampler._format_results()   -- convert to numpy, add live points (trapezoidal weights)
  |
  v
results dict (Dynesty format)     -- consumed by plotting and user code
```

### Key NamedTuples

**WhileLoopNSConfig**

| Field | Type | Default | Description |
|---|---|---|---|
| `nlive` | int | 500 | Number of live points |
| `max_iterations` | int | 10000 | Hard iteration cap |
| `delta_logZ_threshold` | float | 0.01 | Convergence threshold |
| `rwalk_K` | int | 25 | Walk steps per iteration |
| `rwalk_L` | int | 16 | (Unused in while_loop mode) |
| `rwalk_step_scale` | float | 1.0 | Initial proposal scale |
| `target_acceptance` | float | 0.5 | Target acceptance rate |
| `scale_adapt_interval` | int | 1 | Per-walk adaptation |
| `use_ellipsoid` | bool | False | Single-ellipsoid proposals |
| `ellipsoid_update_interval` | int | 500 | Ellipsoid refit frequency |
| `prior_bounds` | array/None | None | Optional rejection bounds |
| `verbose` | bool | True | Print progress |
| `print_progress` | bool | True | tqdm progress bar |
| `bound` | str | 'none' | 'none', 'single', 'multi' |
| `bound_update_interval` | int | 0 | Bound refit frequency (0 = once) |
| `max_ellipsoids` | int | 20 | Max ellipsoids for multi decomposition |

**WhileLoopNSResult**

| Field | Type | Description |
|---|---|---|
| `logZ` | float | Log evidence |
| `logZ_error` | float | Standard error on logZ |
| `H` | float | Information (Kullback-Leibler divergence) |
| `delta_logZ` | float | Final remaining evidence estimate |
| `n_iterations` | int | Iterations completed |
| `runtime` | float | Wall-clock seconds |
| `samples` | ndarray | Dead point samples (physical space) |
| `logL_samples` | ndarray | Log-likelihoods of dead points |
| `delta_logZ_trajectory` | ndarray | Per-iteration convergence trace |
| `scale_trajectory` | ndarray | Per-iteration proposal scale |
| `acceptance_rate` | float | Overall walk acceptance rate |
| `live_x` | ndarray | Final live points (unit cube) |
| `live_logL` | ndarray | Final live point log-likelihoods |

**MultiEllipsoidState**

| Field | Shape | Description |
|---|---|---|
| `centers` | (max_ells, ndim) | Ellipsoid centers |
| `covs` | (max_ells, ndim, ndim) | Covariance matrices |
| `axes` | (max_ells, ndim, ndim) | Axes = eigvecs * sqrt(eigvals) |
| `precision` | (max_ells, ndim, ndim) | Inverse covariance (eigvecs / eigvals @ eigvecs.T) |
| `logvol_ells` | (max_ells,) | Log volume of each ellipsoid |
| `n_active` | int | Number of active (non-empty) ellipsoids |

---

## Key Algorithms

### Random Walk Sampling

1. **n-ball proposal** (`randsphere`): draw `z ~ N(0, I)` for direction, `U^(1/n)` for radius in the unit ball. Matches Dynesty's `randsphere()` formula exactly.
2. **Multi-step walk**: starting from a random live point, take K steps of `x += scale * (axes @ randsphere(...))`. For multi-ellipsoid, `axes` comes from the scheduled ellipsoid; for no-bound, `axes` is the identity.
3. **Constraint enforcement**: accept step only if `logL(proposed) > logL_worst` AND within unit cube / prior bounds. No Metropolis ratio -- pure constraint enforcement matching Dynesty's rwalk sampler.

### Adaptive Scale

Robbins-Munro adaptation, applied every walk (per-iteration):

```
proposed_scale = scale * exp((acceptance_rate - target) / ndim / target)
```

- `target_acceptance` defaults to 0.5 (Dynesty's rwalk default).
- Adaptation is skipped on iteration 0 to allow the walk to establish an acceptance baseline.
- The scale is clamped implicitly by the exponential form (cannot go negative).

### Multi-Ellipsoid Decomposition

Implemented in `_fit_multi_ellipsoid_core` (JIT-compiled via `@functools.partial(jax.jit, static_argnums=(1,))`):

1. Start with one slot containing all live points.
2. Pick the first splittable slot. Fit a bounding ellipsoid via masked covariance + Mahalanobis expansion.
3. Initialize k-means (k=2) with centers at the ellipsoid's major axis endpoints.
4. Run k-means (10 iterations). Check that both clusters have at least `2 * ndim` points.
5. Fit sub-ellipsoids for each cluster. Apply BIC criterion: accept split if `logvol_combined - logvol_parent < -nparam * log(n) / n`.
6. On success: replace slot with left child, add right child to next free slot. On failure: mark slot as leaf.
7. Repeat via `lax.while_loop` until no splittable slots remain or `max_ellipsoids` reached.
8. All operations use masked arrays (fixed-size, padded with inactive slots) to remain JIT-compatible.

**Schedule-based ellipsoid selection** (Bresenham interleaving): instead of per-step `random.choice`, pre-compute a walk schedule of length K using a fractional accumulator. This ensures proportional coverage across K steps without stochastic variance.

**Chunked loop for bound updates**: when `bound_update_interval > 0`, the Python-level loop runs chunks of `bound_update_interval` iterations. Between chunks, `fit_multi_ellipsoid` is called in Python to refit. The compiled chunk function stays the same (no re-JIT) because multi-ellipsoid state is passed as traced arrays.

### Convergence

- Criterion: `delta_logZ < delta_logZ_threshold` (default 0.01).
- `delta_logZ = logsumexp([0, max_live_logL + logX_remaining - logZ])`.
- Checked every iteration via `lax.while_loop` condition function (iteration-granular, not batch-granular).
- The loop also terminates if `iteration >= max_iterations`.

### Evidence Calculation

All arithmetic in log-space to avoid underflow:

- **logsubexp(a, b)**: `a + log1p(-exp(b - a))` for computing `log(exp(a) - exp(b))` when `a > b`.
- **Volume shrinkage**: `logX_i = -i / nlive` (linear approximation). `log_dX_i = logsubexp(logX_{i-1}, logX_i)`.
- **Evidence increment**: `log_dZ_i = logL_i + log_dX_i`.
- **Cumulative evidence**: `logZ = logaddexp(logZ, log_dZ_i)` at each iteration.
- **Final live points**: remaining volume `logX_final = -niter / nlive`, split equally: `log_dX_final = logX_final - log(nlive)`, added with `best_logL`. This is computed post-loop in the results formatting step.
- **Information H**: `H = sum(weights * logL) - logZ`, where weights are normalized from `log_dZ`.
- **Error**: `logZ_error = sqrt(|H| / nlive)` (standard NS approximation).

---

## Parameter Defaults

| Parameter | Default | Explanation |
|---|---|---|
| `nlive` | 500 | Standard for moderate problems. Dynesty uses 500 for static NS. |
| `rwalk_K` | `max(25, ndim + 20)` | Ensures adequate mixing in high dimensions. Matches Dynesty's `walks` default scaling. |
| `rwalk_step_scale` | 1.0 | Dynesty's default initial scale. Adapted during run. |
| `target_acceptance` | 0.5 | Dynesty rwalk default. Balances exploration vs efficiency. |
| `scale_adapt_interval` | 1 | Adapt every walk (per-iteration). Matches Dynesty's per-call adaptation. |
| `bound` | 'none' | No bounding by default. Safest for multi-modal problems. |
| `bound_update_interval` | None (auto) | For `bound='multi'`: `nlive` iterations (Dynesty ratio=1.0). Otherwise: 0 (fit once). |
| `max_ellipsoids` | 20 | Caps decomposition depth. Sufficient for most multi-modal problems. |
| `delta_logZ_threshold` | 0.01 | Standard 1% evidence uncertainty. |
| `max_iterations` | 100000 | Hard cap in `run_nested()`. Config default is 10000. |
| `use_ellipsoid` | False | Legacy single-ellipsoid mode. Not used when `bound='multi'`. |
| `ellipsoid_update_interval` | 500 | Legacy parameter for single-ellipsoid refit frequency. |

**Auto-tuning logic** (in `api.py`):

- `rwalk_K`: `max(25, ndim + 20)` if not specified.
- `rwalk_step_scale`: 1.0 (Dynesty default, was previously `min(1.0, 1/sqrt(ndim))`).
- `bound_update_interval`: `nlive` for multi-ellipsoid, 0 otherwise. Float ratio supported (e.g., `0.5` means `0.5 * nlive`).

---

## Performance Characteristics

### Dimension Scaling (Approximate, GPU)

| ndim | J-Nesty iter/s | Dynesty iter/s | Speedup | Notes |
|---|---|---|---|---|
| 2 | ~500 | ~1000 | 0.5x | JAX dispatch overhead dominates |
| 5 | ~800 | ~500 | 1.6x | Break-even region |
| 20 | ~1200 | ~100 | 12x | GPU parallelism kicks in |
| 100 | ~600 | ~5 | 120x | Full GPU utilization |

These are representative figures for a Gaussian likelihood with `nlive=500`, `rwalk_K=25`. Actual performance depends on likelihood cost and hardware.

### Key Performance Notes

- **GPU crossover**: approximately 50D. Below this, JAX dispatch overhead exceeds computation time.
- **JAX dispatch overhead**: dominates at low dimensions (~2D-10D). Each `lax.while_loop` iteration is a single XLA op, but the per-iteration Python overhead of the progress callback (`io_callback`) can add ~20%.
- **Setting `print_progress=False`** eliminates `io_callback` overhead entirely.
- **Multi-ellipsoid refit**: ~135ms per call (JIT-compiled, first call includes compilation). Auto interval = `nlive` iterations, so refit cost amortizes to ~0.27ms/iteration at `nlive=500`.
- **Chunked loop**: for multi-ellipsoid, the Python-level chunk boundary causes a device-to-host transfer every `bound_update_interval` iterations. This is negligible compared to refit cost.

---

## Dynesty Alignment

### Matches Dynesty Exactly

- **n-ball proposal**: `randsphere` uses `z * (U^(1/n) / ||z||)` with `z ~ N(0,I)`, matching Dynesty's `randsphere()`.
- **Per-walk scale adaptation**: Robbins-Munro update `scale *= exp((f - target) / ndim / target)` applied every iteration (not every step).
- **Pure constraint enforcement**: accept step if `logL > logL_worst` and within bounds. No Metropolis acceptance ratio.
- **Multi-ellipsoid decomposition**: recursive k-means(k=2) splitting, BIC stopping criterion, Mahalanobis-based bounding ellipsoid fitting. Same algorithm as Dynesty's `_bounding_ellipsoids`.
- **Live points added to results**: `_format_results()` implements Dynesty's `add_live_points()` -- remaining live points sorted by logL, with trapezoidal weight computation connecting the last dead point to live points.
- **bound_update_interval semantics**: ratio semantics (float values multiply `nlive`), default ratio=1.0 for multi, matching Dynesty's `update_interval`.
- **Results dictionary format**: keys match Dynesty's `Results` object (`logz`, `logzerr`, `logl`, `logvol`, `logwt`, `samples`, `samples_u`, `nlive`, `niter`).
- **to_dynesty_results()**: produces a genuine `dynesty.utils.Results` object for direct use with Dynesty's plotting functions.

### Intentional Differences

| Aspect | Dynesty | J-Nesty | Reason |
|---|---|---|---|
| Multi-ellipsoid fitting | Python recursion + loops | JIT-compiled `lax.while_loop` with masked arrays | Eliminates Python dispatch overhead for GPU |
| Ellipsoid selection during walk | Per-step `random.choice` proportional to volume | Pre-computed schedule (Bresenham interleaving) | Reduces stochastic variance across K steps; no per-step RNG cost inside scan |
| Bound update mechanism | Python callback between iterations | Chunked loop: compiled chunk runs `bound_update_interval` iterations, then Python refits | Avoids re-JIT on bound updates; state passed as traced arrays |
| Early stopping | Python-level check after each batch | `lax.while_loop` condition checks every iteration | True iteration-granular convergence without wasted iterations |
| Volume accounting (live points) | Individual point volumes | Equal-split: `logX_final - log(nlive)` for all live points | Simpler; matches the standard NS approximation |
| Legacy sampler | N/A | `sampler.py` uses `lax.scan` with Metropolis | Retained for reference; not active |
