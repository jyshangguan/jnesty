---
name: /architecture
description: JNesty code architecture — modules, data flow, key data structures, JAX patterns, and demo scripts.
---

## Module Map

```
src/jnesty/
├── __init__.py            # Public exports
├── jnesty.py              # NestedSampler class (user-facing API)
├── sampler.py             # Core NS loop (run_nested_sampling, two-phase)
├── internal_samplers.py   # Proposal strategies (InternalSampler + RWalkSampler + vmap_queue_refill)
├── bounding.py            # Bounding methods (Bound + UnitCube/Single/Multi)
├── multi_ellipsoid.py     # JIT-compiled multi-ellipsoid fitting
├── results.py             # Results class + format_results() + FITS I/O
├── plotting.py            # Dynesty-style plotting (built-in, no dynesty dependency)
└── utils.py               # Shared utilities (randsphere, logsubexp, mean_and_cov)
```

## Data Flow

```
User code
  │
  ▼
NestedSampler (jnesty.py)          ← User-facing API
  │  __init__: configure parameters, auto-tune defaults
  │  run_nested(): build config, call core loop, format results
  │
  ▼
run_nested_sampling (sampler.py)   ← Core NS loop (two-phase)
  │
  │  Phase 1: Uniform rejection sampling
  │  │  _run_uniform_phase() with lax.while_loop
  │  │  Draws unit_cube_batch_size uniform points per iteration
  │  │  Switches when eff < min_eff% after min_ncall calls
  │  │  All JIT-compiled
  │  │
  │  Transition: Fit initial bound from Phase 1 live points
  │  │  bound_obj.fit(live_x)
  │  │  estimate_batch_size_from_memory() to cap batch_size
  │  │
  │  Phase 2: Random walk with adaptive bounds
  │  │  Legacy mode: lax.while_loop with batch parallel walks
  │  │  Queue mode: lax.while_loop with pre-filled candidate queue
  │  │  Per iteration:
  │  │    a. Find worst live point (loglstar)
  │  │    b. Retry loop: select start + ellipsoid, run walk until valid
  │  │    c. Adapt scale via Robbins-Munro (or at queue drain)
  │  │    d. Update live points and evidence
  │  │    e. Check convergence (delta_logZ < threshold)
  │  │  Chunked loop (multi-ellipsoid): periodic bound refit between chunks
  │  │
  │  Return WhileLoopNSResult
  │
  ▼
format_results (results.py)        ← Raw result → Results object
  │  Converts JAX arrays to numpy
  │  Computes trapezoidal weights
  │  Adds remaining live points (sorted by logL, Dynesty's add_live_points)
  │  Computes cumulative evidence trajectory
  │  Returns Results(dict)
  │
  ▼
Results object                      ← Dict wrapper with attribute access
     save_results() / load_results() for FITS I/O
```

## Key Data Structures

### WhileLoopNSConfig (sampler.py)

NamedTuple carrying all run configuration:

```python
class WhileLoopNSConfig(NamedTuple):
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
    bound_update_interval: int = 0     # in likelihood calls
    max_ellipsoids: int = 20
    batch_size: int = 1                # parallel walks (legacy mode)
    queue_size: int = 0                # 0=legacy, >1=queue mode
    memory_frac: float = 0.9
    ncdim: int = None                  # clustered dimensions (defaults to ndim)
    unit_cube_batch_size: int = 200    # Phase 1 batch size
    min_eff: float = 10.0             # Phase 1 -> 2 efficiency threshold (%)
    min_ncall: int = None             # min calls before phase switch (default 2*nlive)
```

### WhileLoopNSResult (sampler.py)

Raw output from the NS loop:

```python
class WhileLoopNSResult(NamedTuple):
    logZ: float
    logZ_error: float
    H: float
    delta_logZ: float
    n_iterations: int
    runtime: float
    samples: jnp.ndarray          # (niter, ndim) dead point samples (physical)
    logL_samples: jnp.ndarray     # (niter,) dead log-likelihoods
    delta_logZ_trajectory: jnp.ndarray  # (niter,) convergence history
    scale_trajectory: jnp.ndarray       # (niter,) scale adaptation history
    acceptance_rate: float
    live_x: jnp.ndarray = None    # (nlive, ndim) final live points (unit cube)
    live_logL: jnp.ndarray = None # (nlive,) final live log-likelihoods
```

### MultiEllipsoidState (multi_ellipsoid.py)

Fixed-size array container for multi-ellipsoid decomposition:

```python
class MultiEllipsoidState(NamedTuple):
    centers: jnp.ndarray      # (max_ellipsoids, ndim)
    covs: jnp.ndarray         # (max_ellipsoids, ndim, ndim)
    axes: jnp.ndarray         # (max_ellipsoids, ndim, ndim)
    precision: jnp.ndarray    # (max_ellipsoids, ndim, ndim)
    logvol_ells: jnp.ndarray  # (max_ellipsoids,)
    n_active: int             # number of active ellipsoids
```

### Results (results.py)

Dict wrapper providing both `results['logz']` and `results.logz` access. See `/usage` skill for full field listing.

## Internal Loop State Tuple (Phase 2)

The core loop packs all mutable state into a flat tuple for `lax.while_loop`. Base state is 18 elements; queue mode extends to 23:

| Index | Variable | Shape | Description |
|-------|----------|-------|-------------|
| 0 | live_x | (nlive, ndim) | Current live points |
| 1 | live_logL | (nlive,) | Live point log-likelihoods |
| 2 | worst_x_buffer | (max_iterations, ndim) | Dead point storage |
| 3 | worst_logL_buffer | (max_iterations,) | Dead logL storage |
| 4 | delta_logZ_buffer | (max_iterations,) | Convergence trajectory |
| 5 | scale_buffer | (max_iterations,) | Scale trajectory |
| 6 | logZ | scalar | Running evidence |
| 7 | delta_logZ | scalar | Current convergence metric |
| 8 | iteration | scalar | Loop counter |
| 9 | key | PRNGKey | Random state |
| 10 | scale | scalar | Current proposal scale |
| 11 | hist_accept | scalar | Accumulated acceptances (reset at bound update) |
| 12 | hist_total | scalar | Accumulated proposals (reset at bound update) |
| 13 | bound_axes | (ndim, ndim) | Axes from current bound |
| 14 | me_axes | (max_ell, ndim, ndim) | Multi-ellipsoid axes (all slots) |
| 15 | me_logvol_ells | (max_ell,) | Multi-ellipsoid log-volumes |
| 16 | calls_at_update | scalar | total_calls at last bound update |
| 17 | total_calls | scalar | Total likelihood calls (non-resetting) |
| 18 | queue_x | (queue_size, ndim) | Queue: candidate positions |
| 19 | queue_logL | (queue_size,) | Queue: candidate logL |
| 20 | queue_nacc | (queue_size,) | Queue: per-candidate acceptances |
| 21 | queue_ntot | (queue_size,) | Queue: per-candidate totals |
| 22 | queue_head | scalar | Queue: read head index |

Indices 18-22 only present in queue mode (`queue_size > 1`).

## JAX Patterns Used

### lax.while_loop for convergence stopping

```python
final_state = lax.while_loop(cond_fn, body_fn, init_state)
```

`cond_fn` checks both convergence (`delta_logZ >= threshold`) and iteration limit (`iter < max_iterations`). Used in Phase 1, Phase 2 (both modes), and the retry loop within each iteration.

### Chunked loop for periodic bound updates

When using multi-ellipsoid with `bound_update_interval > 0`, the loop is broken into chunks measured in **likelihood calls**:

```python
compiled_chunk = jax.jit(lambda s: lax.while_loop(chunk_cond, body_fn, s))
while total_done < max_iterations:
    current_state = compiled_chunk(state_with_chunk)
    # Refit bound
    me_state = fit_multi_ellipsoid(live_x_cur, ...)
    # Reset scale history, repack state
```

`chunk_cond` checks `(total_calls - calls_at_update) >= bound_update_interval`.

### io_callback for progress bar

```python
jax.experimental.io_callback(progress_cb, None, iteration, delta_logZ, logZ, loglstar, eff)
```

Throttled to every 100 iterations to avoid overhead. Uses `pbar.write()` for bound update messages (not `print()`, which breaks tqdm).

### lax.scan for walk steps

```python
final_state, _ = lax.scan(walk_step, init_state, jnp.arange(n_steps))
```

Used in `_single_walk()` (internal_samplers.py) for each individual random walk.

### jax.vmap for parallel operations

**Batch parallel walks** (legacy mode, `batch_size > 1`):
```python
vmapped_walk = jax.vmap(lambda wk, x0, ax: _single_walk(wk, x0, ..., ax, ...))
```
Each walk starts from a different live point with a different ellipsoid. First valid candidate is selected.

**Queue refill** (queue mode):
```python
jax.vmap(_generate_one)(keys)  # queue_size candidates in parallel
```
Each candidate gets full `rwalk_K` steps from a random starting point with a random ellipsoid.

**Batch likelihood evaluation** (Phase 1):
```python
vmapped_loglike = jax.vmap(loglikelihood_fn)
logL_batch = vmapped_loglike(u_batch)  # unit_cube_batch_size evaluations
```

### Compile-time memory estimation for batch capping

```python
# In sampler.py: estimate_batch_size_from_memory()
trial_fn = jax.vmap(lambda wk: _single_walk(...))
compiled = jax.jit(trial_fn).lower(trial_keys).compile()
peak = compiled.memory_analysis().peak_memory_in_bytes
per_walk = peak // trial_batch
max_batch = available_budget // per_walk
```

Uses XLA's `memory_analysis()` to get compile-time peak memory estimate. Caps `batch_size` to fit within `memory_frac` of GPU memory.

### Retry loop for valid point guarantee

```python
_, _, x_new, logL_new, iter_accepted, iter_total = lax.while_loop(
    _retry_cond, _retry_body, _init_retry)
```

Wraps the walk section to guarantee every NS iteration produces a valid replacement (`logL > loglstar`). Matches Dynesty's `_new_point()` while-True behavior. Re-selects starting point, ellipsoid, and randomness on each retry.

## Registry / Factory Pattern

Samplers and bounds use string-based lookup:

```python
# internal_samplers.py
SAMPLER_REGISTRY = {'rwalk': RWalkSampler}
sampler = get_sampler('rwalk', ndim, target_acceptance=0.5, batch_size=..., ncdim=...)

# bounding.py
BOUND_REGISTRY = {'none': UnitCube, 'single': SingleEllipsoid, 'multi': MultiEllipsoidBound}
bound = get_bound('multi', ndim, max_ellipsoids=20)
```

To add new types: implement the base class, add to registry, wire config through.

## Results Pipeline

```
WhileLoopNSResult (raw, JAX arrays)
  │
  ▼  format_results() in results.py
  │  - JAX → numpy conversion
  │  - Trapezoidal log-weight computation (logwt = logaddexp(logL_i, logL_{i-1}) + logdvol)
  │  - Live point addition: sorted by logL, volumes via Dynesty's formula
  │  - Cumulative evidence trajectory via logaddexp.accumulate(logwt)
  │
  ▼
Results(dict) — dict wrapper with attribute access
  │
  ▼  save_results() / load_results()
FITS file (astropy.io.fits)
```

## Demo Scripts

Located in `dev/demo/`. Each problem has a JNesty and Dynesty variant:

| Demo | Problem | Dimension | Key Test |
|------|---------|-----------|----------|
| 01 | Multi-modal Gaussian mixture | 5D | Multi-modality handling |
| 02 | Rosenbrock banana | 2D | Non-Gaussian elongated posterior |
| 03 | High-D Gaussian | 20D (configurable) | GPU scaling advantage |
| 04 | Gaussian shells | 2D | Thin degenerate ring structures |

Run from `dev/demo/`:
```bash
python 01_multimodal_gaussian_mixture_jnesty.py
```

Output goes to `output_0X_jnesty/` or `output_0X_dynesty/` with FITS results and plots.

## Test Suite

```
tests/
├── unit/
│   ├── test_bounding.py            # UnitCube, SingleEllipsoid, MultiEllipsoid
│   ├── test_internal_samplers.py   # RWalkSampler, _single_walk, apply_reflect
│   ├── test_multi_ellipsoid.py     # Multi-ellipsoid fitting
│   ├── test_plotting.py            # Plotting functions
│   ├── test_results.py             # Results formatting, FITS I/O
│   ├── test_sampler_unit.py        # Sampler unit tests
│   ├── test_utils.py               # Utility functions
│   └── test_queue_mode.py          # Queue mode behavior
├── integration/
│   ├── test_sampler.py             # Full NS loop integration
│   ├── test_queue_and_defaults.py  # Queue mode + auto-defaults
│   └── test_jnesty_api.py          # API compatibility
```

Run with:
```bash
pytest tests/ -v
```

## Legacy Code

`dev/legacy/` contains retired implementations (old `api.py`, `while_loop_sampler.py`, etc.) kept for reference. Do not modify or import from these.
