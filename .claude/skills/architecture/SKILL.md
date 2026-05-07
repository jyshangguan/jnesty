---
name: /architecture
description: JNesty code architecture — modules, data flow, key data structures, JAX patterns, and demo scripts.
---

## Module Map

```
src/jnesty/
├── __init__.py            # Public exports
├── jnesty.py              # NestedSampler class (user-facing API)
├── sampler.py             # Core NS loop (run_nested_sampling)
├── internal_samplers.py   # Proposal strategies (InternalSampler + RWalkSampler)
├── bounding.py            # Bounding methods (Bound + UnitCube/Single/Multi)
├── multi_ellipsoid.py     # JIT-compiled multi-ellipsoid fitting
├── results.py             # Results class + format_results() + FITS I/O
├── plotting.py            # Dynesty-style plotting helpers
└── utils.py               # Shared utilities (randsphere, logsubexp, etc.)
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
run_nested_sampling (sampler.py)   ← Core NS loop
  │  1. Initialize live points from prior
  │  2. Create Bound and InternalSampler via factories
  │  3. Run lax.while_loop (or chunked loop for multi-ellipsoid)
  │     per iteration:
  │       a. Find worst live point
  │       b. Call sampler.sample() to propose replacement
  │       c. Call sampler.tune() to adapt scale
  │       d. Update live points and evidence
  │       e. Check convergence (delta_logZ < threshold)
  │  4. Return WhileLoopNSResult
  │
  ▼
format_results (results.py)        ← Raw result → Results object
  │  Converts JAX arrays to numpy
  │  Computes trapezoidal weights
  │  Adds live points
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
    bound_update_interval: int = 0
    max_ellipsoids: int = 20
    batch_size: int = 1
    memory_frac: float = 0.9
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
    samples: jnp.ndarray          # (niter, ndim) dead point samples
    logL_samples: jnp.ndarray     # (niter,) log-likelihoods
    delta_logZ_trajectory: jnp.ndarray  # (niter,) convergence history
    scale_trajectory: jnp.ndarray       # (niter,) scale adaptation history
    acceptance_rate: float
    live_x: jnp.ndarray = None    # (nlive, ndim) final live points
    live_logL: jnp.ndarray = None # (nlive,) final live log-likelihoods
```

### MultiEllipsoidState (multi_ellipsoid.py)

Fixed-size array container for multi-ellipsoid decomposition:

```python
class MultiEllipsoidState(NamedTuple):
    centers: jnp.ndarray      # (max_ellipsoids, ndim)
    axes: jnp.ndarray         # (max_ellipsoids, ndim, ndim)
    precision: jnp.ndarray    # (max_ellipsoids, ndim, ndim)
    logvol_ells: jnp.ndarray  # (max_ellipsoids,)
    n_active: int             # number of active ellipsoids
```

### Results (results.py)

Dict wrapper providing both `results['logz']` and `results.logz` access. See `/usage` skill for full field listing.

## Internal Loop State Tuple

The core loop packs all mutable state into a flat tuple for `lax.while_loop`:

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
| 11 | acceptance_count | scalar | Total acceptances |
| 12 | total_proposals | scalar | Total proposals |
| 13 | bound_axes | (ndim, ndim) | Axes from current bound |
| 14 | walk_schedule | (rwalk_K,) | Ellipsoid schedule (multi only) |
| 15 | me_axes | (max_ell, ndim, ndim) | Multi-ellipsoid axes |
| 16 | me_logvol_ells | (max_ell,) | Multi-ellipsoid volumes |
| 17 | chunk_start | scalar | Start of current chunk |

## JAX Patterns Used

### lax.while_loop for convergence stopping

```python
final_state = lax.while_loop(cond_fn, body_fn, init_state)
```

`cond_fn` checks both convergence (`delta_logZ >= threshold`) and iteration limit (`iter < max_iterations`).

### Chunked loop for periodic bound updates

When using multi-ellipsoid with `bound_update_interval > 0`, the loop is broken into chunks:

```python
compiled_chunk = jax.jit(lambda s: lax.while_loop(chunk_cond, body_fn, s))
while total_done < max_iterations:
    current_state = compiled_chunk(state_with_chunk)
    # Refit bound
    me_state = fit_multi_ellipsoid(live_x_cur, ...)
    # Update walk schedule and repack state
```

### io_callback for progress bar

```python
jax.experimental.io_callback(progress_cb, None, iteration, delta_logZ, logZ)
```

Throttled to every 100 iterations to avoid overhead.

### lax.scan for walk steps

```python
final_state, _ = lax.scan(walk_step, init_state, jnp.arange(n_steps))
```

Used in `_single_walk()` (internal_samplers.py) for each individual random walk.

### jax.vmap for parallel walks

```python
# batch_size independent walks from same starting point, different keys
vmapped_walk = jax.vmap(lambda wk: _single_walk(wk, x_start, ...))
x_candidates, n_accepted_arr = vmapped_walk(walk_keys)
```

When `batch_size > 1`, `RWalkSampler.sample()` runs `batch_size` independent walks in parallel via `jax.vmap`. Each walk has `rwalk_K // batch_size` steps. The first valid candidate among the batch is selected as the replacement. This parallelizes likelihood evaluation on GPU without introducing sampling bias (each walk is independently correct).

Walk keys are sharded across available GPUs via `NamedSharding` using `gcd(batch_size, num_devices)` devices, spreading memory evenly. Controlled by `CUDA_VISIBLE_DEVICES` for device selection.

### Compile-time memory estimation for batch capping

```python
# In sampler.py: estimate_batch_size_from_memory()
trial_fn = jax.vmap(lambda wk: _single_walk(...))
compiled = jax.jit(trial_fn).lower(trial_keys).compile()
peak = compiled.memory_analysis().peak_memory_in_bytes
per_walk = peak // trial_batch
max_batch = available_budget // per_walk
```

Uses XLA's `memory_analysis()` to get compile-time peak memory estimate (no runtime execution needed). Runs before the main loop to cap `batch_size` from `nlive` to fit within `memory_frac` of GPU memory.

## Registry / Factory Pattern

Samplers and bounds use string-based lookup:

```python
# internal_samplers.py
SAMPLER_REGISTRY = {'rwalk': RWalkSampler}
sampler = get_sampler('rwalk', ndim, target_acceptance=0.5)

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
  │  - Trapezoidal weight computation
  │  - Live point addition (sorted by logL)
  │  - Evidence trajectory computation
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
| 02 | Rosenbrock banana | 5D | Non-Gaussian elongated posterior |
| 03 | High-D Gaussian | 50D | GPU scaling advantage |
| 04 | Gaussian shells | 2D | Thin degenerate ring structures |

Run from `dev/demo/`:
```bash
python 01_multimodal_gaussian_mixture_jnesty.py --nlive 500
```

Output goes to `output_0X_jnesty/` or `output_0X_dynesty/` with FITS results and plots.

## Legacy Code

`dev/legacy/` contains retired implementations (old `api.py`, `while_loop_sampler.py`, etc.) kept for reference. Do not modify or import from these.
