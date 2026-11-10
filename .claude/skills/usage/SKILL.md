---
name: /usage
description: JNesty usage guide — public API, parameters, results access, FITS I/O, bounding methods, tuning, plotting, and the Nautilus importance sampler.
---

## Quick Start

```python
from jnesty import NestedSampler, save_results, load_results
import jax.numpy as jnp

# Define problem
def loglikelihood(x):
    return -0.5 * jnp.sum(x**2)

def prior_transform(u):
    return (u - 0.5) * 10.0  # [0,1] -> [-5,5]

# Run
sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
sampler.run_nested(max_iterations=10000, delta_logZ_threshold=0.01)

# Results
results = sampler.results
print(f"logZ = {results['logz']:.4f} +/- {results['logzerr']:.4f}")

# Save / Load
save_results(results, 'output.fits')
loaded = load_results('output.fits')
```

## NestedSampler Constructor

```python
NestedSampler(
    loglikelihood,        # callable: x (ndim array) -> float
    prior_transform,      # callable: u (unit cube) -> x (physical space)
    ndim,                 # int: number of parameters
    nlive=500,            # int: live points (more = more accurate, slower)
    rwalk_K=None,         # int or None: walk steps per iteration (auto: max(25, ndim+20))
    rwalk_step_scale=None,# float or None: initial scale (auto: 1.0, Dynesty default)
    target_acceptance=0.5,# float: target acceptance rate for scale adaptation
    scale_adapt_interval=1,
    device='gpu',         # 'gpu' or 'cpu'
    verbose=True,
    bound='none',         # 'none', 'single', 'multi'
    bound_update_interval=None,  # None=auto (rwalk_K*nlive calls for multi, 0 otherwise)
    max_ellipsoids=20,    # int: max ellipsoids for multi-ellipsoid
    batch_size=None,      # int or None: parallel walks in legacy mode (auto: ~5 steps/walk)
    queue_size=None,      # int or None: Dynesty-style queue mode (auto: 8 for multi, 0 otherwise)
    memory_frac=0.9,      # float: cap batch_size to fit within this fraction of GPU memory
    unit_cube_batch_size=200,  # int: batch size for Phase 1 uniform rejection
    min_eff=10.0,         # float: efficiency threshold (%) to switch Phase 1 -> Phase 2
    min_ncall=None,       # int or None: min likelihood calls before phase switch (auto: 2*nlive)
)
```

### Sampling Modes

JNesty has two sampling modes, selected automatically:

**Legacy mode** (`queue_size=0`, default for `bound='none'`):
- Runs `batch_size` independent walks in parallel via `jax.vmap`
- Each walk gets `rwalk_K // batch_size` steps
- Scale adapts every iteration

**Queue mode** (`queue_size>1`, default for `bound='multi'`):
- Dynesty-style GPU parallelism
- Generates `queue_size` candidates in parallel, each with full `rwalk_K` steps
- Tests one candidate per iteration against current loglstar
- Scale adapts only when queue drains (matching Dynesty's multiprocessing)
- Leftover candidates preserved across iterations

### Auto-tuning Defaults

| Parameter | Auto value | Logic |
|-----------|-----------|-------|
| `rwalk_K` | `max(25, ndim+20)` | Dynesty's ndim+20 formula |
| `rwalk_step_scale` | `1.0` | Dynesty default |
| `batch_size` | `max(1, rwalk_K // max(1, rwalk_K*10//nlive))` | ~5 steps/walk |
| `queue_size` | `8` for multi, `0` otherwise | Match Dynesty multiprocessing |
| `bound_update_interval` | `rwalk_K*nlive` calls for multi, `0` otherwise | In likelihood calls |

## run_nested() Parameters

```python
sampler.run_nested(
    max_iterations=100000,      # hard iteration limit
    delta_logZ_threshold=0.01,  # convergence criterion
    print_progress=True,        # tqdm progress bar
)
```

### Two-Phase Sampling

The sampler uses a two-phase approach:

1. **Phase 1 (uniform rejection)**: Draws batches of uniform random points from the unit cube, picks the first valid replacement. Fast when efficiency is high. Switches to Phase 2 when efficiency drops below `min_eff`% after `min_ncall` likelihood calls.

2. **Phase 2 (random walk)**: Uses `RWalkSampler` with adaptive bounds. Runs until `delta_logZ < threshold` or `max_iterations` reached.

Progress bar shows: `logZ`, `dlogZ` (with threshold), `logl*` (current worst logL), `eff(%)` (overall efficiency).

## Results Access

The `results` property returns a `Results` object supporting dict-style and attribute access:

```python
results = sampler.results

# Dict-style
results['logz']           # float: log evidence
results['logzerr']        # float: log evidence error
results['information']    # float: information H
results['samples']        # (N, ndim) array: posterior samples (physical space, dead+live)
results['samples_u']      # (N, ndim) array: samples (unit cube space, dead+live)
results['logl']           # (N,) array: log-likelihoods (dead+live, sorted ascending)
results['logwt']          # (N,) array: log importance weights (trapezoidal)
results['logvol']         # (N,) array: log volumes (dead+live)
results['logz_trajectory']     # (N,) array: cumulative evidence at each sample
results['logzerr_trajectory']  # (N,) array: evidence error at each sample
results['delta_logZ_trajectory'] # (niter,) array: convergence trajectory
results['scale_trajectory']     # (niter,) array: proposal scale over time
results['nlive']          # int: number of live points used
results['niter']          # int: total iterations
results['eff']            # float: sampling efficiency (%)
results['acceptance_rate']# float: overall acceptance rate
results['converged']      # bool: did it converge?
results['delta_logz']     # float: final delta_logZ
results['delta_logZ_threshold'] # float: threshold used
results['rwalk_K']        # int: walk steps used
```

### Results Methods

```python
results.summary()             # Print formatted summary
results.samples_equal()       # Equal-weight posterior resampling
```

## FITS I/O

```python
from jnesty import save_results, load_results

save_results(results, 'output.fits')  # Save to FITS
loaded = load_results('output.fits')  # Load from FITS
```

FITS structure:
- HDU 0 (PrimaryHDU): Header with scalar metadata (LOGZ, LOGZERR, H, NLIVE, NITER, etc.)
- HDU 1 (BinTableHDU): Per-sample arrays (LOGL, LOGWT, LOGVOL, SAMPLES, SAMPLES_U, trajectories)

## Bounding Methods

| Method | String | Use Case | Auto-updates |
|--------|--------|----------|-------------|
| Unit cube | `'none'` | Simple unimodal posteriors (default) | No |
| Single ellipsoid | `'single'` | Elongated posteriors | No |
| Multi-ellipsoid | `'multi'` | Multi-modal or complex geometries | Yes (periodic refit) |

Multi-ellipsoid uses recursive k-means splitting with BIC selection. Periodic refitting is measured in **likelihood calls** (default: `rwalk_K * nlive` calls, approximately `nlive` iterations). Queue mode (`queue_size=8`) is auto-enabled for multi-ellipsoid.

## Parameter Tuning

- **nlive**: 500 is a good default. Use 1000+ for high-dimensional or multi-modal problems.
- **rwalk_K**: Auto-tuned to `max(25, ndim+20)`. Higher values improve mixing at cost of speed.
- **rwalk_step_scale**: Start at 1.0. Adapted automatically via Robbins-Munro.
- **target_acceptance**: 0.5 is Dynesty's default. Lower values explore more aggressively.
- **delta_logZ_threshold**: 0.01 is standard. Use 0.1 for quick tests, 0.001 for publication.
- **batch_size** (legacy mode only): Auto-tuned to ~5 steps/walk. Runs `batch_size` independent walks in parallel via `jax.vmap`. Set to 1 to disable parallelism. Larger values benefit expensive likelihoods on GPU.
- **queue_size** (queue mode): 8 by default for `bound='multi'`. Each queue entry gets full `rwalk_K` steps. Scale adapts only at queue drain.
- **memory_frac**: Fraction of GPU memory available for batch walks (default 0.9). Caps auto-tuned `batch_size` if it would exceed this fraction of GPU memory. Uses compile-time memory estimation via XLA's `memory_analysis()`. Ignored on CPU.
- **bound_update_interval**: In likelihood calls (not iterations). Default `rwalk_K * nlive` for multi-ellipsoid. Set to 0 to disable periodic updates. Float values in (0,1) are interpreted as fraction of `rwalk_K * nlive`.
- **min_eff**: Efficiency threshold (default 10%) to switch from Phase 1 to Phase 2.

## Plotting

### Built-in plotting (no Dynesty dependency)

```python
from jnesty import plotting

fig, axes = plotting.runplot(sampler.results, lnz_truth=-10.0)
fig, axes = plotting.traceplot(sampler.results, truths=true_params)
fig, axes = plotting.cornerplot(sampler.results, truths=true_params)
fig, axes = plotting.cornerpoints(sampler.results)
fig, axes = plotting.diagnostics(sampler.results)
```

### Dynesty plotting (requires `dynesty` installed)

```python
sampler.plot_run()          # Evidence evolution
sampler.plot_trace()        # Parameter evolution with 1D marginals
sampler.plot_corner()       # Corner plot of posterior
sampler.plot_diagnostics()  # Convergence diagnostics
```

These convenience methods call `to_dynesty_results()` internally to convert results for Dynesty's plotting functions.

## Convenience Methods

```python
sampler.print_summary()           # Print formatted results
sampler.get_samples()             # Get posterior samples
sampler.get_logz()                # Get (logZ, logZ_error) tuple
sampler.to_dynesty_results()      # Convert to Dynesty Results for its plotting
```

## Typical Usage Patterns

### Simple problem (unimodal, low-dim)

```python
sampler = NestedSampler(loglike, prior_transform, ndim=5, bound='none')
sampler.run_nested(delta_logZ_threshold=0.01)
```

### Multi-modal problem

```python
sampler = NestedSampler(loglike, prior_transform, ndim=5,
                         bound='multi', nlive=1000)
# auto: queue_size=8, bound_update_interval=rwalk_K*nlive calls
sampler.run_nested()
```

### High-dimensional problem (>50D)

```python
sampler = NestedSampler(loglike, prior_transform, ndim=100,
                         nlive=500, bound='none')
# auto: rwalk_K=120, rwalk_step_scale=1.0
sampler.run_nested(delta_logZ_threshold=0.1)
```

## NautilusSampler (importance nested sampling)

`NautilusSampler` is a second sampling engine: our JAX/GPU implementation of
the Nautilus importance nested sampler. The nested-sampling idea is the same
but the mechanics differ — every likelihood evaluation is kept, neural-network
emulator shells are added, and evidence/posterior weights come from shell
volumes. It is **not** a drop-in replacement for `NestedSampler`: different run
controls and result objects.

### Install

```bash
pip install 'jnesty[nautilus]'
```

Plain `import jnesty` never imports the engine or its dependencies. The extra
provides `nautilus-sampler` (upstream CPU sampler — support routines, the
reference emulator, parity tests), `optax`, `scikit-learn`, `threadpoolctl`
and `h5py`.

### Minimal example

```python
import jax.numpy as jnp
from jnesty import NautilusSampler

def loglikelihood(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.01) ** 2)

def prior_transform(u):
    return u  # uniform prior on [0, 1]^ndim

sampler = NautilusSampler(loglikelihood, prior_transform, ndim=2,
                          nlive=500, seed=0)
sampler.run(f_live=0.01, n_eff=10000, verbose=True)

points, log_w, log_l = sampler.posterior()   # log_w are normalised
print(sampler.log_z, sampler.n_eff, sampler.n_like)
```

Argument order is JNesty's — likelihood first, prior transform second (the
native `JaxSampler` takes them the other way round). Constructing the sampler
imports the engine lazily on first use.

### Constructor parameters

```python
NautilusSampler(loglikelihood, prior_transform, ndim, nlive=500,
                seed=None, device=None, context=None, **kwargs)
```

| kwarg | default | meaning |
|-------|---------|---------|
| `nlive` | 500 | live points (native engine default is 2000; the adapter lowers it) |
| `n_update` | `nlive` | likelihood calls between bound updates during exploration |
| `n_batch` | 100 | likelihood evaluations per batched step |
| `n_networks` | 4 | emulator networks per bound; `0` disables neural bounds (diagnostic) |
| `n_like_new_bound` | `10*nlive` | calls a new bound gets before the next update |
| `enlarge_per_dim` | 1.1 | per-dimension bound enlargement factor |
| `split_threshold` | 100 | volume ratio that triggers splitting a bound |
| `periodic` | None | per-dimension periodic flags (bool array/list) |
| `vectorized` | False | likelihood already accepts `(n_points, ...)` batches |
| `pass_dict` | None | auto: False for callable priors, True for `nautilus.Prior` |
| `seed` | None | RNG seed for reproducible runs |
| `device` | None | JAX platform, e.g. `'gpu'` or `'cpu'` |
| `context` | None | extra argument passed to both likelihood and prior (keeps data out of compiled kernels) |
| `fit_backend` | `'numpy'` | MVEE/GMM fitting backend: `numpy`, `numpy_rank1`, `cpu_rank1`, `jax` |
| `neural_backend` | `'jax'` | `'jax'` (device emulator) or `'reference'` (upstream scikit-learn, parity only) |
| `filepath` / `resume` | None / True | HDF5 checkpoint path; resume from it when it exists |
| `blobs_dtype` | None | dtype for structured likelihood blobs |

`**kwargs` are forwarded verbatim to `jnesty.nautilus.sampler.JaxSampler`.

### Run controls

```python
sampler.run(f_live=0.01, n_shell=1, n_eff=10000, n_like_max=np.inf,
            discard_exploration=False, timeout=np.inf, verbose=False)
```

- `f_live`: target live-set fraction at the end of exploration (smaller = more thorough).
- `n_shell`: minimum samples per shell during the refinement phase.
- `n_eff`: target total effective sample size — raise it for a more accurate
  evidence/posterior.
- `n_like_max`: hard cap on likelihood evaluations.
- `discard_exploration`: drop exploration-phase points from `posterior()`.
- `timeout`: wall-clock cap in seconds.
- `verbose`: one-line status refreshed roughly every two seconds.

Returns `True` on normal completion and `False` when a likelihood-call or time
limit stopped the run early.

### Results

- `posterior(return_as_dict=None, equal_weight=False, equal_weight_boost=1.0,
  return_blobs=False)` → `(points, log_w, log_l)` with normalised log-weights;
  `return_blobs=True` adds the blobs as a fourth element.
- `log_z`: estimated log evidence.
- `n_eff`: total effective sample size — the main quality diagnostic.
- `n_like`: total number of likelihood evaluations.
- `eta`: estimated asymptotic sampling efficiency.
- `discard_exploration`: also settable as a property.
- `native_sampler`: the underlying `JaxSampler`; any engine attribute or
  method the adapter does not forward explicitly is reachable there (the
  adapter forwards unknown names via `__getattr__`).

**No `logzerr`**: Nautilus does not produce a Dynesty-style evidence error.
Monitor `n_eff`, and re-seed to gauge run-to-run scatter.

### Plotting nautilus results

The shared plotting helpers work on Nautilus results through the
Dynesty-compatible bridge:

```python
from jnesty import plotting

res = sampler.to_results()          # or the cached snapshot sampler.results
fig, axes = plotting.cornerplot(res)
```

`logzerr` is NaN (importance NS uses `n_eff` as the uncertainty proxy);
`logvol` / `logz_trajectory` in the bridged view are derived quantities
for plotting only, not re-derived evidence estimates.

### Checkpointing

```python
sampler = NautilusSampler(..., filepath='run.h5', resume=True)
sampler.run(...)                          # checkpoints after each batch
sampler.write('run.h5', overwrite=True)   # explicit checkpoint
```

Limitations:
- likelihood **blobs are not stored**: a run with blobs raises `ValueError` on
  write/resume. Use a blob-free likelihood for checkpointed runs.
- `neural_backend='reference'` cannot be checkpointed (scikit-learn weights are
  not serialised) — unsupported by design; use `'jax'`.
- Backend/bound metadata (`fit_backend`, `neural_backend`, `periodic`,
  `n_networks`, ...) is persisted; resume with the same options.
- `h5py` is required (part of the `jnesty[nautilus]` extra).

### JAX x64 and compilation cache

Constructing a sampler — or importing `jnesty.nautilus` — enables JAX float64
mode and an on-disk compilation cache process-wide. Plain `import jnesty` does
not. The cache defaults to `$XDG_CACHE_HOME/nautilus_jax`; set
`NAUTILUS_JAX_CACHE_DIR` to relocate it, or to an empty string to disable it.
The first run compiles several programs (slow start); later runs reuse the
cache. Because x64 is process-global, a mixed session sees float64 defaults
after the first Nautilus sampler is built.

### Which engine to choose

| Situation | Engine |
|-----------|--------|
| Dynesty-compatible API, FITS output, plotting helpers, dynamic NS | `NestedSampler` / `DynamicNestedSampler` |
| High-dimensional, cheap vectorizable likelihood, neural-bound efficiency | `NautilusSampler` |
| You need a Dynesty-style `logzerr` | Dynesty-style samplers (`NautilusSampler` has none) |
| You need bit-for-bit upstream-nautilus behaviour | upstream `nautilus.Sampler` (not re-exported) |
