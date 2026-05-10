---
name: /usage
description: JNesty usage guide — public API, parameters, results access, FITS I/O, bounding methods, tuning, and plotting.
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
