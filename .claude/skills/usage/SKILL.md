---
name: /usage
description: JNesty usage guide — public API, parameters, results access, FITS I/O, bounding methods, tuning, and plotting.
---

## Quick Start

```python
from jnesty import NestedSampler, save_results, load_results

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
    rwalk_step_scale=None,# float or None: initial scale (auto: 1.0)
    target_acceptance=0.5,# float: target acceptance rate for scale adaptation
    scale_adapt_interval=1,
    device='gpu',         # 'gpu' or 'cpu'
    verbose=True,
    bound='none',         # 'none', 'single', 'multi'
    bound_update_interval=None,  # None=auto (nlive for multi, 0 otherwise)
    max_ellipsoids=20,    # int: max ellipsoids for multi-ellipsoid
    batch_size=None,      # int or None: parallel walks (auto: rwalk_K // max(2, rwalk_K*10//nlive))
    memory_frac=0.9,      # float: cap batch_size to fit within this fraction of GPU memory
)
```

## run_nested() Parameters

```python
sampler.run_nested(
    max_iterations=100000,      # hard iteration limit
    delta_logZ_threshold=0.01,  # convergence criterion
    print_progress=True,        # tqdm progress bar
)
```

## Results Access

The `results` property returns a `Results` object supporting dict-style and attribute access:

```python
results = sampler.results

# Dict-style
results['logz']           # float: log evidence
results['logzerr']        # float: log evidence error
results['information']    # float: information H
results['samples']        # (N, ndim) array: posterior samples (physical space)
results['samples_u']      # (N, ndim) array: samples (unit cube space)
results['logl']           # (N,) array: log-likelihoods
results['logwt']          # (N,) array: log weights
results['logvol']         # (N,) array: log volumes
results['logz_trajectory']     # (N,) array: evidence at each sample
results['logzerr_trajectory']  # (N,) array: error at each sample
results['delta_logZ_trajectory'] # (niter,) array: convergence trajectory
results['scale_trajectory']     # (niter,) array: proposal scale over time
results['nlive']          # int: number of live points used
results['niter']          # int: total iterations
results['eff']            # float: sampling efficiency (%)
results['acceptance_rate']# float: overall acceptance rate
results['converged']      # bool: did it converge?
results['delta_logz']     # float: final delta_logZ
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
- HDU 1 (BinTableHDU): Per-sample arrays (LOGL, LOGWT, LOGVOL, SAMPLES, trajectories)

## Bounding Methods

| Method | String | Use Case |
|--------|--------|----------|
| Unit cube | `'none'` | Simple unimodal posteriors (default) |
| Single ellipsoid | `'single'` | Elongated posteriors |
| Multi-ellipsoid | `'multi'` | Multi-modal or complex geometries |

Multi-ellipsoid uses recursive k-means splitting with BIC selection. It requires `bound_update_interval > 0` (default: `nlive`) for periodic refitting.

## Parameter Tuning

- **nlive**: 500 is a good default. Use 1000+ for high-dimensional or multi-modal problems.
- **rwalk_K**: Auto-tuned to `max(25, ndim+20)`. Higher values improve mixing at cost of speed.
- **rwalk_step_scale**: Start at 1.0. Adapted automatically via Robbins-Munro.
- **target_acceptance**: 0.5 is Dynesty's default. Lower values explore more aggressively.
- **delta_logZ_threshold**: 0.01 is standard. Use 0.1 for quick tests, 0.001 for publication.
- **batch_size**: Auto-tuned to `rwalk_K // max(2, rwalk_K * 10 // nlive)`. Runs `batch_size`
  independent walks in parallel via `jax.vmap`, each with `rwalk_K // batch_size` steps. Set to
  1 to disable parallelism. Larger values benefit expensive likelihoods on GPU.
- **memory_frac**: Fraction of GPU memory available for batch walks (default 0.9). Caps
  auto-tuned `batch_size` if it would exceed this fraction of GPU memory. Uses compile-time
  memory estimation via XLA's `memory_analysis()`. Ignored on CPU.

## Plotting

JNesty delegates plotting to Dynesty's plotting module (requires `dynesty` installed):

```python
sampler.plot_run()          # Evidence evolution
sampler.plot_trace()        # Parameter evolution with 1D marginals
sampler.plot_corner()       # Corner plot of posterior
sampler.plot_diagnostics()  # Convergence diagnostics
```

These are convenience methods — they call `to_dynesty_results()` internally.

## Convenience Methods

```python
sampler.print_summary()           # Print formatted results
sampler.get_samples()             # Get posterior samples
sampler.get_logz()                # Get (logZ, logZ_error) tuple
sampler.to_dynesty_results()      # Convert to Dynesty Results for its plotting
```
