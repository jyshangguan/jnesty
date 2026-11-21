# API Reference

## NestedSampler

```{eval-rst}
.. autoclass:: jnesty.NestedSampler
   :members:
   :inherited-members:
   :show-inheritance:
```



## DynamicNestedSampler

```{eval-rst}
.. autoclass:: jnesty.DynamicNestedSampler
   :members:
   :show-inheritance:
```

### Dynamic Module Functions

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.compute_integrals
```

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.kish_ess
```

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.weight_function
```

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.stopping_function
```

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.combine_saved_and_new
```

## NautilusSampler

```{eval-rst}
.. autoclass:: jnesty.NautilusSampler
   :members:
   :show-inheritance:
```

## DNestSampler

```{eval-rst}
.. autoclass:: jnesty.DNestSampler
   :members:
   :inherited-members:
   :show-inheritance:
```

### JAX DNest Engine

The engine behind :class:`jnesty.DNestSampler`: our JAX port of the CDNest
diffusive nested sampler. Importing `jnesty.dfnest` enables JAX float64
mode (CDNest computes in double precision).

```{eval-rst}
.. autoclass:: jnesty.dfnest.JaxDNestSampler
   :members:
   :show-inheritance:
```

#### Postprocessing

```{eval-rst}
.. autofunction:: jnesty.dfnest.dnest_postprocess
```

```{eval-rst}
.. autoclass:: jnesty.dfnest.DNestResult
   :members:
```

### JAX Nautilus Engine

The engine behind :class:`jnesty.NautilusSampler`: our JAX port of the
Nautilus importance nested sampler (the external `nautilus-sampler` package
is only a support dependency and is not re-exported). Importing
`jnesty.nautilus` enables JAX float64 mode and the engine's compilation
cache; see `jnesty.nautilus._config`.

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxSampler
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxEvaluator
   :members:
```

#### Engine Bounds

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxUnitCube
   :members:
```

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxEllipsoid
   :members:
```

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxUnion
   :members:
```

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxNeuralBound
   :members:
```

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxNautilusBound
   :members:
```

#### Neural Emulator

```{eval-rst}
.. autoclass:: jnesty.nautilus.Emulator
   :members:
```

```{eval-rst}
.. autofunction:: jnesty.nautilus.train_emulator
```

#### Checkpointing

```{eval-rst}
.. autofunction:: jnesty.nautilus.checkpoint.write_sampler
```

```{eval-rst}
.. autofunction:: jnesty.nautilus.checkpoint.read_sampler
```

## Results

```{eval-rst}
.. autoclass:: jnesty.Results
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: jnesty.results.format_results
```

```{eval-rst}
.. autofunction:: jnesty.results.format_nautilus_results
```

```{eval-rst}
.. autofunction:: jnesty.results.format_dnest_results
```

### FITS I/O

```{eval-rst}
.. autofunction:: jnesty.save_results
```

```{eval-rst}
.. autofunction:: jnesty.load_results
```

## Plotting

All plotting functions accept a results dict (or `Results` object) as the
first argument and return `(fig, axes)`. For a `NautilusSampler` or
`DNestSampler` run, pass `sampler.to_results()` (the same cached object as
`sampler.results`).

```{eval-rst}
.. autofunction:: jnesty.runplot
```

```{eval-rst}
.. autofunction:: jnesty.traceplot
```

```{eval-rst}
.. autofunction:: jnesty.cornerplot
```

```{eval-rst}
.. autofunction:: jnesty.cornerpoints
```

```{eval-rst}
.. autofunction:: jnesty.diagnostics
```

## Core Sampling Loop

The core loop and its pluggable pieces live in the `jnesty.dynesty` engine
subpackage: `core`, `internal_samplers`, `bounding`, `multi_ellipsoid` and
`_dynamic`. The public classes in `jnesty.dynesty_sampler` are thin wrappers
around it.

### Configuration

```{eval-rst}
.. autoclass:: jnesty.WhileLoopNSConfig
   :members:
```

### Result

```{eval-rst}
.. autoclass:: jnesty.WhileLoopNSResult
   :members:
```

### Run Function

```{eval-rst}
.. autofunction:: jnesty.run_nested_sampling
```

### Memory Estimation

```{eval-rst}
.. autofunction:: jnesty.dynesty.core.estimate_batch_size_from_memory
```

## Bounding Methods

```{eval-rst}
.. autoclass:: jnesty.dynesty.bounding.Bound
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.dynesty.bounding.UnitCube
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.dynesty.bounding.SingleEllipsoid
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.dynesty.bounding.MultiEllipsoidBound
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.bounding.get_bound
```

## Internal Samplers

```{eval-rst}
.. autoclass:: jnesty.dynesty.internal_samplers.InternalSampler
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.dynesty.internal_samplers.RWalkSampler
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.internal_samplers.get_sampler
```

### Queue Mode

```{eval-rst}
.. autofunction:: jnesty.dynesty.internal_samplers.vmap_queue_refill
```

## Utilities

```{eval-rst}
.. autofunction:: jnesty.utils.randsphere
```

```{eval-rst}
.. autofunction:: jnesty.utils.logsubexp
```

```{eval-rst}
.. autofunction:: jnesty.utils.logvol_prefactor
```

```{eval-rst}
.. autofunction:: jnesty.utils.mean_and_cov
```

## Multi-Ellipsoid Fitting

```{eval-rst}
.. autofunction:: jnesty.dynesty.multi_ellipsoid.fit_multi_ellipsoid
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.multi_ellipsoid.fit_bounding_ellipsoid
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.multi_ellipsoid.get_axes_for_rwalk
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.multi_ellipsoid.sample_from_union
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.multi_ellipsoid.contains
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.multi_ellipsoid.contains_all
```
