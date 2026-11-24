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

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.compute_weights
```

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.jitter_run
```

```{eval-rst}
.. autofunction:: jnesty.dynesty._dynamic.estimate_fresh_batch_size_from_memory
```

## NautilusSampler

```{eval-rst}
.. autoclass:: jnesty.NautilusSampler
   :members:
   :show-inheritance:
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

```{eval-rst}
.. autoclass:: jnesty.nautilus.JaxUnitCubeEllipsoidMixture
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.nautilus.PhaseShift
   :members:
   :show-inheritance:
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

## DNestSampler

```{eval-rst}
.. autoclass:: jnesty.DNestSampler
   :members:
   :inherited-members:
   :show-inheritance:
```

### Opt-in enhancements

The opt-in enhancements (`sample_frequency`, `gradient_scales`,
`gradient_interval`, `results_at_temperature`, `posterior_at_temperature`)
are fully documented in the class and method docstrings rendered by the
autoclass above. `sample_frequency` allocates *how often* a coordinate is
proposed while `gradient_scales` modulates *how far* its step reaches (they
compose orthogonally), and with `gradient_scales=True` the `n_like` counter
grows by exactly `n_gradient_events`. See {doc}`methods` for the mechanisms
and measured gains, and {doc}`examples` (Example 6).

### JAX DNest Engine

The engine behind :class:`jnesty.DNestSampler`: our JAX port of the CDNest
diffusive nested sampler. The engine lives in six modules: `state`
(configuration and validation), `levels` (level hierarchy and statistics),
`proposals` (the lane-step Metropolis kernel), `core` (callable plumbing /
compiled block program), `postprocess` (evidence and weights) and `engine`
(the sampler loop). Importing `jnesty.dfnest` enables JAX float64 mode
(CDNest computes in double precision).

```{eval-rst}
.. autoclass:: jnesty.dfnest.JaxDNestSampler
   :members:
   :show-inheritance:
```

#### Configuration and Validation

```{eval-rst}
.. autoclass:: jnesty.dfnest.state.DNestConfig
   :members:
```

```{eval-rst}
.. autofunction:: jnesty.dfnest.state.validate_sample_frequency
```

#### Level Statistics

```{eval-rst}
.. autofunction:: jnesty.dfnest.levels.delta_log_z_estimate
```

#### Postprocessing

```{eval-rst}
.. autofunction:: jnesty.dfnest.dnest_postprocess
```

```{eval-rst}
.. autoclass:: jnesty.dfnest.DNestResult
   :members:
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

## Dynesty-Style Engine (jnesty.dynesty)

The dynesty-style engine and its pluggable pieces live in the
`jnesty.dynesty` subpackage: `core`, `internal_samplers`, `bounding`,
`multi_ellipsoid` and `_dynamic`. The public classes in
`jnesty.dynesty_sampler` are thin wrappers around it.

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

### Compilation Cache

```{eval-rst}
.. autofunction:: jnesty.dynesty.core.clear_compiled_units
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.core.compiled_units_size
```

### Bounding Methods

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

### Internal Samplers

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
.. autoclass:: jnesty.dynesty.internal_samplers.HSliceSampler
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.internal_samplers.validate_rwalk_direction_weights
```

```{eval-rst}
.. autofunction:: jnesty.dynesty.internal_samplers.get_sampler
```

#### Queue Mode

```{eval-rst}
.. autofunction:: jnesty.dynesty.internal_samplers.vmap_queue_refill
```

### Multi-Ellipsoid Fitting

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
