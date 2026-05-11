# API Reference

## NestedSampler

```{eval-rst}
.. autoclass:: jnesty.NestedSampler
   :members:
   :inherited-members:
   :show-inheritance:
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

### FITS I/O

```{eval-rst}
.. autofunction:: jnesty.save_results
```

```{eval-rst}
.. autofunction:: jnesty.load_results
```

## Plotting

All plotting functions accept a results dict (or `Results` object) as the
first argument and return `(fig, axes)`.

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
.. autofunction:: jnesty.sampler.estimate_batch_size_from_memory
```

## Bounding Methods

```{eval-rst}
.. autoclass:: jnesty.bounding.Bound
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.bounding.UnitCube
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.bounding.SingleEllipsoid
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.bounding.MultiEllipsoidBound
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: jnesty.bounding.get_bound
```

## Internal Samplers

```{eval-rst}
.. autoclass:: jnesty.internal_samplers.InternalSampler
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: jnesty.internal_samplers.RWalkSampler
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: jnesty.internal_samplers.get_sampler
```

### Queue Mode

```{eval-rst}
.. autofunction:: jnesty.internal_samplers.vmap_queue_refill
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
.. autofunction:: jnesty.multi_ellipsoid.fit_multi_ellipsoid
```

```{eval-rst}
.. autofunction:: jnesty.multi_ellipsoid.fit_bounding_ellipsoid
```

```{eval-rst}
.. autofunction:: jnesty.multi_ellipsoid.get_axes_for_rwalk
```

```{eval-rst}
.. autofunction:: jnesty.multi_ellipsoid.sample_from_union
```

```{eval-rst}
.. autofunction:: jnesty.multi_ellipsoid.contains
```

```{eval-rst}
.. autofunction:: jnesty.multi_ellipsoid.contains_all
```
