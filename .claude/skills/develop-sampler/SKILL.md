---
name: /develop-sampler
description: Guide for adding a new internal sampler (proposal strategy) to JNesty.
---

## Architecture

Internal samplers live in `src/jnesty/internal_samplers.py`. Each implements the `InternalSampler` interface and is registered in `SAMPLER_REGISTRY` for lookup by string name.

The core NS loop in `src/jnesty/sampler.py` calls `sampler_obj.sample()` to propose replacement live points and `sampler_obj.tune()` to adapt the proposal scale.

## InternalSampler Interface

```python
class InternalSampler:
    """Base class for proposal samplers."""

    def __init__(self, ndim, **kwargs):
        self.ndim = ndim

    def sample(self, key, x_start, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None):
        """
        Generate a replacement point.

        Parameters
        ----------
        key : jax.random.PRNGKey
        x_start : (ndim,) starting point in unit cube space
        logL_constraint : float, minimum log-likelihood
        loglikelihood_fn : callable, log-likelihood (physical space)
        axes : (ndim, ndim) or (max_ellipsoids, ndim, ndim) axes matrix
        scale : float, current proposal scale
        n_steps : int, number of walk/slice steps
        prior_bounds : optional (2, ndim) array for rejection

        Returns
        -------
        (x_new, logL_new, n_accepted) tuple
        """
        raise NotImplementedError

    def tune(self, scale, acceptance_rate, ndim, iteration):
        """Adapt scale based on acceptance rate. Returns new scale."""
        raise NotImplementedError
```

## JAX Constraints

All code inside `sample()` must be JAX-compatible because it runs inside `lax.while_loop`:

- **No Python control flow**: use `jnp.where()`, `jax.lax.cond()`, `jax.lax.select()`
- **No Python print**: use `jax.debug.print()` or `io_callback` for debugging only
- **No side effects**: sample must be a pure function of its inputs
- **Use `lax.scan`** for loops, not Python `for`
- **Key splitting**: always split keys before use: `key, subkey = random.split(key)`

## Reference: RWalkSampler

The existing `RWalkSampler` (lines 57-151 of `internal_samplers.py`) demonstrates the pattern:

1. `sample()` uses `lax.scan` over `n_steps` walk steps
2. Each step proposes via `randsphere()` transformed by axes matrix
3. Accepts if in unit cube AND satisfies logL constraint (no Metropolis)
4. Returns `(x_new, logL_new, n_accepted)` — the final position and total acceptances

Key detail for multi-ellipsoid: `sample()` accepts an optional `walk_schedule` parameter — a pre-computed array of ellipsoid indices used to select which axes to use for each walk step.

## Steps to Add a New Sampler

### 1. Implement the class

Add your class to `src/jnesty/internal_samplers.py`:

```python
class SliceSampler(InternalSampler):
    def __init__(self, ndim, **kwargs):
        super().__init__(ndim, **kwargs)
        # Your initialization

    def sample(self, key, x_start, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None, **kwargs):
        # Your sampling logic
        # Must return (x_new, logL_new, n_accepted)
        pass

    def tune(self, scale, acceptance_rate, ndim, iteration):
        # Your scale adaptation (or return scale unchanged)
        return scale
```

### 2. Register it

Add to the `SAMPLER_REGISTRY` dict at the bottom of the file:

```python
SAMPLER_REGISTRY = {
    'rwalk': RWalkSampler,
    'slice': SliceSampler,  # Add this
}
```

### 3. Wire into the core loop

In `src/jnesty/sampler.py`, the sampler is currently hardcoded:

```python
sampler_obj = get_sampler('rwalk', ndim, ...)  # line ~141
```

To make it configurable, this string should come from the config. The `WhileLoopNSConfig` NamedTuple may need a new `sampler` field (currently `bound` is already configurable this way).

### 4. Wire into the user API

In `src/jnesty/jnesty.py`, the `NestedSampler.__init__` needs a `sampler` parameter (analogous to the existing `bound` parameter):

```python
def __init__(self, ..., sampler='rwalk', ...):
```

### 5. Test

1. `python -m py_compile src/jnesty/internal_samplers.py`
2. Run a demo with the new sampler
3. Compare logZ with Dynesty's equivalent sampler (if available)

## Checklist

- [ ] Class inherits `InternalSampler`
- [ ] `sample()` returns `(x_new, logL_new, n_accepted)`
- [ ] `sample()` is JAX-compatible (no Python control flow in hot path)
- [ ] `tune()` returns a valid scale (or passes through unchanged)
- [ ] Added to `SAMPLER_REGISTRY`
- [ ] Config and API wired through (`WhileLoopNSConfig`, `NestedSampler`)
- [ ] Tested with at least one demo problem
