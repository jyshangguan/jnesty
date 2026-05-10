---
name: /develop-sampler
description: Guide for adding a new internal sampler (proposal strategy) to JNesty.
---

## Architecture

Internal samplers live in `src/jnesty/internal_samplers.py`. Each implements the `InternalSampler` interface and is registered in `SAMPLER_REGISTRY` for lookup by string name.

The core NS loop in `src/jnesty/sampler.py` calls `sampler_obj.sample()` to propose replacement live points and `sampler_obj.tune()` to adapt the proposal scale.

There is also a standalone `vmap_queue_refill()` function for Dynesty-style queue mode that generates multiple candidates in parallel outside of any sampler object.

## InternalSampler Interface

```python
class InternalSampler:
    """Base class for proposal samplers."""

    def __init__(self, ndim, **kwargs):
        self.ndim = ndim

    def sample(self, key, x_starts, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None, walk_schedule=None):
        """
        Generate a replacement point.

        Parameters
        ----------
        key : jax.random.PRNGKey
        x_starts : array
            If batch_size=1: (ndim,) single starting point.
            If batch_size>1: (batch_size, ndim) diverse starting points.
        logL_constraint : float, minimum log-likelihood
        loglikelihood_fn : callable, log-likelihood
        axes : array
            If batch_size=1: (ncdim, ncdim) single axes matrix.
            If batch_size>1: (batch_size, ncdim, ncdim) per-walk axes.
        scale : float, current proposal scale
        n_steps : int, number of walk/slice steps
        prior_bounds : optional (2, ndim) array for rejection
        walk_schedule : optional array, not used in current code (legacy)

        Returns
        -------
        (x_new, logL_new, n_accepted, n_total) tuple
            n_total includes boundary rejections (matches Dynesty semantics)
        """
        raise NotImplementedError

    def tune(self, scale, acceptance_rate, ndim, iteration):
        """Adapt scale based on acceptance rate. Returns new scale."""
        raise NotImplementedError
```

**Important**: `sample()` returns a **4-tuple** `(x_new, logL_new, n_accepted, n_total)`, not a 3-tuple. The `n_total` counts all proposals including boundary rejections.

## JAX Constraints

All code inside `sample()` must be JAX-compatible because it runs inside `lax.while_loop`:

- **No Python control flow**: use `jnp.where()`, `jax.lax.cond()`, `jax.lax.select()`
- **No Python print**: use `jax.debug.print()` or `io_callback` for debugging only
- **No side effects**: sample must be a pure function of its inputs
- **Use `lax.scan`** for loops, not Python `for`
- **Key splitting**: always split keys before use: `key, subkey = random.split(key)`
- **Branchless likelihood evaluation**: use `jnp.where(in_bounds, loglikelihood_fn(x), -inf)` for XLA fusion — `lax.cond` with data-dependent predicates inside `lax.scan` kills GPU kernel fusion

## Reference: RWalkSampler

The existing `RWalkSampler` demonstrates the pattern:

### Key functions

**`_single_walk(key, x_start, logL_constraint, loglikelihood_fn, axes, scale, n_steps, ndim, n_cluster, prior_bounds, walk_schedule)`**
Core walk function using `lax.scan` over `n_steps`. Each step:
1. Proposes via `randsphere()` transformed by axes matrix (clustered dims)
2. Non-clustered dims resampled uniformly
3. Accepts if in unit cube AND satisfies logL constraint (no Metropolis)
4. Returns `(x_final, n_accepted, n_total)`

**`_propose_one(key, x, axes, scale, n_cluster, ndim, walk_schedule, step_idx)`**
Single proposal generation. First `n_cluster` dimensions perturbed via axes transform; remaining dims uniform from [0,1].

**`apply_reflect(u)`**
Iterative reflection into [0,1]. Matches Dynesty's `apply_reflect`.

**`vmap_queue_refill(refill_key, live_x, live_logL, loglstar, loglikelihood_fn, me_axes, me_logvol_ells, bound_axes, scale, rwalk_K, ndim, ncdim, queue_size, use_multi_ellipsoid, prior_bounds=None)`**
Generates `queue_size` candidates in parallel. Each gets full `rwalk_K` steps from a random starting point (live point above loglstar) with a random volume-weighted ellipsoid. Returns `(queue_x, queue_logL, queue_nacc, queue_ntot)`.

### Batch mode (legacy)

When `batch_size > 1`, `RWalkSampler.sample()` runs `batch_size` independent walks in parallel via `jax.vmap`:
- Each walk starts from a different live point
- Each walk uses a different (volume-weighted random) ellipsoid
- Steps per walk: `rwalk_K // batch_size`
- First valid candidate is selected as replacement
- All walks' acceptance stats summed for scale adaptation

### Scale adaptation

`RWalkSampler.tune()` uses Robbins-Munro adaptation matching Dynesty:
```python
proposed_scale = scale * jnp.exp(
    (acceptance_rate - target_acceptance)
    / ncdim / target_acceptance
)
```
Uses `ncdim` (clustered dimensions) not full `ndim`, matching Dynesty.

## Steps to Add a New Sampler

### 1. Implement the class

Add your class to `src/jnesty/internal_samplers.py`:

```python
class SliceSampler(InternalSampler):
    def __init__(self, ndim, **kwargs):
        super().__init__(ndim, **kwargs)

    def sample(self, key, x_starts, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None, walk_schedule=None):
        # Your sampling logic
        # Must return (x_new, logL_new, n_accepted, n_total)
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
sampler_obj = get_sampler('rwalk', ndim, ...)  # line ~400
```

To make it configurable, add a `sampler` field to `WhileLoopNSConfig` and pass it through.

### 4. Wire into the user API

In `src/jnesty/jnesty.py`, add a `sampler` parameter to `NestedSampler.__init__`:

```python
def __init__(self, ..., sampler='rwalk', ...):
```

### 5. Consider queue mode

If the sampler should work with queue mode, you may need a queue-refill function similar to `vmap_queue_refill()`. This function generates `queue_size` candidates in parallel, each using the full `rwalk_K` steps independently.

### 6. Test

1. `python -m py_compile src/jnesty/internal_samplers.py`
2. Run a demo with the new sampler
3. Compare logZ with Dynesty's equivalent sampler (if available)

## Checklist

- [ ] Class inherits `InternalSampler`
- [ ] `sample()` returns `(x_new, logL_new, n_accepted, n_total)` — 4-tuple
- [ ] `sample()` is JAX-compatible (no Python control flow in hot path)
- [ ] `sample()` handles both single and batch modes (x_starts shape varies)
- [ ] `tune()` returns a valid scale (or passes through unchanged)
- [ ] Added to `SAMPLER_REGISTRY`
- [ ] Config and API wired through (`WhileLoopNSConfig`, `NestedSampler`)
- [ ] Tested with at least one demo problem
