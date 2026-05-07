---
name: /develop-bound
description: Guide for adding a new bounding method to JNesty.
---

## Architecture

Bounding methods live in `src/jnesty/bounding.py`. Each implements the `Bound` interface and is registered in `BOUND_REGISTRY` for lookup by string name.

The core NS loop in `src/jnesty/sampler.py` calls `bound_obj.fit()` to update the bound, `bound_obj.get_axes()` for proposal generation, and optionally uses bound-specific state for multi-ellipsoid walk schedules.

## Bound Interface

```python
class Bound:
    """Base class for bounding distributions."""

    def __init__(self, ndim, **kwargs):
        self.ndim = ndim

    def fit(self, points):
        """Fit the bound to live points. Returns self."""
        raise NotImplementedError

    def sample(self, key, n=1):
        """Sample point(s) from the bound."""
        raise NotImplementedError

    def get_axes(self):
        """Return (ndim, ndim) axes matrix for proposal generation."""
        raise NotImplementedError

    def contains(self, point):
        """Check if point is inside the bound."""
        raise NotImplementedError
```

## Existing Implementations

### UnitCube (`'none'`)
- `fit()`: no-op
- `get_axes()`: identity matrix
- `sample()`: uniform from [0,1]^ndim
- Use for: simple problems, baseline comparison

### SingleEllipsoid (`'single'`)
- `fit()`: covariance eigenvalue decomposition
- `get_axes()`: eigenvectors scaled by sqrt(eigenvalues)
- Use for: elongated unimodal posteriors

### MultiEllipsoidBound (`'multi'`)
- `fit()`: delegates to `fit_multi_ellipsoid()` from `multi_ellipsoid.py`
- Stores state as `MultiEllipsoidState` NamedTuple
- `get_axes()`: axes of the largest-volume ellipsoid
- `get_walk_schedule(rwalk_K)`: Bresenham interleaving of ellipsoid indices
- Use for: multi-modal, complex geometries
- Requires periodic refitting via chunked loop mode

## Integration with the Core NS Loop

The bound is used in three places in `src/jnesty/sampler.py`:

1. **Initial fit** (line ~144): `bound_obj.fit(live_physical)` after live points initialized
2. **Axes extraction** (line ~175): `bound_axes = bound_obj.get_axes()` for rwalk proposals
3. **Periodic refit** (chunked loop, line ~345): `fit_multi_ellipsoid(live_x_cur, ...)` for multi-ellipsoid

For multi-ellipsoid, the chunked loop mode is activated when `use_chunked_loop = True` (bound is `'multi'` and `bound_update_interval > 0`). The loop runs chunks of `bound_update_interval` iterations, then refits the bound between chunks.

### State packing for lax.while_loop

The bound state is packed into the flat tuple that `lax.while_loop` carries:

- Index 13: `bound_axes` — (ndim, ndim) axes matrix
- Index 14: `walk_schedule` — (rwalk_K,) int array of ellipsoid indices
- Index 15: `me_axes` — (max_ellipsoids, ndim, ndim) for multi-ellipsoid
- Index 16: `me_logvol_ells` — (max_ellipsoids,) for multi-ellipsoid

A new bound that needs additional state in the loop must extend this tuple and update the packing/unpacking in `sampler.py`.

## Steps to Add a New Bound

### 1. Implement the class

Add your class to `src/jnesty/bounding.py`:

```python
class MyBound(Bound):
    def __init__(self, ndim, **kwargs):
        super().__init__(ndim, **kwargs)
        # Your initialization

    def fit(self, points):
        # Fit to live points
        return self

    def sample(self, key, n=1):
        # Sample from the bound
        pass

    def get_axes(self):
        # Return (ndim, ndim) axes for proposals
        pass

    def contains(self, point):
        # Check membership
        pass
```

### 2. Register it

Add to `BOUND_REGISTRY`:

```python
BOUND_REGISTRY = {
    'none': UnitCube,
    'single': SingleEllipsoid,
    'multi': MultiEllipsoidBound,
    'mybound': MyBound,  # Add this
}
```

### 3. Handle state in the core loop (if needed)

If your bound needs periodic refitting or custom state carried through `lax.while_loop`:

- Add fields to the state tuple in `sampler.py`
- Add a `use_chunked_loop` branch similar to multi-ellipsoid
- Update the body function to use your bound's state

If your bound is stateless or only needs initial fitting, no changes to `sampler.py` are needed — the factory handles it.

### 4. Test

1. `python -m py_compile src/jnesty/bounding.py`
2. Test with demo 01 (multimodal Gaussian) — bound should correctly identify two modes
3. Compare logZ with Dynesty using the same bound type
4. Check acceptance rate is reasonable (>0.2)

## Checklist

- [ ] Class inherits `Bound`
- [ ] All four methods implemented: `fit`, `sample`, `get_axes`, `contains`
- [ ] Added to `BOUND_REGISTRY`
- [ ] Core loop integration handled (state packing if needed)
- [ ] `NestedSampler.__init__` accepts the new bound string
- [ ] Tested with at least one demo problem
- [ ] Acceptance rate and logZ compared with Dynesty
