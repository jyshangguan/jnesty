---
name: /develop-nautilus
description: Guide for developing the JAX Nautilus engine — bounds, emulator, runtime config, checkpoints, and the public adapter.
---

## Scope

This skill covers our JAX/GPU implementation of the Nautilus importance
nested sampler in `src/jnesty/nautilus/` and its thin public adapter
`src/jnesty/nautilus_sampler.py`. The upstream CPU implementation
(`nautilus-sampler` 1.0.6) is a dependency and the parity oracle — never a
target for edits.

## Engine Module Map

```
src/jnesty/nautilus/
├── __init__.py      # Engine exports; calls _config.initialize_runtime() on import
├── sampler.py       # JaxSampler: shell loop, bound updates, refinement, posterior()
├── bounds.py        # PhaseShift, JaxUnitCube, JaxEllipsoid, JaxUnitCubeEllipsoidMixture,
│                    # JaxUnion, JaxNeuralBound, JaxNautilusBound
├── eval.py          # JaxEvaluator: batched prior/likelihood programs, blobs
├── neural.py        # Emulator, train_emulator (optax), ReferenceEmulator
├── fitting.py       # fit_mvee / fit_gmm; backends numpy (default), numpy_rank1,
│                    # cpu_rank1, jax
├── geometry.py      # Unified bound kernels (unified_transform/log_v/contains/sample)
├── gmm.py           # GMM splitting helpers used by JaxUnion (gmm2_fit, ...)
├── _jit.py          # Shape-bucketed JIT cache (bucket_size, padded_eval, stack_eval)
├── random.py        # take_key / finish_key: per-bound PRNG stream
├── checkpoint.py    # HDF5 write_sampler / read_sampler (versioned schema)
└── _config.py       # initialize_runtime(): x64 + on-disk compilation cache
```

Division of labour: `sampler.py` decides *when* (shells, bound updates,
refinement), `bounds.py` + `geometry.py` decide *where* (acceptance region),
`eval.py` evaluates, `neural.py` trains the emulator, `checkpoint.py` persists.
Structural decisions (how many bounds, when to split) stay in Python;
everything numeric runs through JAX kernels.

## Extending a Bound

All bound classes follow one contract:

```python
class Bound:
    @classmethod
    def compute(cls, points, ..., key=None, rng=None, backend='numpy'):
        """Fit the bound; returns a bound instance."""

    def contains(self, points):          # (n, ndim) -> (n,) bool
    def sample(self, n_points=100, key=None):  # (n, ndim) in [0, 1)^ndim
    @property
    def log_v(self): ...                 # scalar log volume
    def reset(self, rng=None): ...       # reset acceptance counters
    def unified(self): ...               # (center, B, B_inv, dim_cube, n_ell, log_v)
```

Rules:
- **JAX-compatible** kernels: no Python `if` on traced values (use
  `lax.cond` / `jnp.where`), no in-place mutation of JAX arrays, fixed shapes
  per call. Batched entry points go through `_jit.padded_eval` /
  `_jit.stack_eval` so one compiled program serves many batch sizes.
- **Explicit PRNG keys**: `sample()` accepts a key. Use `take_key(self, key)`
  and `finish_key(...)` from `random.py` rather than a constant key — the
  helpers keep a per-object stream, so unkeyed calls (bound self-volume
  estimation) do not repeat the same proposals. Never seed inside a kernel.
- **Unified representation**: `unified()` returns the tuple consumed by
  `geometry.unified_*` (`z = (B_inv + 2*diag(dim_cube)) @ (x - center)`).
  Implement it if the bound should compose into mixtures/unions.
- **Composition**: `JaxNautilusBound.compute()` builds non-overlapping
  `JaxUnion` regions, wraps each in a `JaxNeuralBound`, and keeps a separate
  outer `JaxUnitCubeEllipsoidMixture` bound for proposals. A new regional
  bound is plugged in through `JaxUnion.compute(..., bound_class=...)`.
- Numerical code must never import from `develop/`.

## Emulator (`neural.py`)

- `train_emulator(x, y, key, n_networks=4, hidden=..., learning_rate=...,
  batch_size=..., max_epochs=..., patience=..., dtype=jnp.float32,
  parallel=False)` trains an optax (Adam) MLP ensemble with early stopping.
  `parallel=True` trains all members in one compiled program (bit-identical
  results, fewer compilations).
- `Emulator` holds stacked `(w, b)` params plus `mean`/`scale` and predicts
  the ensemble mean. Prediction uses the shape-bucketed JIT cache keyed by
  layer shapes (weights are dynamic arguments, never baked into a program).
- **Network weights are float32 on purpose** — the emulator only ranks and
  rejects proposals, so it must stay cheap; the sampler state (points,
  `log_l`, volumes, evidence) stays float64. Do not "fix" this by promoting
  the weights.
- `ReferenceEmulator` wraps upstream's scikit-learn emulator for
  `neural_backend='reference'` parity runs; prediction runs on the host and is
  much slower. `n_networks=0` disables the emulator entirely (ellipsoid-only
  containment) — useful to isolate emulator vs bound bugs.
- Compilation caches are module-level dicts keyed by shapes and
  hyperparameters; extend the key whenever a new knob changes the program.

## Check the Reference First

The oracle for engine behaviour is upstream `nautilus-sampler` 1.0.6 (the
external `nautilus` package), plus the separate `develop/` tree (vendored at
commit `7de6eb1`) which holds the reference sources and this port's
development history.

- Before changing algorithm code, reproduce the behaviour upstream. If
  upstream agrees with theory, the bug is ours.
- `tests/nautilus/test_jax_parity.py` is the executable form of this rule.
- **Never modify `develop/`** — it is a read-only historical oracle, not part
  of the package.

## Runtime Configuration

`_config.initialize_runtime()` is called on `import jnesty.nautilus` and on
`JaxSampler` construction. It enables JAX x64 and the on-disk compilation
cache (`$XDG_CACHE_HOME/nautilus_jax`; `NAUTILUS_JAX_CACHE_DIR` relocates it,
empty string disables it), and is idempotent — repeated calls are no-ops.

Rules:
- No import-time `XLA_FLAGS` mutation and no other `jax.config.update` at
  import: users may preset `XLA_FLAGS` / JAX config themselves before
  importing. Keep all global configuration inside `initialize_runtime()`.
- Plain `import jnesty` must not touch JAX configuration — see
  `tests/nautilus/test_import_side_effects.py`.
- x64 is process-global: a mixed session sees float64 defaults after the
  first Nautilus construction.

## Checkpointing Limits

`checkpoint.py` writes a versioned HDF5 file: shell arrays, bound geometry,
neural weights, counters, PRNG state, and neural/union backend metadata
(`fit_backend`, `neural_backend`, `periodic`, `n_networks`, ...).

- `JaxSampler(filepath=..., resume=True)` checkpoints after every batch;
  `write(path, overwrite=...)` is the explicit form.
- **Unsupported by design**: runs with likelihood blobs (raise `ValueError` —
  the schema has no blob storage) and `neural_backend='reference'`
  (scikit-learn objects cannot be serialised). Use blob-free likelihoods and
  `neural_backend='jax'`.
- Resume with the same backend/bound options, or the restored bounds will not
  correspond to the original run.
- The layout is versioned (`SCHEMA_VERSION`); bump it when fields change.

## Tests

```bash
python -m pytest tests/nautilus -q                    # quick: 5 passed, 80 deselected
python -m pytest -o addopts='' tests/nautilus -q      # full: 85 passed (~2.5 min)
```

- Engine tests are marked `slow` and start with
  `pytest.importorskip('jnesty.nautilus')`, so the default `python -m pytest`
  deselects them and a missing optional dependency skips instead of failing.
- `tests/nautilus/` has **no `__init__.py` on purpose**: adding one puts that
  directory on `sys.path` during collection and shadows the real upstream
  `nautilus` package imported by the parity tests. Never add one.
- `test_jax_parity.py` → upstream comparison; `test_jax_sampler.py` →
  end-to-end runs; `test_nautilus_adapter.py` → public adapter surface.

## Adapter vs Engine Responsibilities

- Public-API changes (argument names/order, properties, docstrings users
  read) go in `src/jnesty/nautilus_sampler.py`. The adapter keeps the engine's
  call semantics but uses JNesty's convention (likelihood first, prior
  transform second, `ndim`/`nlive`); unknown attributes forward to
  `native_sampler` via `__getattr__`.
- Algorithm changes (bounds, emulator, shell loop, fitting) go in
  `src/jnesty/nautilus/`.
- The upstream CPU `Sampler` and `Prior` are never re-exported: import them
  from `nautilus`, not from `jnesty`.