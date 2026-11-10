---
name: /debug-sampling
description: Diagnosing convergence problems, poor acceptance rates, and incorrect results in JNesty.
---

## General Principle: Check Dynesty First

When investigating sampling issues, **always compare with Dynesty first**. Dynesty has years of battle-tested optimization. If Dynesty gets the right answer and JNesty doesn't, the issue is in JNesty. If both disagree with theory, the issue may be in the problem setup.

See AGENT.md: "Critical Development Principle: Check Dynesty First."

## Check the Reference Nautilus First

For issues in the JAX Nautilus engine (`NautilusSampler` / `JaxSampler`) the
analogue of the Dynesty-first rule is: **compare against the reference
Nautilus first**.

- The reference is the upstream CPU sampler from the external
  `nautilus-sampler` package (importable as `nautilus`). Run the same problem
  with the same `n_live` / `f_live` / `n_eff` there and compare `log_z`,
  `n_like` and the posterior marginals. If upstream agrees with theory and
  `JaxSampler` does not, the bug is in our engine (`jnesty/nautilus/`).
- The separate `develop/` tree (vendored at commit `7de6eb1`) contains the
  reference `nautilus` sources next to the JAX port they seeded, and serves as
  the parity oracle. Treat it as read-only: **never modify `develop/`**, never
  import it from the package.
- `tests/nautilus/test_jax_parity.py` asserts engine-vs-upstream parity; start
  there when a numeric discrepancy appears.

## Common Failure Modes

### Non-convergence (delta_logZ stays above threshold)

**Symptoms**: Run hits `max_iterations` without converging.

**Causes**:
- `nlive` too low for the problem complexity
- `rwalk_K` too small — walker doesn't mix well
- Bad bounding — bound doesn't cover the posterior
- Multi-modal posterior with `bound='none'` — walker can't cross between modes
- Phase 1 running too long (efficiency never drops below `min_eff`)

**Fixes**:
- Increase `nlive` (try 2x current)
- Use `bound='multi'` for multi-modal problems
- Increase `max_iterations`
- Check that `prior_transform` maps correctly to the region of interest
- Lower `min_eff` if Phase 1 is taking too long

### Poor Acceptance Rate (< 0.1)

**Symptoms**: `results['acceptance_rate']` is very low.

**Causes**:
- `rwalk_step_scale` starting too large — proposals jump too far
- Bound is too tight (axes are small relative to posterior)
- Prior volume is huge relative to posterior
- Scale history not resetting properly at bound updates

**Fixes**:
- Lower `target_acceptance` (e.g., 0.3)
- Check scale adaptation: `results['scale_trajectory']` should decrease over time
- Use `bound='single'` or `bound='multi'` to focus proposals
- Check that `hist_accept`/`hist_total` are reset at bound update boundaries

### Wrong logZ (disagrees with Dynesty by > 2 sigma)

**Symptoms**: logZ differs from Dynesty by more than combined uncertainty.

**Causes**:
- Missing a mode (multi-modal posterior with `bound='none'`)
- Incorrect `prior_transform` (wrong mapping range)
- Bug in `format_results()` — weight calculation error
- Phase 1 call counting bug (was counting all batch proposals instead of first_valid+1)
- Live point addition issue in results formatting

**Investigation steps**:
1. Run the same problem with Dynesty and compare logZ
2. Check posterior samples visually (scatter plot of first 2 dims)
3. Check `delta_logZ_trajectory` — should decrease monotonically
4. Check `scale_trajectory` — should adapt and stabilize
5. Verify `prior_transform` with a unit test
6. Check that `total_calls` tracking is correct

### Scale Not Adapting

**Symptoms**: `scale_trajectory` is flat or oscillates wildly.

**Causes**:
- Robbins-Munro formula not converging (check `target_acceptance` vs actual rate)
- `ndim` very high — adaptation needs more iterations
- Scale history not resetting at bound updates (should reset in queue mode)

**Fixes**:
- Increase `rwalk_K` to get better acceptance estimates per iteration
- Check the tune formula in `RWalkSampler.tune()`
- Verify `hist_accept`/`hist_total` reset at queue drain and bound updates

### Phase 1 Issues

**Symptoms**: Phase 1 exits too early or too late.

**Causes**:
- `min_eff` too high → Phase 1 exits before it should
- `min_ncall` too low → Phase 1 exits before live points are well-distributed
- `unit_cube_batch_size` too small → slow Phase 1

**Fixes**:
- Default `min_eff=10.0` works well for most problems
- Default `min_ncall=2*nlive` ensures enough initial exploration
- Increase `unit_cube_batch_size` (default 200) for faster Phase 1 on GPU

## Nautilus Engine Failure Modes

### Likelihood returns NaN (the engine raises ValueError)

The engine validates likelihood outputs: a NaN would propagate into the shell
evidence, so it raises a `ValueError` instead of silently producing a NaN
`log_z`. Return `-inf` (or a large negative finite value) for forbidden
regions; never return NaN.

### f_live stalls, or bounds never shrink

**Symptoms**: `n_like` keeps growing while the live-set fraction stays above
the `f_live` target and the remaining-evidence fraction plateaus.

**Causes / fixes**:
- too few live points for the dimension → increase `nlive` (2x, 4x)
- the neural emulator is rejecting valid points → set `n_networks=0` to
  disable it and isolate emulator vs bound/proposal effects
- the prior transform or likelihood is not JAX-traceable (Python control flow
  on traced values, `np.asarray` on device arrays, in-place mutation) → make
  both pure JAX functions
- `n_update` / `n_like_new_bound` too large for a short run → bound updates
  never fire; shorten them for debugging

### Low n_eff or a biased posterior

- Raise the `n_eff` target passed to `run()` (default 10000) and inspect
  `sampler.n_eff` and the per-shell `shell_n_eff` until the occupation is
  adequate.
- Consider `discard_exploration=True`: exploration-phase points are then
  excluded from `posterior()`, removing the low-likelihood tail that biases a
  short run.
- There is no `logzerr`; judge run-to-run stability by re-seeding and
  comparing `log_z` at equal `n_eff`.

### Checkpoint errors

- `ValueError: Checkpointing a run with likelihood blobs is not supported` —
  the HDF5 schema has no blob storage. Use a plain `log_l` likelihood (no
  blobs) when running with `filepath=` or calling `write()`.
- `ValueError` for `neural_backend='reference'` — the reference emulator keeps
  its weights in a scikit-learn object that cannot be serialised.
  Checkpointing requires `neural_backend='jax'`; this is unsupported by
  design, not a bug.
- Neural/union backend metadata (`fit_backend`, `neural_backend`, periodic
  settings) is persisted; resume with the same options or the restored bounds
  will not match the original run.

### x64 / precision surprises

Importing `jnesty.nautilus` or constructing a `NautilusSampler` enables JAX
float64 mode process-wide, plus the on-disk compilation cache. In a mixed
session every sampler created afterwards sees `jax_enable_x64=True`, so dtype
promotions differ from a Dynesty-only session. Never toggle x64 back off
mid-session — JAX treats it as a process-global setting.

### Upstream classes are not re-exported

`jnesty` exports only `NautilusSampler` (and the Dynesty-style API). The
upstream CPU `Sampler` and `Prior` are **not** re-exported: import them from
`nautilus` (the `nautilus-sampler` package), never from `jnesty`. A
`nautilus.Prior` instance is accepted by the engine, but its transform runs on
the host with SciPy rather than on the device.

## Reading Diagnostic Plots

### Run Plot (`plotting.runplot()`)
- Evidence logZ should converge to a stable value
- delta_logZ should decrease monotonically toward threshold
- If delta_logZ plateaus above threshold, the run didn't converge

### Trace Plot (`plotting.traceplot()`)
- Parameters should explore the posterior region
- If parameters are stuck in one region, the walker isn't mixing
- For multi-modal problems, should see samples from all modes
- Tail should be sparse (live points get accelerating volume shrinkage)

### Corner Plot (`plotting.cornerplot()` or `plotting.cornerpoints()`)
- Marginal distributions should match known posteriors
- Missing modes appear as absent peaks
- Overly broad distributions suggest poor constraint

### Diagnostics (`plotting.diagnostics()`)
- Shows evidence evolution, convergence trajectory, key metrics, efficiency

## Comparing with Dynesty

The demo scripts in `dev/demo/` provide a template for comparison:

```python
# Run Dynesty
from dynesty import NestedSampler as DynestyNS
dynesty_sampler = DynestyNS(loglikelihood, prior_transform, ndim=ndim,
                            nlive=nlive, bound='multi', sample='rwalk')
dynesty_sampler.run_nested(dlogz=0.01)
dynesty_results = dynesty_sampler.results

# Compare
logZ_diff = abs(jnesty_results['logz'] - dynesty_results.logz[-1])
combined_unc = np.sqrt(jnesty_results['logzerr']**2 + dynesty_results.logzerr[-1]**2)
agrees = logZ_diff < combined_unc
```

Key comparison metrics:
- **logZ**: should agree within combined uncertainty
- **H (information)**: should be similar
- **Iterations**: JNesty typically uses fewer iterations but higher per-iteration cost
- **Posterior shape**: visual comparison via scatter/corner plots

Important: When comparing, ensure Dynesty demos use `dlogz=0.01` (not default `dlogz=1.0`) for fair comparison.

## Multi-Ellipsoid Specific Issues

### Too Many/Few Ellipsoids

Check `n_active` in the multi-ellipsoid state:
- If `n_active=1` for a clearly multi-modal posterior, the splitting is too conservative
- If `n_active=max_ellipsoids`, the splitting is too aggressive (overfitting noise)

The BIC criterion in `fit_multi_ellipsoid()` controls splitting. Adjust `max_ellipsoids` if needed.

### Bound Update Interval

Bound update interval is measured in **likelihood calls** (not iterations):
- `bound_update_interval=0`: bound fitted once at start, never updated. Fast but may miss evolving posterior shape.
- `bound_update_interval=rwalk_K*nlive` (default for multi): ~`nlive` iterations between updates. Good balance.
- Float values (0,1) are interpreted as fraction of `rwalk_K*nlive`.
- If acceptance rate drops mid-run, try decreasing the interval.

## Queue Mode Issues

### Queue not draining

If the queue never drains, scale won't adapt. This happens when:
- `queue_size` is too large relative to iteration count
- All candidates are valid (very rare in practice)

### Scale not adapting in queue mode

Scale adapts only at queue drain in queue mode. If `queue_size` is very large, adaptation will be infrequent. Check:
- `queue_size=8` (default) is Dynesty's multiprocessing default
- Scale history resets at each drain (matching Dynesty's per-walk reset)

## Quick Diagnostic Commands

```bash
# Syntax check after code changes
python -m py_compile jnesty/dynesty_sampler.py
python -m py_compile jnesty/dynesty/internal_samplers.py
python -m py_compile jnesty/dynesty/bounding.py
python -m py_compile jnesty/dynesty/core.py
python -m py_compile jnesty/nautilus/sampler.py

# Run unit tests
pytest tests/unit/test_queue_mode.py -v
pytest tests/integration/test_queue_and_defaults.py -v
python -m pytest tests/nautilus -q          # quick engine check (5 passed, 80 deselected)

# Run a quick demo
python dev/demo/01_multimodal_gaussian_mixture_jnesty.py

# Import check
python -c "from jnesty import NestedSampler, save_results, load_results; print('OK')"
python -c "from jnesty import NautilusSampler; print('OK')"
```

For a Nautilus engine bug, add the reference comparison before changing code:

```python
import nautilus                          # upstream CPU oracle
from jnesty.nautilus import JaxSampler

ref = nautilus.Sampler(prior, loglike, n_live=nlive, seed=0)
ref.run(f_live=0.01, n_eff=10000)
ours = JaxSampler(prior, loglike, n_dim=ndim, n_live=nlive, seed=0)
ours.run(f_live=0.01, n_eff=10000)

print(ref.log_z, ours.log_z)             # should agree well within run scatter
print(ref.n_like, ours.n_like)           # similar order of magnitude
```
