---
name: /debug-sampling
description: Diagnosing convergence problems, poor acceptance rates, and incorrect results in JNesty.
---

## General Principle: Check Dynesty First

When investigating sampling issues, **always compare with Dynesty first**. Dynesty has years of battle-tested optimization. If Dynesty gets the right answer and JNesty doesn't, the issue is in JNesty. If both disagree with theory, the issue may be in the problem setup.

See CLAUDE.md: "Critical Development Principle: Check Dynesty First."

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
python -m py_compile src/jnesty/internal_samplers.py
python -m py_compile src/jnesty/bounding.py
python -m py_compile src/jnesty/sampler.py
python -m py_compile src/jnesty/jnesty.py

# Run unit tests
pytest tests/unit/test_queue_mode.py -v
pytest tests/integration/test_queue_and_defaults.py -v

# Run a quick demo
python dev/demo/01_multimodal_gaussian_mixture_jnesty.py

# Import check
python -c "from jnesty import NestedSampler, save_results, load_results; print('OK')"
```
