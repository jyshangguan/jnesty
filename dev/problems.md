# JAXNS Problems and Solutions

## Overview

This document consolidates all problems encountered during JAXNS development and their solutions. Each problem includes description, symptoms, investigation, solution, and lessons learned.

---

## Critical Problems

### Problem #1: Performance Bug - 1000x Slower Than Expected
**Date:** 2026-05-03
**Task:** 006 (Performance Fix)
**Severity:** 🔴 CRITICAL

#### Description
Initial GPU implementation was extremely slow: 1.6 seconds per iteration (should be < 0.1s)

#### Symptoms
- Small 5D problems took ~30 minutes
- Per-iteration time: 1.6s (expected: < 0.1s)
- GPU not being utilized effectively
- Implementation unusable for practical problems

#### Investigation
- **Root Cause:** Python for loop instead of JAX lax.scan
- **Contributing Factors:**
  - No JIT compilation (function arguments breaking JIT)
  - Python data structures (lists) instead of JAX arrays
  - Memory overhead from object creation every iteration
  - No GPU batching

#### Solution
Complete rewrite with proper JAX patterns:
- Replaced Python `for` with `lax.scan` for main NS loop
- JAX-compatible state management with pytrees
- Pre-allocated JAX arrays with `.at[idx].set()` updates
- Closure pattern for likelihood function to enable JIT

#### Results
- **14.3x speedup** (Phase 1 alone)
- Before: 26.639s for 20 iterations (0.8 iter/sec)
- After: 1.858s for 20 iterations (10.8 iter/sec)
- Final: 600-2300 iter/sec (770-2900x total speedup)

#### Lessons Learned
- Always use JAX control flow (lax.scan, lax.while_loop) in hot loops
- Avoid Python data structures in JIT-compiled functions
- Pre-allocate arrays and use in-place updates
- Test per-iteration time early to catch performance issues

#### Related Issues
- Task 006 (Performance Fix)
- INTEGRATION_SUMMARY.md

---

### Problem #2: 100D Complete Failure - 0% Acceptance
**Date:** 2026-05-05
**Task:** 008 (100D Scale Bug)
**Severity:** 🔴 CRITICAL

#### Description
JAXNS completely failed at 100D with 500 live points: 0% acceptance rate, sampler stuck

#### Symptoms
- Acceptance rate: 0%
- Max logL: -310.55 (should be close to 0)
- logZ: -315.31 (should be -138.36)
- Sampler completely stuck, can't explore parameter space
- Dynesty works fine on same problem

#### Investigation
- **Root Cause:** Proposal scale parameter too large for high-dimensional unit cube
- **Math:**
  - For scale = 0.63 in 100D
  - P(one dimension in [0,1]) ≈ 0.63
  - P(all 100 dimensions in [0,1]) = 0.63^100 ≈ 10^-19
  - Result: 0% acceptance, all proposals rejected immediately

#### Solution
Dimension-aware parameters:
```python
# More walk steps for high dimensions (matches Dynesty)
rwalk_K = max(25, ndim + 20)  # 120 for 100D

# Smaller scale for high dimensions
rwalk_step_scale = min(1.0, 1.0 / np.sqrt(ndim))  # 0.1 for 100D
```

#### Results
- **Before:** Acceptance 0%, logZ -315.31 (wrong)
- **After:** Acceptance 40.3%, logZ -139.28 ± 0.42 (correct!)
- Analytical: -138.36
- **JAXNS now 18% faster than Dynesty at 100D!**

#### Lessons Learned
- Proposal scale must scale with dimensionality: 1/√ndim
- Walk steps should increase with dimensionality: ndim + 20
- High-dimensional unit cubes are very restrictive for proposals
- Test acceptance rate early when scaling to new dimensions

#### Related Issues
- Task 008 (100D Scale Bug)
- Demo 03 validation

---

## Major Problems

### Problem #3: Evidence Calculation Numerical Instability
**Date:** 2026-05-03
**Task:** 001 (Milestone 1)
**Severity:** 🟠 MAJOR

#### Description
Evidence calculation produced negative information H values, indicating numerical issues

#### Symptoms
- H values: -2.77 (negative, impossible!)
- logZ values suspicious
- Volume shrinkage calculation incorrect

#### Investigation
- Initial formula used wrong log-space arithmetic
- Indexing error in log_dX calculation
- Missing "- logZ" term in H calculation

#### Solution
Corrected log-space arithmetic:
```python
# Main loop evidence calculation
log_dX = logX_old + jnp.log1p(-jnp.exp(logX_new - logX_old))

# Final evidence calculation
logX_i = -jnp.arange(actual_iterations) / nlive
logX_i_plus_1 = -(jnp.arange(actual_iterations) + 1) / nlive
log_dX_dead = logsubexp(logX_i, logX_i_plus_1)

# H calculation with correction
H = -2.77 → 4.27 (after fix)
```

#### Results
- H values now positive and realistic
- logZ calculations stable
- Agreement with Dynesty and analytical solutions

#### Lessons Learned
- Always use log-space arithmetic for evidence calculations
- Use logsubexp (or log1p) for numerical stability
- Verify H is positive (information can't be negative)

#### Related Issues
- Task 001 (Milestone 1)
- Task 007 (Evidence Calculation Fix)

---

### Problem #4: Demo 02 LogZ Discrepancy
**Date:** 2026-05-04
**Task:** 007 (Evidence & Boundary Fix)
**Severity:** 🟠 MAJOR

#### Description
JAXNS logZ differed from Dynesty by 0.54 in Rosenbrock problem

#### Symptoms
- JAXNS: logZ = -4.94 ± 0.20
- Dynesty: logZ = -4.40 ± 0.18
- Difference: 0.54 (outside expected uncertainty)
- Missing 0.5 factor in likelihood

#### Investigation
- Rosenbrock likelihood implemented as: `L = -(a-x)² - b(y-x²)²`
- Should be: `L = -0.5 * [(a-x)² + b(y-x²)²]` for proper Gaussian
- Missing normalization factor

#### Solution
Added 0.5 factor to match standard Gaussian form:
```python
# BEFORE (WRONG)
logL = -(a - x_val)**2 - b * (y_val - x_val**2)**2

# AFTER (CORRECT)
logL = -0.5 * ((a - x_val)**2 + b * (y_val - x_val**2)**2)
```

#### Results
- Difference reduced to 0.086 (within uncertainty)
- JAXNS: logZ = -4.94 ± 0.20
- Dynesty: logZ = -4.86 ± 0.18
- Proper agreement achieved

#### Lessons Learned
- Always verify likelihood normalization
- Compare with reference implementations (Dynesty)
- Use standard forms for well-known distributions

#### Related Issues
- Task 007 (Evidence & Boundary Fix)
- Demo 02 validation

---

### Problem #5: Edge Clustering in Posterior Samples
**Date:** 2026-05-04
**Task:** 007 (Evidence & Boundary Fix)
**Severity:** 🟠 MAJOR

#### Description
4.6% of posterior samples clustered on edge of prior bounds

#### Symptoms
- Many points at exactly x = ±5.0 (prior boundary)
- Should not see points outside [-5, 5] range
- Visual inspection shows edge clustering

#### Investigation
- Boundary handling used clipping (jnp.clip)
- Clipping doesn't properly enforce uniform prior
- Creates artificial concentration at boundaries
- Dynesty uses rejection, not clipping

#### Solution
Replaced clipping with rejection (matches Dynesty):
```python
# BEFORE (clipping - wrong)
x_proposed = jnp.clip(x_proposed, 0.0, 1.0)

# AFTER (rejection - correct)
in_unit_cube = jnp.all((x_proposed >= 0.0) & (x_proposed <= 1.0))
logL_proposed = jnp.where(
    in_unit_cube,
    loglikelihood_for_jit(x_proposed),
    -jnp.inf  # Reject out-of-bounds proposals
)
accept = in_unit_cube & constraint_satisfied & metropolis_accept
```

#### Results
- Edge clustering eliminated: 0/200000 points outside bounds
- Sample range: Min=-4.996, Max=4.999 (correct uniform distribution)
- Proper uniform sampling maintained

#### Lessons Learned
- Use rejection, not clipping, for boundary enforcement
- Match reference implementation (Dynesty) methodology
- Verify sample distribution visually

#### Related Issues
- Task 007 (Evidence & Boundary Fix)
- Boundary handling in while_loop_sampler.py

---

## Moderate Problems

### Problem #6: Fixed Iteration Count Limitation
**Date:** 2026-05-03
**Task:** 004 (Termination Investigation)
**Severity:** 🟡 MODERATE

#### Description
Original implementation used fixed iteration count with no automatic stopping

#### Symptoms
- Runs all max_iterations even when converged
- Wasteful computation
- No convergence metric
- Difficult to know when to stop

#### Investigation
- Used lax.scan with fixed range
- Needed conditional iteration but maintaining JIT compilation
- Python loops 1000x slower (1.8s vs 1800s for 1000 iterations)

#### Solution
Replaced lax.scan with lax.while_loop:
- Iteration-granular convergence checking
- delta_logZ convergence criterion (like Dynesty)
- Early termination when converged
- Maintains JIT compilation

#### Results
- Automatic stopping at convergence
- No wasted iterations
- GPU speed maintained
- Proper delta_logZ < 0.01 convergence

#### Lessons Learned
- Use lax.while_loop for conditional iteration in JAX
- Dynesty's delta_logZ is robust convergence criterion
- Can have early termination without sacrificing performance

#### Related Issues
- Task 004 (Termination Investigation)
- Task 005 (Adaptive Termination)

---

### Problem #7: Import Path Issues
**Date:** 2026-05-03
**Task:** 002 (Milestone 2)
**Severity:** 🟡 MODERATE

#### Description
Difficulty importing modules from previous task's scripts

#### Symptoms
- Import errors when integrating ellipsoid code
- Relative imports not working
- PYTHONPATH issues

#### Investigation
- Task folders have relative structure
- Scripts need to import from previous tasks
- Python path not set up correctly

#### Solution
- Used absolute paths and PYTHONPATH setup
- Function-level imports for clean integration
- Direct imports for cleaner code

#### Results
- Clean integration across tasks
- No import errors
- Reusable code structure

#### Lessons Learned
- Set up PYTHONPATH early in project
- Use absolute imports for cross-task integration
- Function-level imports can resolve circular dependencies

#### Related Issues
- Task 002 (Milestone 2)

---

### Problem #8: JAX API Differences
**Date:** 2026-05-03
**Task:** 002 (Milestone 2)
**Severity:** 🟡 MODERATE

#### Description
JAX has different API than NumPy in some cases

#### Symptoms
- `jnp.randint` doesn't exist in JAX
- Code breaks when移植ing from NumPy
- Function signature differences

#### Investigation
- JAX random module separate from numpy
- Different function names/signatures
- Need to use JAX-specific functions

#### Solution
- Used `random.randint` from JAX random module
- Check JAX documentation for correct functions
- Write JAX-compatible code from start

#### Results
- No JAX API errors
- Proper random number generation
- Clean JAX code

#### Lessons Learned
- JAX random is separate from numpy.random
- Always check JAX docs, don't assume NumPy compatibility
- Test JAX-specific functions early

#### Related Issues
- Task 002 (Milestone 2)

---

## Minor Problems

### Problem #9: Variable Naming Conflicts
**Date:** 2026-05-03
**Task:** 002 (Milestone 2)
**Severity:** 🟢 MINOR

#### Description
Variable `ellipsoid` used as both module and variable name

#### Symptoms
- Confusion in code
- Potential shadowing issues
- Code harder to read

#### Solution
- Renamed variable to `fitted_ellipsoid`
- Direct function imports for clarity
- Consistent naming conventions

#### Results
- No naming conflicts
- Clearer code
- Easier to maintain

#### Lessons Learned
- Avoid variable names that match module names
- Use descriptive names (fitted_ellipsoid vs ellipsoid)
- Direct imports can help avoid conflicts

#### Related Issues
- Task 002 (Milestone 2)

---

### Problem #10: Single Ellipsoid Failure on Multi-Modal Problems
**Date:** 2026-05-04
**Task:** 006 (Dynesty Strategy)
**Severity:** 🟢 MINOR

#### Description
Single ellipsoid bound doesn't work for multi-modal distributions

#### Symptoms
- Single ellipsoid centered between modes
- Poor sampling of modes
- Wrong logZ results
- Much slower than Gaussian proposals

#### Investigation
- Single ellipsoid cannot capture multi-modal structure
- Centered between modes → poor proposals
- Works well for uni-modal only

#### Solution
- Disabled ellipsoid by default
- Available for users to enable for uni-modal problems
- Gaussian proposals work better for multi-modal

#### Results
- Gaussian proposals work well for multi-modal
- Ellipsoid available for uni-modal problems
- No performance regression

#### Lessons Learned
- Not all proposal methods work for all problem types
- Provide options, use sensible defaults
- Test on diverse problem types

#### Related Issues
- Task 006 (Dynesty Strategy)
- Ellipsoid bounds implementation

---

## Problem Statistics

### by Severity
- 🔴 CRITICAL: 2 (Problems #1, #2)
- 🟠 MAJOR: 3 (Problems #3, #4, #5)
- 🟡 MODERATE: 3 (Problems #6, #7, #8)
- 🟢 MINOR: 2 (Problems #9, #10)

### by Category
- **Performance:** 2 (#1, #2)
- **Numerical:** 2 (#3, #4)
- **Methodology:** 3 (#5, #6, #10)
- **Infrastructure:** 3 (#7, #8, #9)

### Resolution Status
- ✅ **All Resolved** (10/10)
- 🟢 **0 Open Issues**

---

## Prevention Guidelines

### Performance Issues
1. Always profile early (measure per-iteration time)
2. Use JAX control flow in hot loops (lax.scan, lax.while_loop)
3. Avoid Python data structures in JIT-compiled code
4. Pre-allocate arrays, use in-place updates

### Numerical Issues
1. Use log-space arithmetic for evidence calculations
2. Verify likelihood normalization factors
3. Check that information H is positive
4. Use logsubexp/log1p for stability

### Methodology Issues
1. Match reference implementation (Dynesty) methodology
2. Use rejection, not clipping, for boundary enforcement
3. Verify sample distribution visually
4. Test on diverse problem types

### Dimensionality Scaling
1. Scale proposals as 1/√ndim
2. Scale walk steps as ndim + constant
3. Test acceptance rate when scaling dimensions
4. Use dimension-aware defaults

---

**Last Updated:** 2026-05-05
**Total Problems:** 10
**Resolved:** 10
**Open:** 0
**Status:** ✅ **All Issues Resolved**
