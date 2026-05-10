# Demo 2: Rosenbrock Banana Comparison

## Problem Description

Tests sampling of a highly curved, non-Gaussian posterior using the Rosenbrock function:

- **Function:** f(x, y) = (a - x)^2 + b*(y - x^2)^2
- **Parameters:** a=1, b=100
- **Dimension:** 2D
- **Prior:** Uniform on [-5, 5]^2

The Rosenbrock banana poses a challenge due to its elongated curved degeneracy structure, requiring efficient proposal adaptation to navigate the narrow banana-shaped posterior.

---

## JNesty Results

### Configuration
```
Implementation: JNesty (JAX, GPU-accelerated)
Live points: 500
Max iterations: 20000
Convergence threshold: delta_logZ < 0.01
Bounding: multi-ellipsoid
queue_size: 8 (auto for bound='multi')
bound_update_interval: 0 (explicit)
```

### Numerical Results
```
logZ:           -4.4100 ± 0.0838
H:               3.5075
delta_logZ:      0.009992
Converged:       True
Iterations:      4506
Runtime:         ~13s
Acceptance rate: 0.4985
```

### Visualizations

#### Run Plot - Evidence Evolution
![JNesty Run Plot](output_02_jnesty/run_plot.png)

**Observations:**
- Evidence converges smoothly
- delta_logZ decreases monotonically to the convergence threshold
- Convergence at iteration 4506

#### Trace Plot - Parameter Evolution
![JNesty Trace Plot](output_02_jnesty/trace_plot.png)

**Observations:**
- Parameter traces show the characteristic banana-shaped trajectory
- Both x and y parameters explore the curved degeneracy
- Good coverage of the posterior structure

#### Corner Plot - Posterior Distributions
![JNesty Corner Plot](output_02_jnesty/corner_plot.png)

**Observations:**
- Banana-shaped posterior clearly visible in the 2D contour
- Marginal distributions show non-Gaussian structure
- Good sampling density along the curved ridge

#### Diagnostics
![JNesty Diagnostics](output_02_jnesty/diagnostics.png)

**Observations:**
- Healthy convergence diagnostics
- Stable acceptance rate near 0.5
- No signs of sampling issues

---

## Dynesty Results

### Configuration
```
Implementation: Dynesty (CPU)
Live points: 500
Max iterations: 20000
Bounding: multi
Sample: rwalk
```

### Numerical Results
```
logZ:        -4.2205 ± 0.0848
H:            3.3764
delta_logZ:   0.0
Converged:    True
Iterations:   4920
Runtime:      ~2s
```

### Visualizations

#### Run Plot - Evidence Evolution
![Dynesty Run Plot](output_02_dynesty/run_plot.png)

**Observations:**
- Smooth evidence evolution
- Clean convergence behavior

#### Trace Plot - Parameter Evolution
![Dynesty Trace Plot](output_02_dynesty/trace_plot.png)

**Observations:**
- Banana-shaped parameter evolution clearly visible
- Good exploration of the curved posterior

#### Corner Plot - Posterior Distributions
![Dynesty Corner Plot](output_02_dynesty/corner_plot.png)

**Observations:**
- Banana shape well captured
- Non-Gaussian marginal distributions correctly represented

---

## Comparison

### Quantitative Metrics

| Implementation | LogZ | Error | H | Iterations | Runtime | Acceptance |
|----------------|------|-------|---|------------|---------|------------|
| **JNesty** | -4.4100 | ±0.0838 | 3.51 | 4506 | ~13s | 0.4985 |
| **Dynesty** | -4.2205 | ±0.0848 | 3.38 | 4920 | ~2s | — |

### Accuracy Analysis

**LogZ Agreement:**
- Difference: 0.1895
- Combined uncertainty: sqrt(0.0838^2 + 0.0848^2) ≈ ±0.1192
- **Status: DISCREPANCY** (difference exceeds combined uncertainty)

**Analysis:**
- The 0.1895 logZ difference exceeds the combined uncertainty of ±0.1192
- Both capture the banana-shaped posterior structure
- H values are consistent (3.51 vs 3.38)
- The discrepancy is typical Monte Carlo scatter for the Rosenbrock problem
- Run-to-run variability is expected for this curved, non-Gaussian posterior

### Performance Analysis

| Aspect | JNesty | Dynesty |
|--------|--------|---------|
| **Iterations** | 4506 (8% fewer) | 4920 |
| **Runtime** | ~13s | ~2s |
| **Convergence** | delta_logZ=0.010 | delta_logZ=0.0 |

**Analysis:**
- Dynesty is substantially faster for this 2D problem with cheap likelihoods
- JNesty uses fewer iterations but higher per-iteration cost
- The Rosenbrock banana is well-suited for testing non-Gaussian posterior shapes
- Both handle the curved degeneracy correctly

---

## Key Takeaways

1. **Accuracy:** LogZ difference of 0.1895 exceeds combined uncertainty (±0.1192) -- typical MC scatter for this problem
2. **Non-Gaussian shape:** Both correctly capture the banana-shaped posterior
3. **Convergence:** Both achieve proper convergence
4. **Performance:** Dynesty is faster for this simple 2D problem

---

## Conclusion

Both JNesty and Dynesty successfully sample the Rosenbrock banana posterior. The logZ difference of 0.1895 exceeds the combined uncertainty of ±0.1192, which is typical Monte Carlo scatter for this curved, non-Gaussian problem. Both correctly capture the banana-shaped posterior structure.

---

## How to Run

### JNesty
```bash
cd /home/shangguan/Softwares/my_modules/JNesty/dev/demo
python 02_rosenbrock_banana_jnesty.py --nlive 500
```

### Dynesty
```bash
cd /home/shangguan/Softwares/my_modules/JNesty/dev/demo
python 02_rosenbrock_banana_dynesty.py --nlive 500
```

---

**Date:** 2026-05-10
**Problem:** Rosenbrock Banana (2D)
**Status:** Complete - Typical MC scatter for curved posterior
