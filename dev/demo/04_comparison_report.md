# Demo 4: Gaussian Shells Comparison

## Problem Description

Tests sampling of thin, ring-shaped (shell) posteriors separated in parameter space:

- **Shell 1:** radius=2.0, width=0.1, center=(-3.5, 0.0)
- **Shell 2:** radius=2.0, width=0.1, center=(3.5, 0.0)
- **Dimension:** 2D
- **Prior:** Uniform on [-10, 10]^2

Gaussian shells are a challenging nested sampling test because the posterior concentrates on two thin rings rather than solid regions, requiring efficient exploration of degenerate, curving structures.

---

## JNesty Results

### Configuration
```
Implementation: JNesty (JAX, GPU-accelerated)
Live points: 500
Max iterations: 20000
Convergence threshold: delta_logZ < 0.01
Bounding: multi-ellipsoid
```

### Numerical Results
```
logZ:           -1.7823 ± 0.0729
H:               2.6570
delta_logZ:      0.009981
Converged:       True
Iterations:      3884
Runtime:         45.81s
Speed:           84.78 iterations/sec
Acceptance rate: 0.498
```

### Visualizations

#### Run Plot - Evidence Evolution
![JNesty Run Plot](output_04_jnesty/run_plot.png)

**Observations:**
- Evidence converges smoothly
- delta_logZ decreases monotonically
- Convergence at iteration 3884

#### Trace Plot - Parameter Evolution
![JNesty Trace Plot](output_04_jnesty/trace_plot.png)

**Observations:**
- Parameters trace out the shell structures
- Both shells explored by the sampler
- Good coverage of the thin ring structures

#### Posterior 2D
![JNesty Posterior](output_04_jnesty/posterior_2d.png)

**Observations:**
- Two distinct ring-shaped shells clearly visible
- Correct shell radii and centers
- Good sampling density on both shells

---

## Dynesty Results

### Configuration
```
Implementation: Dynesty (CPU)
Live points: 500
Max iterations: 20000
Bounding: none
Sample: rwalk
```

### Numerical Results
```
logZ:        -1.7623 ± 0.0751
H:            2.6566
delta_logZ:   0.0
Converged:    True
Iterations:   4382
Runtime:      2.63s
```

### Visualizations

#### Run Plot - Evidence Evolution
![Dynesty Run Plot](output_04_dynesty/run_plot.png)

**Observations:**
- Smooth evidence evolution
- Clean convergence to the final logZ value

#### Trace Plot - Parameter Evolution
![Dynesty Trace Plot](output_04_dynesty/trace_plot.png)

**Observations:**
- Both shells visible in parameter traces
- Good exploration of the ring structures

#### Corner Plot - Posterior Distributions
![Dynesty Corner Plot](output_04_dynesty/corner_plot.png)

**Observations:**
- Bimodal marginal distributions reflecting the two shells
- Correct spatial structure

#### Posterior 2D
![Dynesty Posterior](output_04_dynesty/posterior_2d.png)

**Observations:**
- Two rings clearly visible
- Good sampling of both shells

---

## Comparison

### Quantitative Metrics

| Implementation | LogZ | Error | H | Iterations | Runtime | Acceptance |
|----------------|------|-------|---|------------|---------|------------|
| **JNesty** | -1.7823 | ±0.0729 | 2.657 | 3884 | 45.81s | 0.498 |
| **Dynesty** | -1.7623 | ±0.0751 | 2.657 | 4382 | 2.63s | — |

### Accuracy Analysis

**LogZ Agreement:**
- Difference: 0.020
- Combined uncertainty: sqrt(0.0729^2 + 0.0751^2) ≈ ±0.105
- **Status: EXCELLENT AGREEMENT** (well within combined uncertainty)

**Analysis:**
- The 0.020 logZ difference is very small relative to the uncertainties
- H values match almost exactly (2.657 vs 2.657)
- Both implementations correctly identify and sample both shells
- This is the closest agreement of all four test problems

### Performance Analysis

| Aspect | JNesty | Dynesty |
|--------|--------|---------|
| **Iterations** | 3884 (11% fewer) | 4382 |
| **Runtime** | 45.81s | 2.63s |
| **Convergence** | delta_logZ=0.010 | delta_logZ=0.0 |

**Analysis:**
- JNesty uses fewer iterations (3884 vs 4382) but higher per-iteration cost
- Dynesty is significantly faster for this 2D problem with cheap likelihoods
- Both correctly handle the thin shell structures
- The shell problem is well-suited for testing degenerate posterior shapes

---

## Key Takeaways

1. **Accuracy:** Excellent logZ agreement (diff=0.020, well within ±0.105 combined uncertainty)
2. **H agreement:** Nearly identical H values (2.657 for both)
3. **Shell structure:** Both correctly sample the thin ring-shaped posteriors
4. **Convergence:** Both achieve proper convergence with similar iteration counts

---

## Conclusion

Both JNesty and Dynesty successfully solve the Gaussian shells problem with excellent agreement. The logZ difference of 0.020 is the smallest of all four test problems, and the H values match almost exactly. This validates JNesty's ability to handle thin, degenerate posterior structures.

---

## How to Run

### JNesty
```bash
cd /home/shangguan/Softwares/my_modules/JNesty/dev/demo
python 04_gaussian_shells_jnesty.py --nlive 500
```

### Dynesty
```bash
cd /home/shangguan/Softwares/my_modules/JNesty/dev/demo
python 04_gaussian_shells_dynesty.py --nlive 500
```

---

**Date:** 2026-05-07
**Problem:** Gaussian Shells (2D)
**Status:** Complete - Excellent agreement with Dynesty
