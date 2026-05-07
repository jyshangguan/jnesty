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
logZ:           -1.7532 ± 0.0728
H:               2.6482
delta_logZ:      0.009990
Converged:       True
Iterations:      3869
Runtime:         ~16s
Acceptance rate: 0.4981
```

### Visualizations

#### Run Plot - Evidence Evolution
![JNesty Run Plot](output_04_jnesty/run_plot.png)

**Observations:**
- Evidence converges smoothly
- delta_logZ decreases monotonically
- Convergence at iteration 3869

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
logZ:        -1.8891 ± 0.0768
H:            2.7789
delta_logZ:   0.0
Converged:    True
Iterations:   4446
Runtime:      ~2.5s
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
| **JNesty** | -1.7532 | ±0.0728 | 2.648 | 3869 | ~16s | 0.4981 |
| **Dynesty** | -1.8891 | ±0.0768 | 2.779 | 4446 | ~2.5s | — |

### Accuracy Analysis

**LogZ Agreement:**
- Difference: 0.1359
- Combined uncertainty: sqrt(0.0728^2 + 0.0768^2) ≈ ±0.1059
- **Status: OUTSIDE** (marginally outside combined uncertainty)

**Analysis:**
- The 0.1359 logZ difference slightly exceeds the combined uncertainty of ±0.1059
- H values differ moderately (2.648 vs 2.779)
- Both implementations correctly identify and sample both shells
- The marginal discrepancy may reflect differences in bounding method (multi-ellipsoid vs none) affecting shell exploration efficiency

### Performance Analysis

| Aspect | JNesty | Dynesty |
|--------|--------|---------|
| **Iterations** | 3869 (13% fewer) | 4446 |
| **Runtime** | ~16s | ~2.5s |
| **Convergence** | delta_logZ=0.010 | delta_logZ=0.0 |

**Analysis:**
- JNesty uses fewer iterations (3869 vs 4446) but higher per-iteration cost
- Dynesty is significantly faster for this 2D problem with cheap likelihoods
- Both correctly handle the thin shell structures
- The shell problem is well-suited for testing degenerate posterior shapes

---

## Key Takeaways

1. **Accuracy:** LogZ difference of 0.1359 marginally exceeds the combined uncertainty of ±0.1059
2. **H values:** Moderate discrepancy (2.648 vs 2.779), both reasonable for this problem
3. **Shell structure:** Both correctly sample the thin ring-shaped posteriors
4. **Convergence:** Both achieve proper convergence with similar iteration counts

---

## Conclusion

Both JNesty and Dynesty successfully solve the Gaussian shells problem, though the logZ difference of 0.1359 is marginally outside the combined uncertainty of ±0.1059. Both correctly identify and sample the thin shell structures. The discrepancy may stem from differences in bounding methods or run-to-run statistical fluctuation in this challenging degenerate geometry.

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
**Status:** Complete - Marginal discrepancy with Dynesty
