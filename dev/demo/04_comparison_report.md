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
queue_size: 8 (auto for bound='multi')
bound_update_interval: auto (default)
```

### Numerical Results
```
logZ:           -1.8256 ± 0.0732
H:               2.6762
delta_logZ:      0.009990
Converged:       True
Iterations:      3905
Runtime:         ~16s
Acceptance rate: 0.4998
```

### Visualizations

#### Run Plot - Evidence Evolution
![JNesty Run Plot](output_04_jnesty/run_plot.png)

**Observations:**
- Evidence converges smoothly
- delta_logZ decreases monotonically
- Convergence at iteration 3905

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
Bounding: multi
Sample: rwalk
```

### Numerical Results
```
logZ:        -1.7419 ± 0.0747
H:            2.6336
delta_logZ:   0.0
Converged:    True
Iterations:   4372
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
| **JNesty** | -1.8256 | ±0.0732 | 2.676 | 3905 | ~16s | 0.4998 |
| **Dynesty** | -1.7419 | ±0.0747 | 2.634 | 4372 | ~2.5s | — |

### Accuracy Analysis

**LogZ Agreement:**
- Difference: 0.0837
- Combined uncertainty: sqrt(0.0732^2 + 0.0747^2) ≈ ±0.1046
- **Status: AGREEMENT** (within combined uncertainty)

**Analysis:**
- The 0.0837 logZ difference is well within the combined uncertainty of ±0.1046
- H values are consistent (2.676 vs 2.634)
- Both implementations correctly identify and sample both shells
- Results agree within Monte Carlo uncertainty

### Performance Analysis

| Aspect | JNesty | Dynesty |
|--------|--------|---------|
| **Iterations** | 3905 (11% fewer) | 4372 |
| **Runtime** | ~16s | ~2.5s |
| **Convergence** | delta_logZ=0.010 | delta_logZ=0.0 |

**Analysis:**
- JNesty uses fewer iterations (3905 vs 4372) but higher per-iteration cost
- Dynesty is significantly faster for this 2D problem with cheap likelihoods
- Both correctly handle the thin shell structures
- The shell problem is well-suited for testing degenerate posterior shapes

---

## Key Takeaways

1. **Accuracy:** LogZ difference of 0.0837 is well within the combined uncertainty of ±0.1046
2. **H values:** Consistent between implementations (2.676 vs 2.634)
3. **Shell structure:** Both correctly sample the thin ring-shaped posteriors
4. **Convergence:** Both achieve proper convergence with similar iteration counts

---

## Conclusion

Both JNesty and Dynesty successfully solve the Gaussian shells problem. The logZ difference of 0.0837 is well within the combined uncertainty of ±0.1046, confirming good agreement. Both correctly identify and sample the thin shell structures.

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

**Date:** 2026-05-10
**Problem:** Gaussian Shells (2D)
**Status:** Complete - Good agreement with Dynesty
