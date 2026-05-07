# Demo 1: Multi-modal Gaussian Mixture Comparison

## Problem Description

Tests multi-modality handling with 2 separated Gaussian modes in 5D:

- **Mode 1:** centered at origin, weight 0.6
- **Mode 2:** centered at (3, 3, ...) with weight 0.4
- **Dimension:** 5D
- **Prior:** Uniform on [-10, 10]^5

This problem challenges nested samplers to properly explore both modes and accurately estimate the evidence.

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
logZ:        -6.9489 ± 0.0881
H:            3.8782
delta_logZ:   0.009985
Converged:    True
Iterations:   5510
Runtime:      55.49s
Speed:        99.30 iterations/sec
```

### Visualizations

#### Run Plot - Evidence Evolution
![JNesty Run Plot](output_01_jnesty/run_plot.png)

**Observations:**
- Evidence (logZ) converges smoothly with decreasing delta_logZ
- Convergence achieved at iteration 5510 with delta_logZ < 0.01
- Log-likelihood increases monotonically as expected

#### Trace Plot - Parameter Evolution
![JNesty Trace Plot](output_01_jnesty/trace_plot.png)

**Observations:**
- Parameter evolution shows exploration of both modes
- Two clusters visible in parameter traces corresponding to the two Gaussian modes
- Proper coverage of the posterior

#### Corner Plot - Posterior Distributions
![JNesty Corner Plot](output_01_jnesty/corner_plot.png)

**Observations:**
- Both modes clearly identified with correct spatial locations
- Proper relative weights between modes
- Good coverage of each mode's posterior structure

#### Diagnostics
![JNesty Diagnostics](output_01_jnesty/diagnostics.png)

**Observations:**
- Convergence metrics look healthy
- Acceptance rate stable near 0.5
- No pathological behavior detected

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
logZ:        -6.9985 ± 0.0914
H:            3.8689
delta_logZ:   0.0
Converged:    True
Iterations:   6038
Runtime:      3.66s
```

### Visualizations

#### Run Plot - Evidence Evolution
![Dynesty Run Plot](output_01_dynesty/run_plot.png)

**Observations:**
- Smooth evidence evolution
- Good convergence behavior

#### Trace Plot - Parameter Evolution
![Dynesty Trace Plot](output_01_dynesty/trace_plot.png)

**Observations:**
- Both modes visible in parameter traces
- Proper exploration of the parameter space

#### Corner Plot - Posterior Distributions
![Dynesty Corner Plot](output_01_dynesty/corner_plot.png)

**Observations:**
- Both modes clearly identified
- Good posterior coverage

---

## Comparison

### Quantitative Metrics

| Implementation | LogZ | Error | H | Iterations | Runtime | Speed (iter/s) |
|----------------|------|-------|---|------------|---------|----------------|
| **JNesty** | -6.9489 | ±0.0881 | 3.88 | 5510 | 55.49s | 99.3 |
| **Dynesty** | -6.9985 | ±0.0914 | 3.87 | 6038 | 3.66s | — |

### Accuracy Analysis

**LogZ Agreement:**
- Difference: 0.050
- Combined uncertainty: sqrt(0.0881^2 + 0.0914^2) ≈ ±0.127
- **Status: EXCELLENT AGREEMENT** (well within combined uncertainty)

**Analysis:**
- Results agree within Monte Carlo uncertainty
- Both implementations correctly identify bi-modality
- H values match closely (3.88 vs 3.87)
- Both achieve convergence (delta_logZ < 0.01)

### Performance Analysis

| Aspect | JNesty | Dynesty |
|--------|--------|---------|
| **Iterations** | 5510 (9% fewer) | 6038 |
| **Runtime** | 55.49s | 3.66s |
| **Convergence** | delta_logZ=0.010 | delta_logZ=0.0 |

**Analysis:**
- Dynesty is significantly faster in wall-clock time for this 5D problem
- JNesty requires fewer iterations but each iteration is more expensive due to JAX overhead
- For low-dimensional problems with cheap likelihoods, CPU-based samplers have a speed advantage
- GPU acceleration benefit will appear for higher-dimensional problems or expensive likelihoods

---

## Key Takeaways

1. **Accuracy:** Both implementations produce consistent logZ estimates within Monte Carlo uncertainty
2. **Multi-modality:** Both correctly identify and sample both Gaussian modes
3. **Convergence:** Both achieve proper convergence with delta_logZ < 0.01
4. **Performance:** Dynesty is faster for this simple 5D problem; JNesty's GPU advantage is not yet relevant

---

## Conclusion

Both JNesty and Dynesty successfully solve this multi-modal Gaussian mixture problem. The logZ estimates agree to within 0.050, well within the combined uncertainty of ±0.127. This validates JNesty's correctness for multi-modal problems.

---

## How to Run

### JNesty
```bash
cd /home/shangguan/Softwares/my_modules/JNesty/dev/demo
python 01_multimodal_gaussian_mixture_jnesty.py --nlive 500
```

### Dynesty
```bash
cd /home/shangguan/Softwares/my_modules/JNesty/dev/demo
python 01_multimodal_gaussian_mixture_dynesty.py --nlive 500
```

---

**Date:** 2026-05-07
**Problem:** Multi-modal Gaussian Mixture (5D)
**Status:** Complete - Excellent agreement with Dynesty
