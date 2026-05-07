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
logZ:           -6.8765 ± 0.0877
H:               3.8429
delta_logZ:      0.009981
Converged:       True
Iterations:      5475
Runtime:         ~16s
Acceptance rate: 0.4975
```

### Visualizations

#### Run Plot - Evidence Evolution
![JNesty Run Plot](output_01_jnesty/run_plot.png)

**Observations:**
- Evidence (logZ) converges smoothly with decreasing delta_logZ
- Convergence achieved at iteration 5475 with delta_logZ < 0.01
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
logZ:        -6.8500 ± 0.0901
H:            3.7601
delta_logZ:   0.0
Converged:    True
Iterations:   5971
Runtime:      ~3s
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

| Implementation | LogZ | Error | H | Iterations | Runtime | Acceptance |
|----------------|------|-------|---|------------|---------|------------|
| **JNesty** | -6.8765 | ±0.0877 | 3.84 | 5475 | ~16s | 0.4975 |
| **Dynesty** | -6.8500 | ±0.0901 | 3.76 | 5971 | ~3s | — |

### Accuracy Analysis

**LogZ Agreement:**
- Difference: 0.0265
- Combined uncertainty: sqrt(0.0877^2 + 0.0901^2) ≈ ±0.1266
- **Status: AGREEMENT** (well within combined uncertainty)

**Analysis:**
- Results agree within Monte Carlo uncertainty
- Both implementations correctly identify bi-modality
- H values are consistent (3.84 vs 3.76)
- Both achieve convergence (delta_logZ < 0.01)

### Performance Analysis

| Aspect | JNesty | Dynesty |
|--------|--------|---------|
| **Iterations** | 5475 (8% fewer) | 5971 |
| **Runtime** | ~16s | ~3s |
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

Both JNesty and Dynesty successfully solve this multi-modal Gaussian mixture problem. The logZ estimates agree to within 0.0265, well within the combined uncertainty of ±0.1266. This validates JNesty's correctness for multi-modal problems.

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
**Status:** Complete - Good agreement with Dynesty
