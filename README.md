# J-Nesty

GPU-accelerated random walk nested sampling with JAX.

A standalone nested sampling package providing a Dynesty-like API with
JAX-based GPU acceleration. Outperforms Dynesty at high dimensions (>50D).

## Quick start

```python
from jnesty import NestedSampler

def loglike(x):
    return -0.5 * sum(x**2)

def prior_transform(u):
    return 20.0 * u - 10.0

sampler = NestedSampler(loglike, prior_transform, ndim=5)
sampler.run_nested()
print(sampler.results)
```

## Installation

```bash
pip install -e .
```

## Performance

| Dimension | J-Nesty (GPU) | Dynesty (CPU) | Ratio |
|-----------|---------------|----------------|-------|
| 2D        | 582 iter/s    | 3731 iter/s   | 0.16x |
| 20D       | 1194 iter/s   | 1353 iter/s   | 0.88x |
| 100D      | 574 iter/s    | 424 iter/s    | **1.35x** |

GPU advantage kicks in above ~50 dimensions.
