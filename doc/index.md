# JNesty Documentation

GPU-accelerated nested sampling with JAX, designed for expensive likelihood
functions. JNesty parallelizes likelihood evaluations across GPU cores using
JAX's `vmap`, making it well-suited for problems where each likelihood call is
computationally costly (forward models, simulations, large-data inferences).

Two sampling engines are available: the Dynesty-style static/dynamic nested
samplers, and our JAX port of the Nautilus importance nested sampler
(`NautilusSampler`, optional `jnesty[nautilus]` dependencies).

The API follows [dynesty](https://dynesty.readthedocs.io/), so switching is
straightforward.

```{toctree}
:maxdepth: 2

methods
examples
api
```

## Installation

```bash
pip install -e .
```

This installs JNesty with JAX CUDA 12 GPU support. You need a CUDA-compatible
GPU and driver. For CPU-only use, set `device='cpu'` when constructing the
sampler.

Optional dependencies for dynesty comparison and plotting interop:

```bash
pip install -e ".[dynesty]"
```

Optional dependencies for the JAX Nautilus importance sampler:

```bash
pip install -e ".[nautilus]"
```

## Quick Start

```python
import jax.numpy as jnp
from jnesty import NestedSampler, plotting

def loglikelihood(x):
    return -0.5 * jnp.sum(x**2)

def prior_transform(u):
    return 20.0 * u - 10.0  # uniform [-10, 10] in each dim

sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
sampler.run_nested(max_iterations=50000, delta_logZ_threshold=0.01)

r = sampler.results
print(f"logZ = {r['logz']:.4f} +/- {r['logzerr']:.4f}")
print(f"Converged: {r['converged']}")

# Built-in plots
fig, axes = plotting.runplot(r)
fig, axes = plotting.traceplot(r)
fig, axes = plotting.cornerplot(r)
```

## Next Steps

- {doc}`methods` — How the sampling and bounding algorithms work
- {doc}`examples` — Five worked examples (multi-modal, Rosenbrock, high-D, shells, dynamic)
- {doc}`api` — Full API reference
