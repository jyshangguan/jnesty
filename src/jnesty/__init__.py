"""
J-Nesty: GPU-accelerated random walk nested sampling with JAX.

A standalone nested sampling package with a Dynesty-like API.

Key features:
- GPU-accelerated likelihood evaluations via JAX
- True iteration-granular adaptive termination (stops when converged)
- Multi-ellipsoid bounding with JIT-compiled fitting
- Dynesty-compatible plotting via matplotlib

Example:
    >>> from jnesty import NestedSampler
    >>> sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
    >>> sampler.run_nested()
    >>> print(f"logZ = {sampler.results['logz']:.4f}")
"""

from .while_loop_sampler import (
    run_nested_sampling_while_loop,
    WhileLoopNSConfig,
    WhileLoopNSResult
)
from .api import NestedSampler
from . import plotting

# Re-export with simpler names
run_nested_sampling = run_nested_sampling_while_loop
NSConfig = WhileLoopNSConfig
NSResult = WhileLoopNSResult

# Plotting functions
runplot = plotting.runplot
traceplot = plotting.traceplot
cornerplot = plotting.cornerplot
cornerpoints = plotting.cornerpoints
diagnostics = plotting.diagnostics

__all__ = [
    # Main API
    'NestedSampler',

    # Function-based API
    'run_nested_sampling',
    'NSConfig',
    'NSResult',

    # Internal names
    'run_nested_sampling_while_loop',
    'WhileLoopNSConfig',
    'WhileLoopNSResult',

    # Plotting
    'runplot',
    'traceplot',
    'cornerplot',
    'cornerpoints',
    'diagnostics',
    'plotting'
]
