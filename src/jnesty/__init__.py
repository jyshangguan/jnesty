"""
JNesty: GPU-accelerated nested sampling with JAX.

A nested sampling package with a Dynesty-like API, pluggable
samplers and bounding methods, and JAX GPU acceleration.

Example:
    >>> from jnesty import NestedSampler
    >>> sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
    >>> sampler.run_nested()
    >>> print(f"logZ = {sampler.results['logz']:.4f}")
"""

from .jnesty import NestedSampler
from .sampler import run_nested_sampling, WhileLoopNSConfig, WhileLoopNSResult
from .results import Results, save_results, load_results
from . import plotting

# Plotting functions
runplot = plotting.runplot
traceplot = plotting.traceplot
cornerplot = plotting.cornerplot
cornerpoints = plotting.cornerpoints
diagnostics = plotting.diagnostics

__all__ = [
    # Main API
    'NestedSampler',
    'Results',
    'save_results',
    'load_results',

    # Low-level
    'run_nested_sampling',
    'WhileLoopNSConfig',
    'WhileLoopNSResult',

    # Plotting
    'runplot',
    'traceplot',
    'cornerplot',
    'cornerpoints',
    'diagnostics',
    'plotting',
]
