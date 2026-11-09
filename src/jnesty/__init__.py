"""
JNesty: GPU-accelerated nested sampling with JAX.

A nested sampling package with a Dynesty-like API, pluggable
samplers and bounding methods, and JAX GPU acceleration. Two sampling
engines are available:

* Dynesty-style nested sampling (:class:`NestedSampler`,
  :class:`DynamicNestedSampler`) with random-walk proposals and
  ellipsoidal bounds.
* Our JAX port of the Nautilus importance nested sampler
  (:class:`NautilusSampler`); needs the optional ``jnesty[nautilus]``
  dependencies.

Example:
    >>> from jnesty import NestedSampler
    >>> sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
    >>> sampler.run_nested()
    >>> print(f"logZ = {sampler.results['logz']:.4f}")
"""

__version__ = "0.1.0"

from .dynesty_sampler import (NestedSampler, DynamicNestedSampler,
                               DEFAULT_QUEUE_SIZE, LEGACY_STEPS_PER_WALK,
                               auto_queue_size)
from .nautilus_sampler import NautilusSampler
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
    'DynamicNestedSampler',
    'NautilusSampler',
    'Results',
    'save_results',
    'load_results',

    # Lane (queue) settings
    'DEFAULT_QUEUE_SIZE',
    'LEGACY_STEPS_PER_WALK',
    'auto_queue_size',

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
