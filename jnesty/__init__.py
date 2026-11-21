"""
JNesty: GPU-accelerated nested sampling with JAX.

A nested sampling package with a Dynesty-like API, pluggable
samplers and bounding methods, and JAX GPU acceleration. Three sampling
engines are available:

* Dynesty-style nested sampling (:class:`NestedSampler`,
  :class:`DynamicNestedSampler`) with random-walk proposals and
  ellipsoidal bounds.
* Our JAX port of the Nautilus importance nested sampler
  (:class:`NautilusSampler`); needs the optional ``jnesty[nautilus]``
  dependencies.
* Our JAX port of the CDNest diffusive nested sampler
  (:class:`DNestSampler`); JAX-only, no extra dependencies.

Example:
    >>> from jnesty import NestedSampler
    >>> sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
    >>> sampler.run_nested()
    >>> print(f"logZ = {sampler.results['logz']:.4f}")
"""

__version__ = "0.2.0"

import os as _os
# JAX routes its C++-side log noise through absl; hide INFO/WARNING by
# default. setdefault: a value exported by the user wins, so debugging
# logs come back with e.g.  ABSL_MIN_LOG_LEVEL=0 python script.py
_os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '3')
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
del _os

from .dynesty_sampler import (NestedSampler, DynamicNestedSampler,
                               DEFAULT_QUEUE_SIZE, LEGACY_STEPS_PER_WALK,
                               auto_queue_size)
from .nautilus_sampler import NautilusSampler
from .dnest_sampler import DNestSampler
from .dynesty.core import (run_nested_sampling, WhileLoopNSConfig,
                           WhileLoopNSResult)
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
    'DNestSampler',
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
