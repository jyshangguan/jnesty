"""
JNesty's dynesty-style nested-sampling engine.

This subpackage holds the internals of the dynesty-style sampler:

* :mod:`jnesty.dynesty.core` — the core nested-sampling while loop
  (live points, evidence accumulation, convergence checking).
* :mod:`jnesty.dynesty.internal_samplers` — proposal strategies for
  generating replacement live points.
* :mod:`jnesty.dynesty.bounding` — bounding methods (unit cube, single
  ellipsoid, multi-ellipsoid decomposition).
* :mod:`jnesty.dynesty.multi_ellipsoid` — the multi-ellipsoid volume /
  sampling math.
* :mod:`jnesty.dynesty._dynamic` — dynamic nested sampling math (weight
  functions, stopping criteria, batch combining).

It is the structural peer of :mod:`jnesty.nautilus`, the other engine:
each sampler family is one public top-level adapter plus one engine
subpackage. The public classes live in :mod:`jnesty.dynesty_sampler`, and
shared results/plotting live in :mod:`jnesty.results` / :mod:`jnesty.plotting`.

This subpackage deliberately does not import the public adapter or the
shared results module: it is a lower layer, so importing it never creates a
cycle.

The external ``dynesty`` package is a different, optional dependency and is
not shadowed: ``import dynesty`` still resolves upstream.
"""

from .core import (
    run_nested_sampling,
    WhileLoopNSConfig,
    WhileLoopNSResult,
)
from .bounding import BOUND_REGISTRY, get_bound
from .internal_samplers import SAMPLER_REGISTRY, get_sampler

__all__ = [
    # Core nested-sampling loop
    'run_nested_sampling',
    'WhileLoopNSConfig',
    'WhileLoopNSResult',

    # Bounding methods
    'BOUND_REGISTRY',
    'get_bound',

    # Proposal strategies
    'SAMPLER_REGISTRY',
    'get_sampler',
]