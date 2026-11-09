"""
JNesty static nested sampler API (compatibility module).

The implementation of :class:`NestedSampler` moved to
:mod:`jnesty.dynesty_sampler`, which now holds both Dynesty-style public
classes. This module re-exports the static API so historical imports such
as ``from jnesty.jnesty import NestedSampler`` keep working.
"""

from .dynesty_sampler import (  # noqa: F401
    DEFAULT_QUEUE_SIZE,
    LEGACY_STEPS_PER_WALK,
    NestedSampler,
    auto_queue_size,
)

__all__ = [
    'NestedSampler',
    'DEFAULT_QUEUE_SIZE',
    'LEGACY_STEPS_PER_WALK',
    'auto_queue_size',
]
