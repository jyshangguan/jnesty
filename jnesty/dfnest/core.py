"""Compiled programs and callable probing for the DNest engine.

This module builds the two JAX programs the engine needs:

* :func:`make_block_fn` -- one compiled *block*: ``num_lanes`` independent
  workers each advance ``thread_steps`` MCMC updates over their local
  particles (``vmap`` over lanes, ``lax.scan`` over the sequential steps),
  with the level arrays, level count, "levels done" flag, the broadcast
  global counters and the likelihood context supplied as traced arguments.
  Every step performs exactly one likelihood evaluation, so a block costs
  ``num_lanes * thread_steps`` evaluations.
* :func:`make_init_fn` -- prior initialisation of all particles (one
  batched likelihood evaluation).

The likelihood context is always passed as a *traced argument* (``None``
when absent), never closed over, so the compiled kernels are keyed on the
context's pytree structure rather than its values and are reused across
datasets.
"""

import jax
import jax.numpy as jnp
from jax import random

from ..utils import call_with_context
from . import proposals
from .state import LaneState

__all__ = ['make_logl_of_u', 'probe_callables', 'make_block_fn',
           'make_init_fn', 'make_to_physical']


def make_logl_of_u(prior_transform, loglikelihood):
    """Compose the user callables into one ``u -> logL`` function."""

    def logl_of_u(u, context):
        x = call_with_context(prior_transform, u, context)
        return call_with_context(loglikelihood, x, context)

    return logl_of_u


def probe_callables(prior_transform, loglikelihood, ndim, context):
    """Trace the user callables once to validate their signatures.

    The prior transform must map a ``(ndim,)`` unit-cube vector to a
    ``(ndim,)`` parameter vector and the likelihood must return one scalar
    per point (blobs/tuple returns are unsupported).  Raises ``TypeError``
    for non-traceable callables and ``ValueError`` for wrong shapes.
    """
    dummy_u = jax.ShapeDtypeStruct((ndim,), jnp.float64)
    try:
        x_struct = jax.eval_shape(
            lambda u: call_with_context(prior_transform, u, context), dummy_u)
    except Exception as exc:
        raise TypeError(
            f'prior_transform is not JAX-traceable: {exc}') from exc
    leaves, _ = jax.tree_util.tree_flatten(x_struct)
    if len(leaves) != 1 or leaves[0].shape != (ndim,):
        raise ValueError(
            f'prior_transform must map a ({ndim},) unit-cube vector to a '
            f'({ndim},) parameter vector')

    try:
        out = jax.eval_shape(
            lambda x: call_with_context(loglikelihood, x, context), x_struct)
    except Exception as exc:
        raise TypeError(
            f'loglikelihood is not JAX-traceable: {exc}') from exc
    leaves, _ = jax.tree_util.tree_flatten(out)
    if len(leaves) != 1 or leaves[0].shape != ():
        raise ValueError(
            'loglikelihood must return a scalar log-likelihood per point '
            '(the DNest engine does not support likelihood blobs)')


def make_block_fn(logl_of_u, config, device=None):
    """Build the jitted one-block transition program.

    Parameters
    ----------
    logl_of_u : callable
        ``logl_of_u(u, context) -> scalar`` from :func:`make_logl_of_u`.
    config : DNestConfig
    device : jax.Device or None

    Returns
    -------
    block_fn : callable
        ``block_fn(state, keys, level_logl, level_logX, push, n_levels,
        levels_done, g_accepts, g_tries, g_visits, g_exceeds, context)
        -> (final_state, ys)`` where ``state``/``keys`` carry a leading
        lane axis and ``ys`` holds the per-step above-threshold records.
    """
    num_lanes = config.num_lanes
    capacity = config.level_capacity
    lane_step = proposals.make_lane_step(
        config.num_particles, config.sample_frequency, config.beta,
        logl_of_u)

    def block_fn(state, keys, level_logl, level_logX, push, n_levels,
                 levels_done, g_accepts, g_tries, g_visits, g_exceeds,
                 context):
        # Broadcast the global level counters to every lane: the JAX
        # analogue of CDNest's MPI_Bcast of the combined levels array.
        state = state._replace(
            accepts=jnp.broadcast_to(g_accepts, (num_lanes, capacity)),
            tries=jnp.broadcast_to(g_tries, (num_lanes, capacity)),
            visits=jnp.broadcast_to(g_visits, (num_lanes, capacity)),
            exceeds=jnp.broadcast_to(g_exceeds, (num_lanes, capacity)),
        )

        def scan_body(s, k):
            return lane_step(s, k, level_logl, level_logX, push, n_levels,
                             levels_done, context)

        def lane_run(s, ks):
            return jax.lax.scan(scan_body, s, ks)

        return jax.vmap(lane_run)(state, keys)

    return jax.jit(block_fn, device=device)


def make_init_fn(logl_of_u, config, device=None):
    """Build the jitted prior initialisation program."""

    num_lanes = config.num_lanes
    num_particles = config.num_particles
    ndim = config.ndim
    capacity = config.level_capacity

    def init_fn(key, context):
        k_u, k_tie = random.split(key, 2)
        u = random.uniform(k_u, (num_lanes, num_particles, ndim),
                           dtype=jnp.float64)
        logl = jax.vmap(
            jax.vmap(lambda ui: logl_of_u(ui, context)))(u)
        logl = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, dtype=jnp.float64), logl)
        tie = random.uniform(k_tie, (num_lanes, num_particles),
                             dtype=jnp.float64)
        level = jnp.zeros((num_lanes, num_particles), dtype=jnp.int32)
        counters = jnp.zeros((num_lanes, capacity), dtype=jnp.int64)
        coord = jnp.zeros((num_lanes, ndim), dtype=jnp.int64)
        return LaneState(
            u=u, logl=logl, tie=tie, level=level,
            accepts=counters, tries=counters, visits=counters,
            exceeds=counters, coord_attempts=coord, coord_accepts=coord)

    return jax.jit(init_fn, device=device)


def make_to_physical(prior_transform, device=None):
    """Build the jitted unit-cube -> physical transform for saved samples."""

    def to_physical(u, context):
        return jax.vmap(
            lambda ui: call_with_context(prior_transform, ui, context))(u)

    return jax.jit(to_physical, device=device)
