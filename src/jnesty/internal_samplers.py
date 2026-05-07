"""
Internal proposal samplers for nested sampling.

Each sampler implements a strategy for generating a replacement live point
given a likelihood constraint. All follow the InternalSampler interface:
- sample(): propose a replacement point via K walk steps
- tune(): adapt the proposal scale based on acceptance rate

Currently implemented:
- RWalkSampler: random walk with n-ball proposals and adaptive scale

Future additions:
- SliceSampler, RSliceSampler, etc.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Optional

from .utils import randsphere


class InternalSampler:
    """Base class for proposal samplers."""

    def __init__(self, ndim, **kwargs):
        self.ndim = ndim

    def sample(self, key, x_start, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None):
        """
        Generate a replacement point via the sampling strategy.

        Parameters
        ----------
        key : jax.random.PRNGKey
        x_start : (ndim,) starting point
        logL_constraint : float, minimum log-likelihood for the new point
        loglikelihood_fn : callable, log-likelihood function
        axes : (ndim, ndim) axes matrix from the current bound
        scale : float, current proposal scale
        n_steps : int, number of walk/slice steps
        prior_bounds : optional (2, ndim) array for rejection

        Returns
        -------
        (x_new, logL_new, n_accepted) tuple
        """
        raise NotImplementedError

    def tune(self, acceptance_rate, ndim):
        """Adapt scale based on acceptance rate. Returns new scale."""
        raise NotImplementedError


def _propose_one(key, x, axes, scale, ndim, walk_schedule, step_idx):
    """Generate a single proposal point from x."""
    dr = randsphere(key, ndim)
    if walk_schedule is not None:
        sel_idx = walk_schedule[step_idx]
        selected_axes = axes[sel_idx] if axes.ndim == 3 else axes
        return x + scale * (selected_axes @ dr)
    else:
        delta = scale * (axes @ dr) if axes is not None else scale * dr
        return x + delta


def _single_walk(key, x_start, logL_constraint, loglikelihood_fn,
                 axes, scale, n_steps, ndim, prior_bounds, walk_schedule):
    """Run a single K-step random walk. Returns (x_final, n_accepted)."""

    def walk_step(walk_state, step_idx):
        x, walk_key, walk_accepted = walk_state
        walk_key, subkey = random.split(walk_key)

        x_proposed = _propose_one(
            subkey, x, axes, scale, ndim, walk_schedule, step_idx
        )
        in_unit_cube = jnp.all(
            (x_proposed >= 0.0) & (x_proposed <= 1.0)
        )
        logL_proposed = jnp.where(
            in_unit_cube,
            loglikelihood_fn(x_proposed),
            -jnp.inf
        )
        in_bounds = True
        if prior_bounds is not None:
            in_bounds = jnp.all(
                (x_proposed >= prior_bounds[0]) &
                (x_proposed <= prior_bounds[1])
            )
        constraint_ok = logL_proposed > logL_constraint
        accept = in_unit_cube & in_bounds & constraint_ok
        x_new = jnp.where(accept, x_proposed, x)
        accepted_new = jnp.where(accept, walk_accepted + 1, walk_accepted)

        return (x_new, walk_key, accepted_new), None

    init_state = (x_start, key, 0)
    final_state, _ = lax.scan(walk_step, init_state, jnp.arange(n_steps))
    return final_state[0], final_state[2]


class RWalkSampler(InternalSampler):
    """
    Random walk sampler with n-ball proposals and adaptive scale.

    Uses randsphere() for isotropic proposals in the unit ball,
    transformed by the bound's axes matrix. Scale is adapted via
    Robbins-Munro following Dynesty's approach.

    When batch_size > 1, runs batch_size independent walks in parallel
    (vmap across GPU), each with n_steps // batch_size steps. This
    reduces wall-clock time by ~batch_size for expensive likelihoods
    while keeping each walk independently correct. Among the batch_size
    candidates, selects the first valid replacement.
    """

    def __init__(self, ndim, target_acceptance=0.5, batch_size=1, **kwargs):
        super().__init__(ndim, **kwargs)
        self.target_acceptance = target_acceptance
        self.batch_size = batch_size

    def sample(self, key, x_start, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None,
               walk_schedule=None):
        """
        K-step random walk using randsphere proposals.

        Parameters
        ----------
        walk_schedule : optional (n_steps,) int array
            Pre-computed ellipsoid index schedule for multi-ellipsoid.
            If None, uses identity axes (unit cube proposals).
        """
        batch_size = self.batch_size
        ndim = self.ndim

        if batch_size == 1:
            # Single walk path
            x_new, n_accepted = _single_walk(
                key, x_start, logL_constraint, loglikelihood_fn,
                axes, scale, n_steps, ndim, prior_bounds, walk_schedule
            )
        else:
            # Parallel walks: B walks with n_steps // B steps each
            steps_per_walk = max(1, n_steps // batch_size)
            walk_keys = random.split(key, batch_size)

            # vmap the walk across batch_size starting points
            # All start from x_start but use different keys
            vmapped_walk = jax.vmap(
                lambda wk: _single_walk(
                    wk, x_start, logL_constraint, loglikelihood_fn,
                    axes, scale, steps_per_walk, ndim,
                    prior_bounds, walk_schedule
                )
            )
            x_candidates, n_accepted_arr = vmapped_walk(walk_keys)

            # Evaluate likelihoods for all candidates
            logL_candidates = jax.vmap(loglikelihood_fn)(x_candidates)

            # Select first valid candidate
            in_unit_cube = jnp.all(
                (x_candidates >= 0.0) & (x_candidates <= 1.0), axis=-1
            )
            logL_candidates = jnp.where(
                in_unit_cube, logL_candidates, -jnp.inf
            )
            if prior_bounds is not None:
                in_bounds = jnp.all(
                    (x_candidates >= prior_bounds[0]) &
                    (x_candidates <= prior_bounds[1]),
                    axis=-1
                )
                logL_candidates = jnp.where(
                    in_bounds, logL_candidates, -jnp.inf
                )

            valid = logL_candidates > logL_constraint
            first_valid = jnp.argmax(valid)
            any_valid = jnp.any(valid)

            x_new = jnp.where(any_valid, x_candidates[first_valid], x_start)
            n_accepted = jnp.where(
                any_valid, n_accepted_arr[first_valid], 0
            )

        logL_new = loglikelihood_fn(x_new)
        return x_new, logL_new, n_accepted

    def tune(self, scale, acceptance_rate, ndim, iteration):
        """
        Robbins-Munro scale adaptation (matches Dynesty).

        Only adapts after the first iteration to avoid premature tuning.
        """
        should_adapt = iteration > 0
        proposed_scale = scale * jnp.exp(
            (acceptance_rate - self.target_acceptance)
            / ndim / self.target_acceptance
        )
        return jnp.where(should_adapt, proposed_scale, scale)


# Registry
SAMPLER_REGISTRY = {
    'rwalk': RWalkSampler,
}


def get_sampler(name, ndim, **kwargs):
    """
    Instantiate a sampler by name string.

    Parameters
    ----------
    name : str
        One of 'rwalk' (future: 'slice', 'rslice').
    ndim : int
        Number of dimensions.
    **kwargs
        Additional arguments passed to the sampler constructor.

    Returns
    -------
    InternalSampler
        Instantiated sampler object.
    """
    if name not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler '{name}'. Choose from {list(SAMPLER_REGISTRY.keys())}"
        )
    return SAMPLER_REGISTRY[name](ndim=ndim, **kwargs)
