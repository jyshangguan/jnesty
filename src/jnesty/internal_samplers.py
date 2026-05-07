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


class RWalkSampler(InternalSampler):
    """
    Random walk sampler with n-ball proposals and adaptive scale.

    Uses randsphere() for isotropic proposals in the unit ball,
    transformed by the bound's axes matrix. Scale is adapted via
    Robbins-Munro following Dynesty's approach.
    """

    def __init__(self, ndim, target_acceptance=0.5, **kwargs):
        super().__init__(ndim, **kwargs)
        self.target_acceptance = target_acceptance

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
        ndim = self.ndim

        def walk_step(walk_state, step_idx):
            x, walk_key, walk_accepted = walk_state
            walk_key, subkey = random.split(walk_key, 2)

            # Propose new point
            if walk_schedule is not None:
                # Multi-ellipsoid: use pre-computed schedule
                sel_idx = walk_schedule[step_idx]
                selected_axes = axes[sel_idx] if axes.ndim == 3 else axes
                dr = randsphere(subkey, ndim)
                du = selected_axes @ dr
                x_proposed = x + scale * du
            else:
                # Default: n-ball proposal with bound axes
                dr = randsphere(subkey, ndim)
                delta = scale * (axes @ dr) if axes is not None else scale * dr
                x_proposed = x + delta

            # Check bounds
            in_unit_cube = jnp.all((x_proposed >= 0.0) & (x_proposed <= 1.0))

            in_bounds = True
            if prior_bounds is not None:
                in_bounds = jnp.all(
                    (x_proposed >= prior_bounds[0]) &
                    (x_proposed <= prior_bounds[1])
                )

            logL_proposed = jnp.where(
                in_unit_cube,
                loglikelihood_fn(x_proposed),
                -jnp.inf
            )

            # Accept if in bounds AND satisfies constraint (no Metropolis)
            constraint_satisfied = logL_proposed > logL_constraint
            accept = in_unit_cube & in_bounds & constraint_satisfied

            x_new = jnp.where(accept, x_proposed, x)
            walk_accepted_new = jnp.where(accept, walk_accepted + 1, walk_accepted)

            return (x_new, walk_key, walk_accepted_new), None

        init_walk_state = (x_start, key, 0)
        final_walk_state, _ = lax.scan(
            walk_step,
            init_walk_state,
            jnp.arange(n_steps)
        )
        x_new, final_key, n_accepted = final_walk_state

        # Evaluate likelihood of final position
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
