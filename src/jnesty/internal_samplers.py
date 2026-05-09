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
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding

from .utils import randsphere


def apply_reflect(u):
    """
    Iteratively reflect values into [0, 1].

    Matches Dynesty's apply_reflect (utils.py:919-944).
    All numbers u = 2n +/- x are mapped to x in [0, 1].

    For the '+' case (even number of reflections): u % 1
    For the '-' case (odd number of reflections): 1 - (u % 1)

    E.g., -0.9, 1.1, and 2.9 all map to 0.9.
    """
    idxs_even = jnp.mod(u, 2) < 1
    return jnp.where(idxs_even, jnp.mod(u, 1), 1 - jnp.mod(u, 1))


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


def _propose_one(key, x, axes, scale, n_cluster, ndim, walk_schedule, step_idx):
    """
    Generate a single proposal point from x.

    Follows Dynesty's propose_ball_point:
    - First n_cluster dimensions: perturbed using axes transform
    - Remaining (ndim - n_cluster) dimensions: resampled uniformly from [0,1]
    - Unit cube check (rejection) instead of reflection — matches Dynesty default

    Returns (u_prop, in_bounds) where in_bounds is True if proposal is within [0,1].
    """
    # Non-clustered dims: resample uniformly (always in [0,1])
    key, subkey = random.split(key)
    u_non_cluster = random.uniform(subkey, (ndim - n_cluster,))
    u_prop = jnp.concatenate([x[:n_cluster], u_non_cluster])

    # Clustered dims: perturbed from current position via axes transform
    if n_cluster > 0:
        key, subkey2 = random.split(key)
        dr = randsphere(subkey2, n_cluster)
        if walk_schedule is not None:
            sel_idx = walk_schedule[step_idx]
            selected_axes = axes[sel_idx] if axes.ndim == 3 else axes
            cluster_axes = selected_axes[:n_cluster, :n_cluster]
        else:
            if axes is not None:
                cluster_axes = axes[:n_cluster, :n_cluster]
            else:
                cluster_axes = None
        du = cluster_axes @ dr if cluster_axes is not None else dr
        u_prop = u_prop.at[:n_cluster].set(x[:n_cluster] + scale * du)

    # Unit cube check — reject proposals outside [0,1] (Dynesty default)
    in_bounds = jnp.all((u_prop > 0) & (u_prop < 1))
    return u_prop, in_bounds


def _single_walk(key, x_start, logL_constraint, loglikelihood_fn,
                 axes, scale, n_steps, ndim, n_cluster, prior_bounds,
                 walk_schedule):
    """Run a single K-step random walk. Returns (x_final, n_accepted, n_total).

    Matches Dynesty's generic_random_walk (internal_samplers.py:866-987):
    - Boundary rejections: proposal outside [0,1] → stay put, count toward n_total
    - Likelihood rejections: logL <= logL_constraint → stay put, count toward n_total
    - n_accepted counts only successful moves (logL > constraint AND in bounds)
    - n_total counts ALL proposals (boundary + likelihood evaluations)
    """

    def walk_step(walk_state, step_idx):
        x, walk_key, walk_accepted, walk_total = walk_state
        walk_key, subkey = random.split(walk_key)

        x_proposed, in_unit_cube = _propose_one(
            subkey, x, axes, scale, n_cluster, ndim, walk_schedule, step_idx
        )

        # Check prior_bounds if provided
        in_prior = True
        if prior_bounds is not None:
            in_prior = jnp.all(
                (x_proposed >= prior_bounds[0]) &
                (x_proposed <= prior_bounds[1])
            )

        in_bounds = in_unit_cube & in_prior

        # Evaluate logL only if in bounds
        logL_proposed = jnp.where(
            in_bounds,
            loglikelihood_fn(x_proposed),
            -jnp.inf
        )

        # Accept only if in bounds AND above likelihood constraint
        accept = logL_proposed > logL_constraint
        x_new = jnp.where(accept, x_proposed, x)
        accepted_new = jnp.where(accept, walk_accepted + 1, walk_accepted)

        return (x_new, walk_key, accepted_new, walk_total + 1), None

    init_state = (x_start, key, 0, 0)
    final_state, _ = lax.scan(walk_step, init_state, jnp.arange(n_steps))
    return final_state[0], final_state[2], final_state[3]


class RWalkSampler(InternalSampler):
    """
    Random walk sampler with n-ball proposals and adaptive scale.

    Matches Dynesty's generic_random_walk / propose_ball_point:
    - Clustered dimensions (first n_cluster): perturbed via axes transform
    - Non-clustered dimensions: resampled uniformly from [0,1]
    - Boundary rejection (matches Dynesty default)
    - Scale adapted via Robbins-Munro using ncdim

    When batch_size > 1, runs batch_size independent walks in parallel
    (vmap across GPU), each with n_steps // batch_size steps.
    Each batch walk can start from a different point and use different axes.
    """

    def __init__(self, ndim, target_acceptance=0.5, batch_size=1,
                 ncdim=None, **kwargs):
        super().__init__(ndim, **kwargs)
        self.target_acceptance = target_acceptance
        self.batch_size = batch_size
        self.ncdim = ncdim if ncdim is not None else ndim

        # Shard parallel walks across available GPUs for even memory usage
        devices = jax.devices()
        n_devices = len(devices)
        if batch_size > 1 and n_devices > 1:
            # Round down to nearest multiple of n_devices for even sharding
            effective_batch = (batch_size // n_devices) * n_devices
            if effective_batch >= n_devices:
                self.batch_size = effective_batch
                self._mesh = Mesh(devices, ('walks',))
                self._sharding = NamedSharding(self._mesh, P('walks'))
            else:
                self._mesh = None
                self._sharding = None
        else:
            self._mesh = None
            self._sharding = None

    def sample(self, key, x_starts, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None,
               walk_schedule=None):
        """
        K-step random walk using randsphere proposals.

        Parameters
        ----------
        x_starts : array
            If batch_size=1: (ndim,) single starting point.
            If batch_size>1: (batch_size, ndim) diverse starting points.
        axes : array
            If batch_size=1: (ncdim, ncdim) single axes matrix.
            If batch_size>1: (batch_size, ncdim, ncdim) per-walk axes.

        Returns (x_new, logL_new, n_accepted, n_total) where n_total includes
        boundary rejections (matches Dynesty's acceptance rate semantics).
        """
        batch_size = self.batch_size
        ndim = self.ndim
        n_cluster = self.ncdim

        if batch_size == 1:
            # Single walk path
            x_new, n_accepted, n_total = _single_walk(
                key, x_starts, logL_constraint, loglikelihood_fn,
                axes, scale, n_steps, ndim, n_cluster,
                prior_bounds, walk_schedule
            )
        else:
            # Parallel walks: B walks with n_steps // B steps each
            steps_per_walk = max(1, n_steps // batch_size)
            walk_keys = random.split(key, batch_size)

            # Distribute across GPUs for even memory usage
            if self._sharding is not None:
                walk_keys = jax.device_put(walk_keys, self._sharding)

            # vmap the walk across batch_size starting points + per-walk axes
            vmapped_walk = jax.vmap(
                lambda wk, x0, ax: _single_walk(
                    wk, x0, logL_constraint, loglikelihood_fn,
                    ax, scale, steps_per_walk, ndim, n_cluster,
                    prior_bounds, walk_schedule
                )
            )
            x_candidates, n_accepted_arr, n_total_arr = vmapped_walk(
                walk_keys, x_starts, axes)

            # Evaluate likelihoods for all candidates
            logL_candidates = jax.vmap(loglikelihood_fn)(x_candidates)

            # Check prior_bounds
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

            x_new = jnp.where(any_valid, x_candidates[first_valid], x_starts[0])
            n_accepted = jnp.where(
                any_valid, n_accepted_arr[first_valid], 0
            )
            n_total = n_total_arr[first_valid]

        logL_new = loglikelihood_fn(x_new)
        return x_new, logL_new, n_accepted, n_total

    def tune(self, scale, acceptance_rate, ndim, iteration):
        """
        Robbins-Munro scale adaptation (matches Dynesty).

        Uses ncdim (number of clustered dimensions) instead of full ndim,
        matching Dynesty's RWalkSampler.tune() at internal_samplers.py:491.
        """
        should_adapt = iteration > 0
        proposed_scale = scale * jnp.exp(
            (acceptance_rate - self.target_acceptance)
            / self.ncdim / self.target_acceptance
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
