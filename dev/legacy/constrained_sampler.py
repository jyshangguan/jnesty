"""
Constrained random walk sampling for nested sampling.

Implements GPU-accelerated random walk sampling with whitening transforms
for efficient exploration of constrained posterior distributions.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, grad
from typing import NamedTuple
import jax.scipy as jsp


class RWalkConfig(NamedTuple):
    """
    Configuration for random walk sampler.

    Fields:
        K: Number of parallel chains (batch size)
        L: Number of steps per chain
        step_scale: Initial step size scale
    """
    K: int = 64
    L: int = 16
    step_scale: float = 0.5


def rwalk_single_step(key, x, logL, logL_constraint, chol, step_scale):
    """
    Single random walk step with constraint checking.

    Args:
        key: PRNG key
        x: Current point [D]
        logL: Current log-likelihood
        logL_constraint: Log-likelihood constraint
        chol: Cholesky decomposition of whitening transform
        step_scale: Step size scale

    Returns:
        new_x, new_logL, accepted
    """
    # Propose new point in whitened space
    key, *subkeys = random.split(key, 3)

    # Generate step in whitened coordinates
    delta = random.normal(subkeys[0], shape=x.shape)
    delta_whitened = chol.T @ delta * step_scale

    # Propose new point
    x_proposed = x + delta_whitened

    return x_proposed


def rwalk_chain_L_steps(key, x_start, logL_start, logL_constraint,
                        chol, step_scale, loglikelihood_fn, L):
    """
    Run L steps of random walk chain, returning only the final point.

    Args:
        key: PRNG key
        x_start: Starting point
        logL_start: Starting log-likelihood
        logL_constraint: Log-likelihood constraint
        chol: Whitening transform
        step_scale: Step size
        loglikelihood_fn: Log-likelihood function
        L: Number of steps

    Returns:
        (x_final, logL_final, n_accept)
    """

    def step_fn(carry, step_idx):
        key, x, logL, n_accept = carry

        # Single proposal
        x_proposed = rwalk_single_step(key, x, logL, logL_constraint,
                                       chol, step_scale)

        # Evaluate constraint
        logL_proposed = loglikelihood_fn(x_proposed)

        # Accept/reject (Metropolis)
        key, *subkeys = random.split(key, 2)
        log_accept_ratio = logL_proposed - logL

        accept = (log_accept_ratio > 0) | (
            jnp.log(random.uniform(subkeys[0])) < log_accept_ratio
        )

        # Update state
        x_new = jnp.where(accept, x_proposed, x)
        logL_new = jnp.where(accept, logL_proposed, logL)
        n_accept_new = n_accept + jnp.where(accept & (logL_proposed > logL_constraint), 1, 0)

        return (key, x_new, logL_new, n_accept_new), None

    # Run L steps
    key_init, *subkeys = random.split(key, 2)
    initial_state = (subkeys[0], x_start, logL_start, 0)

    from jax import lax
    carry_final, outputs = lax.scan(
        lambda carry, step: step_fn(carry, step),
        initial_state,
        jnp.arange(L)
    )

    key_final, x_final, logL_final, n_accept = carry_final

    return x_final, logL_final, n_accept


def batched_rwalk_sampler(key, live_x, live_logL, logL_constraint,
                         chol, config, loglikelihood_fn):
    """
    Batched random walk sampler for parallel chains.

    Args:
        key: PRNG key
        live_x: Live points [nlive, D]
        live_logL: Live point log-likelihoods [nlive]
        logL_constraint: Log-likelihood constraint
        chol: Whitening transform [D, D]
        config: RWalkConfig
        loglikelihood_fn: Log-likelihood function

    Returns:
        candidates_x: Candidate points [K, D]
        candidates_logL: Candidate log-likelihoods [K]
        accept_counts: Acceptance counts per chain [K]
    """
    # Select K starting points randomly from live points
    nlive = live_x.shape[0]
    K_effective = min(config.K, nlive)  # Can't sample more than nlive without replacement

    idx = random.choice(key, nlive, shape=(K_effective,), replace=False)

    x_starts = live_x[idx]
    logL_starts = live_logL[idx]

    # Split keys for each chain
    chain_keys = random.split(key, K_effective)

    # Run K independent chains
    chains_results = vmap(
        lambda key, x_start, logL_start: rwalk_chain_L_steps(
            key, x_start, logL_start, logL_constraint,
            chol, config.step_scale, loglikelihood_fn, config.L
        )
    )(chain_keys, x_starts, logL_starts)

    candidates_x, candidates_logL, accept_counts = chains_results

    return candidates_x, candidates_logL, accept_counts