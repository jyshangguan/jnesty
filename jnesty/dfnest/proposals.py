"""Pure JAX proposal and MCMC kernels for the DNest engine.

Every function here is a device-side transcription of the corresponding
piece of CDNest's ``src/dnest.c`` (commit ``3aaa039f9a0a...``):

* :func:`randh`           -- ``dnest_randh`` heavy-tailed step size.
* :func:`select_coordinate` -- the per-coordinate proposal choice, generalised
  from CDNest's uniform draw to the user's ``sample_frequency`` weights.
* :func:`update_particle` -- ``update_particle_normal``: Metropolis update of
  one local particle plus the level's accepts/tries and the early-break
  visits/exceeds bookkeeping.
* :func:`update_level`    -- ``update_level_assignment_normal``: the
  heavy-tailed level move with the logX/log_push/beta acceptance terms.

All functions are pure (no JIT of their own); :mod:`jnesty.dfnest.core`
composes them into one compiled block program.  Arrays use float64
positions/likelihoods and int64 level counters; the engine enables JAX x64
mode on construction.

Two CDNest details are worth calling out because they are easy to get
subtly wrong:

1. *Visits/exceeds early break.*  After each particle update, CDNest
   increments ``visits[j]`` for ``j`` from the particle's level upward and
   ``exceeds[j]`` only until the first level whose next threshold is *not*
   exceeded, then breaks.  Because level thresholds are ordered, the set of
   levels below a likelihood is a prefix, which turns the loop into the two
   range updates computed by :func:`visits_exceeds_increments`.

2. *Exactly one likelihood evaluation per MCMC step.*  The particle
   proposal (coordinate choice, step, prior transform, likelihood) is
   independent of the update order, so it is evaluated once per step and
   the two Metropolis updates are then applied in the CDNest's randomly
   chosen order.
"""

import jax
import jax.numpy as jnp
from jax import random

__all__ = ['randh', 'wrap_unit', 'select_coordinate',
           'count_thresholds_below', 'visits_exceeds_increments',
           'update_particle', 'update_level', 'make_lane_step']


def wrap_unit(x):
    """Wrap a scalar into ``[0, 1)`` (CDNest's ``dnest_wrap`` on the unit cube)."""
    return jnp.mod(x, 1.0)


def randh(key):
    """CDNest's heavy-tailed step-size ``h = 10**(1.5 - 3|t|) * N(0, 1)``.

    ``t`` is a Student-t with 2 degrees of freedom, constructed as
    ``N(0,1) / sqrt(-log(U))`` exactly as CDNest's own Python examples do.
    Typical steps are of order ``10**-1.5`` but the tails are arbitrarily
    long, which is what lets the diffusive sampler escape and re-explore.
    """
    k_t, k_u, k_n = random.split(key, 3)
    t = random.normal(k_t) / jnp.sqrt(-jnp.log(random.uniform(k_u)))
    return jnp.power(10.0, 1.5 - 3.0 * jnp.abs(t)) * random.normal(k_n)


def select_coordinate(key, weights):
    """Categorical draw over normalised per-coordinate weights.

    Uses the inverse-CDF construction (cumulative sum + search) rather than
    a Gumbel-max trick so that scale-equivalent weight vectors select
    identically under the same seed.
    """
    u = random.uniform(key)
    cumulative = jnp.cumsum(weights)
    index = jnp.searchsorted(cumulative, u, side='right')
    return jnp.minimum(index, weights.shape[0] - 1)


def count_thresholds_below(level_logl, logl, n_levels):
    """``m = #{ active levels whose threshold < logl }``.

    Level thresholds are ordered, so this count is the length of the prefix
    of levels below ``logl``; it identifies the particle's sandwiching level
    without an explicit scan.
    """
    capacity = level_logl.shape[0]
    idx = jnp.arange(capacity)
    return jnp.sum((idx < n_levels) & (level_logl < logl))


def visits_exceeds_increments(level, logl, level_logl, n_levels):
    """Boolean masks reproducing CDNest's early-break visits/exceeds loop.

    CDNest runs ``for j = level .. n-2: visits[j]++; if levels[j+1] <
    logl: exceeds[j]++; else break``.  With ``m`` the number of thresholds
    below ``logl``, the loop visits ``j`` in ``[level, min(m-1, n-2)]`` and
    exceeds in ``[level, min(m-2, n-2)]``; both become range masks.  The
    particle invariant ``levels[level] < logl`` guarantees ``level <= m-1``,
    and a particle at the top level produces empty masks, matching the C
    loop that never starts.
    """
    capacity = level_logl.shape[0]
    j = jnp.arange(capacity)
    m = count_thresholds_below(level_logl, logl, n_levels)
    hi_v = jnp.minimum(m - 1, n_levels - 2)
    hi_e = jnp.minimum(m - 2, n_levels - 2)
    visits_inc = (j >= level) & (j <= hi_v)
    exceeds_inc = (j >= level) & (j <= hi_e)
    return visits_inc.astype(jnp.int64), exceeds_inc.astype(jnp.int64)


def update_particle(state, which, coord, row_new, logl_prop, tie_prop,
                    level_logl, n_levels, key_accept):
    """Apply ``update_particle_normal`` for one lane-local particle.

    Parameters
    ----------
    state : LaneState
        Current lane state (before this update).
    which : int array scalar
        Index of the local particle being updated.
    coord : int array scalar
        Coordinate that the proposal perturbed (for the per-coordinate
        diagnostics).
    row_new, logl_prop, tie_prop
        The fully evaluated proposal: new unit-cube row, its log-likelihood
        and its tiebreaker.
    level_logl : ndarray (capacity,)
        Level thresholds (block constant).
    n_levels : int array scalar
        Number of active levels.
    key_accept
        PRNG key for the Metropolis uniform.

    Returns
    -------
    LaneState
        Updated state.

    Notes
    -----
    The wrapped one-coordinate proposal on the unit cube is symmetric and
    the prior transform is applied outside the kernel, so the Hastings
    ratio is exactly one and the Metropolis uniform is vacuous; CDNest's
    level constraint (strict ``threshold < logl_prop`` in normal mode)
    decides acceptance.  The uniform draw is kept so the structure mirrors
    the C code and a non-trivial ``log_H`` could be dropped in later.
    """
    lvl = state.level[which]
    accept = (random.uniform(key_accept) <= 1.0) & (level_logl[lvl] < logl_prop)
    accept_i = accept.astype(jnp.int64)

    row_old = state.u[which]
    u = state.u.at[which].set(jnp.where(accept, row_new, row_old))
    logl_old = state.logl[which]
    logl_new = jnp.where(accept, logl_prop, logl_old)
    logl = state.logl.at[which].set(logl_new)
    tie = state.tie.at[which].set(jnp.where(accept, tie_prop, state.tie[which]))

    accepts = state.accepts.at[lvl].add(accept_i)
    tries = state.tries.at[lvl].add(1)
    coord_attempts = state.coord_attempts.at[coord].add(1)
    coord_accepts = state.coord_accepts.at[coord].add(accept_i)

    visits_inc, exceeds_inc = visits_exceeds_increments(
        lvl, logl_new, level_logl, n_levels)
    visits = state.visits + visits_inc
    exceeds = state.exceeds + exceeds_inc

    return state._replace(
        u=u, logl=logl, tie=tie, accepts=accepts, tries=tries,
        visits=visits, exceeds=exceeds,
        coord_attempts=coord_attempts, coord_accepts=coord_accepts)


def update_level(state, which, level_logl, level_logX, push, n_levels,
                 levels_done, beta, keys):
    """Apply ``update_level_assignment_normal`` for one local particle.

    The proposed level is ``current + int(10**(2U) * N(0,1))`` (C truncates
    toward zero), forced to ``current +- 1`` when unchanged, then wrapped
    into ``[0, n_levels)`` with mathematical modulo.  The acceptance log
    ratio combines the level volumes ``-logX[prop] + logX[cur]``, the
    ``log_push`` difference (drift toward the highest level while levels
    are being built), and -- once level creation has finished -- CDNest's
    ``beta * log((tries[cur]+1)/(tries[prop]+1))`` equal-push term evaluated
    with the lane's *running* counter values, exactly as the C workers read
    their private copy of ``levels``.
    """
    k_scale, k_jump, k_bump, k_accept = keys
    cur = state.level[which]

    scale = jnp.power(10.0, 2.0 * random.uniform(k_scale))
    jump = jnp.int32(jnp.trunc(scale * random.normal(k_jump)))
    prop_raw = cur + jump
    bump = jnp.where(random.uniform(k_bump) < 0.5,
                     jnp.int32(-1), jnp.int32(1))
    prop_raw = jnp.where(prop_raw == cur, cur + bump, prop_raw)
    prop = jnp.mod(prop_raw, n_levels)

    log_accept = (-level_logX[prop] + level_logX[cur]
                  + push[prop] - push[cur])
    log_accept = log_accept + jnp.where(
        levels_done,
        beta * jnp.log((state.tries[cur] + 1.0) / (state.tries[prop] + 1.0)),
        0.0)
    log_accept = jnp.minimum(log_accept, 0.0)

    accept = ((random.uniform(k_accept) <= jnp.exp(log_accept))
              & (level_logl[prop] < state.logl[which]))
    level = state.level.at[which].set(jnp.where(accept, prop, cur))
    return state._replace(level=level)


def make_lane_step(num_particles, coord_weights, beta, logl_of_u):
    """Build the per-lane, per-step MCMC transition function.

    The returned function has the ``lax.scan`` body signature
    ``(state, key) -> (state, ys)`` with the block constants (level arrays,
    ``n_levels``, ``levels_done``, the likelihood context) supplied
    explicitly, so the compiled program never closes over run data.

    ``ys`` records, per step, the selected particle's final log-likelihood
    and tiebreaker, whether the step emitted an above-threshold record for
    level construction, and whether the *proposal* likelihood was NaN or
    +inf (a user bug that would otherwise be silently rejected).
    """
    weights = jnp.asarray(coord_weights, dtype=jnp.float64)

    def lane_step(state, key, level_logl, level_logX, push, n_levels,
                  levels_done, context):
        (k_which, k_order, k_coord, k_h, k_tie, k_pacc,
         k_scale, k_jump, k_bump, k_lacc) = random.split(key, 10)

        which = random.randint(k_which, (), 0, num_particles)
        particle_first = random.uniform(k_order) <= 0.5

        # ---- order-independent proposal (one likelihood call per step) ----
        row = state.u[which]
        coord = select_coordinate(k_coord, weights)
        h = randh(k_h)
        row_new = row.at[coord].set(jnp.mod(row[coord] + h, 1.0))
        logl_prop = logl_of_u(row_new, context)
        tie_prop = jnp.mod(state.tie[which] + random.uniform(k_tie), 1.0)

        level_keys = (k_scale, k_jump, k_bump, k_lacc)

        def particle_then_level(s):
            s2 = update_particle(s, which, coord, row_new, logl_prop,
                                 tie_prop, level_logl, n_levels, k_pacc)
            return update_level(s2, which, level_logl, level_logX, push,
                                n_levels, levels_done, beta, level_keys)

        def level_then_particle(s):
            s2 = update_level(s, which, level_logl, level_logX, push,
                              n_levels, levels_done, beta, level_keys)
            return update_particle(s2, which, coord, row_new, logl_prop,
                                   tie_prop, level_logl, n_levels, k_pacc)

        state2 = jax.lax.cond(particle_first, particle_then_level,
                              level_then_particle, state)

        # ---- above-threshold emission (CDNest dnest_mcmc_run_normal) ----
        logl_final = state2.logl[which]
        m = count_thresholds_below(level_logl, logl_final, n_levels)
        emit = (~levels_done) & (m == n_levels)
        invalid = jnp.isnan(logl_prop) | jnp.isposinf(logl_prop)
        return state2, (logl_final, state2.tie[which], emit, invalid)

    return lane_step
