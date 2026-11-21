"""
Internal proposal samplers for nested sampling.

Each sampler implements a strategy for generating a replacement live point
given a likelihood constraint. All follow the InternalSampler interface:
- sample(): propose a replacement point via K walk steps
- tune(): adapt the proposal scale based on acceptance rate

Currently implemented:
- RWalkSampler: random walk with n-ball proposals and adaptive scale
- HSliceSampler: "Hamiltonian" slice sampling with gradient reflections
  (jittable port of dynesty's ``sample_hslice`` / ``sample='hslice'``)

Future additions:
- SliceSampler, RSliceSampler, etc.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Optional

from ..utils import call_with_context, randsphere


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
                 walk_schedule, context=None):
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

        # Evaluate logL (branchless jnp.where for XLA fusion — lax.cond with
        # data-dependent predicate inside lax.scan kills GPU kernel fusion).
        logL_proposed = jnp.where(
            in_bounds,
            call_with_context(loglikelihood_fn, x_proposed, context),
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


def vmap_queue_refill(refill_key, live_x, live_logL, loglstar,
                      loglikelihood_fn, me_axes, me_logvol_ells,
                      bound_axes, scale, rwalk_K, ndim, ncdim,
                      queue_size, use_multi_ellipsoid, prior_bounds=None,
                      context=None):
    """Generate queue_size candidate points in parallel via vmap.

    Each candidate is produced by a FULL rwalk_K-step random walk from a
    random starting point (live point above loglstar) using a random ellipsoid
    (or bound_axes if not multi-ellipsoid).  Matches Dynesty's pool.map batch:
    every worker does a full walk independently.

    Returns
    -------
    (queue_x, queue_logL, queue_nacc, queue_ntot) each of leading dim queue_size.
    """
    keys = random.split(refill_key, queue_size)
    n_ells = me_axes.shape[0]
    nlive = live_x.shape[0]

    # Pre-compute ellipsoid probabilities (shared across all candidates)
    if use_multi_ellipsoid:
        ell_log_probs = me_logvol_ells - jax.scipy.special.logsumexp(me_logvol_ells)
        ell_probs = jnp.exp(ell_log_probs)
        ell_probs = jnp.where(jnp.isnan(ell_probs) | (ell_probs < 0), 0.0, ell_probs)
        ell_probs = ell_probs / jnp.maximum(ell_probs.sum(), 1e-30)
    else:
        ell_probs = None

    # Pre-compute above-mask probabilities (shared, but loglstar is fixed for the batch)
    above_mask = live_logL > loglstar
    above_probs = above_mask.astype(jnp.float32)
    above_probs = above_probs / jnp.maximum(above_probs.sum(), 1)

    def _generate_one(k):
        k_start, k_ell, k_walk = random.split(k, 3)

        # Select random starting point above loglstar
        start_idx = random.choice(k_start, nlive, p=above_probs)
        x_start = live_x[start_idx]

        # Select ellipsoid axes
        if use_multi_ellipsoid:
            ell_idx = random.choice(k_ell, n_ells, p=ell_probs)
            axes = me_axes[ell_idx]
        else:
            axes = bound_axes

        # Run full walk
        x_new, n_acc, n_tot = _single_walk(
            k_walk, x_start, loglstar, loglikelihood_fn,
            axes, scale, rwalk_K, ndim, ncdim,
            prior_bounds, None, context
        )
        logL_new = call_with_context(loglikelihood_fn, x_new, context)
        return x_new, logL_new, n_acc, n_tot

    return jax.vmap(_generate_one)(keys)


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

        # GPU sharding: disabled by default — vmap already parallelizes across devices.
        # Explicit sharding with Mesh/NamedSharding adds overhead for small batch sizes
        # and can conflict with the sampler's batch_size expectations.
        self._mesh = None
        self._sharding = None

    def sample(self, key, x_starts, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None,
               walk_schedule=None, context=None):
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
                prior_bounds, walk_schedule, context
            )
        else:
            # Parallel walks: B walks with n_steps // B steps each
            steps_per_walk = max(1, n_steps // batch_size)
            walk_keys = random.split(key, batch_size)

            # vmap the walk across batch_size starting points + per-walk axes
            vmapped_walk = jax.vmap(
                lambda wk, x0, ax: _single_walk(
                    wk, x0, logL_constraint, loglikelihood_fn,
                    ax, scale, steps_per_walk, ndim, n_cluster,
                    prior_bounds, walk_schedule, context
                )
            )
            x_candidates, n_accepted_arr, n_total_arr = vmapped_walk(
                walk_keys, x_starts, axes)

            # Evaluate likelihoods for all candidates
            logL_candidates = jax.vmap(
                lambda x: call_with_context(loglikelihood_fn, x, context))(x_candidates)

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
            # Sum ALL walks' acceptance stats for scale adaptation (matches Dynesty)
            n_accepted = jnp.sum(n_accepted_arr).astype(jnp.int32)
            n_total = jnp.sum(n_total_arr).astype(jnp.int32)

        logL_new = call_with_context(loglikelihood_fn, x_new, context)
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


#: Jitter applied to every trajectory step of the Hamiltonian slice sampler
#: (dynesty's ``jitter`` in ``sample_hslice``).
HS_JITTER = 0.25


class HSliceSampler(InternalSampler):
    """
    "Hamiltonian" slice sampler with gradient reflections (dynesty ``hslice``).

    A jittable port of ``dynesty.sampling.sample_hslice``.  Every
    data-dependent branch is expressed with ``lax.while_loop`` /
    ``lax.fori_loop`` over pre-allocated fixed-size buffers, so ``sample`` can
    be traced once inside the nested-sampling kernels (no Python-level
    control flow depends on traced values).

    One call ``sample(..., n_steps=K)`` performs ``K`` independent slice moves.
    ``n_steps`` is dynesty's ``slices`` keyword, i.e. for this sampler
    ``rwalk_K`` means "number of slices", not "number of walk steps".
    Each slice

    1. draws a direction ``drhat`` on the unit sphere and a velocity
       ``vel = axes @ drhat * scale * 0.01``;
    2. opens the initial window ``u -/+ U(1-j, 1+j) * vel`` with ``j = 0.25``;
    3. walks "forwards in time" one jittered step at a time until the
       loglikelihood drops below the constraint, recording one chord
       (left node, interior "mid" node, right node) per trajectory segment;
    4. computes the autodiff gradient ``h`` of the unit-cube loglikelihood at
       the bracketing point and reflects specularly off the boundary,
       ``vel <- vel - 2 h (vel . h) / |h|^2``, continuing with the reflected
       velocity while the reflection is not degenerate
       (``cos(vel_ref, vel) >= -0.99``);
    5. repeats the walk backwards in time with ``-vel`` (chords are appended
       symmetrically);
    6. slice-samples over all recorded chords: picks a chord with probability
       proportional to its length, proposes ``u_l + t (u_r - u_l)`` with
       ``t ~ U(0, 1)``, accepts when ``logL > logL_constraint`` and otherwise
       shrinks the chord towards its interior "mid" node.

    The initial (fallback) chord is dropped once the trajectory produced at
    least one real chord, matching dynesty.

    Unit-cube handling: a step outside ``(0, 1)^ndim`` -- or outside
    ``prior_bounds`` when given -- counts as ``logL = -inf`` without acting on
    the value (dynesty's ``unitcheck``).  JAX still executes the likelihood on
    the out-of-bounds point; the result is discarded with ``jnp.where``.

    Caps (fixed sizes, hence jittable): ``max_move`` steps per direction,
    ``max_reflects`` reflections per direction, ``max_chords`` chords per
    slice, ``max_contract`` chord proposals per slice.  A slice that exhausts
    its caps keeps its current point, so an invalid proposal is never
    returned.

    Counting / tuning: ``n_total`` counts every likelihood *and* gradient
    evaluation inside the slice (plus the final evaluation of the returned
    point), like dynesty's ``ncall`` which counts gradient evaluations too.
    ``n_accepted`` counts the evaluations that landed *inside* the constraint
    region (``logL >= logL_constraint``).  Their ratio -- the fraction of
    evaluations that are "in" rather than reflected or contracted away -- is
    the quantity ``tune`` drives towards ``target_acceptance``; it plays the
    role of dynesty's ``fmove`` (target fraction of time spent moving rather
    than reflecting).  This is the only self-consistent choice once
    ``n_total`` counts *all* evaluations: a slice-level "did the point move"
    rate would be ``1/ncall`` (a few percent) and the Robbins-Munro update
    ``scale * exp((facc - target)/ncdim/target)`` would collapse the scale to
    zero instead of finding the step size that matches the region.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    target_acceptance : float, optional
        Target in-region fraction of evaluations (see "Counting / tuning"
        above).  Default: 0.5, the shared API default.  Measured calibration
        on a 5-D correlated Gaussian (nlive=150, 5 slices, ``max_move=100``):
        the 7-seed mean evidence error is -0.06 +/- 0.19 at 0.4, and
        -0.25 +/- 0.23 at 0.5 -- i.e. the trajectory step that 0.4 settles on
        spans the constrained region better.  Values >= 0.7 shrink the step
        until the trajectory can no longer cross the region and the evidence
        degrades (no fixed point exists for the cumulative acceptance average
        to reach; see ``tune``).
    batch_size : int, optional
        Must be 1: the Hamiltonian slice trajectory is intrinsically serial.
    ncdim : int, optional
        Number of clustered dimensions (used by ``tune``).  Default: ``ndim``
    max_move : int, optional
        Maximum number of trajectory steps per direction (dynesty's
        ``max_move``).  Default: 100, matching dynesty.  This is the cap that
        sets the *length* of the chord set (``max_move`` steps of length
        ``|axes @ drhat| * scale * 0.01``), and hence how far a slice can move.
        Measured on a 5-D correlated Gaussian (nlive=150, 5 slices): the
        5-seed mean evidence error is -1.4 with ``max_move=30`` (the
        trajectory spans a small fraction of the constrained region and the
        live set cannot refresh) versus -0.28 with 100 and 60/100/200-step
        runs become indistinguishable from the rwalk scatter.
    max_reflects : int, optional
        Maximum number of boundary reflections per direction.  Default: 15
    max_chords : int, optional
        Size of the fixed chord buffer (must exceed
        ``2 * (max_reflects + 1)`` to never truncate a trajectory).
        Default: 64
    max_contract : int, optional
        Maximum number of chord proposals per slice (dynesty caps
        contractions at 60 per slice).  Default: 60
    grad_loglikelihood : callable, optional
        Optional override for the gradient.  It is called in **unit-cube
        space** as ``grad_loglikelihood(u)`` -- or ``grad_loglikelihood(u,
        context)`` when a context is given -- and must return ``d logL / du``
        *including* the prior transform Jacobian.  When omitted, the gradient
        is obtained by autodiff through
        ``loglikelihood_fn(prior_transform(u))``.
    """

    def __init__(self, ndim, target_acceptance=0.5, batch_size=1,
                 ncdim=None, max_move=100, max_reflects=15, max_chords=64,
                 max_contract=60, grad_loglikelihood=None, **kwargs):
        super().__init__(ndim, **kwargs)
        if batch_size != 1:
            raise ValueError(
                "HSliceSampler does not support batch_size != 1 "
                f"(got batch_size={batch_size}); the Hamiltonian slice "
                "trajectory is intrinsically serial.  Use queue_size=0 and "
                "batch_size=1."
            )
        self.target_acceptance = target_acceptance
        self.batch_size = 1
        self.ncdim = ncdim if ncdim is not None else ndim
        self.max_move = int(max_move)
        self.max_reflects = int(max_reflects)
        self.max_chords = int(max_chords)
        self.max_contract = int(max_contract)
        self.grad_loglikelihood = grad_loglikelihood
        # Unexpected kwargs are ignored on purpose so that
        # get_sampler('hslice', ndim, target_acceptance=..., batch_size=...,
        #             ncdim=..., **hslice_kwargs) always works.
        for key in ('memory_frac', 'max_ellipsoids', 'scale'):
            kwargs.pop(key, None)

    # -- sampling ---------------------------------------------------------
    def sample(self, key, x_starts, logL_constraint, loglikelihood_fn,
               axes, scale, n_steps, prior_bounds=None,
               walk_schedule=None, context=None):
        """
        Run ``n_steps`` Hamiltonian slice moves (dynesty's ``slices``).

        Parameters
        ----------
        key : jax.random.PRNGKey
        x_starts : array (ndim,)
            Starting point in the unit cube.
        logL_constraint : float
            Minimum log-likelihood of the returned point.
        loglikelihood_fn : callable
            Unit-cube loglikelihood ``loglikelihood(prior_transform(u))``.
        axes : array (ndim, ndim)
            Axes matrix of the current bound.
        scale : float
            Current proposal scale; the trajectory velocity is
            ``axes @ drhat * scale * 0.01`` (dynesty).
        n_steps : int
            Number of slices.
        prior_bounds : array (2, ndim), optional
            Additional bounds; points outside are treated as ``-inf``.
        walk_schedule : unused
            Kept for interface compatibility with :class:`RWalkSampler`.
        context : optional
            Passed to the user callables as a second argument.

        Returns
        -------
        (x_new, logL_new, n_accepted, n_total)
        """
        ndim = self.ndim
        max_move = self.max_move
        max_reflects = self.max_reflects
        max_chords = self.max_chords
        max_contract = self.max_contract
        jitter = HS_JITTER

        u_start = jnp.asarray(x_starts)
        dtype = jnp.result_type(u_start, 0.0)
        u_start = u_start.astype(dtype)
        if axes is None:
            axes = jnp.eye(ndim, dtype=dtype)
        else:
            axes = jnp.asarray(axes, dtype=dtype)
        scale = jnp.asarray(scale, dtype=dtype)
        constraint = jnp.asarray(logL_constraint)

        if prior_bounds is not None:
            pb = jnp.asarray(prior_bounds, dtype=dtype)

        tiny = jnp.array(1e-300, dtype=dtype)
        zero_i = jnp.array(0, dtype=jnp.int32)
        max_move_idx = jnp.array(max_move - 1, dtype=jnp.int32)

        def _logl(u):
            """Unit-cube loglikelihood: -inf outside the cube/prior bounds."""
            inb = jnp.all((u > 0.0) & (u < 1.0))
            if prior_bounds is not None:
                inb = inb & jnp.all((u >= pb[0]) & (u <= pb[1]))
            val = jnp.asarray(call_with_context(loglikelihood_fn, u, context))
            return jnp.where(inb, val, jnp.array(-jnp.inf)), inb

        # Gradient in unit-cube space (exact through the prior transform).
        if self.grad_loglikelihood is not None:
            def _grad_raw(u):
                return jnp.asarray(
                    call_with_context(self.grad_loglikelihood, u, context))
        else:
            def _grad_raw(u):
                return jax.grad(
                    lambda v: jnp.sum(
                        call_with_context(loglikelihood_fn, v, context)))(u)

        def _grad(u):
            h = jnp.asarray(_grad_raw(u), dtype=dtype)
            # A non-finite (or zero) gradient is unusable: sanitize and let
            # the caller reverse the direction (dynesty's fallback).
            return jnp.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # One directional trajectory pass (dynesty's "forwards"/"backwards"
        # section).  Chords are appended to the shared (max_chords, ndim)
        # buffers L/M/R; the fallback chord already occupies slot 0.
        # ------------------------------------------------------------------
        def _direction(key_in, u_begin, vel0, forward, L, M, R,
                       n_chords, n_acc, n_tot):
            inbuf = jnp.zeros((max_move, ndim), dtype=dtype)
            state = (
                u_begin,                         # 0 current position
                vel0,                            # 1 current velocity
                key_in,                          # 2 key
                zero_i,                          # 3 steps of this direction
                zero_i,                          # 4 reflections
                jnp.array(False),                # 5 done
                n_chords,                        # 6 chords (shared count)
                n_acc,                           # 7
                n_tot,                           # 8
                L, M, R,                         # 9, 10, 11
            )

            def outer_cond(st):
                return ~st[5]

            def outer_body(st):
                (u_cur, vel, ky, ncall, nreflect, _done, n_ch,
                 n_a, n_t, Lb, Mb, Rb) = st
                ky, k_in = random.split(ky)

                # --- inner loop: step until the constraint is bracketed -----
                inner = (
                    u_cur,                      # 0 position
                    vel,                        # 1 velocity
                    k_in,                       # 2 key
                    jnp.array(False),           # 3 reflect (previous step)
                    u_cur,                      # 4 u_out
                    jnp.array(-jnp.inf),        # 5 logl_out
                    jnp.array(False),           # 6 has_out
                    zero_i,                     # 7 n_in
                    inbuf,                      # 8 in-bounds points
                    ncall,                      # 9 steps
                    jnp.array(False),           # 10 aborted
                    n_a,                        # 11
                    n_t,                        # 12
                )

                def inner_cond(s):
                    return (~s[6]) & (~s[10]) & (s[9] < max_move)

                def inner_body(s):
                    (u, vel_, ky2, reflect, u_out, logl_out, has_out,
                     n_in, buf, nc, aborted, n_a, n_t) = s
                    ky2, kk = random.split(ky2)
                    step = random.uniform(kk, (), minval=1.0 - jitter,
                                          maxval=1.0 + jitter)
                    u_new = u + step * vel_
                    logl_new, inb = _logl(u_new)
                    out = logl_new < constraint
                    violated = out & reflect        # reflected and still out
                    newly_out = out & (~reflect)    # first point out of bounds
                    keep = inb & (~out)
                    slot = jnp.minimum(n_in, max_move_idx)
                    buf = jnp.where(keep, buf.at[slot].set(u_new), buf)
                    return (
                        u_new,
                        vel_,
                        ky2,
                        jnp.where(out, reflect, False),
                        jnp.where(newly_out, u_new, u_out),
                        jnp.where(newly_out, logl_new, logl_out),
                        has_out | newly_out,
                        n_in + keep.astype(jnp.int32),
                        buf,
                        nc + 1,
                        aborted | violated,
                        n_a + (~out).astype(jnp.int32),
                        n_t + 1,
                    )

                ist = lax.while_loop(inner_cond, inner_body, inner)
                (_u_last, vel_, ky_i, reflect, u_out, logl_out, has_out,
                 n_in, buf, ncall_new, aborted, n_a, n_t) = ist

                committed = has_out & (~aborted)

                # --- interior "mid" node: random in-bounds point -----------
                ky_i, k_mid = random.split(ky_i)
                r_mid = random.uniform(k_mid, ())
                mid_idx = jnp.minimum(
                    (r_mid * jnp.maximum(n_in, 1).astype(dtype)).astype(
                        jnp.int32),
                    max_move_idx)
                mid = jnp.where(n_in > 0, buf[mid_idx], u_begin)

                # --- append the chord --------------------------------------
                slot = n_ch
                write = committed & (slot < max_chords)
                left = jnp.where(forward, u_cur, u_out)
                right = jnp.where(forward, u_out, u_cur)
                Lb = jnp.where(write, Lb.at[slot].set(left), Lb)
                Mb = jnp.where(write, Mb.at[slot].set(mid), Mb)
                Rb = jnp.where(write, Rb.at[slot].set(right), Rb)
                n_ch = n_ch + write.astype(jnp.int32)

                # --- specular reflection off the likelihood boundary -------
                h = _grad(u_out)
                hh = jnp.dot(h, h)
                vn = jnp.linalg.norm(vel)
                vel_ref = vel - 2.0 * h * jnp.dot(vel, h) / jnp.maximum(hh, tiny)
                vrn = jnp.linalg.norm(vel_ref)
                cosang = jnp.dot(vel_ref, vel) / jnp.maximum(vrn * vn, tiny)
                usable = jnp.isfinite(hh) & (hh > 0.0) & (vrn > 0.0)
                go = (committed
                      & jnp.isfinite(logl_out)
                      & usable
                      & (cosang >= -0.99)
                      & (nreflect < max_reflects)
                      & (n_ch < max_chords))
                # A specular reflection is isometric; renormalize only to
                # protect against round-off (|vel| is preserved).
                vel_new = vel_ref * (vn / jnp.maximum(vrn, tiny))
                vel_new = jnp.where(go, vel_new, vel)
                nreflect = nreflect + go.astype(jnp.int32)

                return (u_out, vel_new, ky_i, ncall_new, nreflect, ~go,
                        n_ch, n_a, n_t + 1, Lb, Mb, Rb)

            st = lax.while_loop(outer_cond, outer_body, state)
            return st[9], st[10], st[11], st[6], st[7], st[8]

        # ------------------------------------------------------------------
        # One slice: trajectory pass forwards + backwards, then chord
        # slice sampling over the recorded chords.
        # ------------------------------------------------------------------
        arc = jnp.arange(max_chords, dtype=jnp.int32)

        def slice_body(_i, carry):
            u, ky, n_a, n_t = carry
            ky, k_dir, k_win = random.split(ky, 3)

            # Direction on the unit sphere, transformed/scaled by the bound.
            z = random.normal(k_dir, (ndim,))
            z = z / jnp.maximum(jnp.linalg.norm(z), tiny)
            axis = (axes @ z) * scale * 0.01

            # Initial window (fallback chord, slot 0).
            k_win, k_l, k_r = random.split(k_win, 3)
            jl = random.uniform(k_l, (), minval=1.0 - jitter,
                                maxval=1.0 + jitter)
            jr = random.uniform(k_r, (), minval=1.0 - jitter,
                                maxval=1.0 + jitter)
            L = jnp.zeros((max_chords, ndim), dtype=dtype).at[0].set(
                u - jl * axis)
            M = jnp.zeros((max_chords, ndim), dtype=dtype).at[0].set(u)
            R = jnp.zeros((max_chords, ndim), dtype=dtype).at[0].set(
                u + jr * axis)
            n_ch = jnp.array(1, dtype=jnp.int32)

            ky, k_f, k_b = random.split(ky, 3)
            L, M, R, n_ch, n_a, n_t = _direction(
                k_f, u, axis, True, L, M, R, n_ch, n_a, n_t)
            L, M, R, n_ch, n_a, n_t = _direction(
                k_b, u, -axis, False, L, M, R, n_ch, n_a, n_t)

            # Drop the initial fallback chord when real chords exist, but
            # keep it when the trajectory never bracketed (dynesty pops it
            # unconditionally, which is only safe because its inner loop is
            # unbounded; with a step cap it is the only usable chord).
            first = jnp.where(n_ch > 1, jnp.array(1, dtype=jnp.int32), zero_i)
            valid = (arc >= first) & (arc < n_ch)
            axlen = jnp.where(valid, jnp.linalg.norm(R - L, axis=1),
                              jnp.zeros((), dtype=dtype))

            cstate = (u, ky, axlen, L, M, R, zero_i, jnp.array(False),
                      n_a, n_t)

            def contract_cond(s):
                return (~s[7]) & (s[6] < max_contract)

            def contract_body(s):
                (u_c, ky_c, alen, Lc, Mc, Rc, n_c, _done, n_a, n_t) = s
                ky_c, k_ch, k_rp = random.split(ky_c, 3)

                total = jnp.sum(alen)
                probs = jnp.where(
                    total > 0.0,
                    alen / jnp.maximum(total, tiny),
                    (arc == first).astype(dtype))
                idx = random.choice(k_ch, max_chords, p=probs)
                rprop = random.uniform(k_rp, ())

                u_l = Lc[idx]
                u_m = Mc[idx]
                u_r = Rc[idx]
                u_hat = u_r - u_l
                u_prop = u_l + rprop * u_hat

                logl_p, _inb = _logl(u_prop)
                okp = logl_p > constraint

                # Shrink towards the interior "mid" node on failure.
                side = jnp.dot(u_prop - u_m, u_hat)
                sh_l = (~okp) & (side < 0.0)
                sh_r = (~okp) & (side > 0.0)
                newlen = jnp.where(sh_l, alen[idx] * (1.0 - rprop),
                                   jnp.where(sh_r, alen[idx] * rprop,
                                             alen[idx]))
                Lc = jnp.where(sh_l, Lc.at[idx].set(u_prop), Lc)
                Rc = jnp.where(sh_r, Rc.at[idx].set(u_prop), Rc)
                alen = jnp.where(sh_l | sh_r, alen.at[idx].set(newlen), alen)

                return (jnp.where(okp, u_prop, u_c), ky_c, alen, Lc, Mc, Rc,
                        n_c + 1, okp, n_a + okp.astype(jnp.int32), n_t + 1)

            cst = lax.while_loop(contract_cond, contract_body, cstate)
            return cst[0], cst[1], cst[8], cst[9]

        init = (u_start, key, zero_i, zero_i)
        u_final, _key, n_accepted, n_total = lax.fori_loop(
            0, n_steps, slice_body, init)

        logL_new = jnp.asarray(
            call_with_context(loglikelihood_fn, u_final, context))
        return u_final, logL_new, n_accepted, n_total + 1

    # -- tuning -----------------------------------------------------------
    def tune(self, scale, acceptance_rate, ndim, iteration):
        """
        Robbins-Munro scale adaptation (same formula as :class:`RWalkSampler`).

        ``acceptance_rate`` is the in-region fraction of evaluations (see the
        class docstring): the probability that a trajectory step or a chord
        proposal lands above the likelihood constraint rather than being
        reflected or contracted away.  Driving that fraction to
        ``target_acceptance`` sets the trajectory step size to the size of
        the constraint region, which is what keeps the chord set useful.
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
    'hslice': HSliceSampler,
}


def get_sampler(name, ndim, **kwargs):
    """
    Instantiate a sampler by name string.

    Parameters
    ----------
    name : str
        One of 'rwalk', 'hslice' (future: 'slice', 'rslice').
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
