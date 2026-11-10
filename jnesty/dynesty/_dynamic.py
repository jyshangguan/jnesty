
"""
Dynamic nested sampling helpers for JNesty.

Numerical machinery for Dynesty-style dynamic nested sampling, kept free of
sampler-class imports so :mod:`jnesty.dynesty_sampler` can build on it
without circular imports. Per-batch sampling is JIT-compiled and
GPU-accelerated; weight / stopping / combine logic is CPU-side numpy,
faithfully ported from ``dynesty.dynamicsampler`` and ``dynesty.utils`` so
numerical results match. The user-facing
:class:`~jnesty.dynesty_sampler.DynamicNestedSampler` lives in
:mod:`jnesty.dynesty_sampler`.

Public API:
    compute_integrals        -- (logl, logvol) -> (logwt, logz, logzvar, h)
    kish_ess                 -- Kish effective sample size from logwt
    compute_weights          -- (zweight, pweight) per sample
    weight_function          -- (logl_min, logl_max) bounds from saved results
    stopping_function        -- bool stop + stop_vals
    jitter_run               -- single MC realisation of prior volumes
    combine_saved_and_new    -- merge a new batch into the saved run
    seed_initial_live_points -- path-A seeding (fresh from prior)
    seed_batch_from_saved    -- path-B seeding (volume-weighted subset)
"""

import math
import numpy as np
from scipy.special import logsumexp

__all__ = [
    "compute_integrals",
    "kish_ess",
    "compute_weights",
    "weight_function",
    "stopping_function",
    "jitter_run",
    "combine_saved_and_new",
    "seed_initial_live_points",
    "seed_batch_from_saved",
]


def compute_integrals(logl, logvol, reweight=None):
    """
    Compute weights, logz, logz variance, and H using the trapezoidal estimator.

    Faithful port of ``dynesty.utils.compute_integrals``.
    """
    logl = np.asarray(logl, dtype=float)
    logvol = np.asarray(logvol, dtype=float)
    # Replace -inf logl with float32-safe large negative value to prevent
    # compute_integrals from producing NaN (exp(0) * (-inf) = NaN).
    safe_min = np.finfo(logl.dtype).min / 2 if hasattr(logl, 'dtype') else -3.4e38
    logl = np.where(np.isneginf(logl), safe_min, logl)
    loglstar_pad = np.concatenate([[-1.0e300], logl])

    dlogvol = np.diff(logvol, prepend=0)
    logdvol = logvol - dlogvol + np.log1p(-np.exp(dlogvol))
    logdvol2 = logdvol + math.log(0.5)

    logwt = np.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
    if reweight is not None:
        logwt = logwt + reweight
    logz = np.logaddexp.accumulate(logwt)
    logzmax = logz[-1]

    h_part1 = np.cumsum(
        np.exp(loglstar_pad[1:] - logzmax + logdvol2) * loglstar_pad[1:] +
        np.exp(loglstar_pad[:-1] - logzmax + logdvol2) * loglstar_pad[:-1]
    )
    h = h_part1 - logzmax * np.exp(logz - logzmax)
    dh = np.diff(h, prepend=0)

    dlogvol_neg = -np.diff(logvol, prepend=0)
    logzvar = np.abs(np.cumsum(dh * dlogvol_neg))
    return logwt, logz, logzvar, h


def kish_ess(logwt):
    """Kish ESS from unnormalised log-weights. Matches dynesty."""
    logwt = np.asarray(logwt, dtype=float)
    if len(logwt) == 0 or np.all(np.isneginf(logwt)):
        return 0.0
    W = np.exp(logwt - np.max(logwt))
    denom = (W ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(W.sum() ** 2 / denom)


def compute_weights(results):
    """Faithful port of ``dynesty.dynamicsampler.compute_weights``."""
    logl = np.asarray(results['logl'], dtype=float)
    logz = np.asarray(results['logz'], dtype=float)
    logvol = np.asarray(results['logvol'], dtype=float)
    logwt = np.asarray(results['logwt'], dtype=float)
    samples_n = np.asarray(results['samples_n'], dtype=float)

    if np.ptp(logz) == 0:
        zweight = np.ones(len(logl)) / len(logl)
    else:
        logz_remain = logl[-1] + logvol[-1]
        logz_tot = np.logaddexp(logz[-1], logz_remain)
        lzones = np.ones_like(logz)
        logzin = logsumexp(
            np.stack([lzones * logz_tot, logz], axis=0),
            axis=0,
            b=np.stack([lzones, -lzones], axis=0),
        )
        logzweight = logzin - np.log(samples_n)
        logzweight = logzweight - logsumexp(logzweight)
        zweight = np.exp(logzweight)

    pweight = np.exp(logwt - logz[-1])
    pweight_sum = pweight.sum()
    if pweight_sum > 0:
        pweight = pweight / pweight_sum
    else:
        pweight = np.ones_like(pweight) / len(pweight)
    return zweight, pweight


def weight_function(results, args=None, return_weights=False):
    """Default weight function. Faithful port of dynesty."""
    if args is None:
        args = {}
    pfrac = args.get('pfrac', 0.8)
    maxfrac = args.get('maxfrac', 0.8)
    lpad = args.get('pad', 1)
    if not 0.0 <= pfrac <= 1.0:
        raise ValueError(f"pfrac {pfrac} not in [0, 1]")
    if not 0.0 <= maxfrac <= 1.0:
        raise ValueError(f"maxfrac {maxfrac} not in [0, 1]")
    if lpad < 0:
        raise ValueError(f"pad {lpad} < 0")

    zweight, pweight = compute_weights(results)
    weight = (1.0 - pfrac) * zweight + pfrac * pweight

    nsamps = len(weight)
    logl = np.asarray(results['logl'], dtype=float)
    bounds = np.nonzero(weight > maxfrac * np.max(weight))[0]
    if len(bounds) == 0:
        bounds = np.array([0, nsamps - 1])
    bounds = (int(bounds[0]) - lpad, int(bounds[-1]) + lpad)
    if bounds[1] > nsamps - 1:
        bounds = [bounds[0] - (bounds[1] - (nsamps - 1)), nsamps - 1]
    if bounds[0] <= 0:
        logl_min = -np.inf
        logl_max = logl[min(bounds[1] - bounds[0], nsamps - 1)]
    else:
        logl_min, logl_max = logl[bounds[0]], logl[bounds[1]]
    if bounds[1] >= nsamps - 1:
        logl_max = np.inf

    if return_weights:
        return (logl_min, logl_max), (pweight, zweight, weight)
    return (logl_min, logl_max)


def _find_decrease(samples_n):
    samples_n = np.asarray(samples_n, dtype=float)
    nsamps = len(samples_n)
    diff = np.diff(samples_n, prepend=samples_n[0])
    nlive_flag = diff >= 0
    nlive_start = []
    bounds = []
    i = 0
    while i < nsamps:
        if not nlive_flag[i]:
            start = i
            while i < nsamps and not nlive_flag[i]:
                i += 1
            nstart = int(samples_n[start - 1]) if start > 0 else int(samples_n[0])
            nlive_start.append(nstart)
            bounds.append((start, i))
        else:
            i += 1
    return nlive_flag, nlive_start, bounds


def jitter_run(results, rstate=None, approx=False):
    """Probe statistical uncertainties via a single prior-volume realisation.

    Faithful port of ``dynesty.utils.jitter_run``.
    """
    if rstate is None:
        rstate = np.random.default_rng()
    results = dict(results)
    logl = np.asarray(results['logl'], dtype=float)
    samples_n = np.asarray(results['samples_n'], dtype=float)
    nsamps = len(logl)

    if approx:
        nlive_flag = np.ones(nsamps, dtype=bool)
        nlive_start, bounds = [], []
    else:
        nlive_flag, nlive_start, bounds = _find_decrease(samples_n)

    t_arr = np.zeros(nsamps)
    t_arr[nlive_flag] = rstate.beta(a=samples_n[nlive_flag], b=1.0)

    for i in range(len(nlive_start)):
        nstart = nlive_start[i]
        bound = bounds[i]
        sn = samples_n[bound[0]:bound[1]]
        y_arr = rstate.exponential(scale=1.0, size=nstart + 1)
        ycsum = y_arr.cumsum()
        ycsum /= ycsum[-1]
        uorder = ycsum[np.append(nstart, sn - 1).astype(int)]
        rorder = uorder[1:] / uorder[:-1]
        t_arr[bound[0]:bound[1]] = rorder

    logvol = np.log(t_arr).cumsum()
    logwt, logz, logzvar, h = compute_integrals(logl=logl, logvol=logvol)
    out = dict(results)
    out['logvol'] = logvol
    out['logwt'] = logwt
    out['logz'] = logz
    out['logzvar'] = logzvar
    out['h'] = h
    out['logzerr'] = np.sqrt(np.maximum(logzvar, 0.0))
    return out


def _get_seed_sequence(rstate, n):
    if rstate is None:
        return [np.random.default_rng() for _ in range(n)]
    seeds = rstate.integers(0, 2 ** 31 - 1, size=n)
    return [np.random.default_rng(int(s)) for s in seeds]


def stopping_function(results, args=None, rstate=None, M=None, return_vals=False):
    """Default stopping function. Faithful port of dynesty."""
    if args is None:
        args = {}
    if M is None:
        M = map
    pfrac = args.get('pfrac', 1.0)
    evid_thresh = args.get('evid_thresh', 0.1)
    target_n_effective = args.get('target_n_effective', 10000)
    n_mc = args.get('n_mc', 0)
    error = args.get('error', 'jitter')

    if n_mc > 1:
        seeds = _get_seed_sequence(rstate, n_mc)
        rlist = [(results, s) for s in seeds]
        if error == 'jitter':
            outputs = list(M(lambda p: jitter_run(p[0], rstate=p[1]), rlist))
        else:
            outputs = list(M(lambda p: jitter_run(p[0], rstate=p[1]), rlist))
        lnz_arr = np.array([res['logz'][-1] for res in outputs])
        lnz_std = float(np.std(lnz_arr))
    else:
        logzvar = np.asarray(results.get('logzvar', np.array([0.0])), dtype=float)
        lnz_std = float(np.sqrt(max(logzvar[-1], 0.0))) if len(logzvar) else 0.0

    stop_evid = lnz_std / max(evid_thresh, 1e-30)
    n_eff = kish_ess(results['logwt'])
    stop_post = target_n_effective / max(n_eff, 1e-30)
    stop = pfrac * stop_post + (1.0 - pfrac) * stop_evid
    if return_vals:
        return stop <= 1.0, (stop_post, stop_evid, stop)
    return stop <= 1.0


def seed_initial_live_points(prior_sample_fn, loglikelihood_fn, ndim, nlive, key,
                              n_attempts=1000):
    """Path-A seeding: uniform from prior with -inf-logL fallback rule."""
    import jax
    import jax.numpy as jnp

    min_npoints = min(nlive, max(ndim + 1, min(nlive - 20, 100)))
    live_u = np.zeros((nlive, ndim))
    live_logl = np.full(nlive, -np.inf)
    _LOWL_VAL_SEED = -1.0e300
    ngood = 0
    iattempt = 0
    ncalls = 0
    logvol_init = 0.0
    while True:
        iattempt += 1
        keys = jax.random.split(key, nlive + 1)
        key = keys[-1]
        cand_u = np.asarray(jnp.stack([prior_sample_fn(k) for k in keys[:nlive]]))
        cand_logl = np.asarray(jax.vmap(loglikelihood_fn)(cand_u))
        ncalls += nlive
        finite = np.isfinite(cand_logl)
        nfinite = int(finite.sum())
        if nfinite > 0:
            idx_finite = np.nonzero(finite)[0]
            nextra = min(nlive - ngood, nfinite)
            sel = idx_finite[:nextra]
            live_u[ngood:ngood + nextra] = cand_u[sel]
            live_logl[ngood:ngood + nextra] = cand_logl[sel]
            ngood += nextra
        if ngood >= min_npoints:
            n_left = nlive - ngood
            if n_left > 0:
                idx_nf = np.nonzero(~finite)[0]
                if len(idx_nf) >= n_left:
                    sel = idx_nf[:n_left]
                    live_u[ngood:ngood + n_left] = cand_u[sel]
                    live_logl[ngood:ngood + n_left] = cand_logl[sel]
            logvol_init = -np.log(iattempt)
            break
        if iattempt >= n_attempts:
            if ngood == 0:
                raise RuntimeError(
                    f"After {n_attempts} attempts no valid logL was found.")
            logvol_init = -np.log(iattempt)
            break
    # Replace any remaining -inf logl with float32-safe large negative value
    safe_min = np.finfo(live_logl.dtype).min / 2 if hasattr(live_logl, 'dtype') else -3.4e38
    live_logl = np.where(np.isneginf(live_logl), safe_min, live_logl)
    return live_u, live_logl, logvol_init, ncalls


def seed_batch_from_saved(saved_results, nlive_new, logl_min, rstate):
    """Path-B seeding: volume-weighted resample of saved dead points above logl_min."""
    saved_logl = np.asarray(saved_results['logl'], dtype=float)
    saved_u = np.asarray(saved_results.get('samples_u',
                                            saved_results.get('u', saved_logl)),
                         dtype=float)
    saved_logvol = np.asarray(saved_results['logvol'], dtype=float)
    saved_scale = np.asarray(saved_results.get('scale',
                                                np.ones_like(saved_logl)),
                              dtype=float)
    n_total = len(saved_logl)

    subset0 = np.nonzero(saved_logl > logl_min)[0]
    if len(subset0) == 0:
        raise RuntimeError("No live points above logl_min")
    if len(subset0) < nlive_new:
        if n_total < nlive_new:
            subset0 = np.arange(n_total)
        else:
            subset0 = np.arange(subset0[-1] - nlive_new + 1, subset0[-1] + 1)
        if subset0[0] > 0:
            logl_min = float(saved_logl[subset0[0] - 1])
        else:
            logl_min = -np.inf

    cur_log_uniwt = saved_logvol[subset0]
    cur_uniwt = np.exp(cur_log_uniwt - cur_log_uniwt.max())
    s = cur_uniwt.sum()
    if s > 0:
        cur_uniwt = cur_uniwt / s
    else:
        cur_uniwt = np.ones_like(cur_uniwt) / len(cur_uniwt)

    n_pos = int((cur_uniwt > 0).sum())
    n_pick = min(nlive_new, n_pos)
    subset = rstate.choice(subset0, size=n_pick, p=cur_uniwt, replace=False)
    live_u = saved_u[subset]
    live_logl = saved_logl[subset]
    scale = float(saved_scale[subset0[0]])
    return live_u, live_logl, scale, logl_min


def combine_saved_and_new(saved_results, new_results, logl_min, logl_max):
    """Merge a new batch run into the saved run.

    Faithful port of ``dynesty.dynamicsampler.combine_runs`` including the
    plateau-aware logvol recompute.
    """
    keys_int = ['logl', 'scale']
    saved_d = {k: np.asarray(saved_results[k]) for k in keys_int}
    saved_d['n'] = np.asarray(saved_results.get(
        'n', saved_results.get('samples_n')))
    saved_d['u'] = np.asarray(saved_results.get(
        'samples_u', saved_results.get('u')))
    saved_d['batch'] = np.asarray(saved_results.get(
        'samples_batch', saved_results.get(
            'batch', np.zeros(len(saved_d['logl'])))))
    new_d = {k: np.asarray(new_results[k]) for k in keys_int}
    new_d['n'] = np.asarray(new_results.get(
        'n', new_results.get('samples_n')))
    new_d['u'] = np.asarray(new_results.get('samples_u', new_results.get('u')))
    new_d['batch'] = np.asarray(new_results.get(
        'samples_batch', new_results.get(
            'batch', np.ones(len(new_d['logl'])))))

    nsaved = len(saved_d['n'])
    nnew = len(new_d['n'])
    ntot = nsaved + nnew

    merged = {k: [] for k in ['u', 'logl', 'n', 'scale', 'batch']}
    idx_s, idx_n = 0, 0
    logl_s = float(saved_d['logl'][0]) if nsaved > 0 else np.inf
    logl_n = float(new_d['logl'][0]) if nnew > 0 else np.inf
    nlive_s = float(saved_d['n'][0]) if nsaved > 0 else 0.0
    nlive_n = float(new_d['n'][0]) if nnew > 0 else 0.0

    for _ in range(ntot):
        if logl_s > logl_min:
            nlive = nlive_s + nlive_n
        else:
            nlive = nlive_s
        if logl_s <= logl_n:
            merged['batch'].append(int(saved_d['batch'][idx_s]))
            merged['u'].append(saved_d['u'][idx_s])
            merged['logl'].append(saved_d['logl'][idx_s])
            merged['n'].append(nlive)
            merged['scale'].append(saved_d['scale'][idx_s])
            idx_s += 1
        else:
            merged['batch'].append(int(new_d['batch'][idx_n]))
            merged['u'].append(new_d['u'][idx_n])
            merged['logl'].append(new_d['logl'][idx_n])
            merged['n'].append(nlive)
            merged['scale'].append(new_d['scale'][idx_n])
            idx_n += 1
        try:
            logl_s = float(saved_d['logl'][idx_s])
            nlive_s = float(saved_d['n'][idx_s])
        except IndexError:
            logl_s = np.inf
            nlive_s = 0.0
        try:
            logl_n = float(new_d['logl'][idx_n])
            nlive_n = float(new_d['n'][idx_n])
        except IndexError:
            logl_n = np.inf
            nlive_n = 0.0

    logl_arr = np.asarray(merged['logl'], dtype=float)
    nlive_merged_arr = np.asarray(merged['n'], dtype=float)
    # Use combined nlive per merged point (matching dynesty's combine_runs).
    # The logvol decrement log((n+1)/n) uses the total number of live points
    # at each step in the merged chain, giving a smooth, monotonic trajectory.
    n_arr_int = nlive_merged_arr.astype(int)

    plateau_mode = False
    plateau_counter = 0
    plateau_logdvol = 0.0
    logvol = 0.0
    logvol_list = []
    for i, (cur_logl, nlive) in enumerate(zip(logl_arr, n_arr_int)):
        if (not plateau_mode and i != len(n_arr_int) - 1
                and logl_arr[i] == logl_arr[i + 1]):
            plateau_mask = (logl_arr[i:] == cur_logl)
            nplateau = int(plateau_mask.sum())
            if nplateau > 1:
                plateau_counter = nplateau
                plateau_logdvol = logvol + math.log(1.0 / (nlive + 1))
                plateau_mode = True
        if not plateau_mode:
            logvol -= math.log((nlive + 1.0) / nlive)
        else:
            logvol = logvol + np.log1p(-np.exp(plateau_logdvol - logvol))
        logvol_list.append(logvol)
        if plateau_mode:
            plateau_counter -= 1
            if plateau_counter == 0:
                plateau_mode = False
    logvol_arr = np.asarray(logvol_list, dtype=float)

    logwt, logz, logzvar, h = compute_integrals(logl=logl_arr, logvol=logvol_arr)
    out = {
        'samples_u': np.asarray(merged['u']),
        'u': np.asarray(merged['u']),
        'logl': logl_arr,
        'samples_n': np.asarray(merged['n']),
        'n': np.asarray(merged['n']),
        'scale': np.asarray(merged['scale']),
        'samples_batch': np.asarray(merged['batch']),
        'batch': np.asarray(merged['batch']),
        'logvol': logvol_arr,
        'logwt': logwt,
        'logz': logz,
        'logzvar': logzvar,
        'h': h,
        'logzerr': np.sqrt(np.maximum(logzvar, 0.0)),
        'information': float(h[-1]) if hasattr(h, '__len__') else float(h),
    }
    return out


def _batch_logL_eval_chunked(logl_fn, u, batch_size=50):
    """Evaluate a vmapped log-likelihood in fixed-size chunks.

    Identical strategy to sampler._batch_logL_eval: never evaluates the whole
    candidate batch at once, so peak GPU memory is bounded by batch_size
    simultaneous forward models instead of len(u).
    """
    n = len(u)
    if n <= batch_size:
        return np.asarray(logl_fn(u))
    results = []
    for i in range(0, n, batch_size):
        results.append(np.asarray(logl_fn(u[i:i + batch_size])))
    return np.concatenate(results)


def estimate_fresh_batch_size_from_memory(loglikelihood_fn, prior_transform_fn,
                                          ndim, requested_batch_size=50,
                                          memory_frac=0.9, verbose=True):
    """Cap the fresh-point chunk size based on available GPU memory.

    Identical strategy to sampler.estimate_batch_size_from_memory: compile a
    small (2-point) trial vmap of the actual likelihood, read the per-point
    peak memory from XLA's memory_analysis(), and cap the chunk so that it
    fits within memory_frac of total GPU memory. Falls back to
    requested_batch_size whenever probing is unavailable.
    """
    import jax

    device = jax.devices()[0]
    stats = device.memory_stats()
    if stats is None:
        return requested_batch_size

    def loglike_wrapped(x):
        return loglikelihood_fn(prior_transform_fn(x))

    trial_u = jax.random.uniform(jax.random.PRNGKey(0), (2, ndim))
    try:
        trial_fn = jax.jit(jax.vmap(loglike_wrapped))
        compiled = trial_fn.lower(trial_u).compile()
    except Exception as e:
        if verbose:
            print(f"WARNING: memory probe failed ({e}). "
                  f"Using fresh_batch_size={requested_batch_size}.")
        return requested_batch_size

    ma = compiled.memory_analysis() if hasattr(compiled, 'memory_analysis') else None
    if ma is None:
        # memory_analysis() unavailable in some jaxlib versions.
        return requested_batch_size

    peak = getattr(ma, 'peak_memory_in_bytes', None)
    if peak is None:
        # fallback: temp + argument + output as a rough upper bound
        peak = (getattr(ma, 'temp_size_in_bytes', 0)
                + getattr(ma, 'argument_size_in_bytes', 0)
                + getattr(ma, 'output_size_in_bytes', 0))
        if peak <= 0:
            return requested_batch_size

    per_point = max(1, peak // 2)  # trial used 2 points
    available = int(stats['bytes_limit'] * memory_frac)
    max_chunk = max(1, available // per_point)
    capped = min(requested_batch_size, max_chunk)

    if verbose and capped < requested_batch_size:
        print(f"  Memory cap: fresh_batch_size {requested_batch_size} -> {capped} "
              f"(per-point: {per_point / 1024:.0f} KB, "
              f"budget: {available / (1024**2):.0f} MB, "
              f"GPU: {stats['bytes_limit'] / (1024**3):.1f} GB)")

    return capped


def generate_fresh_batch_points(loglikelihood_fn, prior_transform_fn,
                                subset_u, logl_min, ndim, nlive_batch,
                                bound_type, scale, key,
                                fresh_batch_size=50, _compiled_logl=None):
    """Generate nlive_batch fresh live points above logl_min.

    Matches dynesty's _new_point(logl_min) intent: independent uniform
    samples from the constrained prior. Samples nlive_batch*8 candidate
    points from the unit cube and keeps the first nlive_batch above
    logl_min.

    The candidate batch is evaluated in memory-bounded chunks (identical
    strategy to the static sampler's _batch_logL_eval); the chunk size is
    capped by estimate_fresh_batch_size_from_memory.
    """
    import jax
    import jax.numpy as jnp

    # Build or reuse compiled logL evaluator
    if _compiled_logl is None:
        def loglike_wrapped(x):
            return loglikelihood_fn(prior_transform_fn(x))
        _compiled_logl = jax.jit(jax.vmap(loglike_wrapped))

    # Single large batch: 8x oversampling
    n_sample = nlive_batch * 8
    sk, key = jax.random.split(key)
    cand_u = jax.random.uniform(sk, shape=(n_sample, ndim))
    cand_logl = _batch_logL_eval_chunked(_compiled_logl, cand_u,
                                         batch_size=fresh_batch_size)
    cand_u_np = np.asarray(cand_u)

    # Filter above logl_min
    valid = cand_logl > float(logl_min)
    n_valid = int(valid.sum())

    if n_valid >= nlive_batch:
        # Enough fresh points in one shot
        fresh_u = cand_u_np[valid][:nlive_batch]
        fresh_logl = cand_logl[valid][:nlive_batch]
    else:
        # Fallback: one more batch, then fill from subset
        sk2, key = jax.random.split(key)
        cand_u2 = jax.random.uniform(sk2, shape=(n_sample, ndim))
        cand_logl2 = _batch_logL_eval_chunked(_compiled_logl, cand_u2,
                                              batch_size=fresh_batch_size)
        cand_u2_np = np.asarray(cand_u2)
        valid2 = cand_logl2 > float(logl_min)
        all_valid_u = np.concatenate([cand_u_np[valid], cand_u2_np[valid2]])
        all_valid_logl = np.concatenate([cand_logl[valid], cand_logl2[valid2]])
        if len(all_valid_u) >= nlive_batch:
            fresh_u = all_valid_u[:nlive_batch]
            fresh_logl = all_valid_logl[:nlive_batch]
        else:
            # Last resort: use subset-selected points for remaining slots
            n_from_subset = nlive_batch - len(all_valid_u)
            fresh_u = np.concatenate([all_valid_u, np.asarray(subset_u)[:n_from_subset]])
            fresh_logl = np.concatenate([all_valid_logl,
                                         np.asarray([_compiled_logl(jnp.asarray([subset_u[i]]))[0]
                                                      for i in range(n_from_subset)])])

    # Replace -inf with float32-safe sentinel
    safe_min = np.finfo(fresh_logl.dtype).min / 2
    fresh_logl = np.where(np.isneginf(fresh_logl), safe_min, fresh_logl)

    return fresh_u, fresh_logl, _compiled_logl
