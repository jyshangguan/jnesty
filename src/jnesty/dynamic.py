
"""
Dynamic nested sampling for JNesty.

This module implements Dynesty-style dynamic nested sampling on top of
JNesty's static engine. Per-batch sampling is JIT-compiled and GPU-accelerated;
weight / stopping / combine logic is CPU-side numpy, faithfully ported from
``dynesty.dynamicsampler`` and ``dynesty.utils`` so numerical results match.

Public API:
    DynamicNestedSampler     -- user-facing class
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
    "DynamicNestedSampler",
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
    # Store the source nlive alongside each merged point: use the original
    # run's nlive rather than the sum.  When runs are interleaved above
    # logl_min, the combined nlive is saved_n + batch_n — but each dead
    # point was removed from *one* independent run, so the evidence
    # compression should use that run's live-point count.
    source_nlive = np.empty(ntot, dtype=float)
    idx_s, idx_n = 0, 0
    logl_s = float(saved_d['logl'][0]) if nsaved > 0 else np.inf
    logl_n = float(new_d['logl'][0]) if nnew > 0 else np.inf
    nlive_s = float(saved_d['n'][0]) if nsaved > 0 else 0.0
    nlive_n = float(new_d['n'][0]) if nnew > 0 else 0.0
    for t in range(ntot):
        if logl_s <= logl_n:
            source_nlive[t] = nlive_s
            idx_s += 1
        else:
            source_nlive[t] = nlive_n
            idx_n += 1
        try:
            logl_s = float(saved_d['logl'][idx_s])
            nlive_s = float(saved_d['n'][idx_s])
        except IndexError:
            logl_s = np.inf; nlive_s = 0.0
        try:
            logl_n = float(new_d['logl'][idx_n])
            nlive_n = float(new_d['n'][idx_n])
        except IndexError:
            logl_n = np.inf; nlive_n = 0.0
    n_arr_int = source_nlive.astype(int)

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


def _build_static_sampler(loglikelihood, prior_transform, ndim,
                           nlive, bound, **kwargs):
    """Construct a jnesty.NestedSampler with kwargs suitably defaulted."""
    from .jnesty import NestedSampler
    accepted_kwargs = {k: v for k, v in kwargs.items() if k != 'max_iterations'}
    return NestedSampler(loglikelihood, prior_transform, ndim,
                          nlive=nlive, bound=bound, **accepted_kwargs)


class DynamicNestedSampler:
    """Dynesty-style dynamic nested sampler.

    Wraps JNesty's static engine for both the base run and each batch.

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood in physical parameter space.
    prior_transform : callable
        Maps a unit-cube sample to physical space.
    ndim : int
    nlive : int, default 500
    bound : {'none', 'single', 'multi'}, default 'none'
    **static_kwargs :
        Forwarded to NestedSampler.
    """

    def __init__(self, loglikelihood, prior_transform, ndim,
                 nlive=500, bound='none', **static_kwargs):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.nlive0 = nlive
        self.bound = bound
        self.static_kwargs = static_kwargs
        self._results = None
        self.saved_run = None
        self.batch_bounds_log = []
        self.batch_nlive_log = []

    def run_nested(self,
                   nlive_init=None,
                   dlogz_init=0.01,
                   nlive_batch=None,
                   maxbatch=100,
                   pfrac=0.8,
                   maxfrac=0.8,
                   pad=1,
                   n_effective=None,
                   evid_thresh=0.1,
                   n_mc=0,
                   use_stop=True,
                   print_progress=True,
                   seed=None):
        """Run the dynamic nested sampling loop."""
        import jax.numpy as jnp
        from .results import format_results  # noqa: F401

        nlive_init = nlive_init or self.nlive0
        nlive_batch = nlive_batch or self.nlive0
        if n_effective is None:
            n_effective = max(self.ndim * self.ndim, 10000)

        rstate = np.random.default_rng(seed)

        # === Base run ===
        if print_progress:
            print(f"[dynamic] base run: nlive_init={nlive_init}, dlogz={dlogz_init}")
        base_sampler = _build_static_sampler(
            self.loglikelihood, self.prior_transform, self.ndim,
            nlive=nlive_init, bound=self.bound, **self.static_kwargs)
        base_sampler.run_nested(
            delta_logZ_threshold=dlogz_init,
            print_progress=False,
        )
        base_res = self._normalize_static_results(base_sampler.results, nlive_init, 0)
        self.saved_run = base_res
        self.batch_bounds_log = [(-np.inf, np.inf)]
        self.batch_nlive_log = [nlive_init]

        # === Batch loop ===
        wt_kwargs = {'pfrac': pfrac, 'maxfrac': maxfrac, 'pad': pad}
        stop_kwargs = {'pfrac': pfrac, 'evid_thresh': evid_thresh,
                       'target_n_effective': n_effective, 'n_mc': n_mc}

        for n in range(maxbatch):
            res = self._build_results()
            if use_stop:
                stop, stop_vals = stopping_function(res, stop_kwargs,
                                                     rstate=rstate,
                                                     return_vals=True)
                stop_val = stop_vals[2]
            else:
                stop = False
                stop_val = float('nan')

            if stop:
                if print_progress:
                    print(f"[dynamic] stop after batch {n} (stop_val={stop_val:.3f})")
                break

            logl_bounds = weight_function(res, args=wt_kwargs)
            logl_min_b, logl_max_b = logl_bounds

            if print_progress:
                ess_now = kish_ess(res['logwt'])
                print(f"[dynamic] batch {n+1}/{maxbatch}: "
                      f"logl_bounds=({logl_min_b:.2f}, {logl_max_b:.2f}), "
                      f"ESS={ess_now:.0f}/{n_effective}, stop_val={stop_val:.3f}")

            # Seed batch live points
            if np.isneginf(logl_min_b):
                init_x = None
                init_logL = None
            else:
                live_u, live_logl, scale, logl_min_eff = seed_batch_from_saved(
                    res, nlive_batch, logl_min_b, rstate)
                init_x = jnp.asarray(live_u)
                init_logL = jnp.asarray(live_logl)
                logl_min_b = logl_min_eff

            batch_sampler = _build_static_sampler(
                self.loglikelihood, self.prior_transform, self.ndim,
                nlive=nlive_batch, bound=self.bound, **self.static_kwargs)
            batch_sampler.run_nested(
                delta_logZ_threshold=dlogz_init,
                print_progress=False,
                init_live_x=init_x,
                init_live_logL=init_logL,
                logl_min=logl_min_b,
                logl_max=logl_max_b,
            )
            new_res = self._normalize_static_results(
                batch_sampler.results, nlive_batch, n + 1)

            self.saved_run = combine_saved_and_new(
                self.saved_run, new_res, logl_min_b, logl_max_b)
            self.batch_bounds_log.append((logl_min_b, logl_max_b))
            self.batch_nlive_log.append(nlive_batch)

        self._results = self._build_results()
        if print_progress:
            self.print_summary()

    @staticmethod
    def _normalize_static_results(res, nlive, batch_idx):
        """Convert a static-format Results dict into the dynamic-compatible form.

        Static format_results() appends the remaining live points to the dead
        points, producing arrays of length niter + nlive. For those tail points
        samples_n decreases from (nlive-1) down to 0, matching dynesty's
        add_live_points() semantics.
        """
        res = dict(res)
        n_total = len(res['logl'])
        niter = int(res.get('niter', n_total))
        n_live_tail = n_total - niter  # usually == nlive

        # Convert scalar logz to trajectory if needed
        if 'logz_trajectory' in res and not hasattr(res.get('logz'), '__len__'):
            res['logz'] = np.asarray(res['logz_trajectory'], dtype=float)
        elif not hasattr(res.get('logz', None), '__len__'):
            logwt = np.asarray(res['logwt'], dtype=float)
            res['logz'] = np.logaddexp.accumulate(logwt)
        res['logz'] = np.asarray(res['logz'], dtype=float)

        # samples_n: nlive for the main loop; dynesty uses n=nlive-it for the
        # tail (it=0..nlive-1) giving [nlive, nlive-1, ..., 1].
        if 'samples_n' not in res:
            samples_n = np.full(niter, nlive, dtype=float)
            if n_live_tail > 0:
                tail_len = n_live_tail
                start = nlive
                end = max(nlive - tail_len + 1, 1)
                tail = np.arange(start, end - 1, -1, dtype=float)
                if len(tail) < tail_len:
                    tail = np.concatenate([tail, np.ones(tail_len - len(tail))])
                samples_n = np.concatenate([samples_n, tail])
            res['samples_n'] = samples_n
        if 'samples_batch' not in res:
            res['samples_batch'] = np.full(n_total, batch_idx, dtype=int)
        # scale history (extend to n_total if needed)
        if 'scale' not in res:
            traj = res.get('scale_trajectory', None)
            if traj is None or len(traj) < n_total:
                traj = np.ones(n_total)
            res['scale'] = np.asarray(traj)
        # aliases used by combine_saved_and_new
        res['n'] = res['samples_n']
        res['u'] = np.asarray(res.get('samples_u', np.zeros((n_total, 1))))
        res['batch'] = res['samples_batch']
        # Sanitize -inf logl values (float32-safe replacement)
        if np.any(np.isneginf(np.asarray(res['logl']))):
            logl_arr = np.asarray(res['logl'], dtype=float)
            safe_min = np.finfo(logl_arr.dtype).min / 2
            logl_arr = np.where(np.isneginf(logl_arr), safe_min, logl_arr)
            res['logl'] = logl_arr
        # logzerr trajectory: dynesty stores per-iter variance
        if 'logzerr_trajectory' in res and len(res['logzerr_trajectory']) == n_total:
            logzerr_traj = np.asarray(res['logzerr_trajectory'], dtype=float)
            res['logzerr'] = logzerr_traj
            res['logzvar'] = logzerr_traj ** 2
        elif 'logzerr' not in res or not hasattr(res['logzerr'], '__len__'):
            try:
                _, _, logzvar_traj, _ = compute_integrals(
                    logl=res['logl'], logvol=res['logvol'])
                res['logzvar'] = logzvar_traj
                res['logzerr'] = np.sqrt(np.maximum(logzvar_traj, 0.0))
            except Exception:
                res['logzvar'] = np.zeros(n_total)
                res['logzerr'] = np.zeros(n_total)
        # also expose information as h for consistency with combine output
        res.setdefault('h', np.asarray(res.get('information', 0.0)))
        return res

    def _build_results(self):
        if self.saved_run is None:
            return None
        res = dict(self.saved_run)
        res['batch_nlive'] = list(self.batch_nlive_log)
        res['batch_bounds'] = list(self.batch_bounds_log)
        return res

    @property
    def results(self):
        if self._results is None:
            self._results = self._build_results()
        return self._results

    def to_dynesty_results(self):
        """Convert the dynamic results dict to a dynesty Results object.

        Needed so dynesty.plotting.runplot/traceplot/cornerplot can be used
        directly on JNesty dynamic output.
        """
        try:
            from dynesty.utils import Results as DynestyResults
        except ImportError:
            raise ImportError("dynesty required. Install with: pip install dynesty")
        r = self.results
        if r is None:
            raise RuntimeError("run_nested() has not been called yet")
        n = len(r['logl'])
        samples_u = np.asarray(r.get('samples_u', r.get('u', np.zeros((n, 1)))))
        items = [
            ('samples_u', samples_u),
            ('samples_id', np.arange(n)),
            ('logl', np.asarray(r['logl'], dtype=float)),
            ('samples', samples_u),  # physical == unit cube for prior_ident; caller can override
            ('samples_n', np.asarray(r['samples_n'], dtype=float)),
            ('niter', n),
            ('logwt', np.asarray(r['logwt'], dtype=float)),
            ('logvol', np.asarray(r['logvol'], dtype=float)),
            ('logz', np.asarray(r['logz'], dtype=float)),
            ('logzerr', np.asarray(r['logzerr'], dtype=float)),
            ('information', np.asarray(r['h'], dtype=float)),
            ('batch_nlive', list(r.get('batch_nlive', []))),
            ('batch_logl_bounds', list(r.get('batch_bounds', []))),
            ('samples_batch', np.asarray(r['samples_batch'], dtype=int)),
        ]
        return DynestyResults(items)

    def print_summary(self):
        if self._results is None:
            print("[dynamic] no results yet")
            return
        r = self._results
        logz = float(np.asarray(r['logz'])[-1])
        logzerr = float(np.asarray(r['logzerr'])[-1])
        h = float(np.asarray(r['h'])[-1])
        print(f"[dynamic] niter={len(r['logl'])}, logz={logz:.4f} "
              f"± {logzerr:.4f}, h={h:.2f}, "
              f"ESS={kish_ess(r['logwt']):.0f}, "
              f"nbatches={len(self.batch_nlive_log)-1}")
