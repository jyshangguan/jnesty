"""
Results data structure and formatting for JNesty.

Provides a Dynesty-compatible Results class and the format_results()
function that converts raw sampler output into a structured results object.
format_nautilus_results() is the equivalent bridge for the Nautilus engine,
so both samplers can share :mod:`jnesty.plotting`.
Also provides save_results() and load_results() for FITS I/O.
"""

import os
import numpy as np
from typing import Optional

from .utils import call_with_context, _convert_to_numpy


class Results:
    """
    Dynesty-compatible results object wrapping a dict.

    Supports both dict-style access (results['logz']) and
    attribute access (results.logz) for convenience.
    """

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        keys = list(self._data.keys())
        return f"Results({len(keys)} fields: {', '.join(keys[:6])}...)"

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Results has no field '{name}'")

    def summary(self):
        """Print formatted summary of results."""
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Log evidence: logZ = {self._data['logz']:.4f} "
              f"± {self._data['logzerr']:.4f}")
        print(f"Information: H = {self._data['information']:.4f}")
        print(f"Iterations: {self._data['niter']}")
        print(f"Efficiency: {self._data['eff']:.1f}%")
        print(f"Acceptance rate: {self._data['acceptance_rate']:.1%}")
        print(f"Converged: {self._data['converged']}")
        print(f"Final delta_logZ: {self._data['delta_logz']:.6f}")
        print("=" * 70)

    def samples_equal(self):
        """Return equal-weight posterior samples via resampling."""
        logwt = self._data['logwt']
        logwt_max = np.max(logwt)
        weights = np.exp(logwt - logwt_max)
        weights /= weights.sum()
        idx = np.random.choice(len(weights), size=len(weights), p=weights)
        return self._data['samples'][idx]


def format_results(raw_result, prior_transform, ndim, nlive, rwalk_K,
                   delta_logZ_threshold, context=None):
    """
    Convert raw sampler output (WhileLoopNSResult) into a Results object.

    Handles JAX-to-numpy conversion, trapezoidal weight computation,
    live point addition, and efficiency calculation.

    Parameters
    ----------
    raw_result : WhileLoopNSResult
        Raw output from the nested sampling loop.
    prior_transform : callable
        Prior transform function (unit cube -> physical space).
    ndim : int
        Number of dimensions.
    nlive : int
        Number of live points.
    rwalk_K : int
        Number of random walk steps per iteration.
    delta_logZ_threshold : float
        Convergence threshold used.

    Returns
    -------
    Results
        Dynesty-compatible results object.
    """
    r = raw_result

    # Convert dead point samples to numpy
    samples = _convert_to_numpy(r.samples)
    logL_samples = _convert_to_numpy(r.logL_samples)
    delta_logZ_trajectory = _convert_to_numpy(r.delta_logZ_trajectory)
    n_dead = len(logL_samples)

    # Calculate efficiency
    n_calls = r.n_iterations * rwalk_K
    eff = 100.0 * r.n_iterations / n_calls if n_calls > 0 else 0.0

    # Compute logvol for dead points: logX_i = -(i+1)/nlive
    logvol_dead = -(np.arange(n_dead) + 1) / nlive

    # Compute trapezoidal logwt for dead points
    logvol_padded = np.concatenate([[0.0], logvol_dead])
    dlogvol = np.diff(logvol_padded)
    logdvol = logvol_dead - dlogvol + np.log1p(-np.exp(dlogvol))
    logdvol2 = logdvol + np.log(0.5)

    logL_padded = np.concatenate([[-1e300], logL_samples])
    logwt_dead = np.logaddexp(logL_padded[1:], logL_padded[:-1]) + logdvol2

    # Add remaining live points (matching Dynesty's add_live_points())
    live_x = r.live_x
    live_logL = r.live_logL
    has_live_points = live_x is not None and live_logL is not None

    if has_live_points:
        live_logL_np = _convert_to_numpy(live_logL)
        live_x_np = _convert_to_numpy(live_x)

        # Transform live points to physical space
        live_samples = np.array([call_with_context(prior_transform, x, context)
                                 for x in live_x_np])

        # Sort live points by logL (ascending)
        sort_idx = np.argsort(live_logL_np)
        live_logL_sorted = live_logL_np[sort_idx]
        live_samples_sorted = live_samples[sort_idx]
        live_x_sorted = live_x_np[sort_idx]

        # Volume accounting for live points
        logvol_last_dead = logvol_dead[-1] if n_dead > 0 else 0.0
        logvol_live = np.log(1.0 - (np.arange(nlive) + 1.0) / (nlive + 1.0))
        logvol_live += logvol_last_dead

        # Build full sequence
        logL_all = np.concatenate([logL_samples, live_logL_sorted])
        logvol_all = np.concatenate([logvol_dead, logvol_live])
        samples_all = np.concatenate([samples, live_samples_sorted], axis=0)
        # samples_u should be unit-cube coords: dead_u from raw result, live are already unit cube
        dead_u = _convert_to_numpy(r.samples_u) if r.samples_u is not None else samples
        samples_u_all = np.concatenate([dead_u, live_x_sorted], axis=0)
        n_total = len(logL_all)

        # Recompute logwt for full sequence (trapezoidal)
        logvol_padded_all = np.concatenate([[0.0], logvol_all])
        dlogvol_all = np.diff(logvol_padded_all)
        logdvol_all = logvol_all - dlogvol_all + np.log1p(-np.exp(dlogvol_all))
        logdvol2_all = logdvol_all + np.log(0.5)

        logL_padded_all = np.concatenate([[-1e300], logL_all])
        logwt_all = np.logaddexp(logL_padded_all[1:], logL_padded_all[:-1]) + logdvol2_all

        logz_trajectory = np.logaddexp.accumulate(logwt_all)
        logzerr_trajectory = np.full(n_total, r.logZ_error)
    else:
        logL_all = logL_samples
        logvol_all = logvol_dead
        samples_all = samples
        dead_u2 = _convert_to_numpy(r.samples_u) if r.samples_u is not None else samples
        samples_u_all = dead_u2
        logwt_all = logwt_dead
        logz_trajectory = np.logaddexp.accumulate(logwt_dead)
        logzerr_trajectory = np.full(n_dead, r.logZ_error)
        n_total = n_dead

    # Replace any lingering -inf logL values with dynesty's _LOWL_VAL
    # so downstream integrals and plotting functions produce correct results.
    # Replace -inf with a large negative float32-safe value
    _LOWL_VAL = np.finfo(logL_all.dtype).min / 2 if hasattr(logL_all, 'dtype') else -1.0e300
    logL_all = np.where(np.isneginf(logL_all), _LOWL_VAL, logL_all)

    results_dict = {
        # Evidence
        'logz': float(r.logZ),
        'logzerr': float(r.logZ_error),
        'logz_trajectory': logz_trajectory,
        'logzerr_trajectory': logzerr_trajectory,

        # Information
        'information': float(r.H),

        # Samples and likelihoods
        'logl': logL_all,
        'logwt': logwt_all,
        'samples': samples_all,
        'samples_u': samples_u_all,

        # Volumes
        'logvol': logvol_all,

        # Trajectories for plotting
        'delta_logZ_trajectory': delta_logZ_trajectory,
        'scale_trajectory': _convert_to_numpy(r.scale_trajectory),

        # Diagnostics
        'nlive': nlive,
        'niter': int(r.n_iterations),
        # Likelihood (+ gradient) evaluations spent by the run.  For rwalk this
        # is the accumulated proposal count; for sample='hslice' it also
        # includes the autodiff gradient evaluations, matching dynesty's ncall.
        'ncall': int(getattr(r, 'ncall', 0)),
        'eff': eff,

        # JNesty-specific
        'acceptance_rate': float(r.acceptance_rate),
        'converged': bool(r.delta_logZ < delta_logZ_threshold),
        'delta_logz': float(r.delta_logZ),
        'delta_logZ_threshold': delta_logZ_threshold,
        'rwalk_K': rwalk_K,
    }

    return Results(results_dict)


def format_nautilus_results(native):
    """
    Build a Dynesty-compatible :class:`Results` from a JAX Nautilus sampler.

    Uses only the native engine's public API, so it works for the real
    :class:`~jnesty.nautilus.sampler.JaxSampler` as well as for stand-ins
    that provide the same attributes: ``posterior()``, ``log_z``,
    ``n_live``, ``n_like`` and ``n_eff``.

    This is a *plotting bridge*, not a re-derivation of the Nautilus
    evidence estimate: the native importance-nested-sampling semantics are
    untouched. In particular there is no ``logzerr`` (importance nested
    sampling has no run-to-run evidence error; the uncertainty proxy is the
    effective sample size ``n_eff``), and ``logvol`` /
    ``logz_trajectory`` are derived quantities that exist so the shared
    :mod:`jnesty.plotting` functions (which expect Dynesty-style arrays)
    can be used for Nautilus results.

    Parameters
    ----------
    native : JaxSampler
        A finished sampler, as returned by
        :attr:`jnesty.NautilusSampler.native_sampler`.

    Returns
    -------
    Results
        Results object with the per-sample arrays sorted by ascending
        ``logl`` and ``nan`` placeholders for the fields that only the
        Dynesty-style engine defines (``logzerr``, ``information``,
        ``eff``, ``acceptance_rate``, ``delta_logz``).

    Notes
    -----
    ``logwt`` holds the normalised log-weights returned by
    ``native.posterior()``; ``logvol = logwt - logl`` is therefore the
    per-point log prior volume up to the constant evidence normalisation.
    Since ``posterior()`` already normalises the weights,
    ``logz_trajectory`` (the running log-sum of the weights) converges to
    zero and is meaningful only as a plotting aid -- the evidence itself is
    the ``logz`` key, taken from ``native.log_z``.
    """
    points, log_w, log_l = native.posterior()
    points = _convert_to_numpy(points)
    log_w = _convert_to_numpy(log_w)
    log_l = _convert_to_numpy(log_l)

    # Sort every per-point array by ascending log-likelihood: the plotting
    # conventions in jnesty.plotting (runplot's -ln X axis) assume the
    # standard likelihood-ordered nested-sampling sequence.
    order = np.argsort(log_l)
    points = points[order]
    log_l = log_l[order]
    log_w = log_w[order]

    # Per-point log prior volume. Nautilus reports normalised log-weights and
    # no volume array, so derive it; minimum.accumulate keeps -ln X
    # monotonic despite the numerical jitter of the importance weights.
    logvol = np.minimum.accumulate(log_w - log_l)

    # Running log-sum of the per-point weights (plotting aid only).
    logz_trajectory = np.logaddexp.accumulate(log_w)

    log_z = native.log_z
    logz = float('nan') if log_z is None else float(log_z)

    n_live = int(native.n_live)
    n_like = int(native.n_like)

    results_dict = {
        # Evidence (no error estimate for importance nested sampling)
        'logz': logz,
        'logzerr': float('nan'),
        'logz_trajectory': logz_trajectory,

        # This engine has no information / efficiency / acceptance diagnostics
        'information': float('nan'),
        'eff': float('nan'),
        'acceptance_rate': float('nan'),
        # Summary() compat: a completed Nautilus run stops on the live-set
        # fraction / n_eff targets, so there is no delta-logZ criterion.
        'converged': True,
        'delta_logz': float('nan'),

        # Samples and likelihoods (physical space, ascending logl)
        'logl': log_l,
        'logwt': log_w,
        'samples': points,

        # Derived volumes
        'logvol': logvol,

        # Diagnostics
        'nlive': n_live,
        'niter': n_like,
        'n_like': n_like,
        'n_eff': float(native.n_eff),

        # Provenance
        'sampler': 'nautilus',
    }

    return Results(results_dict)


def format_dnest_results(native):
    """
    Build a Dynesty-compatible :class:`Results` from a DNest sampler.

    Uses only the native engine's public surface (``native_result`` plus
    diagnostics attributes), so it works for the real
    :class:`~jnesty.dfnest.engine.JaxDNestSampler` as well as for
    stand-ins that provide the same attributes.

    This is a *plotting bridge*, not a re-derivation of the DNest
    evidence: the native diffusive-nested-sampling semantics are
    untouched.  In particular:

    * ``logvol`` is the native postprocessed per-sample log prior volume
      (level-sandwich order statistics), not a ``-i/nlive`` sequence;
    * ``logwt`` holds the *unnormalised* evidence weights
      ``logp + logl`` so that ``logsumexp(logwt) == logz`` exactly, while
      ``posterior()`` returns the normalised weights;
    * ``logz_trajectory`` is the cumulative log-sum in likelihood order
      (a plotting aid that ends at ``logz``);
    * there is no ``logzerr`` for a single run (CDNest reports none; the
      uncertainty proxy is the entropy effective sample size ``n_eff``);
    * ``converged`` mirrors whether the save target was reached, which is
      a run-length statement, not an evidence-convergence claim.

    Parameters
    ----------
    native : JaxDNestSampler
        A finished engine, as returned by
        :attr:`jnesty.DNestSampler.native_sampler`.

    Returns
    -------
    Results
        Results object with the per-sample arrays sorted by ascending
        ``(logl, tiebreaker)`` and ``NaN`` placeholders for the fields
        that only the Dynesty-style engine defines (``logzerr``,
        ``eff``, ``delta_logz``).
    """
    result = native.native_result
    if result is None:
        raise RuntimeError(
            'the DNest sampler has no results yet; run() must reach its '
            'save target first')

    # Likelihood/tiebreaker ordering, matching the plotting conventions.
    order = np.lexsort((result.tie, result.logl))
    samples = _convert_to_numpy(result.samples)[order]
    samples_u = _convert_to_numpy(result.samples_u)[order]
    logl = _convert_to_numpy(result.logl)[order]
    logvol = _convert_to_numpy(result.logx)[order]
    logp = _convert_to_numpy(result.logp)[order]
    sample_levels = _convert_to_numpy(result.assignment)[order]

    # Unnormalised evidence weights: logsumexp(logwt) == logz exactly.
    logwt = logp + logl
    logz_trajectory = np.logaddexp.accumulate(logwt)

    max_num_saves = native.max_num_saves
    results_dict = {
        # Evidence (no error estimate for a single diffusive-NS run)
        'logz': float(result.logz),
        'logzerr': float('nan'),
        'logz_trajectory': logz_trajectory,

        # Information
        'information': float(result.information),

        # Samples and likelihoods (physical space, ascending logl)
        'logl': logl,
        'logwt': logwt,
        'samples': samples,
        'samples_u': samples_u,

        # Native prior volumes and level assignments
        'logvol': logvol,
        'sample_levels': sample_levels,

        # Diagnostics (no 'nlive': diffusive nested sampling has no live
        # point count, and the shared plotting handles its absence)
        'niter': int(native.n_steps),
        'n_steps': int(native.n_steps),
        'n_like': int(native.n_like),
        'n_saves': int(native.n_saves),
        'n_levels': int(native.n_levels),
        'n_eff': float(result.n_eff),
        'num_lanes': int(native.num_lanes),
        'num_particles': int(native.num_particles),
        'total_particles': int(native.total_particles),
        'acceptance_rate': float(native.acceptance_rate),
        'levels_finalized': bool(native.levels_finalized),

        # Summary()/compat placeholders: 'converged' here only means the
        # save target was reached (a run-length statement).
        'converged': bool(native.completed),
        'completed': bool(native.completed),
        'eff': float('nan'),
        'delta_logz': float('nan'),

        # Provenance and native options
        'sampler': 'dnest',
        'ptol': float(native.ptol),
        'max_num_saves': (None if max_num_saves is None
                          else int(max_num_saves)),
    }

    return Results(results_dict)


# ============================================================================
# FITS I/O
# ============================================================================

# Scalar keys stored in the PrimaryHDU header.
# (dict key, FITS keyword, comment)
_SCALAR_KEYS = [
    ('logz', 'LOGZ', 'Log evidence'),
    ('logzerr', 'LOGZERR', 'Log evidence error'),
    ('information', 'H', 'Information'),
    ('nlive', 'NLIVE', 'Number of live points'),
    ('niter', 'NITER', 'Number of iterations'),
    ('eff', 'EFF', 'Sampling efficiency (%)'),
    ('acceptance_rate', 'ACCRATE', 'Acceptance rate'),
    ('converged', 'CONVERGD', 'Convergence flag'),
    ('delta_logz', 'DLZ', 'Final delta logZ'),
    ('delta_logZ_threshold', 'DLZTHR', 'Delta logZ threshold'),
    ('rwalk_K', 'RWALKK', 'Random walk steps'),
]

# Optional engine-specific scalar keys, written only when present so the
# static/Dynesty-style and Nautilus results are unaffected.
_DNEST_SCALAR_KEYS = [
    ('sampler', 'SAMPLER', 'Sampler engine'),
    ('n_eff', 'NEFF', 'Effective sample size'),
    ('n_like', 'NLIKE', 'Likelihood evaluations'),
    ('n_steps', 'NSTEPS', 'MCMC updates'),
    ('n_saves', 'NSAVES', 'Saved particles'),
    ('n_levels', 'NLEVELS', 'Number of levels'),
    ('num_lanes', 'NLANES', 'Worker lanes'),
    ('num_particles', 'NPART', 'Particles per lane'),
    ('max_num_saves', 'MAXNSAVE', 'Save target'),
    ('ptol', 'PTOL', 'Level PTol'),
]


def save_results(results, filepath):
    """
    Save a Results object to a FITS file.

    Parameters
    ----------
    results : Results
        Results object to save.
    filepath : str
        Output file path.
    """
    from astropy.io import fits

    data = results._data if isinstance(results, Results) else results

    # Primary HDU with scalar metadata in header
    hdr = fits.Header()
    hdr['JNESTY'] = True
    for scalar_keys in (_SCALAR_KEYS, _DNEST_SCALAR_KEYS):
        for key, keyword, comment in scalar_keys:
            if key in data and data[key] is not None:
                val = data[key]
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, (float, np.floating)) and np.isnan(val):
                    # FITS headers cannot store NaN; use an undefined card
                    # value and restore NaN on load.
                    val = None
                hdr[keyword] = (val, comment)

    primary = fits.PrimaryHDU(header=hdr)

    # Binary table with per-sample arrays
    nsamples = len(data['logl'])
    ndim = data['samples'].shape[1] if data['samples'].ndim > 1 else 1

    cols = [
        fits.Column(name='LOGL', format='D', array=data['logl']),
        fits.Column(name='LOGWT', format='D', array=data['logwt']),
        fits.Column(name='LOGVOL', format='D', array=data['logvol']),
        fits.Column(name='SAMPLES', format=f'{ndim}D', array=data['samples']),
    ]

    if 'sample_levels' in data and data['sample_levels'] is not None:
        cols.append(
            fits.Column(name='LEVEL', format='J',
                        array=np.asarray(data['sample_levels'],
                                         dtype=np.int32))
        )

    if 'samples_u' in data and data['samples_u'] is not None:
        cols.append(
            fits.Column(name='SAMPLES_U', format=f'{ndim}D',
                        array=data['samples_u'])
        )

    if 'logz_trajectory' in data:
        cols.append(
            fits.Column(name='LOGZ_TRJ', format='D',
                        array=data['logz_trajectory'])
        )

    if 'logzerr_trajectory' in data:
        cols.append(
            fits.Column(name='LOGZERR_', format='D',
                        array=data['logzerr_trajectory'])
        )

    if 'delta_logZ_trajectory' in data:
        cols.append(
            fits.Column(name='DLZ_TRJ', format='D',
                        array=data['delta_logZ_trajectory'])
        )

    if 'scale_trajectory' in data:
        cols.append(
            fits.Column(name='SCALE_TR', format='D',
                        array=data['scale_trajectory'])
        )

    table = fits.BinTableHDU.from_columns(cols)
    hdu_list = fits.HDUList([primary, table])

    # Atomic write
    tmp = filepath + '.tmp'
    hdu_list.writeto(tmp, overwrite=True)
    os.replace(tmp, filepath)


def load_results(filepath):
    """
    Load a Results object from a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.

    Returns
    -------
    Results
        Loaded Results object.
    """
    from astropy.io import fits

    with fits.open(filepath) as hdul:
        hdr = hdul[0].header
        table = hdul[1]

        data = {}

        # Read scalars from header
        for scalar_keys in (_SCALAR_KEYS, _DNEST_SCALAR_KEYS):
            for key, keyword, _ in scalar_keys:
                if keyword in hdr:
                    value = hdr[keyword]
                    if value is None or isinstance(value, fits.Undefined):
                        value = float('nan')
                    data[key] = value

        # Convert boolean
        if 'converged' in data:
            data['converged'] = bool(data['converged'])

        # Read arrays from table
        data['logl'] = np.array(table.data['LOGL'])
        data['logwt'] = np.array(table.data['LOGWT'])
        data['logvol'] = np.array(table.data['LOGVOL'])
        data['samples'] = np.array(table.data['SAMPLES'])

        if 'SAMPLES_U' in table.columns.names:
            data['samples_u'] = np.array(table.data['SAMPLES_U'])

        if 'LOGZ_TRJ' in table.columns.names:
            data['logz_trajectory'] = np.array(table.data['LOGZ_TRJ'])

        if 'LOGZERR_' in table.columns.names:
            data['logzerr_trajectory'] = np.array(table.data['LOGZERR_'])

        if 'DLZ_TRJ' in table.columns.names:
            data['delta_logZ_trajectory'] = np.array(table.data['DLZ_TRJ'])

        if 'SCALE_TR' in table.columns.names:
            data['scale_trajectory'] = np.array(table.data['SCALE_TR'])

        if 'LEVEL' in table.columns.names:
            data['sample_levels'] = np.array(table.data['LEVEL'])

    return Results(data)
