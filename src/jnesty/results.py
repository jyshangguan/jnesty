"""
Results data structure and formatting for JNesty.

Provides a Dynesty-compatible Results class and the format_results()
function that converts raw sampler output into a structured results object.
"""

import numpy as np
from typing import Optional


def _convert_to_numpy(arr):
    """Convert JAX arrays or other array-like to numpy."""
    if hasattr(arr, '__array__'):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        return np.asarray(arr)


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
                   delta_logZ_threshold):
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
        live_samples = np.array([prior_transform(x) for x in live_x_np])

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
        samples_u_all = np.concatenate([samples, live_x_sorted], axis=0)
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
        samples_u_all = samples
        logwt_all = logwt_dead
        logz_trajectory = np.logaddexp.accumulate(logwt_dead)
        logzerr_trajectory = np.full(n_dead, r.logZ_error)
        n_total = n_dead

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
        'eff': eff,

        # JNesty-specific
        'acceptance_rate': float(r.acceptance_rate),
        'converged': bool(r.delta_logZ < delta_logZ_threshold),
        'delta_logz': float(r.delta_logZ),
        'delta_logZ_threshold': delta_logZ_threshold,
        'rwalk_K': rwalk_K,
    }

    return Results(results_dict)
