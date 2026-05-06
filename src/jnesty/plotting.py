"""
Plotting utilities for JAXNS following Dynesty's API conventions.

This module provides comprehensive plotting functions for visualizing nested
sampling results, designed to match Dynesty's API where possible.

Functions:
- runplot() - Evidence evolution and convergence diagnostics
- traceplot() - Particle evolution with 1D marginal posteriors
- cornerplot() - Professional corner plots with KDE
- cornerpoints() - Simple corner plots without KDE
- diagnostics() - Comprehensive diagnostic plots

Example:
-------
>>> from jaxns.gpu_rwalk import NestedSampler, plotting
>>> sampler = NestedSampler(loglike, prior_transform, ndim)
>>> sampler.run_nested()
>>> fig, axes = plotting.runplot(sampler.results)
>>> fig, axes = plotting.cornerplot(sampler.results, truths=true_params)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
import warnings


# Optional dependencies
try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False
    corner = None

try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    gaussian_kde = None


# ============================================================================
# Helper Functions
# ============================================================================

def _convert_to_numpy(arr: Any) -> np.ndarray:
    """
    Convert JAX arrays or other array-like to numpy arrays for plotting.

    Parameters
    ----------
    arr : array-like
        Input array (could be DeviceArray, numpy array, list, etc.)

    Returns
    -------
    np.ndarray
        Numpy array suitable for plotting.
    """
    if hasattr(arr, '__array__'):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        return np.asarray(arr)


def _get_weights(results: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Calculate normalized weights from log weights in results.

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS.

    Returns
    -------
    weights : np.ndarray or None
        Normalized weights (sum to 1), or None if logwt not available.
    """
    logwt = results.get('logwt', None)
    if logwt is None:
        return None

    logwt = _convert_to_numpy(logwt)

    # Handle empty arrays
    if len(logwt) == 0:
        return None

    # Subtract max for numerical stability
    logwt_norm = logwt - np.max(logwt)
    weights = np.exp(logwt_norm)
    weights /= np.sum(weights)

    return weights


def _downsample(samples: np.ndarray,
                weights: Optional[np.ndarray] = None,
                max_samples: int = 10000,
                rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Downsample large datasets for faster plotting.

    Parameters
    ----------
    samples : np.ndarray
        Sample array (n_samples, ndim).
    weights : np.ndarray, optional
        Sample weights (n_samples,).
    max_samples : int
        Maximum number of samples to return.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    samples_downsampled : np.ndarray
        Downsampled samples.
    weights_downsampled : np.ndarray or None
        Downsampled weights (or None if input weights was None).
    """
    n_samples = len(samples)
    if n_samples <= max_samples:
        return samples, weights

    if rng is None:
        rng = np.random.default_rng()

    # Weighted random sampling
    if weights is not None:
        indices = rng.choice(n_samples, max_samples, replace=False, p=weights)
        samples_downsampled = samples[indices]
        weights_downsampled = weights[indices]
        weights_downsampled /= np.sum(weights_downsampled)
        return samples_downsampled, weights_downsampled
    else:
        indices = rng.choice(n_samples, max_samples, replace=False)
        return samples[indices], None


def _get_plot_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and convert plotting data from results dictionary.

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS.

    Returns
    -------
    data : dict
        Dictionary with numpy arrays ready for plotting.
    """
    data = {}

    # Core data
    if 'samples' in results:
        data['samples'] = _convert_to_numpy(results['samples'])

    if 'samples_u' in results:
        data['samples_u'] = _convert_to_numpy(results['samples_u'])

    if 'logl' in results:
        data['logl'] = _convert_to_numpy(results['logl'])

    if 'logwt' in results:
        data['logwt'] = _convert_to_numpy(results['logwt'])

    if 'logz' in results:
        # logz might be a scalar or array
        logz_val = results['logz']
        if np.isscalar(logz_val):
            data['logz'] = np.array([logz_val])
        else:
            data['logz'] = _convert_to_numpy(logz_val)

    if 'logzerr' in results:
        logzerr_val = results['logzerr']
        if np.isscalar(logzerr_val):
            data['logzerr'] = np.array([logzerr_val])
        else:
            data['logzerr'] = _convert_to_numpy(logzerr_val)

    # Compute logvol (log prior volume) if not present
    if 'logvol' in results:
        data['logvol'] = _convert_to_numpy(results['logvol'])
    elif 'nlive' in results and 'niter' in results:
        # Compute logvol from iteration info
        # IMPORTANT: Use same formula as api.py to ensure consistency
        # Formula: logX_i = -(i+1)/nlive for i-th sample (0-indexed)
        # This ensures x-axis starts at 1/nlive (not 0), matching api.py
        nlive = results['nlive']
        niter = results['niter']
        iterations = np.arange(niter)
        logvol = -(iterations + 1) / nlive
        data['logvol'] = logvol

    # Compute weights if not present
    if 'weights' not in data:
        data['weights'] = _get_weights(results)

    # Metadata
    for key in ['nlive', 'niter', 'logz', 'logzerr', 'information']:
        if key in results and key not in data:
            data[key] = results[key]

    return data


# ============================================================================
# Core Plotting Functions
# ============================================================================

def runplot(results: Dict[str, Any],
            lnz_truth: Optional[float] = None,
            fig: Optional[plt.Figure] = None,
            axes: Optional[np.ndarray] = None,
            color: str = 'blue',
            lnz_color: str = 'red',
            truth_color: str = 'gray',
            plot_kwargs: Optional[Dict] = None,
            **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot summary of the nested sampling run (Dynesty-style).

    Creates a 4x1 subplot grid showing:
    1. Live Points vs -ln X
    2. Likelihood (normalized) vs -ln X
    3. Importance Weight vs -ln X
    4. Cumulative log evidence (ln Z) vs -ln X

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS (e.g., sampler.results).
    lnz_truth : float, optional
        True log evidence for comparison.
    fig : Figure, optional
        Figure to plot on.
    axes : ndarray, optional
        Array of axes to plot on (4x1).
    color : str
        Color for main plots.
    lnz_color : str
        Color for evidence plot.
    truth_color : str
        Color for truth values.
    plot_kwargs : dict, optional
        Additional plotting arguments.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of matplotlib axes (4x1).

    Examples
    --------
    >>> fig, axes = plotting.runplot(sampler.results, lnz_truth=-10.5)
    >>> plt.savefig('run_plot.png')
    """
    # Extract data
    data = _get_plot_data(results)

    # Check required data
    if 'logvol' not in data:
        raise ValueError("Cannot create run plot: 'logvol' not in results. "
                         "Ensure results contains prior volume information.")

    if 'logl' not in data:
        raise ValueError("Cannot create run plot: 'logl' not in results. "
                         "Ensure results contains log-likelihood information.")

    logvol = data['logvol']
    logl = data['logl']
    nlive = data.get('nlive', 500)
    niter = data.get('niter', len(logvol))

    # Setup default plot kwargs to match Dynesty styling
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault('linewidth', 5)
    plot_kwargs.setdefault('alpha', 0.7)

    # Setup figure - 4x1 layout like Dynesty
    if fig is None or axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    else:
        if axes.shape != (4, 1):
            raise ValueError("axes must be a 4x1 array")

    # Negative log prior volume for x-axis (Dynesty convention)
    neg_logvol = -logvol

    # Plot 1: Live Points vs -ln X
    ax = axes[0]
    # Create nlive trajectory matching thinned data length
    nlive_traj = np.full(len(logvol), nlive)
    ax.plot(neg_logvol, nlive_traj, color=color, **plot_kwargs)
    ax.set_ylabel('Live Points')
    ax.set_title('Live Points')
    ax.grid(True, alpha=0.3)

    # Plot 2: Likelihood (normalized) vs -ln X
    ax = axes[1]
    # Normalize likelihood (like Dynesty)
    logl_norm = logl - np.max(logl)
    ax.plot(neg_logvol, np.exp(logl_norm), color=color, **plot_kwargs)
    ax.set_ylabel('Likelihood (normalized)')
    ax.set_title('Likelihood')
    ax.grid(True, alpha=0.3)

    # Plot 3: Importance Weight vs -ln X
    ax = axes[2]
    if 'weights' in data:
        weights = data['weights']
        ax.plot(neg_logvol, weights, color=color, **plot_kwargs)
        ax.set_ylabel('Importance Weight')
    else:
        # Compute weights if not available
        logwt = data.get('logwt', None)
        if logwt is not None:
            logwt_norm = logwt - np.max(logwt)
            weights = np.exp(logwt_norm)
            ax.plot(neg_logvol, weights, color=color, **plot_kwargs)
            ax.set_ylabel('Importance Weight')
        else:
            ax.text(0.5, 0.5, 'Weights not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Importance Weight')
    ax.set_title('Importance Weight')
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative ln Z vs -ln X
    ax = axes[3]

    # Compute cumulative log evidence if not provided
    if 'logz_trajectory' in results:
        logz_cum = _convert_to_numpy(results['logz_trajectory'])
    else:
        # Approximate cumulative log evidence from samples
        logz_cum = np.zeros(len(logl))
        current_logz = -np.inf
        for i, (l, x) in enumerate(zip(logl, logvol)):
            if i == 0:
                logdZ = l + x
            else:
                logdX = x + np.log1p(-np.exp(logvol[i] - logvol[i-1]))
                logdZ = l + logdX
            current_logz = np.logaddexp(current_logz, logdZ)
            logz_cum[i] = current_logz

    # Plot cumulative evidence (create separate kwargs to avoid conflicts)
    evidence_kwargs = plot_kwargs.copy() if plot_kwargs else {}
    evidence_kwargs['linewidth'] = 2  # Override linewidth for evidence plot
    ax.plot(neg_logvol, logz_cum, color=lnz_color,
            label='JAXNS', **evidence_kwargs)

    # Add truth line
    if lnz_truth is not None:
        ax.axhline(y=lnz_truth, color=truth_color, linestyle='--',
                  label='Truth', linewidth=3)

    ax.set_xlabel(r'$-\ln X$')
    ax.set_ylabel('ln Z')
    ax.set_title('Evidence')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig, axes


def traceplot(results: Dict[str, Any],
              dims: Optional[Tuple[int, ...]] = None,
              truths: Optional[np.ndarray] = None,
              truth_color: str = 'red',
              show_titles: bool = True,
              trace_cmap: str = 'plasma',
              connect: bool = False,
              quantiles: Optional[Tuple[float, ...]] = None,
              fig: Optional[plt.Figure] = None,
              axes: Optional[np.ndarray] = None,
              max_samples: int = 5000,
              **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot trace plots for each parameter.

    Shows particle evolution over iterations (top panel) and
    1D marginal posterior distribution (bottom panel) for each dimension.

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS.
    dims : tuple of int, optional
        Which dimensions to plot (default: first min(ndim, 5) dimensions).
    truths : array_like, optional
        True parameter values for overplotting.
    truth_color : str
        Color for truth values.
    show_titles : bool
        Show parameter labels in titles.
    trace_cmap : str
        Colormap for trace points (colors by importance weight).
    connect : bool
        Connect points to show particle paths (not yet implemented).
    quantiles : tuple of float, optional
        Quantiles to show on 1D marginals (default: (0.16, 0.5, 0.84)).
    fig : Figure, optional
        Figure to plot on.
    axes : ndarray, optional
        Array of axes to plot on (ndim x 2).
    max_samples : int
        Maximum samples to plot (downsample if larger).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of matplotlib axes (ndim x 2).

    Examples
    --------
    >>> fig, axes = plotting.traceplot(sampler.results, truths=np.zeros(ndim))
    >>> plt.savefig('trace_plot.png')
    """
    # Extract data
    data = _get_plot_data(results)

    if 'samples' not in data:
        raise ValueError("Cannot create trace plot: 'samples' not in results.")

    samples = data['samples']
    weights = data.get('weights', None)
    ndim = samples.shape[1]

    # Determine which dimensions to plot
    if dims is None:
        dims = tuple(range(min(ndim, 5)))

    # Default quantiles
    if quantiles is None:
        quantiles = (0.16, 0.5, 0.84)

    # Setup figure
    n_dims_plot = len(dims)
    if fig is None or axes is None:
        fig, axes = plt.subplots(n_dims_plot, 2, figsize=(12, 3 * n_dims_plot))
    else:
        if axes.shape != (n_dims_plot, 2):
            raise ValueError(f"axes must be {n_dims_plot}x2 array")

    # Get colormap
    cmap = plt.get_cmap(trace_cmap)

    # Get logvol for x-axis (Dynesty convention)
    logvol = data.get('logvol', None)

    # Downsample if needed
    if len(samples) > max_samples:
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        indices = rng.choice(len(samples), max_samples, replace=False)
        samples = samples[indices]
        if weights is not None:
            weights = weights[indices]
        if logvol is not None:
            logvol = logvol[indices]

    # Get logvol for x-axis (Dynesty convention)
    if logvol is None:
        nlive = results.get('nlive', 500)
        logvol = -(np.arange(len(samples)) + 1) / nlive

    neg_logvol = -logvol

    # Plot each dimension
    for i, dim in enumerate(dims):
        # Top panel: Trace plot
        ax_trace = axes[i, 0]

        # Color by importance weight if available
        if weights is not None:
            # Normalize weights for colormap
            norm_weights = weights / np.max(weights)
            colors = cmap(norm_weights)
            ax_trace.scatter(neg_logvol, samples[:, dim], c=colors,
                           s=1, alpha=0.5, rasterized=True)
        else:
            ax_trace.plot(neg_logvol, samples[:, dim],
                         color='gray', alpha=0.5, linewidth=0.5)

        # Add truth line
        if truths is not None and dim < len(truths):
            ax_trace.axhline(y=truths[dim], color=truth_color,
                            linestyle='--', linewidth=2)

        ax_trace.set_xlabel(r'$-\ln X$')
        ax_trace.set_ylabel(f'Parameter {dim}')
        if show_titles:
            ax_trace.set_title(f'Parameter {dim} Trace')
        ax_trace.grid(True, alpha=0.3)

        # Set x-axis limits to match Dynesty
        ax_trace.set_xlim([0., np.max(neg_logvol)])

        # Bottom panel: 1D marginal posterior
        ax_marg = axes[i, 1]

        # Histogram
        ax_marg.hist(samples[:, dim], bins=30, density=True,
                    alpha=0.7, color='blue', edgecolor='black')

        # Add KDE if scipy available
        if SCIPY_AVAILABLE:
            try:
                kde = gaussian_kde(samples[:, dim], weights=weights)
                x_grid = np.linspace(samples[:, dim].min(),
                                   samples[:, dim].max(), 200)
                ax_marg.plot(x_grid, kde(x_grid), color='red',
                            linewidth=2, label='KDE')
            except Exception:
                pass  # Fall back to just histogram

        # Add truth line
        if truths is not None and dim < len(truths):
            ax_marg.axvline(x=truths[dim], color=truth_color,
                           linestyle='--', linewidth=2, label='Truth')

        # Add quantile markers
        quant_vals = np.quantile(samples[:, dim], quantiles)
        for q_val in quant_vals:
            ax_marg.axvline(x=q_val, color='green', linestyle=':',
                           linewidth=1, alpha=0.7)

        ax_marg.set_xlabel(f'Parameter {dim}')
        ax_marg.set_ylabel('Density')
        if show_titles:
            ax_marg.set_title(f'Parameter {dim} Marginal')
        ax_marg.grid(True, alpha=0.3)

        # Add legend if we added KDE or truth
        if (SCIPY_AVAILABLE) or (truths is not None and dim < len(truths)):
            ax_marg.legend(fontsize='small')

    plt.tight_layout()

    return fig, axes


def cornerplot(results: Dict[str, Any],
              dims: Optional[Tuple[int, ...]] = None,
              truths: Optional[np.ndarray] = None,
              truth_color: str = 'red',
              show_titles: bool = True,
              quantiles: Optional[Tuple[float, ...]] = None,
              levels: Optional[Tuple[float, ...]] = None,
              title_kwargs: Optional[Dict] = None,
              color: str = 'blue',
              quiet: bool = False,
              fig: Optional[plt.Figure] = None,
              weights: Optional[np.ndarray] = None,
              max_samples: int = 10000,
              **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Make a corner plot showing 1D and 2D marginal posteriors.

    Uses the 'corner' package if available, otherwise falls back to
    a simple implementation using matplotlib.

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS.
    dims : tuple of int, optional
        Which dimensions to plot.
    truths : array_like, optional
        True parameter values.
    truth_color : str
        Color for truth values.
    show_titles : bool
        Show titles on 1D marginal plots.
    quantiles : tuple of float, optional
        Quantiles to show on 1D plots (default: (0.16, 0.5, 0.84)).
    levels : tuple of float, optional
        Credible levels for 2D contours (default: (0.68, 0.95)).
    title_kwargs : dict, optional
        Keyword arguments for titles.
    color : str
        Color for plots.
    quiet : bool
        Suppress warnings about corner module.
    fig : Figure, optional
        Figure to plot on.
    weights : array_like, optional
        Sample weights for plotting (default: use results['logwt']).
    max_samples : int
        Maximum samples to plot (downsample if larger).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of matplotlib axes.

    Examples
    --------
    >>> fig, axes = plotting.cornerplot(sampler.results, truths=np.zeros(ndim))
    >>> plt.savefig('corner_plot.png')
    """
    # Extract data
    data = _get_plot_data(results)

    if 'samples' not in data:
        raise ValueError("Cannot create corner plot: 'samples' not in results.")

    samples = data['samples']

    # Use provided weights or get from results
    if weights is None:
        weights = data.get('weights', None)

    # Downsample if needed
    if len(samples) > max_samples:
        rng = np.random.default_rng(42)
        samples, weights = _downsample(samples, weights, max_samples, rng)

    # Determine which dimensions to plot
    if dims is None:
        dims = tuple(range(samples.shape[1]))

    # Extract only requested dimensions
    samples_plot = samples[:, dims]

    # Handle truths for selected dimensions
    if truths is not None:
        truths_plot = truths[list(dims)] if len(truths) > max(dims) else None
    else:
        truths_plot = None

    # Default quantiles and levels
    if quantiles is None:
        quantiles = (0.16, 0.5, 0.84)
    if levels is None:
        levels = (0.68, 0.95)

    # Try using corner package if available
    if CORNER_AVAILABLE:
        try:
            fig = corner.corner(
                samples_plot,
                weights=weights,
                truths=truths_plot,
                truth_color=truth_color,
                show_titles=show_titles,
                quantiles=quantiles,
                levels=levels,
                title_kwargs=title_kwargs or {},
                color=color,
                fig=fig,
                **kwargs
            )
            axes = np.array(fig.axes).reshape(samples_plot.shape[1], samples_plot.shape[1])
            return fig, axes
        except Exception as e:
            if not quiet:
                warnings.warn(f"corner package failed: {e}. Falling back to simple implementation.")
            # Fall through to simple implementation

    # Fallback: Simple corner plot implementation
    return _cornerplot_simple(samples_plot, weights=weights,
                             truths=truths_plot, truth_color=truth_color,
                             show_titles=show_titles, quantiles=quantiles,
                             color=color, fig=fig)


def _cornerplot_simple(samples: np.ndarray,
                      weights: Optional[np.ndarray] = None,
                      truths: Optional[np.ndarray] = None,
                      truth_color: str = 'red',
                      show_titles: bool = True,
                      quantiles: Tuple[float, ...] = (0.16, 0.5, 0.84),
                      color: str = 'blue',
                      alpha: float = 0.3,
                      fig: Optional[plt.Figure] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Simple corner plot implementation (fallback when corner package not available).

    Parameters
    ----------
    samples : np.ndarray
        Samples (n_samples, ndim).
    weights : np.ndarray, optional
        Sample weights.
    truths : np.ndarray, optional
        True parameter values.
    truth_color : str
        Color for truth values.
    show_titles : bool
        Show titles on plots.
    quantiles : tuple of float
        Quantiles to show on 1D plots.
    color : str
        Color for plots.
    fig : Figure, optional
        Figure to plot on.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of matplotlib axes.
    """
    ndim = samples.shape[1]

    # Setup figure
    if fig is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim))
    else:
        axes = np.array(fig.axes).reshape(ndim, ndim)

    # Plot each panel
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(samples[:, j], bins=30, density=True,
                       alpha=0.7, color=color, edgecolor='black')

                # Add quantile markers
                quant_vals = np.quantile(samples[:, j], quantiles)
                for q_val in quant_vals:
                    ax.axvline(x=q_val, color='green', linestyle=':',
                              linewidth=1, alpha=0.7)

                # Add truth line
                if truths is not None and j < len(truths):
                    ax.axvline(x=truths[j], color=truth_color,
                              linestyle='--', linewidth=2)

                if show_titles:
                    ax.set_title(f'Param {j}')

                ax.set_ylabel('Density')

            else:
                # Off-diagonal: 2D scatter plot
                if weights is not None:
                    # Weighted scatter (color by weight)
                    sc = ax.scatter(samples[:, j], samples[:, i],
                                   c=weights, s=1, alpha=0.3,
                                  cmap='viridis', rasterized=True)
                else:
                    ax.scatter(samples[:, j], samples[:, i],
                             s=1, alpha=0.3, color=color, rasterized=True)

                # Add truth lines
                if truths is not None:
                    if j < len(truths):
                        ax.axvline(x=truths[j], color=truth_color,
                                  linestyle='--', linewidth=1, alpha=0.7)
                    if i < len(truths):
                        ax.axhline(y=truths[i], color=truth_color,
                                  linestyle='--', linewidth=1, alpha=0.7)

            ax.grid(True, alpha=0.3)

            # Only show axis labels on edge plots
            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(f'Param {j}')

            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f'Param {i}')

    plt.tight_layout()

    return fig, axes


def cornerpoints(results: Dict[str, Any],
                dims: Optional[Tuple[int, ...]] = None,
                truths: Optional[np.ndarray] = None,
                truth_color: str = 'red',
                show_titles: bool = True,
                color: str = 'blue',
                alpha: float = 0.3,
                fig: Optional[plt.Figure] = None,
                max_samples: int = 10000,
                **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Make a corner plot showing sample positions (no KDE, faster).

    Faster alternative to cornerplot() for large sample sets.

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS.
    dims : tuple of int, optional
        Which dimensions to plot.
    truths : array_like, optional
        True parameter values.
    truth_color : str
        Color for truth values.
    show_titles : bool
        Show titles on plots.
    color : str
        Color for points.
    alpha : float
        Transparency for points.
    fig : Figure, optional
        Figure to plot on.
    max_samples : int
        Maximum samples to plot (downsample if larger).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of matplotlib axes.

    Examples
    --------
    >>> fig, axes = plotting.cornerpoints(sampler.results)
    >>> plt.savefig('corner_points.png')
    """
    # Extract data
    data = _get_plot_data(results)

    if 'samples' not in data:
        raise ValueError("Cannot create corner points plot: 'samples' not in results.")

    samples = data['samples']
    weights = data.get('weights', None)

    # Downsample if needed
    if len(samples) > max_samples:
        rng = np.random.default_rng(42)
        samples, weights = _downsample(samples, weights, max_samples, rng)

    # Determine which dimensions to plot
    if dims is None:
        dims = tuple(range(samples.shape[1]))

    # Extract only requested dimensions
    samples_plot = samples[:, dims]

    # Handle truths for selected dimensions
    if truths is not None:
        truths_plot = truths[list(dims)] if len(truths) > max(dims) else None
    else:
        truths_plot = None

    # Use simple implementation (no KDE)
    return _cornerplot_simple(samples_plot, weights=None,
                             truths=truths_plot, truth_color=truth_color,
                             show_titles=show_titles, color=color,
                             alpha=alpha, fig=fig)


def diagnostics(results: Dict[str, Any],
                fig: Optional[plt.Figure] = None,
                **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Comprehensive diagnostic plot showing convergence and efficiency.

    Creates a 2x2 subplot grid showing:
    1. Evidence evolution (logZ vs iteration)
    2. Convergence metric (delta_logZ vs iteration)
    3. Acceptance rate evolution
    4. Sampling efficiency

    Parameters
    ----------
    results : dict
        Results dictionary from JAXNS.
    fig : Figure, optional
        Figure to plot on.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of matplotlib axes (2x2).

    Examples
    --------
    >>> fig, axes = plotting.diagnostics(sampler.results)
    >>> plt.savefig('diagnostics.png')
    """
    # Setup figure
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        axes = np.array(fig.axes).reshape(2, 2)

    # Extract data
    data = _get_plot_data(results)

    # Plot 1: Evidence evolution
    ax1 = axes[0, 0]
    if 'delta_logZ_trajectory' in results:
        delta_traj = _convert_to_numpy(results['delta_logZ_trajectory'])
        iterations = np.arange(len(delta_traj))

        # Compute cumulative evidence (simplified)
        logz_traj = np.zeros(len(delta_traj))
        current_logz = -np.inf
        for i, dlogz in enumerate(delta_traj):
            if i == 0:
                current_logz = dlogz
            else:
                current_logz = np.logaddexp(current_logz, dlogz)
            logz_traj[i] = current_logz

        ax1.plot(iterations, logz_traj, color='blue', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cumulative logZ')
        ax1.set_title('Evidence Evolution')
        ax1.grid(True, alpha=0.3)

        # Add final value
        if 'logz' in data:
            logz_final = _convert_to_numpy(data['logz'])
            if hasattr(logz_final, 'item'):
                logz_final = float(logz_final.item())
            else:
                logz_final = float(logz_final)
            ax1.axhline(y=logz_final, color='red', linestyle='--',
                       label=f'Final: {logz_final:.3f}')
            ax1.legend()

    # Plot 2: Convergence metric
    ax2 = axes[0, 1]
    if 'delta_logZ_trajectory' in results:
        delta_traj = _convert_to_numpy(results['delta_logZ_trajectory'])
        iterations = np.arange(len(delta_traj))

        ax2.plot(iterations, delta_traj, color='green', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('delta_logZ')
        ax2.set_title('Convergence Metric')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # Add convergence threshold
        if 'delta_logZ_threshold' in results or 'delta_logz' in results:
            threshold = results.get('delta_logZ_threshold', 0.01)
            threshold = results.get('delta_logz', threshold)
            threshold = _convert_to_numpy(threshold)
            if hasattr(threshold, 'item'):
                threshold = float(threshold.item())
            else:
                threshold = float(threshold)
            ax2.axhline(y=threshold, color='red', linestyle='--',
                       label=f'Threshold: {threshold:.3f}')
            ax2.legend()

    # Plot 3: Key Metrics Bar Chart
    ax3 = axes[1, 0]

    # Helper function to safely convert to Python native types
    def safe_float(val):
        """Convert JAX arrays or numpy arrays to Python float."""
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)

    def safe_int(val):
        """Convert JAX arrays or numpy arrays to Python int."""
        if hasattr(val, 'item'):
            return int(val.item())
        return int(val)

    # Collect key metrics
    metrics = []
    labels = []

    if 'logz' in data:
        labels.append('logZ')
        metrics.append(safe_float(data['logz']))

    if 'information' in data:
        labels.append('Information H')
        metrics.append(safe_float(data['information']))

    if 'niter' in data:
        labels.append('Iterations')
        metrics.append(safe_float(data['niter']))

    if 'acceptance_rate' in results:
        labels.append('Acc. Rate (%)')
        metrics.append(safe_float(results['acceptance_rate']) * 100)

    if 'nlive' in data:
        labels.append('Live Points')
        metrics.append(safe_float(data['nlive']))

    # Create horizontal bar chart
    if metrics:
        y_pos = np.arange(len(metrics))
        bars = ax3.barh(y_pos, metrics, color='steelblue', alpha=0.7)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics)):
            if 'logZ' in labels[i] or 'Information H' in labels[i]:
                label_text = f'{val:.3f}'
            elif 'Acc. Rate' in labels[i]:
                label_text = f'{val:.1f}%'
            else:
                label_text = f'{int(val)}'
            ax3.text(val + max(metrics) * 0.01, bar.get_y() + bar.get_height()/2,
                   label_text, va='center', fontsize=10)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        ax3.set_xlabel('Value')
        ax3.set_title('Key Metrics')
        ax3.grid(True, alpha=0.3, axis='x')

        # Add convergence status as text annotation
        if 'converged' in results:
            status = "✓ Converged" if results['converged'] else "✗ Not converged"
            color = 'green' if results['converged'] else 'red'
            ax3.text(0.98, 0.02, status, transform=ax3.transAxes,
                   ha='right', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2),
                   color=color)

    # Plot 4: Efficiency metrics
    ax4 = axes[1, 1]
    if 'nlive' in data and 'niter' in data:
        nlive = data['nlive']
        niter = data['niter']

        # Compute efficiency
        # Efficiency = 1 / (K * nlive) where K is walk steps
        # This is approximate
        if 'rwalk_K' in results:
            K = results['rwalk_K']
            theoretical_eff = 100.0 / (K * nlive)
        else:
            theoretical_eff = None

        metrics = []
        labels = []

        if 'niter' in data:
            labels.append('Iterations')
            metrics.append(data['niter'])

        if theoretical_eff is not None:
            labels.append('Efficiency (%)')
            metrics.append(theoretical_eff)

        if metrics:
            y_pos = np.arange(len(metrics))
            ax4.barh(y_pos, metrics, color='steelblue')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(labels)
            ax4.set_xlabel('Value')
            ax4.set_title('Efficiency Metrics')
            ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    return fig, axes
