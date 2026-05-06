"""
Simple Dynesty-style API for JAXNS GPU random walk nested sampling.

This module provides a user-friendly class-based interface that follows
Dynesty's design philosophy:
- Simple 3-argument initialization
- Auto-tuned parameters based on dimensionality
- Hidden complexity (rwalk_K, scale tuning)
- Progressive disclosure for advanced users
- Dictionary-based results

Example:
    >>> from jaxns.gpu_rwalk import NestedSampler
    >>>
    >>> def loglike(x):
    ...     return -0.5 * np.sum(x**2)
    >>>
    >>> def prior_transform(u):
    ...     return (u - 0.5) * 10.0
    >>>
    >>> # Simple usage (auto-tuned parameters)
    >>> sampler = NestedSampler(loglike, prior_transform, ndim=5)
    >>> sampler.run_nested()
    >>> results = sampler.results
    >>> print(f"logZ = {results['logz']:.4f} ± {results['logzerr']:.4f}")
"""

from typing import Optional, Callable, Dict, Any, Union
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import warnings

from .while_loop_sampler import run_nested_sampling_while_loop, WhileLoopNSConfig


class NestedSampler:
    """
    Simple nested sampler following Dynesty's API design.

    This provides a user-friendly interface for GPU-accelerated nested sampling
    with automatic parameter tuning based on dimensionality.

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood function. Must accept a single parameter x (physical space)
        and return log-likelihood as a float.
        Signature: loglikelihood(x) -> float

    prior_transform : callable
        Prior transform function. Must accept a single parameter u (unit cube [0,1]^ndim)
        and return transformed parameters in physical space.
        Signature: prior_transform(u) -> array

    ndim : int
        Number of dimensions of the parameter space.

    nlive : int, optional
        Number of live points. Default: 500

    rwalk_K : int or None, optional
        Number of random walk steps per iteration. If None, automatically set to
        max(25, ndim + 20) following Dynesty's approach. Default: None

    rwalk_step_scale : float or None, optional
        Initial proposal scale. If None, automatically set to
        min(1.0, 1.0 / sqrt(ndim)) for high-dimensional robustness. Default: None

    target_acceptance : float, optional
        Target acceptance rate for scale adaptation. Default: 0.5

    scale_adapt_interval : int, optional
        Iterations between scale adaptation steps. Default: 100

    device : str, optional
        Device to use ('gpu' or 'cpu'). Default: 'gpu'

    verbose : bool, optional
        Whether to print progress information. Default: True

    bound : str, optional
        Bounding method. One of 'none', 'single', 'multi'. Default: 'none'

    bound_update_interval : int, float, or None, optional
        Controls periodic bound refitting. Default: None (automatic).
        - None: for ``bound='multi'`` updates every ``nlive`` iterations
          (matching Dynesty); otherwise 0 (fit once at start).
        - int > 0: absolute iterations between updates.
        - float > 0: ratio × nlive iterations.
        - 0: fit once at start, no periodic updates.

    max_ellipsoids : int, optional
        Maximum ellipsoids for multi-ellipsoid decomposition. Default: 20

    Examples
    --------
    Simple usage (auto-tuned parameters):

    >>> def loglike(x):
    ...     return -0.5 * np.sum(x**2)
    >>> def prior_transform(u):
    ...     return 20. * u - 10.
    >>> sampler = NestedSampler(loglike, prior_transform, ndim=3)
    >>> sampler.run_nested()
    >>> results = sampler.results
    >>> print(f"logZ = {results['logz']} ± {results['logzerr']}")

    Expert usage (manual control):

    >>> sampler = NestedSampler(
    ...     loglike, prior_transform, ndim=3,
    ...     nlive=1000,
    ...     rwalk_K=50,
    ...     rwalk_step_scale=0.5
    ... )
    >>> sampler.run_nested(max_iterations=200000)
    """

    def __init__(
        self,
        loglikelihood: Callable[[jnp.ndarray], float],
        prior_transform: Callable[[jnp.ndarray], jnp.ndarray],
        ndim: int,
        nlive: int = 500,
        rwalk_K: Optional[int] = None,
        rwalk_step_scale: Optional[float] = None,
        target_acceptance: float = 0.5,
        scale_adapt_interval: int = 1,  # Per-walk adaptation (matches Dynesty)
        device: str = 'gpu',
        verbose: bool = True,
        bound: str = 'none',
        bound_update_interval: Optional[Union[int, float]] = None,
        max_ellipsoids: int = 20
    ):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.nlive = nlive
        self.target_acceptance = target_acceptance
        self.scale_adapt_interval = scale_adapt_interval
        self.device = device
        self.verbose = verbose
        self.bound = bound
        self.max_ellipsoids = max_ellipsoids

        # Resolve bound_update_interval
        # None: automatic default (matches Dynesty's behavior)
        #   - bound='multi': update every nlive iterations (ratio=1.0)
        #   - otherwise: fit once at start (0)
        # int > 0: absolute iterations between updates
        # float > 0: ratio × nlive iterations between updates
        # 0: fit once at start, never update
        if bound_update_interval is None:
            if bound == 'multi':
                bound_update_interval = nlive  # Dynesty default: ratio=1.0
            else:
                bound_update_interval = 0
        elif isinstance(bound_update_interval, float) and bound_update_interval > 0:
            bound_update_interval = int(bound_update_interval * nlive)

        self.bound_update_interval = bound_update_interval

        if verbose and bound == 'multi':
            if bound_update_interval == 0:
                print(f"Bound update: once at start (no periodic updates)")
            else:
                print(f"Bound update interval: {bound_update_interval} iterations "
                      f"({bound_update_interval / nlive:.1f} × nlive)")

        # Auto-tune parameters if not specified
        if rwalk_K is None:
            rwalk_K = max(25, ndim + 20)
            if verbose:
                print(f"Auto-tuned rwalk_K = {rwalk_K} (based on ndim={ndim})")

        if rwalk_step_scale is None:
            rwalk_step_scale = 1.0
            if verbose:
                print(f"Auto-tuned rwalk_step_scale = {rwalk_step_scale:.4f} (Dynesty default)")

        self.rwalk_K = rwalk_K
        self.rwalk_step_scale = rwalk_step_scale

        # Set device
        if device == 'cpu':
            jax.config.update('jax_platform_name', 'cpu')

        # Storage for results
        self._results = None
        self._raw_result = None

    def run_nested(
        self,
        max_iterations: int = 100000,
        delta_logZ_threshold: float = 0.01,
        print_progress: bool = True
    ) -> None:
        """
        Run the nested sampling algorithm.

        This executes the nested sampling algorithm with the configured
        parameters. Results are stored internally and accessed via the
        `results` property.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations. Default: 100000

        delta_logZ_threshold : float, optional
            Convergence threshold for delta_logZ. Default: 0.01

        print_progress : bool, optional
            Whether to display progress bar. Default: True
            Set to False for maximum performance (eliminates ~20% overhead).

        Examples
        --------
        >>> sampler = NestedSampler(loglike, prior_transform, ndim=3)
        >>> sampler.run_nested()
        >>> results = sampler.results

        >>> # With custom stopping criteria
        >>> sampler.run_nested(max_iterations=50000, delta_logZ_threshold=0.1)

        >>> # Without progress bar (faster)
        >>> sampler.run_nested(print_progress=False)
        """
        if self.verbose:
            print("="*70)
            print("JAXNS Nested Sampling")
            print("="*70)
            print(f"Dimensions: {self.ndim}")
            print(f"Live points: {self.nlive}")
            print(f"Max iterations: {max_iterations}")
            print(f"Device: {self.device.upper()}")
            print(f"Walk steps per iteration: {self.rwalk_K}")
            print(f"Initial step scale: {self.rwalk_step_scale:.4f}")
            print(f"Convergence threshold: delta_logZ < {delta_logZ_threshold}")
            print("="*70)
            print()

        # Store threshold for later use in results formatting
        self._delta_logZ_threshold = delta_logZ_threshold

        # Create prior_sample function (samples from unit cube)
        def prior_sample(key):
            return random.uniform(key, shape=(self.ndim,), minval=0.0, maxval=1.0)

        # Create NSConfig
        config = WhileLoopNSConfig(
            nlive=self.nlive,
            max_iterations=max_iterations,
            delta_logZ_threshold=delta_logZ_threshold,
            rwalk_K=self.rwalk_K,
            rwalk_step_scale=self.rwalk_step_scale,
            target_acceptance=self.target_acceptance,
            scale_adapt_interval=self.scale_adapt_interval,
            verbose=self.verbose,
            print_progress=print_progress,
            bound=self.bound,
            bound_update_interval=self.bound_update_interval,
            max_ellipsoids=self.max_ellipsoids
        )

        # Run sampling
        key = random.PRNGKey(42)  # Fixed seed for reproducibility
        self._raw_result = run_nested_sampling_while_loop(
            loglikelihood_fn=self.loglikelihood,
            prior_sample_fn=prior_sample,
            ndim=self.ndim,
            config=config,
            key=key,
            prior_transform_fn=self.prior_transform
        )

        if self.verbose:
            print()
            print("="*70)
            print("✅ Sampling Complete!")
            print("="*70)

    @property
    def results(self) -> Dict[str, Any]:
        """
        Results dictionary following Dynesty's format.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'logz': log evidence
            - 'logzerr': error on log evidence
            - 'information': information H
            - 'logl': log-likelihoods of samples
            - 'logvol': log prior volumes
            - 'samples': samples in physical space
            - 'samples_u': samples in unit cube
            - 'nlive': number of live points
            - 'niter': number of iterations
            - 'eff': sampling efficiency (%)
            - 'acceptance_rate': final acceptance rate
            - 'final_scale': final proposal scale
            - 'converged': convergence status
            - 'delta_logz': final delta_logZ

        Raises
        ------
        RuntimeError
            If run_nested() has not been called yet.

        Examples
        --------
        >>> sampler.run_nested()
        >>> results = sampler.results
        >>> print(f"logZ = {results['logz']}")
        """
        if self._raw_result is None:
            raise RuntimeError("Must call run_nested() before accessing results")

        if self._results is None:
            self._results = self._format_results()

        return self._results

    def _format_results(self) -> Dict[str, Any]:
        """Format raw results into Dynesty-style dictionary."""
        r = self._raw_result

        # Convert dead point samples to numpy
        samples = np.array(r.samples)
        logL_samples = np.array(r.logL_samples)
        delta_logZ_trajectory = np.array(r.delta_logZ_trajectory)
        n_dead = len(logL_samples)

        # Calculate efficiency (approximate)
        n_calls = r.n_iterations * self.rwalk_K
        eff = 100.0 * r.n_iterations / n_calls if n_calls > 0 else 0.0

        # Compute logvol for dead points
        # logX_i = -(i+1)/nlive (linear approximation)
        logvol_dead = -(np.arange(n_dead) + 1) / self.nlive

        # Compute proper cumulative logZ and logwt for dead points (trapezoidal)
        logvol_padded = np.concatenate([[0.0], logvol_dead])
        dlogvol = np.diff(logvol_padded)
        logdvol = logvol_dead - dlogvol + np.log1p(-np.exp(dlogvol))
        logdvol2 = logdvol + np.log(0.5)

        logL_padded = np.concatenate([[-1e300], logL_samples])
        logwt_dead = np.logaddexp(logL_padded[1:], logL_padded[:-1]) + logdvol2

        # Add remaining live points (matching Dynesty's add_live_points())
        # This creates the sparse tail in trace plots that matches Dynesty's appearance
        live_x = r.live_x
        live_logL = r.live_logL
        has_live_points = live_x is not None and live_logL is not None

        if has_live_points:
            live_logL_np = np.array(live_logL)
            live_x_np = np.array(live_x)

            # Transform live points to physical space
            live_samples = np.array([self.prior_transform(x) for x in live_x_np])

            # Sort live points by logL (ascending)
            sort_idx = np.argsort(live_logL_np)
            live_logL_sorted = live_logL_np[sort_idx]
            live_samples_sorted = live_samples[sort_idx]
            live_x_sorted = live_x_np[sort_idx]

            # Volume accounting for live points (Dynesty's formula)
            # logvols = log(1 - (i+1)/(nlive+1)) + logvol_last_dead
            logvol_last_dead = logvol_dead[-1] if n_dead > 0 else 0.0
            logvol_live = np.log(1.0 - (np.arange(self.nlive) + 1.0) / (self.nlive + 1.0))
            logvol_live += logvol_last_dead

            # Compute logwt for live points (trapezoidal with last dead point)
            # Connect last dead point to first live point
            logL_last_dead = logL_samples[-1] if n_dead > 0 else -1e300

            # Build full logL sequence: dead points + live points
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

        return {
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
            'scale_trajectory': np.array(r.scale_trajectory),

            # Diagnostics
            'nlive': self.nlive,
            'niter': int(r.n_iterations),
            'eff': eff,

            # JAXNS-specific
            'acceptance_rate': float(r.acceptance_rate),
            'converged': bool(r.delta_logZ < self._delta_logZ_threshold),
            'delta_logz': float(r.delta_logZ),
            'delta_logZ_threshold': self._delta_logZ_threshold,
            'rwalk_K': self.rwalk_K,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the results."""
        if self._raw_result is None:
            print("No results yet. Call run_nested() first.")
            return

        r = self.results

        print()
        print("="*70)
        print("Summary")
        print("="*70)
        print(f"Log evidence: logZ = {r['logz']:.4f} ± {r['logzerr']:.4f}")
        print(f"Information: H = {r['information']:.4f}")
        print(f"Iterations: {r['niter']}")
        print(f"Efficiency: {r['eff']:.1f}%")
        print(f"Acceptance rate: {r['acceptance_rate']:.1%}")
        print(f"Converged: {r['converged']}")
        print(f"Final delta_logZ: {r['delta_logz']:.6f}")
        print("="*70)

    def get_samples(self, weights: bool = False) -> np.ndarray:
        """
        Get posterior samples.

        Parameters
        ----------
        weights : bool, optional
            If True, return samples with importance weights.
            For now, this returns unweighted samples. Default: False

        Returns
        -------
        samples : ndarray
            Posterior samples in physical space.
        """
        if weights:
            print("Warning: Weighted sampling not yet implemented. Returning unweighted samples.")

        return self.results['samples']

    def get_logz(self) -> tuple[float, float]:
        """
        Get log evidence and its error.

        Returns
        -------
        logz : float
            Log evidence
        logzerr : float
            Error on log evidence
        """
        r = self.results
        return r['logz'], r['logzerr']

    def to_dynesty_results(self):
        """
        Convert results to Dynesty Results object for use with Dynesty plotting.

        This creates a Dynesty-compatible Results object from the JAXNS results,
        allowing use of Dynesty's mature plotting functions.

        Returns
        -------
        dynesty_results : dynesty.utils.Results
            Dynesty Results object compatible with dynesty.plotting functions

        Examples
        --------
        >>> sampler.run_nested()
        >>> dynesty_results = sampler.to_dynesty_results()
        >>> from dynesty import plotting as dyplot
        >>> fig, axes = dyplot.traceplot(dynesty_results)
        """
        try:
            from dynesty.utils import Results
        except ImportError:
            raise ImportError("dynesty package is required to use this method. "
                            "Install with: pip install dynesty")

        r = self.results

        # Create samples_id (sequential IDs for each sample)
        # This is required by Dynesty but not used unless connect=True in traceplot
        samples_id = np.arange(len(r['samples']))

        # Build results dictionary with Dynesty-required keys
        # Required keys: samples_u, samples_id, logl, samples
        dynesty_dict = {
            'samples_u': r['samples_u'],  # Unit cube samples
            'samples_id': samples_id,  # Sample IDs
            'logl': r['logl'],  # Log-likelihoods
            'samples': r['samples'],  # Physical space samples
            'nlive': r['nlive'],  # Number of live points (static nested sampling)
            'niter': r['niter'],  # Number of iterations
            'logwt': r['logwt'],  # Log importance weights
            'logvol': r['logvol'],  # Log prior volumes
            'logz': r['logz_trajectory'],  # Cumulative logZ (required for importance_weights)
            'logzerr': r['logzerr_trajectory'],  # Cumulative logZ error
        }

        # Create Dynesty Results object
        return Results(dynesty_dict)

    # ============================================================================
    # Plotting Convenience Methods (using Dynesty)
    # ============================================================================

    def plot_run(self, lnz_truth: Optional[float] = None, **kwargs):
        """
        Create run plot showing evidence evolution.

        Convenience method that uses Dynesty's plotting.runplot().

        Parameters
        ----------
        lnz_truth : float, optional
            True log evidence for comparison.
        **kwargs
            Additional arguments passed to dynesty.plotting.runplot().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : ndarray
            Array of matplotlib axes

        Examples
        --------
        >>> sampler.run_nested()
        >>> fig, axes = sampler.plot_run(lnz_truth=-10.5)
        >>> plt.savefig('run_plot.png')
        """
        try:
            from dynesty import plotting as dyplot
        except ImportError:
            raise ImportError("dynesty package is required for plotting. "
                            "Install with: pip install dynesty")

        dynesty_results = self.to_dynesty_results()
        return dyplot.runplot(dynesty_results, lnz_truth=lnz_truth, **kwargs)

    def plot_trace(self, truths: Optional[np.ndarray] = None, dims: Optional[list] = None,
                   thin: int = 1, **kwargs):
        """
        Create trace plot showing particle evolution.

        Convenience method that uses Dynesty's plotting.traceplot().

        Parameters
        ----------
        truths : array_like, optional
            True parameter values for overplotting.
        dims : list of int, optional
            Which dimensions to plot. If None, plots all dimensions.
        thin : int, optional
            Thin the samples by plotting every n-th sample. Default: 1 (no thinning).
            Use thin=5 or higher for large datasets to improve performance.
        **kwargs
            Additional arguments passed to dynesty.plotting.traceplot().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : ndarray
            Array of matplotlib axes

        Examples
        --------
        >>> sampler.run_nested()
        >>> fig, axes = sampler.plot_trace(truths=np.zeros(ndim), thin=5)
        >>> plt.savefig('trace_plot.png')

        >>> # Plot only first 3 dimensions
        >>> fig, axes = sampler.plot_trace(dims=[0, 1, 2])
        """
        try:
            from dynesty import plotting as dyplot
        except ImportError:
            raise ImportError("dynesty package is required for plotting. "
                            "Install with: pip install dynesty")

        dynesty_results = self.to_dynesty_results()
        return dyplot.traceplot(dynesty_results, truths=truths, dims=dims, thin=thin, **kwargs)

    def plot_corner(self, truths: Optional[np.ndarray] = None, **kwargs):
        """
        Create corner plot showing posterior distributions.

        Convenience method that uses Dynesty's plotting.cornerplot().

        Parameters
        ----------
        truths : array_like, optional
            True parameter values for overplotting.
        **kwargs
            Additional arguments passed to dynesty.plotting.cornerplot().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : ndarray
            Array of matplotlib axes

        Examples
        --------
        >>> sampler.run_nested()
        >>> fig, axes = sampler.plot_corner(truths=true_params)
        >>> plt.savefig('corner_plot.png')
        """
        try:
            from dynesty import plotting as dyplot
        except ImportError:
            raise ImportError("dynesty package is required for plotting. "
                            "Install with: pip install dynesty")

        dynesty_results = self.to_dynesty_results()
        return dyplot.cornerplot(dynesty_results, truths=truths, **kwargs)

    def plot_diagnostics(self, **kwargs):
        """
        Create comprehensive diagnostic plot.

        Convenience method that uses Dynesty's plotting.runplot()
        to show evidence evolution and convergence.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to dynesty.plotting.runplot().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : ndarray
            Array of matplotlib axes

        Examples
        --------
        >>> sampler.run_nested()
        >>> fig, axes = sampler.plot_diagnostics()
        >>> plt.savefig('diagnostics.png')

        Note
        ----
        This method uses Dynesty's runplot which provides comprehensive
        diagnostic information including evidence evolution, log-likelihood,
        and weight trajectories.
        """
        try:
            from dynesty import plotting as dyplot
        except ImportError:
            raise ImportError("dynesty package is required for plotting. "
                            "Install with: pip install dynesty")

        dynesty_results = self.to_dynesty_results()
        return dyplot.runplot(dynesty_results, **kwargs)
