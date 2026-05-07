"""
JNesty public API — Dynesty-style nested sampler interface.

Thin wrapper that configures the sampler, runs the NS loop,
and formats results via the results module.
"""

from typing import Optional, Callable, Dict, Any, Union
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import warnings

from .sampler import run_nested_sampling, WhileLoopNSConfig
from .results import Results, format_results


class NestedSampler:
    """
    Simple nested sampler following Dynesty's API design.

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood function. Signature: loglikelihood(x) -> float
    prior_transform : callable
        Prior transform. Signature: prior_transform(u) -> array
    ndim : int
        Number of dimensions.
    nlive : int, optional
        Number of live points. Default: 500
    rwalk_K : int or None, optional
        Random walk steps per iteration. Auto-tuned if None.
    rwalk_step_scale : float or None, optional
        Initial proposal scale. Auto-tuned if None.
    target_acceptance : float, optional
        Target acceptance rate for scale adaptation. Default: 0.5
    scale_adapt_interval : int, optional
        Iterations between scale adaptation. Default: 1
    device : str, optional
        Device to use ('gpu' or 'cpu'). Default: 'gpu'
    verbose : bool, optional
        Print progress. Default: True
    bound : str, optional
        Bounding method: 'none', 'single', 'multi'. Default: 'none'
    bound_update_interval : int, float, or None, optional
        Controls periodic bound refitting. Default: None (automatic).
    max_ellipsoids : int, optional
        Maximum ellipsoids for multi-ellipsoid. Default: 20
    batch_size : int or None, optional
        Number of parallel walks for GPU-parallel likelihood evaluation.
        None (default): auto-tuned as rwalk_K // max(2, rwalk_K * 10 // nlive).
        Set to 1 to disable parallelism.
    memory_frac : float, optional
        Fraction of GPU memory to use for batch walks. Default: 0.9.
        Caps batch_size if it would exceed this fraction of GPU memory.
        Ignored on CPU.
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
        scale_adapt_interval: int = 1,
        device: str = 'gpu',
        verbose: bool = True,
        bound: str = 'none',
        bound_update_interval: Optional[Union[int, float]] = None,
        max_ellipsoids: int = 20,
        batch_size: Optional[int] = None,
        memory_frac: float = 0.9,
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
        self.memory_frac = memory_frac

        # Resolve bound_update_interval
        if bound_update_interval is None:
            if bound == 'multi':
                bound_update_interval = nlive
            else:
                bound_update_interval = 0
        elif isinstance(bound_update_interval, float) and bound_update_interval > 0:
            bound_update_interval = int(bound_update_interval * nlive)
        self.bound_update_interval = bound_update_interval

        if verbose and bound == 'multi':
            if bound_update_interval == 0:
                print("Bound update: once at start (no periodic updates)")
            else:
                print(f"Bound update interval: {bound_update_interval} iterations "
                      f"({bound_update_interval / nlive:.1f} x nlive)")

        # Auto-tune parameters
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

        # Auto-tune batch_size
        if batch_size is None:
            batch_size = rwalk_K // max(2, rwalk_K * 10 // nlive)
            if verbose:
                print(f"Auto-tuned batch_size = {batch_size} "
                      f"(rwalk_K={rwalk_K}, nlive={nlive})")
        self.batch_size = batch_size

        if device == 'cpu':
            jax.config.update('jax_platform_name', 'cpu')

        self._results = None
        self._raw_result = None

    def run_nested(
        self,
        max_iterations: int = 100000,
        delta_logZ_threshold: float = 0.01,
        print_progress: bool = True,
    ) -> None:
        """
        Run the nested sampling algorithm.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations. Default: 100000
        delta_logZ_threshold : float, optional
            Convergence threshold. Default: 0.01
        print_progress : bool, optional
            Show progress bar. Default: True
        """
        if self.verbose:
            print("=" * 70)
            print("JNesty Nested Sampling")
            print("=" * 70)
            print(f"Dimensions: {self.ndim}")
            print(f"Live points: {self.nlive}")
            print(f"Max iterations: {max_iterations}")
            print(f"Device: {self.device.upper()}")
            print(f"Walk steps per iteration: {self.rwalk_K}")
            print(f"Initial step scale: {self.rwalk_step_scale:.4f}")
            print(f"Convergence threshold: delta_logZ < {delta_logZ_threshold}")
            print("=" * 70)
            print()

        self._delta_logZ_threshold = delta_logZ_threshold

        def prior_sample(key):
            return random.uniform(key, shape=(self.ndim,), minval=0.0, maxval=1.0)

        config = WhileLoopNSConfig(
            nlive=self.nlive,
            max_iterations=max_iterations,
            delta_logZ_threshold=delta_logZ_threshold,
            rwalk_K=self.rwalk_K,
            rwalk_step_scale=self.rwalk_step_scale,
            target_acceptance=self.target_acceptance,
            verbose=self.verbose,
            print_progress=print_progress,
            bound=self.bound,
            bound_update_interval=self.bound_update_interval,
            max_ellipsoids=self.max_ellipsoids,
            batch_size=self.batch_size,
            memory_frac=self.memory_frac,
        )

        key = random.PRNGKey(42)
        self._raw_result = run_nested_sampling(
            loglikelihood_fn=self.loglikelihood,
            prior_sample_fn=prior_sample,
            ndim=self.ndim,
            config=config,
            key=key,
            prior_transform_fn=self.prior_transform,
        )

        if self.verbose:
            print()
            print("=" * 70)
            print("Sampling Complete!")
            print("=" * 70)

    @property
    def results(self) -> Results:
        """
        Results object following Dynesty's format.

        Raises RuntimeError if run_nested() has not been called.
        """
        if self._raw_result is None:
            raise RuntimeError("Must call run_nested() before accessing results")

        if self._results is None:
            self._results = format_results(
                self._raw_result, self.prior_transform, self.ndim,
                self.nlive, self.rwalk_K, self._delta_logZ_threshold,
            )

        return self._results

    def print_summary(self) -> None:
        """Print a formatted summary of the results."""
        if self._raw_result is None:
            print("No results yet. Call run_nested() first.")
            return
        self.results.summary()

    def get_samples(self, weights: bool = False) -> np.ndarray:
        """Get posterior samples in physical space."""
        if weights:
            print("Warning: Weighted sampling not yet implemented. "
                  "Returning unweighted samples.")
        return self.results['samples']

    def get_logz(self) -> tuple:
        """Get (logZ, logZ_error)."""
        r = self.results
        return r['logz'], r['logzerr']

    def to_dynesty_results(self):
        """
        Convert results to a Dynesty Results object for use with
        Dynesty's plotting functions.
        """
        try:
            from dynesty.utils import Results as DynestyResults
        except ImportError:
            raise ImportError("dynesty package required. Install with: pip install dynesty")

        r = self.results
        samples_id = np.arange(len(r['samples']))

        dynesty_dict = {
            'samples_u': r['samples_u'],
            'samples_id': samples_id,
            'logl': r['logl'],
            'samples': r['samples'],
            'nlive': r['nlive'],
            'niter': r['niter'],
            'logwt': r['logwt'],
            'logvol': r['logvol'],
            'logz': r['logz_trajectory'],
            'logzerr': r['logzerr_trajectory'],
        }

        return DynestyResults(dynesty_dict)

    # Plotting convenience methods (delegate to Dynesty's plotting)

    def plot_run(self, lnz_truth=None, **kwargs):
        """Run plot via Dynesty's plotting.runplot()."""
        from dynesty import plotting as dyplot
        return dyplot.runplot(self.to_dynesty_results(), lnz_truth=lnz_truth, **kwargs)

    def plot_trace(self, truths=None, dims=None, thin=1, **kwargs):
        """Trace plot via Dynesty's plotting.traceplot()."""
        from dynesty import plotting as dyplot
        return dyplot.traceplot(self.to_dynesty_results(), truths=truths,
                                dims=dims, thin=thin, **kwargs)

    def plot_corner(self, truths=None, **kwargs):
        """Corner plot via Dynesty's plotting.cornerplot()."""
        from dynesty import plotting as dyplot
        return dyplot.cornerplot(self.to_dynesty_results(), truths=truths, **kwargs)

    def plot_diagnostics(self, **kwargs):
        """Diagnostics via Dynesty's plotting.runplot()."""
        from dynesty import plotting as dyplot
        return dyplot.runplot(self.to_dynesty_results(), **kwargs)
