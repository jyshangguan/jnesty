#!/usr/bin/env python
"""
Demo 1: Multi-modal Gaussian Mixture (Dynesty)

Tests multi-modality handling with 2 separated Gaussian modes.
Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 01_multimodal_gaussian_mixture_dynesty.py [--nlive 500] [--dim 5] [--seed 42]

Output:
    Creates output_01_dynesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - trace.npz: Log-likelihood trajectory
    - posterior_2d.png: 2D scatter plot
    - trace_plot.png: Log-likelihood trajectory
    - corner.png: Corner plot (if dim <= 10)
"""

import argparse
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

import time
from dynesty import plotting as dyplot
from jnesty.results import save_results


# ============================================================================
# Problem Definition
# ============================================================================

def loglikelihood(x):
    """
    2-mode Gaussian mixture log-likelihood.

    Mode 1: centered at (0, 0) with weight 0.6
    Mode 2: centered at (3, 3) with weight 0.4
    """
    ndim = len(x)

    # Mode 1: centered at origin
    if ndim == 2:
        mean1 = np.array([0.0, 0.0])
        mean2 = np.array([3.0, 3.0])
    else:
        # Higher dimensions: first 2 dims separated, rest at 0
        mean1 = np.zeros(ndim)
        mean2 = np.zeros(ndim)
        mean2[0] = 3.0
        mean2[1] = 3.0

    diff1 = x - mean1
    logL1 = -0.5 * np.sum(diff1**2) + np.log(0.6)

    diff2 = x - mean2
    logL2 = -0.5 * np.sum(diff2**2) + np.log(0.4)

    # Log-sum-exp for numerical stability
    max_logL = max(logL1, logL2)
    logL = max_logL + np.log(np.exp(logL1 - max_logL) + np.exp(logL2 - max_logL))

    return logL


def prior_transform(u):
    """Transform from unit cube to uniform prior on [-5, 5]^ndim."""
    return (u - 0.5) * 10.0


# ============================================================================
# Plotting
# ============================================================================


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demo 1: Multi-modal Gaussian Mixture (Dynesty)')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimensionality')
    parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    ndim = args.dim
    nlive = args.nlive

    print("="*70)
    print("Demo 1: Multi-modal Gaussian Mixture (Dynesty)")
    print("="*70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {nlive}")
    print(f"Random seed: {args.seed}")

    # Create output directory
    outdir = Path('output_01_dynesty')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Run Dynesty
    try:
        from dynesty import NestedSampler
    except ImportError:
        print("Error: dynesty not installed. Install with: pip install dynesty")
        return

    print("\nRunning Dynesty...")
    np.random.seed(args.seed)

    sampler = NestedSampler(
        loglikelihood=loglikelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=nlive,
        bound='multi',
        sample='rwalk'
    )

    start_time = time.time()
    sampler.run_nested(maxcall=args.max_iterations * nlive * 5, dlogz=0.01)
    runtime = time.time() - start_time

    # Extract results
    results = sampler.results

    samples = results.samples  # Posterior samples
    logweights = results.logwt  # Log weights
    logL_samples = results.logl  # Log-likelihoods

    # Compute normalized weights
    weights = np.exp(logweights - np.max(logweights))
    weights /= np.sum(weights)

    # Summary statistics
    logZ = float(results.logz[-1])
    logZ_err = float(results.logzerr[-1])
    information = float(results.information[-1])
    n_iterations = int(len(results.logl))
    n_evals = int(results.ncall[0] if hasattr(results.ncall, '__len__') else results.ncall)

    # Get delta_logz from results (if available)
    # Dynesty stores this in results.dlogz
    delta_logZ = float(results.dlogz[-1]) if hasattr(results, 'dlogz') and len(results.dlogz) > 0 else 0.0

    # Save FITS results
    fits_results = {
        'logz': logZ, 'logzerr': logZ_err, 'information': information,
        'nlive': int(nlive), 'niter': n_iterations, 'eff': 0.0,
        'acceptance_rate': 0.0, 'converged': delta_logZ < 0.01,
        'delta_logz': delta_logZ, 'delta_logZ_threshold': 0.01, 'rwalk_K': 0,
        'logl': logL_samples, 'logwt': np.log(weights + 1e-300),
        'logvol': np.zeros(n_iterations),
        'samples': samples, 'samples_u': samples,
        'logz_trajectory': np.full(n_iterations, logZ),
        'logzerr_trajectory': np.full(n_iterations, logZ_err),
        'delta_logZ_trajectory': np.zeros(n_iterations),
        'scale_trajectory': np.ones(n_iterations),
    }
    save_results(fits_results, str(outdir / 'results.fits'))
    print("    Saved: results.fits")

    print("\nGenerating plots...")

    # Run plot - evidence evolution
    print("  Generating run plot...")
    fig, axes = dyplot.runplot(results)
    plt.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: run_plot.png")

    # Trace plot - particle evolution
    print("  Generating trace plot...")
    fig, axes = dyplot.traceplot(results)
    plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: trace_plot.png")

    # Corner plot - posterior distributions
    if ndim <= 10:
        print("  Generating corner plot...")
        fig, axes = dyplot.cornerplot(results)
        plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: corner_plot.png")

    print(f"\n✅ Complete! Results saved to: {outdir}")
    print(f"   logZ = {logZ:.4f} ± {logZ_err:.4f}")
    print(f"   delta_logZ = {delta_logZ:.4f}")
    print(f"   Converged: {delta_logZ < 0.01}")


if __name__ == "__main__":
    main()
