#!/usr/bin/env python
"""
Demo 3: High-Dimensional Gaussian (Dynesty)

Tests scalability to higher dimensions with a simple Gaussian likelihood.
Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 03_highd_gaussian_dynesty.py [--nlive 500] [--ndim 20] [--seed 42]

Output:
    Creates output_03_dynesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - trace.npz: Log-likelihood trajectory
    - trace_plot.png: Log-likelihood trajectory
"""

import sys
import os
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
    High-dimensional Gaussian log-likelihood.

    L(x) = -0.5 * ||x||^2
    """
    return -0.5 * np.sum(x**2)


def prior_transform(u):
    """Transform from unit cube [0, 1]^ndim to physical space [-5, 5]^ndim."""
    return (u - 0.5) * 10.0  # Maps [0, 1] → [-5, 5]


# ============================================================================
# Plotting
# ============================================================================


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demo 3: High-D Gaussian (Dynesty)')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--ndim', type=int, default=20, help='Problem dimensionality')
    parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    ndim = args.ndim
    nlive = args.nlive

    print("="*70)
    print("Demo 3: High-D Gaussian (Dynesty)")
    print("="*70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {nlive}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Random seed: {args.seed}")

    # Create output directory
    outdir = Path('output_03_dynesty')
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

    # Get delta_logz from results
    delta_logZ = float(results.dlogz[-1]) if hasattr(results, 'dlogz') and len(results.dlogz) > 0 else 0.0

    # Analytical solution for comparison
    # For N-dimensional Gaussian with sigma=1 on [-5,5]^N:
    # Z ≈ (1/√(2π))^N * volume_correction
    analytical_H = ndim  # Information for unit Gaussian

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

    # Trace plot - particle evolution (first 5 dimensions)
    print("  Generating trace plot...")
    fig, axes = dyplot.traceplot(results, dims=list(range(min(ndim, 5))))
    plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: trace_plot.png")

    # Corner plot - posterior distributions (first 6 dimensions if ndim <= 10)
    if ndim <= 10:
        print("  Generating corner plot...")
        fig, axes = dyplot.cornerplot(results, dims=tuple(range(min(ndim, 6))))
        plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: corner_plot.png")

    print("\n" + "="*70)
    print("✅ Complete! Results saved to:", outdir)
    print("="*70)
    print(f"   logZ = {logZ:.4f} ± {logZ_err:.4f}")
    print(f"   Analytical logZ ≈ {ndim * np.log(5.0):.4f} (approximate)")
    print(f"   Difference: {logZ - ndim * np.log(5.0):.4f}")
    print(f"   H = {information:.4f} (analytical: {analytical_H:.4f})")
    print(f"   delta_logZ = {delta_logZ:.6f}")
    print(f"   Converged: {delta_logZ < 0.01}")
    print(f"   Iterations: {n_iterations} (max: {args.max_iterations})")
    print(f"   Runtime: {runtime:.2f}s")

    iterations_per_sec = n_iterations / runtime if runtime > 0 else 0
    print(f"   Performance:")
    print(f"     {iterations_per_sec:.1f} iterations/sec")
    print(f"     {1000/iterations_per_sec if iterations_per_sec > 0 else 0:.2f} ms/iteration")

    if delta_logZ >= 0.01:
        import warnings
        warnings.warn(
            f"Run did NOT converge after {n_iterations} iterations. "
            f"Final delta_logZ ({delta_logZ:.6f}) >= threshold (0.01). "
            f"Increase max_iterations."
        )
    else:
        print(f"   ✅ Converged at iteration {n_iterations}")

    print("="*70)


if __name__ == '__main__':
    main()
