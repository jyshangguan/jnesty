#!/usr/bin/env python
"""
Demo 3: High-Dimensional Gaussian (J-Nesty)

Tests GPU acceleration on high-dimensional Gaussian problem.
Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 03_highd_gaussian_jnesty.py [--ndim 20] [--nlive 500]

Output:
    Creates output_03_jnesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - trace_plot.png: Log-likelihood trajectory
    - scaling.txt: Timing information
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import jax
import jax.numpy as jnp
from jax import random
from jnesty import NestedSampler, save_results
import matplotlib.pyplot as plt

import time
import numpy as np


# ============================================================================
# Problem Definition
# ============================================================================

def loglikelihood(x):
    """
    Standard Gaussian log-likelihood in ndim dimensions.

    logL = -0.5 * sum(x^2)

    This is a simple, well-understood problem for testing
    scalability and GPU acceleration.
    """
    return -0.5 * jnp.sum(x**2)


def prior_sample(key, ndim):
    """Sample from uniform prior on [-5, 5]^ndim."""
    return random.uniform(key, shape=(ndim,), minval=-5.0, maxval=5.0)


def prior_transform(u):
    """Transform from unit cube [0, 1]^ndim to physical space [-5, 5]^ndim."""
    return (u - 0.5) * 10.0  # Maps [0, 1] → [-5, 5]


# ============================================================================
# Plotting
# ============================================================================


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='J-Nesty on High-D Gaussian')
    parser.add_argument('--ndim', type=int, default=20, help='Problem dimensionality')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--max_iterations', type=int, default=20000, help='Max iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to use')
    args = parser.parse_args()

    # Set device
    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = args.ndim

    print("="*70)
    print("Demo 3: High-Dimensional Gaussian (J-Nesty)")
    print("="*70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {args.nlive}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device.upper()}")

    # Create output directory
    outdir = Path('output_03_jnesty')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Run J-Nesty with new simple API
    print("\nRunning J-Nesty with new simple API...")
    print("Note: Parameters are now auto-tuned based on dimensionality")

    sampler = NestedSampler(
        loglikelihood,
        prior_transform,
        ndim=ndim,
        nlive=args.nlive,
        device=args.device,
        verbose=True,
        bound='multi',
        bound_update_interval=0
    )

    start_time = time.time()
    sampler.run_nested(max_iterations=args.max_iterations, delta_logZ_threshold=0.01)
    runtime = time.time() - start_time

    # Get results
    results = sampler.results

    # Convert to numpy arrays for saving
    samples = results['samples']
    logL_samples = results['logl']

    # Get delta_logZ_trajectory from raw result
    delta_logZ_trajectory = np.array(sampler._raw_result.delta_logZ_trajectory)

    # Check convergence
    final_delta_logZ = results['delta_logz']
    converged = results['converged']

    # Analytical solution for unnormalized Gaussian in [-5, 5]^ndim
    # Z = (sqrt(2*pi) * erf(5/sqrt(2)))^ndim
    from math import erf, sqrt, log, pi
    Z_1d = (1.0/10.0) * sqrt(2.0 * pi) * erf(5.0 / sqrt(2.0))
    logZ_analytical = ndim * log(Z_1d)
    H_analytical = ndim / 2.0

    # Save FITS results
    save_results(results, str(outdir / 'results.fits'))
    print("    Saved: results.fits")

    # Save auto-tuned parameters for reference
    with open(outdir / 'auto_tuning.txt', 'w') as f:
        f.write(f"Auto-tuned parameters:\n")
        f.write(f"  rwalk_K: {sampler.rwalk_K}\n")
        f.write(f"  rwalk_step_scale: {sampler.rwalk_step_scale:.4f}\n")

    # Save timing info
    with open(outdir / 'scaling.txt', 'w') as f:
        f.write(f"Dimension: {ndim}\n")
        f.write(f"Runtime: {runtime:.2f}s\n")
        f.write(f"Iterations: {results['niter']}\n")
        f.write(f"Speed: {results['niter'] / runtime:.1f} iter/s\n")
        f.write(f"Time per iteration: {runtime / results['niter'] * 1000:.2f} ms\n")

    # Generate plots
    print("\nGenerating plots...")

    # Run plot - evidence evolution
    print("  Generating run plot...")
    fig, axes = sampler.plot_run()
    plt.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: run_plot.png")

    # Trace plot - particle evolution (first 5 dimensions)
    print("  Generating trace plot...")
    fig, axes = sampler.plot_trace(thin=10)
    plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: trace_plot.png")

    # Corner plot - posterior distributions (first 6 dimensions)
    if ndim <= 10:
        print("  Generating corner plot...")
        fig, axes = sampler.plot_corner(dims=tuple(range(min(ndim, 6))))
        plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: corner_plot.png")

    # Diagnostics plot
    print("  Generating diagnostics plot...")
    fig, axes = sampler.plot_diagnostics()
    plt.savefig(outdir / 'diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: diagnostics.png")

    # Print summary
    print("\n" + "="*70)
    print("✅ Complete! Results saved to:", outdir)
    print(f"   logZ = {results['logz']:.4f} ± {results['logzerr']:.4f}")
    print(f"   Analytical logZ = {logZ_analytical:.4f}")
    print(f"   Difference: {results['logz'] - logZ_analytical:.4f}")
    print(f"   H = {results['information']:.4f} (analytical: {H_analytical:.4f})")
    print(f"   delta_logZ = {final_delta_logZ:.6f}")
    print(f"   Converged: {converged}")
    print(f"   Iterations: {results['niter']} (max: {args.max_iterations})")
    print(f"   Runtime: {runtime:.2f}s")
    print()
    print(f"   Performance:")
    print(f"     {results['niter'] / runtime:.1f} iterations/sec")
    print(f"     {runtime / results['niter'] * 1000:.2f} ms/iteration")
    print()
    print(f"   Auto-tuned parameters:")
    print(f"     rwalk_K: {sampler.rwalk_K}")
    print(f"     rwalk_step_scale: {sampler.rwalk_step_scale:.4f}")
    print()
    print(f"   Convergence details:")
    print(f"     Final delta_logZ: {final_delta_logZ:.6f}")
    print(f"     Efficiency: {100.0 if converged else 0.0:.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()
