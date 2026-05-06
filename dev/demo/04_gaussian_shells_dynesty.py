#!/usr/bin/env python
"""
Demo 4: Gaussian Shells (Dynesty)

Classic test case for multi-ellipsoid bounding from Dynesty's examples.
Two thin annular (ring-shaped) distributions separated in 2D.

Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 04_gaussian_shells_dynesty.py [--nlive 500] [--seed 42]

Output:
    Creates output_04_dynesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - posterior_2d.png: 2D scatter plot showing shell structure
    - trace_plot.png: Log-likelihood trajectory
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import json
import time


# ============================================================================
# Problem Definition
# ============================================================================

# Gaussian shell parameters (matching Dynesty's example)
r = 2.0                      # shell radius
w = 0.1                      # shell width
c1 = np.array([-3.5, 0.0])   # center of shell 1
c2 = np.array([3.5, 0.0])    # center of shell 2
const = np.log(1.0 / np.sqrt(2.0 * np.pi * w**2))


def loglikelihood(x):
    """
    Two Gaussian shells log-likelihood.

    Each shell is a thin annular Gaussian centered at c_i with radius r
    and width w. Total log-likelihood is log-sum-exp of both shells.
    """
    def logcirc(theta, c):
        d = np.sqrt(np.sum((theta - c)**2))
        return const - (d - r)**2 / (2.0 * w**2)
    return np.logaddexp(logcirc(x, c1), logcirc(x, c2))


def prior_transform(u):
    """Transform from unit cube [0, 1]^2 to physical space [-6, 6]^2."""
    return 12.0 * u - 6.0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demo 4: Gaussian Shells (Dynesty)')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--max_iterations', type=int, default=2000, help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    ndim = 2
    nlive = args.nlive

    print("=" * 70)
    print("Demo 4: Gaussian Shells (Dynesty)")
    print("=" * 70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {nlive}")
    print(f"Shell radius: {r}, width: {w}")
    print(f"Shell centers: {c1}, {c2}")
    print(f"Random seed: {args.seed}")

    # Create output directory
    outdir = Path('output_04_dynesty')
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

    samples = results.samples
    logweights = results.logwt
    logL_samples = results.logl

    # Compute normalized weights
    weights = np.exp(logweights - np.max(logweights))
    weights /= np.sum(weights)

    # Summary statistics
    logZ = float(results.logz[-1])
    logZ_err = float(results.logzerr[-1])
    information = float(results.information[-1])
    n_iterations = int(len(results.logl))
    n_evals = int(results.ncall[0] if hasattr(results.ncall, '__len__') else results.ncall)
    delta_logZ = float(results.dlogz[-1]) if hasattr(results, 'dlogz') and len(results.dlogz) > 0 else 0.0

    # Save numerical results
    summary = {
        'implementation': 'dynesty',
        'problem': 'gaussian_shells',
        'dimension': int(ndim),
        'nlive': int(nlive),
        'max_iterations': args.max_iterations,
        'seed': args.seed,
        'shell_radius': float(r),
        'shell_width': float(w),
        'shell_center1': [float(c1[0]), float(c1[1])],
        'shell_center2': [float(c2[0]), float(c2[1])],
        'logZ': float(logZ),
        'logZ_error': float(logZ_err),
        'H': information,
        'delta_logZ': float(delta_logZ),
        'converged': bool(delta_logZ < 0.01),
        'n_iterations': int(n_iterations),
        'n_likelihood_evals': n_evals,
        'runtime': float(runtime),
    }

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save samples
    np.savez(outdir / 'samples.npz', samples=samples, weights=weights, logL=logL_samples)

    # Save trace
    np.savez(outdir / 'trace.npz', logL_trajectory=logL_samples, logZ=np.full(n_iterations, logZ))

    # Generate plots
    print("\nGenerating plots...")

    # 2D posterior scatter plot
    print("  Generating 2D posterior plot...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Dynesty: Gaussian Shells Posterior')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw true shell locations
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(c1[0] + r * np.cos(theta), c1[1] + r * np.sin(theta),
            'k--', alpha=0.5, label='True shell locations')
    ax.plot(c2[0] + r * np.cos(theta), c2[1] + r * np.sin(theta),
            'k--', alpha=0.5)
    ax.legend()
    plt.savefig(outdir / 'posterior_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: posterior_2d.png")

    # Dynesty native plots
    try:
        from dynesty import plotting as dyplot

        print("  Generating run plot...")
        fig, axes = dyplot.runplot(results)
        plt.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: run_plot.png")

        print("  Generating trace plot...")
        fig, axes = dyplot.traceplot(results)
        plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: trace_plot.png")

        print("  Generating corner plot...")
        fig, axes = dyplot.cornerplot(results)
        plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: corner_plot.png")
    except Exception as e:
        print(f"  Warning: Could not generate Dynesty native plots: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Complete! Results saved to: {outdir}")
    print(f"  logZ = {logZ:.4f} +/- {logZ_err:.4f}")
    print(f"  H = {information:.4f}")
    print(f"  delta_logZ = {delta_logZ:.6f}")
    print(f"  Converged: {delta_logZ < 0.01}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
