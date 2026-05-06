#!/usr/bin/env python
"""
Demo 2: Rosenbrock Banana (Dynesty)

Tests constrained sampling on a curved, degenerate likelihood surface.
Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 02_rosenbrock_banana_dynesty.py [--nlive 500] [--a 1] [--b 100] [--seed 42]

Output:
    Creates output_02_dynesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - trace.npz: Log-likelihood trajectory
    - posterior_2d.png: 2D scatter plot showing banana shape
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
from dynesty import plotting as dyplot


# ============================================================================
# Problem Definition
# ============================================================================

def loglikelihood(x, a=1, b=100):
    """
    Rosenbrock "banana" function log-likelihood.

    L(x, y) = -0.5 * ((a - x)^2 + b * (y - x^2)^2)

    Parameters:
        a: Controls position of minimum (default: 1)
        b: Controls curvature/banana shape (default: 100)
    """
    x_, y = x[0], x[1]
    logL = -0.5 * ((a - x_)**2 + b * (y - x_**2)**2)
    return logL


def prior_transform(u):
    """Transform from unit cube [0, 1]^2 to physical space [-3, 3]^2."""
    return (u - 0.5) * 6.0  # Maps [0, 1] → [-3, 3]


# ============================================================================
# Plotting
# ============================================================================


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demo 2: Rosenbrock Banana (Dynesty)')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--a', type=float, default=1, help='Rosenbrock a parameter')
    parser.add_argument('--b', type=float, default=100, help='Rosenbrock b parameter')
    parser.add_argument('--max_iterations', type=int, default=2000, help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    ndim = 2
    nlive = args.nlive

    print("="*70)
    print("Demo 2: Rosenbrock Banana (Dynesty)")
    print("="*70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {nlive}")
    print(f"Rosenbrock parameters: a={args.a}, b={args.b}")
    print(f"Random seed: {args.seed}")

    # Create output directory
    outdir = Path('output_02_dynesty')
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

    # Fix loglikelihood parameters
    def loglikelihood_fixed(x):
        return loglikelihood(x, args.a, args.b)

    sampler = NestedSampler(
        loglikelihood=loglikelihood_fixed,
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

    # Save numerical results
    summary = {
        'implementation': 'dynesty',
        'problem': 'rosenbrock_banana',
        'dimension': int(ndim),
        'nlive': int(nlive),
        'a': float(args.a),
        'b': float(args.b),
        'max_iterations': args.max_iterations,
        'seed': args.seed,
        'logZ': float(logZ),
        'logZ_error': float(logZ_err),
        'H': information,
        'delta_logZ': float(delta_logZ),
        'converged': bool(delta_logZ < 0.01),
        'n_iterations': int(n_iterations),
        'n_likelihood_evals': n_evals,
        'runtime': float(runtime),
        'evals_per_sec': float(n_evals / runtime if runtime > 0 else 0)
    }

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save samples
    np.savez(outdir / 'samples.npz', samples=samples, weights=weights, logL=logL_samples)

    # Save trace
    np.savez(outdir / 'trace.npz', logL_trajectory=logL_samples, logZ=np.full(n_iterations, logZ))

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

    # Corner plot - posterior distributions (2D)
    print("  Generating corner plot...")
    fig, axes = dyplot.cornerplot(results)
    plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: corner_plot.png")

    print("\n" + "="*70)
    print("✅ Complete! Results saved to:", outdir)
    print("="*70)
    print(f"   logZ = {logZ:.4f} ± {logZ_err:.4f}")
    print(f"   H = {information:.4f}")
    print(f"   delta_logZ = {delta_logZ:.6f}")
    print(f"   Converged: {delta_logZ < 0.01}")
    print(f"   Iterations: {n_iterations} (max: {args.max_iterations})")
    print(f"   Runtime: {runtime:.2f}s")

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
