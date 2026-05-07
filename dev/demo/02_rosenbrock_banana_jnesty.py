#!/usr/bin/env python
"""
Demo 2: Rosenbrock Banana (J-Nesty)

Tests highly correlated posterior with Rosenbrock "banana" function.
Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 02_rosenbrock_banana_jnesty.py [--nlive 500] [--a 1] [--b 100]

Output:
    Creates output_02_jnesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - posterior_2d.png: 2D scatter plot showing banana shape
    - trace_plot.png: Log-likelihood trajectory
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
from jnesty import NestedSampler, plotting
import matplotlib.pyplot as plt
import json
import time
import numpy as np


# ============================================================================
# Problem Definition
# ============================================================================

def loglikelihood(x, a=1, b=100):
    """
    Rosenbrock "banana" function (negated for maximization).

    Standard form: f(x, y) = (a - x)^2 + b(y - x^2)^2

    Parameters:
        a: Controls distance from origin (default: 1)
        b: Controls "banana" curvature (default: 100)

    The posterior is highly correlated and has a curved "banana" shape.
    This is challenging for simple random walk samplers.
    """
    x_val = x[0]
    y_val = x[1]

    # Rosenbrock function (negated for maximization)
    # Standard form with 0.5 factor for proper Gaussian log-likelihood
    logL = -0.5 * ((a - x_val)**2 + b * (y_val - x_val**2)**2)

    return logL


def prior_sample(key, a=1, b=100):
    """
    Sample from uniform prior.

    Prior bounds:
    - x: [-3, 3] (covers minimum at x=a)
    - y: [-10, 10] (wide enough for banana curve)
    """
    return random.uniform(key, shape=(2,), minval=-3.0, maxval=3.0)


def prior_transform(u):
    """Transform from unit cube [0, 1]^2 to physical space [-3, 3]^2."""
    return (u - 0.5) * 6.0  # Maps [0, 1] → [-3, 3]


# ============================================================================
# Plotting
# ============================================================================


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='J-Nesty on Rosenbrock Banana')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--max_iterations', type=int, default=20000, help='Max iterations')
    parser.add_argument('--a', type=float, default=1, help='Rosenbrock a parameter')
    parser.add_argument('--b', type=float, default=100, help='Rosenbrock b parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to use')
    args = parser.parse_args()

    # Set device
    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = 2

    print("="*70)
    print("Demo 2: Rosenbrock Banana (J-Nesty)")
    print("="*70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {args.nlive}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Rosenbrock parameters: a={args.a}, b={args.b}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device.upper()}")

    # Create output directory
    outdir = Path('output_02_jnesty')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Fix loglikelihood parameters
    def loglikelihood_fixed(x):
        return loglikelihood(x, args.a, args.b)

    # Run J-Nesty with new simple API
    print("\nRunning J-Nesty with new simple API...")

    sampler = NestedSampler(
        loglikelihood_fixed,
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
    delta_logZ_trajectory = np.array(sampler._raw_result.delta_logZ_trajectory)

    # Check convergence
    final_delta_logZ = results['delta_logz']
    converged = results['converged']

    # Save numerical results
    summary = {
        "implementation": "jnesty",
        "problem": "rosenbrock_banana",
        "dimension": ndim,
        "nlive": args.nlive,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "rosenbrock_a": args.a,
        "rosenbrock_b": args.b,
        "logZ": float(results['logz']),
        "logZ_error": float(results['logzerr']),
        "H": float(results['information']),
        "delta_logZ": final_delta_logZ,
        "converged": bool(converged),
        "n_iterations": int(results['niter']),
        "runtime": runtime,
        "iterations_per_sec": float(results['niter']) / runtime if runtime > 0 else 0,
        "acceptance_rate": float(results['acceptance_rate'])
    }

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save samples
    np.savez(outdir / 'samples.npz', samples=samples, logL=logL_samples)

    # Save trace
    np.savez(outdir / 'trace.npz',
             logL_samples=logL_samples,
             delta_logZ_trajectory=delta_logZ_trajectory)

    # Generate plots
    print("\nGenerating plots...")

    # Run plot - evidence evolution
    print("  Generating run plot...")
    fig, axes = sampler.plot_run()
    plt.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: run_plot.png")

    # Trace plot - particle evolution
    print("  Generating trace plot...")
    fig, axes = sampler.plot_trace(thin=5)
    plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: trace_plot.png")

    # Corner plot - posterior distributions (2D)
    print("  Generating corner plot...")
    fig, axes = sampler.plot_corner()
    plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: corner_plot.png")

    # Diagnostics plot
    print("  Generating diagnostics plot...")
    fig, axes = plotting.diagnostics(results)
    plt.savefig(outdir / 'diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: diagnostics.png")

    # Print summary
    print("\n" + "="*70)
    print("✅ Complete! Results saved to:", outdir)
    print(f"   logZ = {results['logz']:.4f} ± {results['logzerr']:.4f}")
    print(f"   H = {results['information']:.4f}")
    print(f"   delta_logZ = {results['delta_logz']:.6f}")
    print(f"   Converged: {converged}")
    print(f"   Iterations: {results['niter']} (max: {args.max_iterations})")
    print(f"   Runtime: {runtime:.2f}s")
    print(f"   Auto-tuned parameters: rwalk_K={sampler.rwalk_K}, scale={sampler.rwalk_step_scale:.4f}")
    print()
    print(f"   Convergence details:")
    print(f"     Final delta_logZ: {final_delta_logZ:.6f}")
    print(f"     Efficiency: {100.0 if converged else 0.0:.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()
