#!/usr/bin/env python
"""
Demo 4: Gaussian Shells (J-Nesty)

Classic test case for multi-ellipsoid bounding from Dynesty's examples.
Two thin annular (ring-shaped) distributions separated in 2D.

The multi-ellipsoid decomposition should split these into 2 separate
ellipsoids, demonstrating proper multi-modal handling.

Usage:
    python 04_gaussian_shells_jnesty.py [--nlive 500] [--seed 42]

Output:
    Creates output_04_jnesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - posterior_2d.png: 2D scatter plot showing shell structure
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
from jnesty import NestedSampler
import matplotlib.pyplot as plt
import json
import time
import numpy as np


# ============================================================================
# Problem Definition
# ============================================================================

# Gaussian shell parameters (matching Dynesty's example)
r = 2.0                      # shell radius
w = 0.1                      # shell width
c1 = jnp.array([-3.5, 0.0])  # center of shell 1
c2 = jnp.array([3.5, 0.0])   # center of shell 2
const = jnp.log(1.0 / jnp.sqrt(2.0 * jnp.pi * w**2))


def loglikelihood(x):
    """
    Two Gaussian shells log-likelihood.

    Each shell is a thin annular (ring-shaped) Gaussian centered at c_i
    with radius r and width w:

        logL_i(x) = const - (|x - c_i| - r)^2 / (2*w^2)

    The total log-likelihood is the log-sum-exp of both shells.
    """
    def logcirc(theta, c):
        d = jnp.sqrt(jnp.sum((theta - c)**2))
        return const - (d - r)**2 / (2.0 * w**2)
    return jnp.logaddexp(logcirc(x, c1), logcirc(x, c2))


def prior_sample(key):
    """Sample from unit cube [0, 1]^2."""
    return random.uniform(key, shape=(2,), minval=0.0, maxval=1.0)


def prior_transform(u):
    """Transform from unit cube [0, 1]^2 to physical space [-6, 6]^2."""
    return 12.0 * u - 6.0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demo 4: Gaussian Shells (J-Nesty)')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device')

    args = parser.parse_args()

    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = 2

    print("=" * 70)
    print("Demo 4: Gaussian Shells (J-Nesty)")
    print("=" * 70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {args.nlive}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Shell radius: {r}, width: {w}")
    print(f"Shell centers: {np.array(c1)}, {np.array(c2)}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device.upper()}")

    # Create output directory
    outdir = Path('output_04_jnesty')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Run J-Nesty
    print("\nRunning J-Nesty with multi-ellipsoid bounding...")

    sampler = NestedSampler(
        loglikelihood,
        prior_transform,
        ndim=ndim,
        nlive=args.nlive,
        device=args.device,
        verbose=True,
        bound='multi',
        # bound_update_interval=None (default: auto = nlive = 500)
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
        'implementation': 'jnesty',
        'problem': 'gaussian_shells',
        'dimension': ndim,
        'nlive': args.nlive,
        'max_iterations': args.max_iterations,
        'seed': args.seed,
        'shell_radius': float(r),
        'shell_width': float(w),
        'shell_center1': [float(c1[0]), float(c1[1])],
        'shell_center2': [float(c2[0]), float(c2[1])],
        'logZ': float(results['logz']),
        'logZ_error': float(results['logzerr']),
        'H': float(results['information']),
        'delta_logZ': float(final_delta_logZ),
        'converged': bool(converged),
        'n_iterations': int(results['niter']),
        'runtime': float(runtime),
        'iterations_per_sec': float(results['niter'] / runtime) if runtime > 0 else 0,
        'acceptance_rate': float(results['acceptance_rate'])
    }

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save samples
    np.savez(outdir / 'samples.npz', samples=samples, logL=logL_samples)

    # Save trace
    np.savez(outdir / 'trace.npz',
             logL_trajectory=logL_samples,
             delta_logZ_trajectory=delta_logZ_trajectory)

    # Generate plots
    print("\nGenerating plots...")

    # 2D posterior scatter plot
    print("  Generating 2D posterior plot...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('J-Nesty: Gaussian Shells Posterior')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw true shell locations for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(float(c1[0]) + r * np.cos(theta), float(c1[1]) + r * np.sin(theta),
            'r--', alpha=0.5, label='True shell locations')
    ax.plot(float(c2[0]) + r * np.cos(theta), float(c2[1]) + r * np.sin(theta),
            'r--', alpha=0.5)
    ax.legend()
    plt.savefig(outdir / 'posterior_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: posterior_2d.png")

    # Run plot
    print("  Generating run plot...")
    fig, axes = sampler.plot_run()
    plt.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: run_plot.png")

    # Trace plot
    print("  Generating trace plot...")
    fig, axes = sampler.plot_trace()
    plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: trace_plot.png")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Complete! Results saved to: {outdir}")
    print(f"  logZ = {results['logz']:.4f} +/- {results['logzerr']:.4f}")
    print(f"  H = {results['information']:.4f}")
    print(f"  delta_logZ = {final_delta_logZ:.6f}")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {results['niter']} (max: {args.max_iterations})")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Speed: {results['niter'] / runtime:.1f} iter/s")
    print(f"  Acceptance rate: {results['acceptance_rate']:.1%}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
