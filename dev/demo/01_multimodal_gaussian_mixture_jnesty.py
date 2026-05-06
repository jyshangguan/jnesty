#!/usr/bin/env python
"""
Demo 1: Multi-modal Gaussian Mixture (J-Nesty)

Tests multi-modality handling with 2 separated Gaussian modes.
Uses multi-ellipsoid bounding (bound='multi') with random walk sampling.

Usage:
    python 01_multimodal_gaussian_mixture_jnesty.py [--nlive 500] [--dim 5] [--seed 42]

Output:
    Creates output_01_jnesty/ with:
    - summary.json: Numerical results
    - samples.npz: Posterior samples
    - trace.npz: Log-likelihood trajectory
    - posterior_2d.png: 2D scatter plot
    - trace_plot.png: Log-likelihood trajectory
    - corner.png: Corner plot (if dim <= 10)
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

def loglikelihood(x):
    """
    2-mode Gaussian mixture log-likelihood.

    Mode 1: centered at (0, 0) with weight 0.6
    Mode 2: centered at (3, 3) with weight 0.4
    """
    ndim = len(x)

    # Mode 1: centered at origin
    if ndim == 2:
        mean1 = jnp.array([0.0, 0.0])
        mean2 = jnp.array([3.0, 3.0])
    else:
        # Higher dimensions: first 2 dims separated, rest at 0
        mean1 = jnp.zeros(ndim)
        mean2 = jnp.zeros(ndim)
        mean2 = mean2.at[0].set(3.0).at[1].set(3.0)

    diff1 = x - mean1
    logL1 = -0.5 * jnp.sum(diff1**2) + jnp.log(0.6)

    diff2 = x - mean2
    logL2 = -0.5 * jnp.sum(diff2**2) + jnp.log(0.4)

    return jax.scipy.special.logsumexp(jnp.array([logL1, logL2]))


def prior_sample(key):
    """Sample from unit cube [0, 1]^ndim (Dynesty-style)."""
    ndim = 5  # Will be set correctly in main
    return random.uniform(key, shape=(ndim,), minval=0.0, maxval=1.0)


def prior_transform(u):
    """Transform from unit cube to physical space [-5, 5]^ndim."""
    return (u - 0.5) * 10.0  # Maps [0, 1] → [-5, 5]


# ============================================================================
# Plotting
# ============================================================================


def generate_comparison_plots(jaxns_outdir, dynesty_dir, jaxns_summary, jaxns_samples, jaxns_logL, jaxns_delta):
    """Generate comparison plots between J-Nesty and Dynesty results."""

    # Load Dynesty results
    try:
        with open(dynesty_dir / 'summary.json', 'r') as f:
            dynesty_summary = json.load(f)
        dynesty_data = np.load(dynesty_dir / 'samples.npz')
        dynesty_samples = dynesty_data['samples']
        dynesty_logL = dynesty_data['logL']

        # Load trace if available
        try:
            dynesty_trace = np.load(dynesty_dir / 'trace.npz')
            dynesty_delta = dynesty_trace.get('logZ', np.array([]))
            if len(dynesty_delta) > 0:
                dynesty_delta = np.abs(np.diff(dynesty_delta, prepend=0))
        except:
            dynesty_delta = np.array([])
    except Exception as e:
        raise Exception(f"Could not load Dynesty results: {e}")

    print("  ✓ Loaded Dynesty results")

    # Create comparison directory
    comp_dir = jaxns_outdir / 'comparison'
    comp_dir.mkdir(exist_ok=True)

    # 1. Summary comparison table
    fig, ax = plt.subplots(figsize=(12, 8))

    metrics = [
        ('Evidence (logZ)', 'logZ', 'logZ_error'),
        ('Information (H)', 'H', None),
        ('Iterations', 'n_iterations', None),
        ('Runtime (s)', 'runtime', None),
        ('Speed (iter/s)', 'iterations_per_sec', None),
        ('Converged', 'converged', None),
    ]

    table_data = []
    for label, key, error_key in metrics:
        jaxns_val = jaxns_summary.get(key, 0)
        dynesty_val = dynesty_summary.get(key, 0)

        # Format values
        if key == 'converged':
            jaxns_str = '✓' if jaxns_val else '✗'
            dynesty_str = '✓' if dynesty_val else '✗'
        elif key == 'logZ':
            jaxns_str = f'{jaxns_val:.4f} ± {jaxns_summary[error_key]:.4f}'
            dynesty_str = f'{dynesty_val:.4f} ± {dynesty_summary.get(error_key, 0):.4f}'
        elif key in ['H', 'iterations_per_sec', 'runtime']:
            jaxns_str = f'{jaxns_val:.2f}'
            dynesty_str = f'{dynesty_val:.2f}'
        elif key == 'n_iterations':
            jaxns_str = f'{int(jaxns_val)}'
            dynesty_str = f'{int(dynesty_val)}'
        else:
            jaxns_str = str(jaxns_val)
            dynesty_str = str(dynesty_val)

        # Calculate difference for logZ
        if key == 'logZ':
            diff = abs(jaxns_val - dynesty_val)
            diff_str = f'Δ = {diff:.4f}'
        else:
            diff_str = ''

        table_data.append([label, jaxns_str, dynesty_str, diff_str])

    # Plot table
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'J-Nesty', 'Dynesty', 'Difference'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style the header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Style alternating rows
    for i in range(1, len(metrics) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')

    ax.axis('off')
    ax.set_title('J-Nesty vs Dynesty: Quantitative Comparison\nMulti-Modal Gaussian Mixture (5D)',
                 fontsize=14, weight='bold', pad=20)

    # Add interpretation
    logZ_diff = abs(jaxns_summary['logZ'] - dynesty_summary['logZ'])
    combined_unc = np.sqrt(jaxns_summary['logZ_error']**2 + dynesty_summary.get('logZ_error', 0)**2)

    interpretation = f"Interpretation:\n"
    interpretation += f"• logZ difference: {logZ_diff:.4f}\n"
    interpretation += f"• Combined uncertainty: ±{combined_unc:.4f}\n"
    if logZ_diff < combined_unc:
        interpretation += f"• ✅ Results AGREE within Monte Carlo uncertainty"
    else:
        interpretation += f"• ⚠️  Results differ by more than uncertainty"

    ax.text(0.5, 0.02, interpretation,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(comp_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: comparison/summary.png")

    # 2. Posterior comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # J-Nesty
    axes[0].scatter(jaxns_samples[:, 0], jaxns_samples[:, 1],
                    s=1, alpha=0.3, c='blue')
    axes[0].set_xlabel('Parameter 0')
    axes[0].set_ylabel('Parameter 1')
    axes[0].set_title('J-Nesty Posterior')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)

    # Dynesty
    axes[1].scatter(dynesty_samples[:, 0], dynesty_samples[:, 1],
                    s=1, alpha=0.3, c='red')
    axes[1].set_xlabel('Parameter 0')
    axes[1].set_ylabel('Parameter 1')
    axes[1].set_title('Dynesty Posterior')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)

    fig.suptitle('Posterior Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(comp_dir / 'posterior.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: comparison/posterior.png")

    # 3. Trace comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # J-Nesty logL
    axes[0, 0].plot(jaxns_logL, linewidth=0.5, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Log-Likelihood')
    axes[0, 0].set_title('J-Nesty: Log-Likelihood')
    axes[0, 0].grid(True, alpha=0.3)

    # Dynesty logL
    axes[0, 1].plot(dynesty_logL, linewidth=0.5, color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Log-Likelihood')
    axes[0, 1].set_title('Dynesty: Log-Likelihood')
    axes[0, 1].grid(True, alpha=0.3)

    # J-Nesty delta_logZ
    axes[1, 0].semilogy(jaxns_delta, linewidth=0.5, color='blue', alpha=0.7)
    axes[1, 0].axhline(y=0.01, color='green', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('delta_logZ')
    axes[1, 0].set_title('J-Nesty: Convergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Dynesty delta_logZ (if available)
    if len(dynesty_delta) > 0:
        axes[1, 1].semilogy(dynesty_delta, linewidth=0.5, color='red', alpha=0.7)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axhline(y=0.01, color='green', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('delta_logZ')
    axes[1, 1].set_title('Dynesty: Convergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Trace Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(comp_dir / 'trace.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: comparison/trace.png")

    # Print summary
    print("\n" + "="*70)
    print("Comparison Summary:")
    print("="*70)
    print(f"{'Metric':<30} {'J-Nesty':<20} {'Dynesty':<20}")
    print("-"*70)
    print(f"{'logZ':<30} {jaxns_summary['logZ']:.4f} ± {jaxns_summary['logZ_error']:.4f}  {dynesty_summary['logZ']:.4f} ± {dynesty_summary.get('logZ_error', 0):.4f}")
    print(f"{'H':<30} {jaxns_summary['H']:.4f}              {dynesty_summary['H']:.4f}")
    print(f"{'Iterations':<30} {jaxns_summary['n_iterations']}              {dynesty_summary['n_iterations']}")
    print(f"{'Runtime (s)':<30} {jaxns_summary['runtime']:.2f}              {dynesty_summary['runtime']:.2f}")
    print("-"*70)
    print(f"{'logZ difference':<30} {logZ_diff:.4f}")
    print(f"{'Combined uncertainty':<30} ±{combined_unc:.4f}")
    print(f"{'Agreement':<30} {'✅ YES' if logZ_diff < combined_unc else '⚠️  NO'}")
    print("="*70)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demo 1: Multi-modal Gaussian Mixture (J-Nesty)')
    parser.add_argument('--nlive', type=int, default=500, help='Number of live points')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimensionality')
    parser.add_argument('--max_iterations', type=int, default=10000, help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device')

    args = parser.parse_args()

    # Setup
    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = args.dim
    nlive = args.nlive
    max_iterations = args.max_iterations

    print("="*70)
    print("Demo 1: Multi-modal Gaussian Mixture (J-Nesty)")
    print("="*70)
    print(f"Dimension: {ndim}")
    print(f"Live points: {nlive}")
    print(f"Max iterations: {max_iterations}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device.upper()}")

    # Create output directory
    outdir = Path('output_01_jnesty')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Run J-Nesty with new simple API
    print("\nRunning J-Nesty with new simple API...")

    sampler = NestedSampler(
        loglikelihood,
        prior_transform,
        ndim=ndim,
        nlive=nlive,
        device=args.device,
        verbose=True,
        bound='multi',
        bound_update_interval=0
    )

    start_time = time.time()
    sampler.run_nested(max_iterations=max_iterations, delta_logZ_threshold=0.01)
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
        'problem': 'multi_modal_gaussian_mixture',
        'dimension': int(ndim),
        'nlive': nlive,
        'max_iterations': max_iterations,
        'seed': args.seed,
        'logZ': float(results['logz']),
        'logZ_error': float(results['logzerr']),
        'H': float(results['information']),
        'delta_logZ': float(final_delta_logZ),
        'converged': bool(converged),
        'n_iterations': int(results['niter']),
        'runtime': float(runtime),
        'iterations_per_sec': float(results['niter'] / runtime)
    }

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save samples
    np.savez(outdir / 'samples.npz', samples=samples, logL=logL_samples)

    # Save trace
    np.savez(outdir / 'trace.npz',
             logL_trajectory=logL_samples,
             delta_logZ_trajectory=delta_logZ_trajectory)

    print("\nGenerating plots...")

    # Run plot - evidence evolution
    print("  Generating run plot...")
    fig, axes = sampler.plot_run()
    plt.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: run_plot.png")

    # Trace plot - particle evolution
    print("  Generating trace plot...")
    # Use thin parameter to reduce overplotting for large datasets
    fig, axes = sampler.plot_trace(thin=5)
    plt.savefig(outdir / 'trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: trace_plot.png")

    # Corner plot - posterior distributions
    if ndim <= 10:
        print("  Generating corner plot...")
        fig, axes = sampler.plot_corner()
        plt.savefig(outdir / 'corner_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: corner_plot.png")

    # Diagnostics plot
    print("  Generating diagnostics plot...")
    fig, axes = sampler.plot_diagnostics()
    plt.savefig(outdir / 'diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: diagnostics.png")

    print(f"\n✅ Complete! Results saved to: {outdir}")
    print(f"   logZ = {results['logz']:.4f} ± {results['logzerr']:.4f}")
    print(f"   H = {results['information']:.4f}")
    print(f"   delta_logZ = {results['delta_logz']:.6f}")
    print(f"   Converged: {converged}")
    print(f"   Iterations: {results['niter']} (max: {max_iterations})")
    print(f"   Runtime: {runtime:.2f}s")
    print(f"   Auto-tuned parameters: rwalk_K={sampler.rwalk_K}, scale={sampler.rwalk_step_scale:.4f}")

    # Print convergence details
    if len(delta_logZ_trajectory) > 0:
        print(f"\n   Convergence details:")
        print(f"     Initial delta_logZ: {delta_logZ_trajectory[0]:.4f}")
        print(f"     Final delta_logZ: {delta_logZ_trajectory[-1]:.6f}")

        # Find when it converged
        converged_indices = np.where(delta_logZ_trajectory < 0.01)[0]
        if len(converged_indices) > 0:
            first_converged = converged_indices[0]
            print(f"     First converged at: iteration {first_converged}")
            print(f"     Efficiency: {100 * first_converged / len(delta_logZ_trajectory):.1f}%")

    # Generate comparison with Dynesty if results exist
    dynesty_dir = Path('output_01_dynesty')
    if dynesty_dir.exists():
        print("\n" + "="*70)
        print("Generating comparison with Dynesty...")
        print("="*70)

        try:
            generate_comparison_plots(outdir, dynesty_dir, summary, samples, logL_samples, delta_logZ_trajectory)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not generate comparison plots: {e}")
            print("   (This is okay if Dynesty results are not available)")


if __name__ == "__main__":
    main()
