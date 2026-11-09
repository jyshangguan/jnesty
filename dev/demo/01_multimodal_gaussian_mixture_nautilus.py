#!/usr/bin/env python
"""
Demo 1: Multi-modal Gaussian Mixture (JNesty Nautilus)

Importance nested sampling with our JAX Nautilus engine on the same two-mode
Gaussian mixture as ``01_multimodal_gaussian_mixture_jnesty.py``.

Usage:
    python 01_multimodal_gaussian_mixture_nautilus.py [--nlive 200] [--dim 5]

Output (in output_01_nautilus/):
    summary.json, samples.npz (points, log_w, log_l), posterior_2d.png
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from jnesty import NautilusSampler


# ============================================================================
# Problem Definition
# ============================================================================

def loglikelihood(x):
    """
    Two-mode unit-variance Gaussian mixture (weights 0.6 and 0.4).

    Mode 1 sits at the origin, mode 2 at (3, 3, 0, ..., 0).
    """
    mean1 = jnp.zeros(len(x))
    mean2 = mean1.at[0].set(3.0).at[1].set(3.0)
    logL1 = -0.5 * jnp.sum((x - mean1)**2) + jnp.log(0.6)
    logL2 = -0.5 * jnp.sum((x - mean2)**2) + jnp.log(0.4)
    return jax.scipy.special.logsumexp(jnp.stack([logL1, logL2]))


def prior_transform(u):
    """Transform from unit cube [0, 1]^ndim to physical space [-5, 5]^ndim."""
    return (u - 0.5) * 10.0  # Maps [0, 1] → [-5, 5]


def analytic_logz(ndim):
    """Exact log-evidence of the mixture over the [-5, 5]^ndim prior.

    Both modes sit well inside the box, so the likelihood integrates to
    (2 pi)**(ndim/2) and the prior volume is 10**ndim.
    """
    return 0.5 * ndim * np.log(2.0 * np.pi) - ndim * np.log(10.0)


# ============================================================================
# Plotting
# ============================================================================

def plot_posterior_2d(points, log_w, ndim, log_z, path):
    """Weighted 2D scatter of the first two dimensions."""
    weights = np.exp(log_w - np.max(log_w))
    wmax = float(weights.max())
    # Exploration-phase samples have weights orders of magnitude below the
    # posterior's: draw them faint, colour the rest by weight and point size.
    relevant = weights > wmax * 1e-4
    idx = np.flatnonzero(relevant)
    idx = idx[np.argsort(weights[idx])]  # heavy samples on top
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(points[~relevant, 0], points[~relevant, 1], c='0.8', s=3,
               alpha=0.4, edgecolors='none', label='negligible weight')
    sc = ax.scatter(points[idx, 0], points[idx, 1], c=weights[idx],
                    cmap='viridis', s=2 + 30 * np.sqrt(weights[idx] / wmax),
                    alpha=0.75, edgecolors='none',
                    norm=LogNorm(vmin=wmax * 1e-4, vmax=wmax))
    ax.plot([0.0, 3.0], [0.0, 3.0], 'r+', markersize=12, markeredgewidth=2,
            label='true mode centres')
    # The 2D marginal of each unit-variance mode is a standard normal, so its
    # 1σ/2σ contours are circles in any dimension.
    for centre in [(0.0, 0.0), (3.0, 3.0)]:
        for radius, style in [(1.0, '-'), (2.0, '--')]:
            ax.add_patch(Circle(centre, radius, fill=False, edgecolor='r',
                                linestyle=style, linewidth=1.2, alpha=0.8))
    ax.set(xlabel='Parameter 0', ylabel='Parameter 1', xlim=(-5, 5),
           ylim=(-5, 5), aspect='equal',
           title=f'{ndim}D two-mode Gaussian mixture\nlog Z = {log_z:.3f}')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('relative posterior weight')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo 1: Multi-modal Gaussian Mixture (JNesty Nautilus)')
    parser.add_argument('--nlive', type=int, default=200,
                        help='Number of live points')
    parser.add_argument('--dim', type=int, default=5,
                        help='Problem dimensionality')
    parser.add_argument('--networks', type=int, default=2,
                        help='Neural networks in the emulator ensemble '
                             '(0 disables the neural bounds)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'], help='Device to use')
    parser.add_argument('--f_live', type=float, default=0.01,
                        help='Final live-point fraction of the exploration')
    parser.add_argument('--n_eff', type=int, default=1000,
                        help='Target effective sample size')
    args = parser.parse_args()

    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = args.dim
    log_z_ref = analytic_logz(ndim)

    print("="*70)
    print("Demo 1: Multi-modal Gaussian Mixture (JNesty Nautilus)")
    print("="*70)
    print(f"Dimension: {ndim}, live points: {args.nlive}, "
          f"seed: {args.seed}, device: {args.device.upper()}")
    print(f"Neural networks: {args.networks}"
          + (" (neural bounds disabled)" if args.networks == 0 else ""))
    print(f"f_live: {args.f_live}, n_eff target: {args.n_eff}")

    outdir = Path('output_01_nautilus')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    print("\nRunning Nautilus...")
    sampler = NautilusSampler(
        loglikelihood, prior_transform, ndim=ndim, nlive=args.nlive,
        n_networks=args.networks, seed=args.seed, device=args.device)

    start_time = time.time()
    success = sampler.run(f_live=args.f_live, n_eff=args.n_eff)
    runtime = time.time() - start_time

    # Posterior samples with normalised log-weights (JAX → NumPy).
    points, log_w, log_l = sampler.posterior()
    points = np.asarray(points, dtype=float)
    log_w = np.asarray(log_w, dtype=float)
    log_l = np.asarray(log_l, dtype=float)

    log_z = float(sampler.log_z)
    summary = {
        'implementation': 'jnesty.nautilus',
        'problem': 'multi_modal_gaussian_mixture',
        'dimension': int(ndim),
        'nlive': int(args.nlive),
        'n_networks': int(args.networks),
        'seed': int(args.seed),
        'device': args.device,
        'f_live': float(args.f_live),
        'n_eff_target': int(args.n_eff),
        'log_z': log_z,
        'log_z_analytic': float(log_z_ref),
        'log_z_minus_analytic': float(log_z - log_z_ref),
        'n_eff': float(sampler.n_eff),
        'n_like': int(sampler.n_like),
        'eta': float(sampler.eta),
        'n_samples': int(len(points)),
        'success': bool(success),
        'runtime': float(runtime),
    }

    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez(outdir / 'samples.npz', points=points, log_w=log_w, log_l=log_l)
    plot_posterior_2d(points, log_w, ndim, log_z, outdir / 'posterior_2d.png')

    print("\n" + "="*70)
    print(f"✅ Complete! Results saved to: {outdir}")
    print(f"   logZ = {log_z:.4f}  "
          f"(analytic = {log_z_ref:.4f}, Δ = {log_z - log_z_ref:+.4f})")
    print(f"   n_eff = {sampler.n_eff:.1f}, n_like = {sampler.n_like}, "
          f"eta = {sampler.eta:.4f}")
    print(f"   success = {success}, runtime = {runtime:.2f}s")
    print("="*70)


if __name__ == "__main__":
    main()