#!/usr/bin/env python
"""
Demo 2: Rosenbrock Banana (JNesty Nautilus)

Importance nested sampling with our JAX Nautilus engine on the same
highly-correlated Rosenbrock "banana" as ``02_rosenbrock_banana_jnesty.py``.

Usage:
    python 02_rosenbrock_banana_nautilus.py [--nlive 200] [--a 1] [--b 100]

Output (in output_02_nautilus/):
    summary.json, samples.npz (points, log_w, log_l),
    posterior_2d.png, trace_plot.png
"""

import argparse
import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from jnesty import NautilusSampler


# ============================================================================
# Problem Definition
# ============================================================================

def loglikelihood(x, a=1, b=100):
    """
    Rosenbrock "banana" f(x, y) = (a - x)^2 + b(y - x^2)^2, negated.

    The 0.5 factor makes it a proper Gaussian log-likelihood; the posterior
    is highly correlated and curved ("banana").
    """
    x_val, y_val = x[0], x[1]
    return -0.5 * ((a - x_val)**2 + b * (y_val - x_val**2)**2)


def prior_transform(u):
    """Transform from unit cube [0, 1]^2 to physical space [-3, 3]^2."""
    return (u - 0.5) * 6.0  # Maps [0, 1] → [-3, 3]


# ============================================================================
# Plotting
# ============================================================================

def plot_posterior_2d(points, log_w, log_z, a, b, path):
    """Weighted scatter of the posterior samples, showing the banana."""
    weights = np.exp(log_w - np.max(log_w))
    wmax = float(weights.max())
    # Exploration samples have far smaller weights: fade them, colour the rest.
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
    x_curve = np.linspace(-3.0, 3.0, 200)
    ax.plot(x_curve, x_curve**2, 'r--', linewidth=1.5,
            label='Rosenbrock minimum y = x²')
    ax.set(xlabel='Parameter 0 (x)', ylabel='Parameter 1 (y)',
           xlim=(-3, 3), ylim=(-3, 3), aspect='equal',
           title=f'Rosenbrock banana (a={a:g}, b={b:g})\nlog Z = {log_z:.3f}')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('relative posterior weight')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trace(log_l, log_w, path):
    """Log-likelihood and log-weight traces in nested-shell order.

    ``posterior()`` returns the samples shell by shell, so the index stands
    in for the iteration; exploration values are far below the posterior's.
    """
    index = np.arange(len(log_l))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(index, log_l, linewidth=0.6, color='C0')
    axes[0].set(ylabel='Log-likelihood')
    axes[0].set_yscale('symlog', linthresh=10)
    axes[0].grid(alpha=0.3)
    axes[1].plot(index, log_w, linewidth=0.6, color='C1')
    axes[1].set(ylabel='Log-weight  log w',
                xlabel='Posterior sample index (nested-shell order)')
    axes[1].set_yscale('symlog', linthresh=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle('Rosenbrock banana: posterior traces', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo 2: Rosenbrock Banana (JNesty Nautilus)')
    parser.add_argument('--nlive', type=int, default=200,
                        help='Number of live points')
    parser.add_argument('--networks', type=int, default=2,
                        help='Neural networks in the emulator ensemble '
                             '(0 disables the neural bounds)')
    parser.add_argument('--a', type=float, default=1, help='Rosenbrock a')
    parser.add_argument('--b', type=float, default=100, help='Rosenbrock b')
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

    ndim = 2

    print("="*70)
    print("Demo 2: Rosenbrock Banana (JNesty Nautilus)")
    print("="*70)
    print(f"Dimension: {ndim}, live points: {args.nlive}, "
          f"seed: {args.seed}, device: {args.device.upper()}")
    print(f"Neural networks: {args.networks}"
          + (" (neural bounds disabled)" if args.networks == 0 else ""))
    print(f"Rosenbrock parameters: a={args.a}, b={args.b}, "
          f"f_live: {args.f_live}, n_eff target: {args.n_eff}")

    outdir = Path('output_02_nautilus')
    outdir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    def loglikelihood_fixed(x):
        return loglikelihood(x, args.a, args.b)

    print("\nRunning Nautilus...")
    sampler = NautilusSampler(
        loglikelihood_fixed, prior_transform, ndim=ndim, nlive=args.nlive,
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
        'problem': 'rosenbrock_banana',
        'dimension': int(ndim),
        'nlive': int(args.nlive),
        'n_networks': int(args.networks),
        'a': float(args.a),
        'b': float(args.b),
        'seed': int(args.seed),
        'device': args.device,
        'f_live': float(args.f_live),
        'n_eff_target': int(args.n_eff),
        'log_z': log_z,
        'n_eff': float(sampler.n_eff),
        'n_like': int(sampler.n_like),
        'eta': float(sampler.eta),
        'n_samples': int(len(points)),
        'success': bool(success),
        'runtime': float(runtime),
    }

    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez(outdir / 'samples.npz', points=points, log_w=log_w, log_l=log_l)
    plot_posterior_2d(points, log_w, log_z, args.a, args.b,
                      outdir / 'posterior_2d.png')
    plot_trace(log_l, log_w, outdir / 'trace_plot.png')

    print("\n" + "="*70)
    print(f"✅ Complete! Results saved to: {outdir}")
    print(f"   logZ = {log_z:.4f}")
    print(f"   n_eff = {sampler.n_eff:.1f}, n_like = {sampler.n_like}, "
          f"eta = {sampler.eta:.4f}")
    print(f"   success = {success}, runtime = {runtime:.2f}s")
    print("="*70)


if __name__ == "__main__":
    main()