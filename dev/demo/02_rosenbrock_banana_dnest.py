#!/usr/bin/env python
"""
Demo 2: Rosenbrock Banana (JNesty DNest)

Diffusive nested sampling (the CDNest algorithm) on the Rosenbrock
banana, using the identical likelihood and prior definitions as
``02_rosenbrock_banana_jnesty.py``.

Usage:
    python 02_rosenbrock_banana_dnest.py [--num-lanes 8]
        [--num-particles 2] [--max-num-saves 20000] [--ptol 0.1]
        [--a 1] [--b 100] [--sample-frequency 1] [--seed 42]
        [--device gpu]

Output (in output_02_dnest/):
    summary.json, samples.npz, results.fits, posterior_2d.png, run_plot.png
"""

import argparse
import json
import time
from pathlib import Path

from jnesty import DNestSampler, plotting, save_results

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Problem Definition (identical to the other demo variants)
# ============================================================================

def loglikelihood(x, a=1, b=100):
    """Rosenbrock 'banana' (negated for maximisation)."""
    x_val = x[0]
    y_val = x[1]
    return -0.5 * ((a - x_val) ** 2 + b * (y_val - x_val ** 2) ** 2)


def prior_transform(u):
    """Unit cube [0, 1]^2 -> physical space [-3, 3]^2."""
    return (u - 0.5) * 6.0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo 2: Rosenbrock Banana (JNesty DNest)')
    parser.add_argument('--num-lanes', type=int, default=8,
                        help='Independent workers')
    parser.add_argument('--num-particles', type=int, default=2,
                        help='Particles per lane')
    parser.add_argument('--max-num-saves', type=int, default=20000,
                        help='Total save target')
    parser.add_argument('--ptol', type=float, default=0.1)
    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=100)
    parser.add_argument('--sample-frequency', type=str, default='1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = 2
    _freq = [float(v) for v in args.sample_frequency.split(',')]
    frequency = _freq[0] if len(_freq) == 1 else _freq

    print('=' * 70)
    print('Demo 2: Rosenbrock Banana (JNesty DNest)')
    print('=' * 70)
    print(f'Lanes: {args.num_lanes}, particles/lane: {args.num_particles}, '
          f'saves: {args.max_num_saves}, PTol: {args.ptol}')
    print(f'Rosenbrock a={args.a}, b={args.b}, seed={args.seed}, '
          f'device={args.device.upper()}')
    print(f'sample_frequency: {frequency}')

    outdir = Path('output_02_dnest')
    outdir.mkdir(exist_ok=True)
    print(f'\nOutput directory: {outdir}')

    def loglikelihood_fixed(x):
        return loglikelihood(x, args.a, args.b)

    sampler = DNestSampler(
        loglikelihood_fixed, prior_transform, ndim=ndim,
        num_lanes=args.num_lanes, num_particles=args.num_particles,
        sample_frequency=frequency, ptol=args.ptol, seed=args.seed,
        device=args.device)

    start_time = time.time()
    log_z = sampler.run(max_num_saves=args.max_num_saves, verbose=True)
    runtime = time.time() - start_time

    points, log_w, log_l = sampler.posterior()
    points = np.asarray(points)
    log_w = np.asarray(log_w)
    weights = np.exp(log_w)

    results = sampler.results
    save_results(results, str(outdir / 'results.fits'))

    summary = {
        'implementation': 'jnesty.dfnest',
        'problem': 'rosenbrock_banana',
        'sampler': 'dnest',
        'dimension': 2,
        'num_lanes': int(args.num_lanes),
        'num_particles': int(args.num_particles),
        'max_num_saves': int(args.max_num_saves),
        'ptol': float(args.ptol),
        'sample_frequency': frequency,
        'seed': int(args.seed),
        'device': args.device,
        'a': float(args.a),
        'b': float(args.b),
        'log_z': float(log_z),
        'information': float(sampler.information),
        'n_eff': float(sampler.n_eff),
        'n_like': int(sampler.n_like),
        'n_levels': int(sampler.n_levels),
        'levels_finalized': bool(sampler.levels_finalized),
        'acceptance_rate': float(sampler.acceptance_rate),
        'weighted_mean': (points * weights[:, None]).sum(axis=0).tolist(),
        'runtime': float(runtime),
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez(outdir / 'samples.npz', points=points, log_w=log_w,
             log_l=np.asarray(log_l), log_x=results['logvol'],
             sample_levels=results['sample_levels'])

    print('\nGenerating plots...')
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(points[:, 0], points[:, 1], c=weights, s=3,
                    cmap='viridis', alpha=0.5)
    ax.set(xlabel='x', ylabel='y', xlim=(-3, 3), ylim=(-2, 4),
           title='DNest posterior (weighted)')
    fig.colorbar(sc, ax=ax, label='posterior weight')
    fig.savefig(outdir / 'posterior_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plotting.runplot(results)
    fig.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{"=" * 70}')
    print(f'Complete! Results saved to: {outdir}')
    print(f'   logZ = {log_z:.4f} (no analytic reference for the banana; '
          f'compare across engines)')
    print(f'   H = {sampler.information:.4f}, ESS = {sampler.n_eff:.0f}, '
          f'levels = {sampler.n_levels}')
    print(f'   Saves = {sampler.n_saves}, likelihood calls = '
          f'{sampler.n_like}, runtime = {runtime:.1f}s')
    print(f'   Weighted mean = {summary["weighted_mean"]}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
