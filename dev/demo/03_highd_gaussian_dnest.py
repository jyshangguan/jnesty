#!/usr/bin/env python
"""
Demo 3: High-Dimensional Gaussian (JNesty DNest)

Diffusive nested sampling (the CDNest algorithm) on the standard Gaussian
in ``ndim`` dimensions, using the identical likelihood and prior
definitions as ``03_highd_gaussian_jnesty.py``.

The analytic reference uses the *correct* truncated-prior information:

    Z_1 = sqrt(2 pi) erf(5/sqrt(2)) / 10
    v   = 1 - 10 phi(5) / erf(5/sqrt(2))          (truncated variance)
    H_1 = -v/2 - log(Z_1)      (= 0.883654566694516 per dimension)

so logZ = ndim log(Z_1) and H = ndim H_1.  (The naive H = ndim/2
approximation used by the older demos ignores the prior truncation and
the likelihood normalisation.)

Usage:
    python 03_highd_gaussian_dnest.py [--ndim 20] [--num-lanes 16]
        [--num-particles 2] [--max-num-saves 30000] [--ptol 0.1]
        [--sample-frequency 1] [--seed 42] [--device gpu]

Output (in output_03_dnest/):
    summary.json, samples.npz, results.fits, run_plot.png
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
from scipy.stats import norm


# ============================================================================
# Problem Definition (identical to the other demo variants)
# ============================================================================

def loglikelihood(x):
    """Standard Gaussian log-likelihood: logL = -0.5 sum(x^2)."""
    return -0.5 * jnp.sum(x ** 2)


def prior_transform(u):
    """Unit cube [0, 1]^ndim -> physical space [-5, 5]^ndim."""
    return (u - 0.5) * 10.0


def analytic_reference(ndim):
    """Truncated-Gaussian evidence, information and marginal variance."""
    q = norm.cdf(5.0) - norm.cdf(-5.0)
    z1 = np.sqrt(2.0 * np.pi) * q / 10.0
    variance = 1.0 - 2.0 * 5.0 * norm.pdf(5.0) / q
    h1 = -variance / 2.0 - np.log(z1)
    return float(ndim * np.log(z1)), float(ndim * h1), float(variance)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo 3: High-Dimensional Gaussian (JNesty DNest)')
    parser.add_argument('--ndim', type=int, default=20)
    parser.add_argument('--num-lanes', type=int, default=16)
    parser.add_argument('--num-particles', type=int, default=2)
    parser.add_argument('--max-num-saves', type=int, default=30000)
    parser.add_argument('--ptol', type=float, default=0.1)
    parser.add_argument('--sample-frequency', type=str, default='1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = args.ndim
    _freq = [float(v) for v in args.sample_frequency.split(',')]
    frequency = _freq[0] if len(_freq) == 1 else _freq
    logz_ref, h_ref, var_ref = analytic_reference(ndim)

    print('=' * 70)
    print('Demo 3: High-Dimensional Gaussian (JNesty DNest)')
    print('=' * 70)
    print(f'Dimension: {ndim}, lanes: {args.num_lanes}, '
          f'particles/lane: {args.num_particles}')
    print(f'Max saves: {args.max_num_saves}, PTol: {args.ptol}, '
          f'seed: {args.seed}, device: {args.device.upper()}')
    print(f'sample_frequency: {frequency}')
    print(f'Analytic: logZ = {logz_ref:.4f}, H = {h_ref:.4f}, '
          f'marginal variance = {var_ref:.6f}')

    outdir = Path('output_03_dnest')
    outdir.mkdir(exist_ok=True)
    print(f'\nOutput directory: {outdir}')

    sampler = DNestSampler(
        loglikelihood, prior_transform, ndim=ndim,
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

    mean = (points * weights[:, None]).sum(axis=0)
    variance = (points ** 2 * weights[:, None]).sum(axis=0) - mean ** 2

    summary = {
        'implementation': 'jnesty.dfnest',
        'problem': 'highd_gaussian',
        'sampler': 'dnest',
        'dimension': int(ndim),
        'num_lanes': int(args.num_lanes),
        'num_particles': int(args.num_particles),
        'max_num_saves': int(args.max_num_saves),
        'ptol': float(args.ptol),
        'sample_frequency': frequency,
        'seed': int(args.seed),
        'device': args.device,
        'log_z': float(log_z),
        'log_z_analytic': logz_ref,
        'log_z_minus_analytic': float(log_z - logz_ref),
        'information': float(sampler.information),
        'information_analytic': h_ref,
        'marginal_mean_max_abs': float(np.max(np.abs(mean))),
        'marginal_variance_mean': float(variance.mean()),
        'marginal_variance_analytic': var_ref,
        'n_eff': float(sampler.n_eff),
        'n_like': int(sampler.n_like),
        'n_levels': int(sampler.n_levels),
        'levels_finalized': bool(sampler.levels_finalized),
        'acceptance_rate': float(sampler.acceptance_rate),
        'coordinate_attempts': sampler.coordinate_attempts.tolist(),
        'coordinate_accepts': sampler.coordinate_accepts.tolist(),
        'runtime': float(runtime),
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez(outdir / 'samples.npz', points=points, log_w=log_w,
             log_l=np.asarray(log_l), log_x=results['logvol'],
             sample_levels=results['sample_levels'])

    print('\nGenerating plots...')
    fig, axes = plotting.runplot(results, lnz_truth=logz_ref)
    fig.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{"=" * 70}')
    print(f'Complete! Results saved to: {outdir}')
    print(f'   logZ = {log_z:.4f} (analytic = {logz_ref:.4f}, '
          f'delta = {log_z - logz_ref:+.4f})')
    print(f'   H = {sampler.information:.4f} (analytic = {h_ref:.4f})')
    print(f'   Marginal variance = {variance.mean():.4f} '
          f'(analytic = {var_ref:.4f})')
    print(f'   ESS = {sampler.n_eff:.0f}, levels = {sampler.n_levels} '
          f'(finalized: {sampler.levels_finalized})')
    print(f'   Saves = {sampler.n_saves}, likelihood calls = '
          f'{sampler.n_like}, runtime = {runtime:.1f}s')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
