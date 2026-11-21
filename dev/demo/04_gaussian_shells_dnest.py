#!/usr/bin/env python
"""
Demo 4: Gaussian Shells (JNesty DNest)

Diffusive nested sampling (the CDNest algorithm) on the two-shell
problem, using the identical likelihood and prior definitions as
``04_gaussian_shells_jnest.py`` (shells at (-3.5, 0) and (3.5, 0),
radius 2, width 0.1, prior [-6, 6]^2).

The analytic reference integrates each shell's ring Gaussian in polar
coordinates; by symmetry each shell carries exactly half the posterior
mass.

Usage:
    python 04_gaussian_shells_dnest.py [--num-lanes 8]
        [--num-particles 2] [--max-num-saves 30000] [--ptol 0.1]
        [--sample-frequency 1] [--seed 42] [--device gpu]

Output (in output_04_dnest/):
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
from scipy.integrate import quad


# ============================================================================
# Problem Definition (identical to the other demo variants)
# ============================================================================

r_shell = 2.0
w = 0.1
c1 = jnp.array([-3.5, 0.0])
c2 = jnp.array([3.5, 0.0])
const = jnp.log(1.0 / jnp.sqrt(2.0 * np.pi * w ** 2))


def loglikelihood(x):
    """Two thin Gaussian shells (log-sum-exp of both rings)."""

    def logcirc(theta, c):
        d = jnp.sqrt(jnp.sum((theta - c) ** 2))
        return const - (d - r_shell) ** 2 / (2.0 * w ** 2)

    return jnp.logaddexp(logcirc(x, c1), logcirc(x, c2))


def prior_transform(u):
    """Unit cube [0, 1]^2 -> physical space [-6, 6]^2."""
    return 12.0 * u - 6.0


def analytic_logz():
    """One-dimensional polar quadrature of the two-shell evidence.

    Z = 2 * integral of exp(logcirc) over the [-6, 6]^2 prior (the two
    shells are symmetric and the ring Gaussian falls off long before the
    box edge or the other shell).
    """
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * w * 144.0)

    def radial(rho):
        return rho * np.exp(-(rho - r_shell) ** 2 / (2.0 * w ** 2))

    integral, _ = quad(radial, 0.0, 20.0)
    evidence = 2.0 * 2.0 * np.pi * prefactor * integral
    return float(np.log(evidence))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo 4: Gaussian Shells (JNesty DNest)')
    parser.add_argument('--num-lanes', type=int, default=8)
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

    ndim = 2
    _freq = [float(v) for v in args.sample_frequency.split(',')]
    frequency = _freq[0] if len(_freq) == 1 else _freq
    logz_ref = analytic_logz()

    print('=' * 70)
    print('Demo 4: Gaussian Shells (JNesty DNest)')
    print('=' * 70)
    print(f'Lanes: {args.num_lanes}, particles/lane: {args.num_particles}, '
          f'saves: {args.max_num_saves}, PTol: {args.ptol}')
    print(f'Shell radius: {r_shell}, width: {w}, centers: '
          f'({float(c1[0])}, 0) and ({float(c2[0])}, 0)')
    print(f'seed: {args.seed}, device: {args.device.upper()}, '
          f'sample_frequency: {frequency}')
    print(f'Analytic logZ (polar quadrature) = {logz_ref:.4f}')

    outdir = Path('output_04_dnest')
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

    distance1 = np.linalg.norm(points - np.array([-3.5, 0.0]), axis=1)
    distance2 = np.linalg.norm(points - np.array([3.5, 0.0]), axis=1)
    on_shell1 = distance1 < distance2
    mass_shell1 = float(weights[on_shell1].sum())
    all_distance = np.where(on_shell1, distance1, distance2)
    radial_mean = float((all_distance * weights).sum())
    radial_std = float(np.sqrt(
        (weights * (all_distance - radial_mean) ** 2).sum()))

    summary = {
        'implementation': 'jnesty.dfnest',
        'problem': 'gaussian_shells',
        'sampler': 'dnest',
        'dimension': 2,
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
        'n_eff': float(sampler.n_eff),
        'n_like': int(sampler.n_like),
        'n_levels': int(sampler.n_levels),
        'levels_finalized': bool(sampler.levels_finalized),
        'acceptance_rate': float(sampler.acceptance_rate),
        'shell_mass_shell1': mass_shell1,
        'shell_mass_shell2': 1.0 - mass_shell1,
        'radial_mean': radial_mean,
        'radial_std': radial_std,
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
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(-3.5 + r_shell * np.cos(theta), r_shell * np.sin(theta),
            'r--', alpha=0.6, label='true shells')
    ax.plot(3.5 + r_shell * np.cos(theta), r_shell * np.sin(theta),
            'r--', alpha=0.6)
    ax.set(xlabel='x', ylabel='y', xlim=(-6, 6), ylim=(-6, 6),
           aspect='equal', title='DNest posterior (weighted)')
    ax.legend()
    fig.colorbar(sc, ax=ax, label='posterior weight')
    fig.savefig(outdir / 'posterior_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plotting.runplot(results, lnz_truth=logz_ref)
    fig.savefig(outdir / 'run_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{"=" * 70}')
    print(f'Complete! Results saved to: {outdir}')
    print(f'   logZ = {log_z:.4f} (analytic = {logz_ref:.4f}, '
          f'delta = {log_z - logz_ref:+.4f})')
    print(f'   Shell masses = {mass_shell1:.3f}/{1 - mass_shell1:.3f} '
          f'(analytic 0.5/0.5)')
    print(f'   Radial mean/std = {radial_mean:.4f}/{radial_std:.4f} '
          f'(true radius 2.0)')
    print(f'   ESS = {sampler.n_eff:.0f}, levels = {sampler.n_levels}')
    print(f'   Saves = {sampler.n_saves}, likelihood calls = '
          f'{sampler.n_like}, runtime = {runtime:.1f}s')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
