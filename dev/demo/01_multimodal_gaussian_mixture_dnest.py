#!/usr/bin/env python
"""
Demo 1: Multi-modal Gaussian Mixture (JNesty DNest)

Diffusive nested sampling (the CDNest algorithm) on the same two-mode
Gaussian mixture as ``01_multimodal_gaussian_mixture_jnesty.py``, using
the identical likelihood and prior definitions.

Usage:
    python 01_multimodal_gaussian_mixture_dnest.py [--nlive-unused]
        [--dim 5] [--num-lanes 8] [--num-particles 2]
        [--max-num-saves 20000] [--ptol 0.1] [--seed 42]
        [--sample-frequency 1] [--device gpu]

Output (in output_01_dnest/):
    summary.json, samples.npz, results.fits, posterior_2d.png, run_plot.png
"""

import argparse
import json
import time
from pathlib import Path

# Import jnesty before jax so the quiet JAX log defaults apply.
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
    """Two-mode unit-variance Gaussian mixture (weights 0.6 and 0.4)."""
    mean1 = jnp.zeros(len(x))
    mean2 = mean1.at[0].set(3.0).at[1].set(3.0)
    logL1 = -0.5 * jnp.sum((x - mean1) ** 2) + jnp.log(0.6)
    logL2 = -0.5 * jnp.sum((x - mean2) ** 2) + jnp.log(0.4)
    return jax.scipy.special.logsumexp(jnp.stack([logL1, logL2]))


def prior_transform(u):
    """Unit cube [0, 1]^ndim -> physical space [-5, 5]^ndim."""
    return (u - 0.5) * 10.0


def analytic_evidence(ndim):
    """Exact finite-box evidence and component masses.

    Z = sum_m w_m (2 pi)^(d/2) prod_j [Phi(5 - mu_mj) - Phi(-5 - mu_mj)]
        / 10^d
    The (3, 3, ...) mode is truncated by the [-5, 5]^d prior, so its mass
    is below its weight 0.4.
    """
    total = 0.0
    masses = []
    for weight, mean in ((0.6, np.zeros(ndim)),
                         (0.4, np.array([3.0, 3.0] + [0.0] * (ndim - 2)))):
        box = np.prod(norm.cdf(5.0 - mean) - norm.cdf(-5.0 - mean))
        total += weight * box
        masses.append(weight * box)
    logz = 0.5 * ndim * np.log(2.0 * np.pi) - ndim * np.log(10.0) \
        + np.log(total)
    return float(logz), [m / total for m in masses]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo 1: Multi-modal Gaussian Mixture (JNesty DNest)')
    parser.add_argument('--dim', type=int, default=5,
                        help='Problem dimensionality')
    parser.add_argument('--num-lanes', type=int, default=8,
                        help='Independent workers (JAX analogue of MPI ranks)')
    parser.add_argument('--num-particles', type=int, default=2,
                        help='Particles per lane (CDNest NumberParticles)')
    parser.add_argument('--max-num-saves', type=int, default=20000,
                        help='Total save target (run length)')
    parser.add_argument('--ptol', type=float, default=0.1,
                        help='Level-spacing stop for level creation')
    parser.add_argument('--max-num-levels', type=int, default=0,
                        help='Fixed level count (0: PTol-controlled)')
    parser.add_argument('--sample-frequency', type=str, default='1',
                        help='Per-coordinate proposal weights: one float or '
                             'comma-separated dim floats')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'], help='Device to use')
    args = parser.parse_args()

    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    ndim = args.dim
    _freq = [float(v) for v in args.sample_frequency.split(',')]
    frequency = _freq[0] if len(_freq) == 1 else _freq
    logz_ref, mode_masses = analytic_evidence(ndim)

    print('=' * 70)
    print('Demo 1: Multi-modal Gaussian Mixture (JNesty DNest)')
    print('=' * 70)
    print(f'Dimension: {ndim}, lanes: {args.num_lanes}, '
          f'particles/lane: {args.num_particles}')
    print(f'Total diffusive particles: '
          f'{args.num_lanes * args.num_particles}')
    print(f'Max saves: {args.max_num_saves}, PTol: {args.ptol}, '
          f'seed: {args.seed}, device: {args.device.upper()}')
    print(f'sample_frequency: {frequency}')
    print(f'Analytic logZ = {logz_ref:.4f}, mode masses = '
          f'{mode_masses[0]:.3f}/{mode_masses[1]:.3f}')

    outdir = Path('output_01_dnest')
    outdir.mkdir(exist_ok=True)
    print(f'\nOutput directory: {outdir}')

    sampler = DNestSampler(
        loglikelihood, prior_transform, ndim=ndim,
        num_lanes=args.num_lanes, num_particles=args.num_particles,
        sample_frequency=frequency, ptol=args.ptol,
        max_num_levels=args.max_num_levels, seed=args.seed,
        device=args.device)

    start_time = time.time()
    log_z = sampler.run(max_num_saves=args.max_num_saves, verbose=True)
    runtime = time.time() - start_time

    points, log_w, log_l = sampler.posterior()
    points = np.asarray(points)
    log_w = np.asarray(log_w)
    log_l = np.asarray(log_l)
    weights = np.exp(log_w)

    results = sampler.results
    save_results(results, str(outdir / 'results.fits'))

    # Weighted mode masses (finite-box truth, not the raw 0.6/0.4 weights).
    distance = np.linalg.norm(
        points[:, :2] - np.array([0.0, 0.0]), axis=1)
    threshold = 3.0 / np.sqrt(2.0)
    mass_mode1 = float(weights[distance < threshold].sum())
    mass_mode2 = 1.0 - mass_mode1

    summary = {
        'implementation': 'jnesty.dfnest',
        'problem': 'multi_modal_gaussian_mixture',
        'sampler': 'dnest',
        'dimension': int(ndim),
        'num_lanes': int(args.num_lanes),
        'num_particles': int(args.num_particles),
        'total_particles': int(args.num_lanes * args.num_particles),
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
        'n_steps': int(sampler.n_steps),
        'n_saves': int(sampler.n_saves),
        'n_levels': int(sampler.n_levels),
        'levels_finalized': bool(sampler.levels_finalized),
        'acceptance_rate': float(sampler.acceptance_rate),
        'coordinate_attempts': sampler.coordinate_attempts.tolist(),
        'coordinate_accepts': sampler.coordinate_accepts.tolist(),
        'mode_mass_mode1': mass_mode1,
        'mode_mass_mode2': mass_mode2,
        'mode_mass_analytic': mode_masses,
        'runtime': float(runtime),
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez(outdir / 'samples.npz', points=points, log_w=log_w,
             log_l=log_l, log_x=results['logvol'],
             sample_levels=results['sample_levels'])

    print('\nGenerating plots...')
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(points[:, 0], points[:, 1], c=weights, s=3,
                    cmap='viridis', alpha=0.5)
    for center in [(0, 0), (3, 3)]:
        ax.plot(center[0], center[1], 'r+', markersize=12)
    ax.set(xlabel='Parameter 0', ylabel='Parameter 1', xlim=(-5, 5),
           ylim=(-5, 5), title='DNest posterior (weighted)')
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
    print(f'   H = {sampler.information:.4f}, ESS = {sampler.n_eff:.0f}, '
          f'levels = {sampler.n_levels} (finalized: '
          f'{sampler.levels_finalized})')
    print(f'   Mode masses = {mass_mode1:.3f}/{mass_mode2:.3f} '
          f'(analytic {mode_masses[0]:.3f}/{mode_masses[1]:.3f})')
    print(f'   Saves = {sampler.n_saves}, likelihood calls = '
          f'{sampler.n_like}, runtime = {runtime:.1f}s')
    print(f'   (no per-run logzerr; CDNest recommends repeated runs)')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
