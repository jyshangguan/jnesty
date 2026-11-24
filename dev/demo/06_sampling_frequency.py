#!/usr/bin/env python
"""
Demo 6: Per-Coordinate Sampling Frequency (rwalk_direction_weights)

An 8-D two-scale problem mimicking a grid BLR fit: 2 'physical' coordinates
with sigma=1 and 6 'amplitude-like' coordinates with sigma=0.05 (diagonal
covariance; analytic logZ).  Three arms are compared with identical seeds
and budgets:

* ``unweighted``  -- rwalk_direction_weights=None (isotropic walk)
* ``aligned``     -- w = [1, 1, 0.04 x 6]  (physical parameters at the
                     anchor; amplitude latents explored at 4% of the rate)
* ``misaligned``  -- w = [0.04, 0.04, 1 x 6] (inverted: corrupts the walk)

Usage:
    python 06_sampling_frequency.py [--bound single|none] [--nlive 150]

Output (in output_06_sampling_frequency/):
    summary.json, comparison.png
"""

import argparse
import json
import os
import time
from math import erf, log, pi, sqrt

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnesty import NestedSampler

SIGMA = np.array([1.0, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
LOGZ_REF = sum(log((1.0 / 10.0) * sqrt(2 * pi) * s * erf(5.0 / (sqrt(2) * s)))
               for s in SIGMA)
SEEDS = [42, 7, 123]

_sig = jnp.asarray(SIGMA)


def loglikelihood(x):
    return -0.5 * jnp.sum((jnp.asarray(x) / _sig) ** 2)


def prior_transform(u):
    return (jnp.asarray(u) - 0.5) * 10.0


WEIGHTS = {
    'unweighted': None,
    'aligned': [1.0, 1.0, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
    'misaligned': [0.04, 0.04, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def run_arm(bound, nlive, config, seed):
    sampler = NestedSampler(
        loglikelihood, prior_transform, 8, nlive=nlive, bound=bound,
        queue_size=0, rwalk_K=25,
        rwalk_direction_weights=WEIGHTS[config], verbose=False)
    t0 = time.perf_counter()
    sampler.run_nested(max_iterations=20000, delta_logZ_threshold=0.05,
                       print_progress=False, seed=seed)
    wall = time.perf_counter() - t0
    res = sampler.results
    samples = np.asarray(res['samples'], dtype=np.float64)
    logwt = np.asarray(res['logwt'], dtype=np.float64)
    logz = float(res['logz'])
    w = np.exp(logwt - logz)
    w /= w.sum()
    mean = w @ samples
    logzerr = float(res['logzerr'])
    return dict(config=config, seed=seed, logz=logz,
                logz_err=logz - LOGZ_REF, logzerr=logzerr,
                tol=max(0.35, 3 * logzerr), ncall=int(res['ncall']),
                wall_s=wall, phys_max_abs_mean=float(np.max(np.abs(mean[:2]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bound', default='single', choices=['single', 'none'])
    parser.add_argument('--nlive', type=int, default=150)
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output_06_sampling_frequency')
    os.makedirs(outdir, exist_ok=True)

    records = []
    for config in ('unweighted', 'aligned', 'misaligned'):
        for seed in SEEDS:
            r = run_arm(args.bound, args.nlive, config, seed)
            records.append(r)
            print(f"{config:<11} seed {seed:<4} logZ err {r['logz_err']:+.2f} "
                  f"(tol {r['tol']:.2f})  phys max|mean| "
                  f"{r['phys_max_abs_mean']:.3f}  ncall {r['ncall']}", flush=True)

    # Summary table.
    print(f"\n=== 8-D two-scale demo, bound='{args.bound}', "
          f"nlive={args.nlive} (3 seeds) ===")
    print(f"{'arm':<12}{'logZ err (mean)':>17}{'pass':>6}"
          f"{'phys max|mean|':>16}{'ncall':>9}")
    summary = []
    for config in ('unweighted', 'aligned', 'misaligned'):
        rows = [r for r in records if r['config'] == config]
        errs = np.array([r['logz_err'] for r in rows])
        npass = sum(1 for r in rows if abs(r['logz_err']) < r['tol'])
        pm = np.mean([r['phys_max_abs_mean'] for r in rows])
        nc = np.mean([r['ncall'] for r in rows])
        print(f"{config:<12}{errs.mean():>+12.2f}        {npass}/3"
              f"{pm:>14.3f}{nc:>9.0f}")
        summary.append(dict(config=config, mean_logz_err=float(errs.mean()),
                            n_pass=int(npass),
                            mean_phys_max_abs_mean=float(pm),
                            mean_ncall=float(nc)))

    # Comparison figure: moment bias + per-seed logZ error.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    configs = ['unweighted', 'aligned', 'misaligned']
    colors = ['#BBBBBB', '#009988', '#CC3311']
    a1.bar(range(3), [s['mean_phys_max_abs_mean'] for s in summary], 0.55,
           color=colors, alpha=0.9)
    a1.set_xticks(range(3))
    a1.set_xticklabels(configs)
    a1.set_ylabel('max |mean| of physical coords')
    a1.set_title('Posterior-moment bias (lower is better)')
    for i, s in enumerate(summary):
        a1.text(i, s['mean_phys_max_abs_mean'] + 0.03,
                f"{s['mean_phys_max_abs_mean']:.2f}", ha='center', fontsize=9)
    for i, config in enumerate(configs):
        rows = [r for r in records if r['config'] == config]
        a2.plot([i] * 3, [r['logz_err'] for r in rows], 'o', ms=6,
                color=colors[i], alpha=0.8)
        tols = [r['tol'] for r in rows]
        a2.vlines(i, -max(tols), max(tols), color=colors[i], ls='--', lw=1,
                  alpha=0.5)
    a2.axhline(0, color='k', lw=0.8)
    a2.set_xticks(range(3))
    a2.set_xticklabels(configs)
    a2.set_ylabel(r'$\log Z - \log Z_{\rm ref}$')
    a2.set_title('Evidence error per seed (dashed = tolerance)')
    fig.suptitle('rwalk_direction_weights: aligned helps, misaligned hurts')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'comparison.png'), dpi=150)
    plt.close(fig)

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(dict(bound=args.bound, nlive=args.nlive,
                       logz_reference=LOGZ_REF, summary=summary,
                       records=records), f, indent=1)
    print(f"\nwrote {outdir}/summary.json and comparison.png")


if __name__ == '__main__':
    main()
