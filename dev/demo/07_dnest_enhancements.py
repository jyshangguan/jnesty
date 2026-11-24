#!/usr/bin/env python
"""
Demo 7: DNest enhancements -- one enhanced G1 fit.

G1 is the 128-dimensional GMRF problem of the T1c Phase-1 evidence study
(grid (4, 8, 4) second-difference precision, sigma_z = 1, likelihood
N(z; m, 0.5^2 I) with m = 0.7 * default_rng(5)); its analytic logZ is
recreated inline and guarded below.  One enhanced fit:
``gradient_scales=True``, parameter 0 slowed to half rate via
``sample_frequency``, temperature-1 postprocessing.  The historical
off/on + temperature-ladder comparison this demo was reduced from is
documented in ``doc/examples.md`` (Example 6).

Usage:
    python 07_dnest_enhancements.py [--saves 1500] [--seed 42]
        [--frequency 0.5]

Output (in output_07_dnest_enhancements/):
    summary.json, runplot_T1.png
"""

import json
import os
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')

import jax

# x64 BEFORE the problem is built: with x64 off, jnp.asarray silently
# rounds the whitening matrix to float32 -- a different (still valid)
# problem whose reference logZ is -139.7191 instead of -139.4596.
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.linalg import solve_triangular
import jax.scipy.special as jsp

from jnesty import DNestSampler, plotting

# The frozen G1 reference (task03 diag_evidence.py -> T1c common_g1.py);
# the construction guard below must reproduce it exactly.
G1_TRUTH = -138.55200433888717


# ============================================================================
# Problem Definition (G1, 128-d GMRF; recreated verbatim from T1c)
# ============================================================================
def build_g1():
    """Return (loglikelihood, prior_transform, truth) for G1.

    z ~ N(0, Q^-1) via GMRF whitening of iid standard normals;
    Q is the (4, 8, 4) second-difference precision with 0.3/0.3/0.3
    smoothness, ridge 1e-2, rescaled so the mean marginal prior variance
    is sigma_z^2 = 1.
    """
    shape = (4, 8, 4)
    lam_r, lam_phi, lam_eta = 0.3, 0.3, 0.3
    n_r, n_phi, n_eta = shape
    n_cells = n_r * n_phi * n_eta
    s_r = lam_r * n_r ** 2 / np.sqrt(max(1, n_r - 2))
    s_phi = lam_phi * n_phi ** 2 / np.sqrt(max(1, n_phi))
    s_eta = lam_eta * n_eta ** 2 / np.sqrt(max(1, n_eta - 2))
    n_rows = ((n_r - 2) * n_phi * n_eta + (n_eta - 2) * n_r * n_phi
              + n_phi * n_r * n_eta)
    D = np.zeros((n_rows, n_cells))
    cell = lambda i, p, e: (i * n_phi + p) * n_eta + e  # noqa: E731
    row = 0
    for i in range(1, n_r - 1):          # radial interior triples
        for p in range(n_phi):
            for e in range(n_eta):
                D[row, cell(i - 1, p, e)] += s_r
                D[row, cell(i, p, e)] += -2.0 * s_r
                D[row, cell(i + 1, p, e)] += s_r
                row += 1
    for e in range(1, n_eta - 1):        # elevation interior triples
        for i in range(n_r):
            for p in range(n_phi):
                D[row, cell(i, p, e - 1)] += s_eta
                D[row, cell(i, p, e)] += -2.0 * s_eta
                D[row, cell(i, p, e + 1)] += s_eta
                row += 1
    for k in range(n_phi):               # azimuth (periodic) triples
        for i in range(n_r):
            for e in range(n_eta):
                D[row, cell(i, (k - 1) % n_phi, e)] += s_phi
                D[row, cell(i, k, e)] += -2.0 * s_phi
                D[row, cell(i, (k + 1) % n_phi, e)] += s_phi
                row += 1
    DtD = D.T @ D
    Q0 = DtD + 1e-2 * float(np.mean(np.diag(DtD))) * np.eye(n_cells)
    c = float(np.mean(np.diag(np.linalg.inv(Q0)))) / 1.0 ** 2
    Q = c * Q0
    L = np.linalg.cholesky(Q)
    Lt = jnp.asarray(np.ascontiguousarray(L.T))
    m = 0.7 * np.random.default_rng(5).standard_normal(n_cells)

    def prior_transform(u):
        u = jnp.clip(jnp.asarray(u), 1e-12, 1. - 1e-12)
        return solve_triangular(Lt, jsp.ndtri(u), lower=False)

    def loglikelihood(x):
        r = (jnp.asarray(x) - jnp.asarray(m)) / 0.5
        return (-0.5 * jnp.sum(r ** 2)
                - 0.5 * n_cells * np.log(2. * np.pi * 0.5 ** 2))

    C = np.linalg.inv(Q)
    tot = C + 0.25 * np.eye(n_cells)
    sign, logdet = np.linalg.slogdet(2. * np.pi * tot)
    truth = -0.5 * (logdet + m @ np.linalg.solve(tot, m))
    return loglikelihood, prior_transform, truth


# ============================================================================
# One enhanced run
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saves', type=int, default=1500,
                        help='save budget (default 1500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='run seed (default 42)')
    parser.add_argument('--frequency', type=float, default=0.5,
                        help='proposal weight of parameter 0; '
                             'all others get weight 1.0')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output_07_dnest_enhancements')
    os.makedirs(outdir, exist_ok=True)

    loglike, prior, truth = build_g1()
    assert abs(truth - G1_TRUTH) < 1e-9, (truth, G1_TRUTH)  # guard
    print(f'G1 128-d GMRF: analytic logZ = {truth:.4f}')
    print(f'(construction guard OK; x64 = {jax.config.jax_enable_x64})\n')

    frequency = np.ones(128)
    frequency[0] = args.frequency
    print(f'sample_frequency: parameter 0 weight {args.frequency:g} '
          f'(others 1.0) -> ~{args.frequency:g}x the uniform proposal '
          f'rate\n', flush=True)

    print(f'running one enhanced fit ({args.saves} saves, '
          f'seed {args.seed}) ...', flush=True)
    sampler = DNestSampler(
        loglike, prior, ndim=128, num_lanes=8, num_particles=1,
        thread_steps_factor=1.0, new_level_interval_factor=2.0,
        save_interval_factor=2.0, seed=args.seed,
        gradient_scales=True, gradient_interval=100,
        sample_frequency=frequency)
    t0 = time.perf_counter()
    logz = sampler.run(max_num_saves=args.saves)
    wall = time.perf_counter() - t0

    mult = np.asarray(sampler.gradient_scale_multipliers)
    att = np.asarray(sampler.coordinate_attempts)
    ratio = att[0] / att[1:].mean()
    record = dict(
        logz=float(logz), logz_err=float(logz - G1_TRUTH),
        n_eff=float(sampler.n_eff),
        acceptance=float(sampler.acceptance_rate),
        n_like=int(sampler.n_like), n_saves=int(sampler.n_saves),
        n_levels=int(sampler.n_levels), n_steps=int(sampler.n_steps),
        n_gradient_events=int(sampler.n_gradient_events),
        multiplier_min=float(mult.min()),
        multiplier_max=float(mult.max()),
        attempts_param0=int(att[0]),
        attempts_mean_others=float(att[1:].mean()),
        attempts_ratio=float(ratio),
        wall_s=wall)

    print('\n=== Enhanced run record ===')
    print(f"logZ {record['logz']:.4f} (err {record['logz_err']:+.2f})  "
          f"n_eff {record['n_eff']:.1f}  acc {record['acceptance']:.3f}")
    print(f"n_like {record['n_like']}  n_levels {record['n_levels']}  "
          f"n_gradient_events {record['n_gradient_events']}  "
          f"multipliers [{record['multiplier_min']:.3g}, "
          f"{record['multiplier_max']:.3g}]  "
          f"wall {record['wall_s']:.1f} s")
    print(f"parameter 0 attempted {record['attempts_param0']} times vs "
          f"mean {record['attempts_mean_others']:.1f} of the others "
          f"(ratio {record['attempts_ratio']:.2f})")

    print('\n=== Temperature T = 1 postprocessing '
          '(results_at_temperature) ===')
    res = sampler.results_at_temperature(1)
    assert res['logz'] == logz  # T=1 view is bit-identical to run()'s logz
    print(f"logZ {res['logz']:.4f}  H {res['information']:.4f}  "
          f"n_eff {res['n_eff']:.1f}")
    temperature_1 = dict(logz=float(res['logz']),
                         information=float(res['information']),
                         n_eff=float(res['n_eff']))
    fig, _ = plotting.runplot(res)
    fig.suptitle('G1 128-d at temperature T = 1', y=1.02)
    fig.savefig(os.path.join(outdir, 'runplot_T1.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(outdir, 'summary.json'), 'w') as fh:
        json.dump(dict(problem='G1 128-d GMRF (T1c Phase-1)',
                       truth=truth, saves=args.saves, seed=args.seed,
                       frequency=args.frequency, run=record,
                       temperature_1=temperature_1),
                  fh, indent=1)
    print(f'\nwrote {outdir}/summary.json, runplot_T1.png')


if __name__ == '__main__':
    main()
