#!/usr/bin/env python
"""
Demo 7: DNest Enhancements (gradient-informed proposals + temperature)

The 128-dimensional GMRF problem "G1" of the T1c Phase-1 evidence study
(grid shape (4, 8, 4) second-difference precision, sigma_z = 1, likelihood
N(z; m, 0.5^2 I) with m = 0.7 * default_rng(5), analytic logZ).  The
construction is recreated inline (verified bit-identical to the blr_mc
``gmrf_precision`` output; truth guard below).

Two arms at the exact T1c dfnest configuration (8 lanes x 1 particle,
thread_steps_factor 1.0, new/save-level interval factors 2.0, 1500 saves,
seed 42):

* ``off``  -- gradient_scales=False (the default; reproduces the T1c
              dfnest reference logZ = -139.4596, err = -0.91
              bit-exactly on any machine)
* ``on``   -- gradient_scales=True, gradient_interval=100.  The task_007
              study measured err +0.60 / n_eff 132.8 -> 140.2; this
              demo's CPU run lands at err +0.10 / n_eff 204.5.  The
              gradient arm's exact trajectory depends on the
              compilation environment (the committed test asserts the
              |err| <= 2 evidence band, not a digest), but its
              accounting is exact: n_like grows by precisely
              n_gradient_events (30720 here) and the multipliers stay
              inside [1/100, 100].

then a temperature ladder (T = 1, 2, 5, 10) through
``results_at_temperature`` on the gradient-scales arm, one runplot per
rung: the CDNest convention (logl / T for evidence and weights, level
placement untempered) flattens the posterior toward the prior as T
grows (logZ rises toward the prior total mass, H shrinks).

JAX float64 (x64) is enabled at import BEFORE the problem constants are
built, and the demo defaults to CPU (JAX_PLATFORMS=cpu, overridable):
those are the reference-number conventions of the study.

Usage:
    python 07_dnest_enhancements.py [--saves 1500] [--seed 42]
        [--gradient-interval 100] [--temperatures 1,2,5,10]

Output (in output_07_dnest_enhancements/):
    summary.json, comparison.png, runplot_T*.png
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
# Arms
# ============================================================================
def run_arm(loglike, prior, saves, seed, gradient_scales,
            gradient_interval):
    """One G1 arm at the T1c dfnest configuration."""
    sampler = DNestSampler(
        loglike, prior, ndim=128, num_lanes=8, num_particles=1,
        thread_steps_factor=1.0, new_level_interval_factor=2.0,
        save_interval_factor=2.0, seed=seed,
        gradient_scales=gradient_scales,
        gradient_interval=gradient_interval)
    t0 = time.perf_counter()
    logz = sampler.run(max_num_saves=saves)
    wall = time.perf_counter() - t0

    record = dict(gradient_scales=bool(gradient_scales),
                  logz=float(logz), logz_err=float(logz - G1_TRUTH),
                  n_eff=float(sampler.n_eff),
                  acceptance=float(sampler.acceptance_rate),
                  n_like=int(sampler.n_like), n_saves=int(sampler.n_saves),
                  n_levels=int(sampler.n_levels),
                  n_steps=int(sampler.n_steps), wall_s=wall)
    if gradient_scales:
        mult = np.asarray(sampler.gradient_scale_multipliers)
        record.update(n_gradient_events=int(sampler.n_gradient_events),
                      multiplier_min=float(mult.min()),
                      multiplier_max=float(mult.max()))
    return sampler, record


def efficiency_figure(records, outdir):
    """Off-vs-on bars: evidence error, n_eff, acceptance, wall."""
    arms = ['off', 'on']
    colors = ['#BBBBBB', '#009988']
    panels = [('logz_err', r'$\log Z - \log Z_{\rm truth}$',
               'Evidence error (grey band: |err| <= 2)', '{:+.2f}'),
              ('n_eff', 'n_eff',
               'Entropy effective sample size', '{:.1f}'),
              ('acceptance', 'acceptance',
               'MCMC acceptance rate', '{:.3f}'),
              ('wall_s', 'wall (s)', 'Wall clock', '{:.0f}')]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, (key, ylabel, title, fmt) in zip(axes, panels):
        vals = [r[key] for r in records]
        ax.bar(range(2), vals, 0.55, color=colors)
        span = max(abs(v) for v in vals) or 1.0
        for i, v in enumerate(vals):
            ax.text(i, v + (0.06 * span if v >= 0 else -0.12 * span),
                    fmt.format(v), ha='center', fontsize=9)
        if key == 'logz_err':
            ax.axhspan(-2, 2, color='0.9', zorder=0)
            ax.axhline(0, color='k', lw=0.8)
        if key == 'acceptance':
            ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(2))
        ax.set_xticklabels([f'gradient_scales={a}' for a in arms],
                           fontsize=9)
    fig.suptitle('G1 128-d GMRF: gradient-informed proposal scales '
                 '(same seed and save budget)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'comparison.png'), dpi=150)
    plt.close(fig)


def temperature_ladder(sampler, temperatures, outdir):
    """runplot at each T through results_at_temperature (per-T cached)."""
    rows = []
    for T in temperatures:
        res = sampler.results_at_temperature(T)
        fig, _ = plotting.runplot(res)
        fig.suptitle(f'G1 128-d at temperature T = {T:g}', y=1.02)
        fig.savefig(os.path.join(outdir, f'runplot_T{T:g}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        rows.append(dict(T=float(T), logz=float(res['logz']),
                         information=float(res['information']),
                         n_eff=float(res['n_eff'])))
        print(f"  T = {T:<5g} logZ {res['logz']:>10.4f}   H "
              f"{res['information']:>7.4f}   n_eff {res['n_eff']:>6.1f}",
              flush=True)
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saves', type=int, default=1500,
                        help='save budget per arm (default 1500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='run seed (default 42)')
    parser.add_argument('--gradient-interval', type=int, default=100,
                        help='MCMC steps between gradient events')
    parser.add_argument('--temperatures', type=str, default='1,2,5,10',
                        help='comma-separated temperature ladder')
    args = parser.parse_args()
    temperatures = [float(t) for t in args.temperatures.split(',')]

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output_07_dnest_enhancements')
    os.makedirs(outdir, exist_ok=True)

    loglike, prior, truth = build_g1()
    assert abs(truth - G1_TRUTH) < 1e-9, (truth, G1_TRUTH)  # guard
    print(f'G1 128-d GMRF: analytic logZ = {truth:.4f}')
    print(f'(construction guard OK; x64 = {jax.config.jax_enable_x64})\n')

    records = []
    samplers = {}
    for gradient_scales in (False, True):
        label = 'on' if gradient_scales else 'off'
        print(f'running arm gradient_scales={label} '
              f'({args.saves} saves, seed {args.seed}) ...', flush=True)
        sampler, record = run_arm(
            loglike, prior, args.saves, args.seed, gradient_scales,
            args.gradient_interval)
        samplers[label] = sampler
        records.append(record)
        extra = ''
        if gradient_scales:
            extra = (f"  events {record['n_gradient_events']}, "
                     f"multipliers [{record['multiplier_min']:.3g}, "
                     f"{record['multiplier_max']:.3g}]")
        print(f"  logZ {record['logz']:.4f} (err "
              f"{record['logz_err']:+.2f})  n_eff {record['n_eff']:.1f}  "
              f"acc {record['acceptance']:.3f}  "
              f"n_like {record['n_like']}{extra}\n", flush=True)

    print('=== Efficiency: gradient_scales off vs on '
          '(same seed, same budget) ===')
    header = (f"{'arm':<5}{'logZ':>11}{'err':>7}{'n_eff':>8}"
              f"{'acc':>7}{'n_like':>10}{'levels':>8}{'wall_s':>9}")
    print(header)
    for record in records:
        print(f"{'on' if record['gradient_scales'] else 'off':<5}"
              f"{record['logz']:>11.4f}{record['logz_err']:>+7.2f}"
              f"{record['n_eff']:>8.1f}{record['acceptance']:>7.3f}"
              f"{record['n_like']:>10d}{record['n_levels']:>8d}"
              f"{record['wall_s']:>9.1f}")

    efficiency_figure(records, outdir)

    print('\n=== Temperature ladder (gradient-scales arm; '
          'results_at_temperature, per-T cached) ===')
    ladder = temperature_ladder(samplers['on'], temperatures, outdir)

    print('\n=== Summary ===')
    print(f"truth logZ = {truth:.4f}")
    for record in records:
        label = 'on' if record['gradient_scales'] else 'off'
        print(f"gradient_scales {label}: logZ {record['logz']:.4f} "
              f"(err {record['logz_err']:+.2f}), n_eff "
              f"{record['n_eff']:.1f}, wall {record['wall_s']:.0f} s")
    print(f"wrote {outdir}/summary.json, comparison.png, "
          f"runplot_T*.png ({len(ladder)} rungs)")

    with open(os.path.join(outdir, 'summary.json'), 'w') as fh:
        json.dump(dict(problem='G1 128-d GMRF (T1c Phase-1)',
                       truth=truth, saves=args.saves, seed=args.seed,
                       gradient_interval=args.gradient_interval,
                       arms=records, temperature_ladder=ladder),
                  fh, indent=1)


if __name__ == '__main__':
    main()
