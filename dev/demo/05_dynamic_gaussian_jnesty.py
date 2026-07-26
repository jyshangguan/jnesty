#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling — JNesty vs dynesty quantitative comparison

Runs the same 3-D correlated Gaussian likelihood with both JNesty and
dynesty (same nlive, same bound, same dlogz, same maxbatch) and produces
overlaid dyplot.runplot figures so the two implementations can be
compared side-by-side (static case + dynamic 80/20 case).

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 500] [--maxbatch 4]

Output:
    Creates output_05_jnesty/ with:
    - summary.json                    : quantitative comparison table
    - runplot_static_jnesty_vs_dynesty.png  : overlaid static runplot
    - runplot_dynamic_jnesty_vs_dynesty.png : overlaid dynamic runplot
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynesty import plotting as dyplot
from dynesty import NestedSampler as DynestyNestedSampler
from dynesty import DynamicNestedSampler as DynestyDynamicNestedSampler
from dynesty.utils import get_neff_from_logwt

from jnesty import NestedSampler as JNestyNestedSampler
from jnesty import DynamicNestedSampler as JNestyDynamicNestedSampler
from jnesty.dynamic import kish_ess as jnesty_kish_ess  # noqa: F401

# ============================================================================
# Problem: 3-D correlated multivariate normal (matching dynesty docs)
# ============================================================================

NDIM = 3

C_np = np.identity(NDIM)
C_np[C_np == 0] = 0.95
Cinv_np = np.linalg.inv(C_np)
lnorm_np = -0.5 * (np.log(2 * np.pi) * NDIM + np.log(np.linalg.det(C_np)))

# JAX-compatible version for JNesty
import jax.numpy as jnp
_Cinv = jnp.asarray(Cinv_np)
_lnorm = float(lnorm_np)


def loglikelihood_jnesty(x):
    """JAX-compatible 3-D correlated Gaussian."""
    return -0.5 * jnp.dot(x, jnp.dot(_Cinv, x)) + _lnorm


def loglikelihood_dynesty(x):
    """Numpy 3-D correlated Gaussian (for dynesty)."""
    return -0.5 * np.dot(x, np.dot(Cinv_np, x)) + lnorm_np


def prior_transform(u):
    return 20.0 * u - 10.0


# ---- helpers ----------------------------------------------------------------

def run_dynesty_static(nlive, dlogz, maxiter):
    """Run dynesty static."""
    print("\n  dynesty static ...")
    t0 = time.time()
    s = DynestyNestedSampler(loglikelihood_dynesty, prior_transform,
                              ndim=NDIM, nlive=nlive, bound="single")
    s.run_nested(dlogz=dlogz, maxiter=maxiter, print_progress=False)
    rt = time.time() - t0
    res = s.results
    logz = float(res.logz[-1])
    logzerr = float(res.logzerr[-1])
    ess = get_neff_from_logwt(res.logwt)
    print(f"    logZ={logz:.4f} +/- {logzerr:.4f}, ESS={ess:.0f}, "
          f"niter={res.niter}, ncalls={res.ncall}, rt={rt:.1f}s")
    return {
        "sampler": "dynesty", "mode": "static",
        "logZ": logz, "logZ_err": logzerr,
        "ESS": float(ess), "niter": res.niter,
        "ncall": int(res.ncall) if not hasattr(res.ncall, '__len__')
                 else int(np.sum(res.ncall)),
        "runtime": rt, "results": res,
    }


def run_jnesty_static(nlive, dlogz, maxiter):
    """Run JNesty static."""
    print("\n  jnesty static ...")
    t0 = time.time()
    s = JNestyNestedSampler(loglikelihood_jnesty, prior_transform,
                             ndim=NDIM, nlive=nlive, bound="single",
                             verbose=False)
    s.run_nested(max_iterations=maxiter, delta_logZ_threshold=dlogz,
                  print_progress=False)
    rt = time.time() - t0
    r = s.results
    logz = float(r["logz"])
    logzerr = float(r["logzerr"])
    logwt = np.asarray(r["logwt"])
    ess = get_neff_from_logwt(logwt)
    print(f"    logZ={logz:.4f} +/- {logzerr:.4f}, ESS={ess:.0f}, "
          f"niter={r['niter']}, rt={rt:.1f}s")
    return {
        "sampler": "jnesty", "mode": "static",
        "logZ": logz, "logZ_err": logzerr,
        "ESS": float(ess), "niter": r["niter"],
        "ncall": None,
        "runtime": rt, "results": s.to_dynesty_results(),
    }


def run_dynesty_dynamic(nlive, maxbatch, n_effective):
    """Run dynesty dynamic (default 80/20)."""
    print("\n  dynesty dynamic (80/20) ...")
    t0 = time.time()
    s = DynestyDynamicNestedSampler(loglikelihood_dynesty, prior_transform,
                                     ndim=NDIM, nlive=nlive, bound="single")
    s.run_nested(nlive_init=nlive, nlive_batch=nlive,
                  maxbatch=maxbatch,
                  n_effective=n_effective,
                  print_progress=False)
    rt = time.time() - t0
    res = s.results
    logz = float(res.logz[-1])
    logzerr = float(res.logzerr[-1])
    ess = get_neff_from_logwt(res.logwt)
    ncall = int(res.ncall) if not hasattr(res.ncall, '__len__')             else int(np.sum(res.ncall))
    print(f"    logZ={logz:.4f} +/- {logzerr:.4f}, ESS={ess:.0f}, "
          f"niter={res.niter}, ncalls={ncall}, rt={rt:.1f}s")
    return {
        "sampler": "dynesty", "mode": "dynamic_80_20",
        "logZ": logz, "logZ_err": logzerr,
        "ESS": float(ess), "niter": res.niter,
        "ncall": ncall,
        "runtime": rt, "results": res,
    }


def run_jnesty_dynamic(nlive, maxbatch, n_effective):
    """Run JNesty dynamic (80/20)."""
    print("\n  jnesty dynamic (80/20) ...")
    t0 = time.time()
    s = JNestyDynamicNestedSampler(loglikelihood_jnesty, prior_transform,
                                    ndim=NDIM, nlive=nlive, bound="single")
    s.run_nested(nlive_init=nlive, nlive_batch=nlive,
                  maxbatch=maxbatch,
                  n_effective=n_effective,
                  pfrac=0.8,
                  use_stop=True,
                  print_progress=False, seed=0)
    rt = time.time() - t0
    r = s.results
    logz_traj = np.asarray(r["logz"])
    logz = float(logz_traj[-1])
    logwt = np.asarray(r["logwt"])
    info = r.get("information", r.get("h"))
    if info is not None:
        info = float(np.asarray(info).flat[-1] if np.asarray(info).ndim
                      else np.asarray(info))
        logzerr = float(np.sqrt(max(abs(info), 1e-30) / nlive))
    else:
        logzerr = float('nan')
    ess = get_neff_from_logwt(logwt)
    nbatches = len(s.batch_nlive_log) - 1
    print(f"    logZ={logz:.4f} +/- {logzerr:.4f}, ESS={ess:.0f}, "
          f"niter={len(logwt)}, nbatches={nbatches}, rt={rt:.1f}s")
    return {
        "sampler": "jnesty", "mode": "dynamic_80_20",
        "logZ": logz, "logZ_err": logzerr,
        "ESS": float(ess), "niter": len(logwt),
        "nbatches": nbatches,
        "runtime": rt, "results": s.to_dynesty_results(),
    }


# ---- plots ----------------------------------------------------------------

def plot_overlaid_runplot(res_a, res_b, label_a, label_b, color_a, color_b,
                           lnz_truth, out_path, title):
    """
    Overlay two runs (a and b) on one dyplot.runplot figure.
    """
    fig, axes = dyplot.runplot(res_a, color=color_a,
                                mark_final_live=False,
                                logplot=True)
    # Clear the default labels dyplot adds; re-set our own
    fig, axes = dyplot.runplot(res_b, color=color_b,
                                logplot=True,
                                lnz_truth=lnz_truth,
                                truth_color="orange",
                                fig=(fig, axes))
    # Build manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_a, lw=2, label=label_a),
        Line2D([0], [0], color=color_b, lw=2, label=label_b),
        Line2D([0], [0], color="orange", lw=1, ls="--",
               label=f"truth (lnZ={lnz_truth:.3f})"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8, loc="best")
    fig.suptitle(title, y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--maxbatch", type=int, default=4)
    p.add_argument("--dlogz", type=float, default=0.01)
    p.add_argument("--maxiter", type=int, default=20000)
    args = p.parse_args()

    cfg = {
        "nlive": args.nlive, "maxbatch": args.maxbatch,
        "dlogz": args.dlogz, "maxiter": args.maxiter,
        "ndim": NDIM, "analytic lnZ": round(NDIM * -np.log(20.0), 4),
    }
    print("Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    out = Path(__file__).resolve().parent / "output_05_jnesty"
    out.mkdir(exist_ok=True)
    lnz_truth = NDIM * -np.log(20.0)
    n_effective = max(10000, NDIM * NDIM)

    # --- Static runs (JNesty + dynesty) ---
    print("\n=== Static ===")
    r_static_jnesty = run_jnesty_static(args.nlive, args.dlogz, args.maxiter)
    r_static_dynesty = run_dynesty_static(args.nlive, args.dlogz, args.maxiter)

    # --- Dynamic runs (JNesty + dynesty, both pfrac=0.8) ---
    use_stop_dynesty = False
    print(f"\n=== Dynamic 80/20 (maxbatch={args.maxbatch}) ===")
    r_dyn_jnesty = run_jnesty_dynamic(args.nlive, args.maxbatch, n_effective)
    r_dyn_dynesty = run_dynesty_dynamic(args.nlive, args.maxbatch, n_effective)

    # --- Overlaid runplot: static ---
    print("\nGenerating overlaid static runplot ...")
    plot_overlaid_runplot(
        r_static_dynesty["results"], r_static_jnesty["results"],
        f"dynesty static (ESS={r_static_dynesty['ESS']:.0f})",
        f"jneesty static (ESS={r_static_jnesty['ESS']:.0f})",
        "black", "red",
        lnz_truth,
        out / "runplot_static_jnesty_vs_dynesty.png",
        f"Static NS: dynesty (black) vs jnesty (red)  |  "
        f"nlive={args.nlive}, dlogz={args.dlogz}",
    )

    # --- Overlaid runplot: dynamic ---
    print("Generating overlaid dynamic runplot ...")
    plot_overlaid_runplot(
        r_dyn_dynesty["results"], r_dyn_jnesty["results"],
        f"dynesty dyn 80/20 (ESS={r_dyn_dynesty['ESS']:.0f})",
        f"jneesty dyn 80/20 (ESS={r_dyn_jnesty['ESS']:.0f})",
        "black", "red",
        lnz_truth,
        out / "runplot_dynamic_jnesty_vs_dynesty.png",
        f"Dynamic NS 80/20: dynesty (black) vs jnesty (red)  |  "
        f"nlive={args.nlive}, maxbatch={args.maxbatch}",
    )

    # --- Quantitative comparison table ---
    print("\n" + "=" * 80)
    print("QUANTITATIVE COMPARISON")
    print("=" * 80)
    header = (f"{'Run':<30s} {'logZ':>8s} {'+/- err':>8s} "
              f"{'Δ from truth':>12s} {'ESS':>8s} {'time':>8s}")
    print(header)
    print("-" * len(header))
    entries = [r_static_dynesty, r_static_jnesty,
               r_dyn_dynesty, r_dyn_jnesty]
    for e in entries:
        delta = e["logZ"] - lnz_truth
        label = f"{e['sampler']} {e['mode']}"
        print(f"{label:<30s} {e['logZ']:8.4f} {e['logZ_err']:8.4f} "
              f"{delta:12.4f} {e['ESS']:8.0f} {e['runtime']:7.1f}s")

    print(f"\n  truth lnZ = {lnz_truth:.4f}")

    # --- Pass/fail gates ---
    print("\n--- Statistical consistency gates ---")
    all_pass = True
    gates = []

    # Gate 1: static logZ agreement
    dz_static = abs(r_static_jnesty["logZ"] - r_static_dynesty["logZ"])
    err_static = np.sqrt(r_static_jnesty["logZ_err"]**2
                         + r_static_dynesty["logZ_err"]**2)
    g1 = dz_static < 3 * err_static
    all_pass &= g1
    gates.append(("static JNesty vs dynesty logZ agree (3σ)",
                  g1, f"Δ={dz_static:.4f}, 3σ={3*err_static:.4f}"))

    # Gate 2: dynamic logZ agreement
    dz_dyn = abs(r_dyn_jnesty["logZ"] - r_dyn_dynesty["logZ"])
    err_dyn = np.sqrt(r_dyn_jnesty["logZ_err"]**2
                      + r_dyn_dynesty["logZ_err"]**2)
    g2 = dz_dyn < 3 * err_dyn
    all_pass &= g2
    gates.append(("dynamic JNesty vs dynesty logZ agree (3σ)",
                  g2, f"Δ={dz_dyn:.4f}, 3σ={3*err_dyn:.4f}"))

    # Gate 3: ESS ratio
    if r_static_dynesty["ESS"] > 0:
        ess_ratio = r_static_jnesty["ESS"] / r_static_dynesty["ESS"]
        g3 = 0.3 < ess_ratio < 3.0
        all_pass &= g3
        gates.append(("static ESS ratio (JNesty/dynesty) in [0.3, 3.0]",
                      g3, f"ratio={ess_ratio:.2f}"))

    # Gate 4: static |logZ - truth|
    g4 = abs(r_static_jnesty["logZ"] - lnz_truth) < 5 * r_static_jnesty["logZ_err"]
    all_pass &= g4
    gates.append(("static JNesty logZ within 5σ of truth",
                  g4, f"Δ={abs(r_static_jnesty['logZ']-lnz_truth):.4f}, "
                  f"5σ={5*r_static_jnesty['logZ_err']:.4f}"))

    for name, ok, detail in gates:
        print(f"  [{('PASS' if ok else 'FAIL'):>4s}] {name}")
        if not ok or True:
            print(f"         {detail}")

    print(f"\n=== VERDICT: {'PASS' if all_pass else 'FAIL'} ===")

    # --- Save summary ---
    summary = {
        "config": cfg,
        "lnz_truth": lnz_truth,
        "runs": entries,
        "gates": [{"name": n, "pass": ok, "detail": d} for n, ok, d in gates],
        "verdict": "PASS" if all_pass else "FAIL",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nOutput: {out}/")
    print("  - runplot_static_jnesty_vs_dynesty.png")
    print("  - runplot_dynamic_jnesty_vs_dynesty.png")
    print("  - summary.json")


if __name__ == "__main__":
    main()
