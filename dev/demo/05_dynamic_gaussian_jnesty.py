#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling — dynesty-docs overlay comparison

Mirrors the canonical example from the dynesty dynamic NS documentation
(https://dynesty.readthedocs.io/en/latest/dynamic.html) so the JNesty
dynamic sampler can be visually compared one-to-one against the standard
dynesty plots.

Uses the same 3-D correlated multivariate Gaussian likelihood and the
same runplot-overlay style (static=black, default-dyn=red,
posterior-dyn=blue, evidence-dyn=limegreen, truth=orange).

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 1000] [--maxbatch 5]

Output:
    Creates output_05_jnesty/ with:
    - runplot_overlay.png: Overlaid dynamic runplot matching dynesty docs
    - runplot_static.png:   Static-only runplot for comparison
    - summary.json:         logZ, ESS, niter, runtime
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynesty import plotting as dyplot

from jnesty import NestedSampler, DynamicNestedSampler
from jnesty.dynamic import kish_ess

# ============================================================================
# Problem: 3-D correlated multivariate normal (matching dynesty docs)
# ============================================================================

NDIM = 3

# Covariance matrix with 0.95 off-diagonal (materialised as JAX arrays)
C_np = np.identity(NDIM)
C_np[C_np == 0] = 0.95
Cinv_np = np.linalg.inv(C_np)
lnorm_np = -0.5 * (np.log(2 * np.pi) * NDIM + np.log(np.linalg.det(C_np)))

# JAX-compatible versions for use inside the JIT-compiled loglikelihood
Cinv = jnp.asarray(Cinv_np)
lnorm = float(lnorm_np)


def loglikelihood(x):
    """3-D correlated multivariate normal (log-likelihood, JAX-compatible)."""
    return -0.5 * jnp.dot(x, jnp.dot(Cinv, x)) + lnorm


def prior_transform(u):
    """Uniform prior [-10, 10] per dimension."""
    return 20.0 * u - 10.0


def run_jnesty_static(nlive, maxiter, dlogz):
    """Run static JNesty alongside the equivalent dynesty setup."""
    print("\n=== Static JNesty run ===")
    t0 = time.time()
    s = NestedSampler(loglikelihood, prior_transform, ndim=NDIM,
                       nlive=nlive, bound="single", verbose=False)
    s.run_nested(max_iterations=maxiter,
                  delta_logZ_threshold=dlogz,
                  print_progress=False)
    rt = time.time() - t0
    r = s.results
    logz = float(r["logz"])
    logzerr = float(r["logzerr"])
    ess = kish_ess(np.asarray(r["logwt"]) - logz)
    print(f"  JNesty static:  logZ={logz:.3f} +/- {logzerr:.3f}, "
          f"ESS={ess:.0f}, niter={r['niter']}, rt={rt:.1f}s")
    return s.to_dynesty_results(), logz, logzerr, ess, r['niter'], rt


def run_jnesty_dynamic(nlive, maxbatch, pfrac, use_stop, label):
    """Run one dynamic JNesty configuration."""
    print(f"\n=== Dynamic JNesty {label} ===")
    t0 = time.time()
    s = DynamicNestedSampler(loglikelihood, prior_transform, ndim=NDIM,
                              nlive=nlive, bound="single")
    s.run_nested(nlive_init=nlive, nlive_batch=nlive,
                  maxbatch=maxbatch, pfrac=pfrac,
                  use_stop=use_stop,
                  print_progress=False, seed=0)
    rt = time.time() - t0
    dres = s.to_dynesty_results()
    logz_arr = np.asarray(s.results["logz"])
    logz = float(logz_arr.flat[-1] if logz_arr.ndim else logz_arr)
    # Use standard NS error estimate: sqrt(information / nlive)
    # (dynesty's compute_integrals logzvar can produce NaN for large combined runs;
    #  the sqrt(H/nlive) approximation is robust and standard.)
    info_val = s.results.get("information", None)
    if info_val is None:
        info_val = s.results.get("h", 1.0)
    info_val = np.asarray(info_val)
    H = float(info_val.flat[-1] if info_val.ndim else info_val)
    logzerr = float(np.sqrt(max(abs(H), 1e-30) / nlive))
    ess = kish_ess(np.asarray(s.results["logwt"]) - logz)
    niter = len(np.asarray(s.results["logl"]))
    nbatches = len(s.batch_nlive_log) - 1
    print(f"  JNesty dyn {label:>20s}: logZ={logz:.3f} +/- {logzerr:.3f}, "
          f"ESS={ess:.0f}, niter={niter}, nb={nbatches}, rt={rt:.1f}s")
    return dres, s, logz, logzerr, ess, niter, nbatches, rt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlive", type=int, default=1000)
    p.add_argument("--maxbatch", type=int, default=5)
    p.add_argument("--dlogz", type=float, default=0.01)
    p.add_argument("--maxiter", type=int, default=20000)
    args = p.parse_args()

    cfgs = {
        "nlive": args.nlive,
        "maxbatch": args.maxbatch,
        "dlogz": args.dlogz,
        "maxiter": args.maxiter,
        "ndim": NDIM,
    }
    print("Configuration:")
    for k, v in cfgs.items():
        print(f"  {k}: {v}")
    print(f"  analytic lnZ: {NDIM * -np.log(2 * 10.):.3f}")

    out = Path(__file__).resolve().parent / "output_05_jnesty"
    out.mkdir(exist_ok=True)

    # Analytic evidence
    lnz_truth = NDIM * -np.log(2 * 10.0)

    # --- Static run ---
    dres_static, logz_s, logzerr_s, ess_s, niter_s, rt_s =         run_jnesty_static(args.nlive, args.maxiter, args.dlogz)

    # --- Dynamic runs (matching dynesty docs: use_stop=False, fixed budget) ---
    # Default:  80/20 posterior/evidence  (red)
    dres_def, s_def, logz_def, logzerr_def, ess_def, niter_def, nb_def, rt_def =         run_jnesty_dynamic(args.nlive, args.maxbatch, 0.8, False, "pfrac=0.8 (default)")
    # Posterior: 100% posterior (blue)
    dres_p, s_p, logz_p, logzerr_p, ess_p, niter_p, nb_p, rt_p =         run_jnesty_dynamic(args.nlive, args.maxbatch, 1.0, False, "pfrac=1.0 (posterior)")
    # Evidence:  100% evidence (limegreen)
    dres_z, s_z, logz_z, logzerr_z, ess_z, niter_z, nb_z, rt_z =         run_jnesty_dynamic(args.nlive, args.maxbatch, 0.0, False, "pfrac=0.0 (evidence)")

    # --- Overlaid runplot (matching dynesty docs exactly) ---
    print("\nBuilding overlaid runplot ...")
    fig, axes = dyplot.runplot(dres_static, color="black",
                                mark_final_live=False,
                                logplot=True)           # static
    fig, axes = dyplot.runplot(dres_def, color="red",
                                logplot=True,
                                fig=(fig, axes))         # default dynamic
    fig, axes = dyplot.runplot(dres_p, color="blue",
                                logplot=True,
                                fig=(fig, axes))         # posterior dynamic
    fig, axes = dyplot.runplot(dres_z, color="limegreen",
                                logplot=True,
                                lnz_truth=lnz_truth,
                                truth_color="orange",
                                fig=(fig, axes))         # evidence dynamic

    # Add legend with ESS info
    legend_labels = [
        f"JNesty static        (ESS={ess_s:.0f})",
        f"JNesty dyn 80/20     (ESS={ess_def:.0f})",
        f"JNesty dyn posterior (ESS={ess_p:.0f})",
        f"JNesty dyn evidence  (ESS={ess_z:.0f})",
    ]
    legend_colors = ["black", "red", "blue", "limegreen"]
    for ax in axes:
        lines = ax.get_lines()
        # keep only the first N lines matching our runs (one per color)
        for line, label, c in zip(
                lines[:4], legend_labels, legend_colors):
            line.set_label(label)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "JNesty Dynamic NS vs Static  |  3-D Correlated Gaussian  |  "
        f"nlive={args.nlive}, maxbatch={args.maxbatch}",
        y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "runplot_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> runplot_overlay.png")

    # Also save the static-only runplot for side-by-side comparison
    fig2, _ = dyplot.runplot(dres_static, color="black",
                              mark_final_live=False, logplot=True,
                              lnz_truth=lnz_truth, truth_color="orange")
    fig2.suptitle("JNesty static (for reference)", y=1.005, fontsize=11)
    fig2.tight_layout()
    fig2.savefig(out / "runplot_static.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  -> runplot_static.png")

    # --- Summary ---
    summary = {
        "config": cfgs,
        "lnz_truth": lnz_truth,
        "static": {
            "logZ": logz_s, "logZ_err": logzerr_s,
            "ESS": ess_s, "niter": niter_s, "rt": rt_s,
        },
        "dynamic_default": {
            "logZ": logz_def, "logZ_err": logzerr_def,
            "ESS": ess_def, "niter": niter_def,
            "nbatches": nb_def, "rt": rt_def, "pfrac": 0.8,
        },
        "dynamic_posterior": {
            "logZ": logz_p, "logZ_err": logzerr_p,
            "ESS": ess_p, "niter": niter_p,
            "nbatches": nb_p, "rt": rt_p, "pfrac": 1.0,
        },
        "dynamic_evidence": {
            "logZ": logz_z, "logZ_err": logzerr_z,
            "ESS": ess_z, "niter": niter_z,
            "nbatches": nb_z, "rt": rt_z, "pfrac": 0.0,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n\nOutput: {out}/")
    print(f"  - runplot_overlay.png  (static + 3 dynamic overlaid)")
    print(f"  - runplot_static.png   (static only, reference)")
    print(f"  - summary.json")
    print(f"\nSummary:")
    print(f"  lnZ truth  = {lnz_truth:.3f}")
    print(f"  static     : logZ={logz_s:.3f} +/- {logzerr_s:.3f}, "
          f"ESS={ess_s:.0f}, niter={niter_s}")
    for name, z, ze, ess, niter, nb in [
            ("dyn 80/20", logz_def, logzerr_def, ess_def, niter_def, nb_def),
            ("dyn post", logz_p, logzerr_p, ess_p, niter_p, nb_p),
            ("dyn evid", logz_z, logzerr_z, ess_z, niter_z, nb_z)]:
        print(f"  {name:>12s}: logZ={z:.3f} +/- {ze:.3f}, "
              f"ESS={ess:.0f}, niter={niter}, nbatches={nb}")


if __name__ == "__main__":
    main()
