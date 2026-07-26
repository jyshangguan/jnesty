#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling (JNesty)

Runs DynamicNestedSampler on a 2D Gaussian at three pfrac settings
(0.8 = 80/20 posterior/evidence, 1.0 = 100% posterior, 0.0 = 100%
evidence) and produces side-by-side trace plots to illustrate how the
weight allocation sculpts sampling density along the logl axis.

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 200] [--maxbatch 3]

Output:
    Creates output_05_jnesty/ with:
    - summary.json: Numerical results for static + 3 dynamic runs
    - trace_trio.png: 3-panel logl vs -logvol trace (one panel per pfrac)
    - trace_overlay.png: All three traces overlaid for direct comparison
    - weight_components.png: pweight/zweight/weight = pfrac*p + (1-pfrac)*z
                              for the three pfrac values
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jnesty import DynamicNestedSampler, NestedSampler
from jnesty.dynamic import kish_ess, weight_function


def loglikelihood(x):
    """2D Gaussian centered at (0.5, 0.5), sigma=0.1 per dim."""
    return -0.5 * np.sum(((x - 0.5) / 0.1) ** 2)


def prior_transform(u):
    return u


def run_dynamic(nlive, maxbatch, pfrac, seed=0):
    """Run one dynamic configuration. Returns (results_dict, runtime, nbatches)."""
    t0 = time.time()
    s = DynamicNestedSampler(loglikelihood, prior_transform, ndim=2,
                              nlive=nlive, bound="none")
    s.run_nested(nlive_init=nlive, nlive_batch=nlive,
                  maxbatch=maxbatch, pfrac=pfrac,
                  use_stop=False,  # force exactly maxbatch batches for fair visual comparison
                  print_progress=False, seed=seed)
    rt = time.time() - t0
    res = s.results
    logz = float(np.asarray(res["logz"])[-1])
    logzerr = float(np.asarray(res["logzerr"])[-1])
    ess = kish_ess(np.asarray(res["logwt"]) - logz)
    nbatches = len(s.batch_nlive_log) - 1
    summary = {
        "pfrac": pfrac,
        "logZ": logz,
        "logZ_err": logzerr,
        "ESS": float(ess),
        "niter": len(np.asarray(res["logl"])),
        "rt": rt,
        "nbatches": nbatches,
    }
    return res, summary


def plot_trace_trio(runs, out_path):
    """3-panel trace: one panel per pfrac setting."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    titles = {
        0.8: r"pfrac=0.8  (80% posterior / 20% evidence)",
        1.0: r"pfrac=1.0  (100% posterior)",
        0.0: r"pfrac=0.0  (100% evidence)",
    }
    colors = {0.8: "tab:blue", 1.0: "tab:green", 0.0: "tab:red"}
    for ax, pfrac in zip(axes, [0.8, 1.0, 0.0]):
        res = runs[pfrac][0]
        logl = np.asarray(res["logl"])
        logvol = np.asarray(res["logvol"])
        n = len(logl)
        ax.plot(-logvol, logl, ".", ms=2, color=colors[pfrac], alpha=0.6)
        # Mark the batch logl bounds if present
        bounds = res.get("batch_bounds", None)
        if bounds and len(bounds) > 1:
            # bounds[0] is (-inf, inf) for the base; subsequent are batch bounds
            for (lmin, lmax) in bounds[1:]:
                if np.isfinite(lmin):
                    ax.axhline(lmin, color="k", ls="--", alpha=0.3, lw=0.7)
                if np.isfinite(lmax):
                    ax.axhline(lmax, color="k", ls=":", alpha=0.3, lw=0.7)
        ax.set_xlabel("-logvol")
        ax.set_title(titles[pfrac] + f'\n  (n={n}, ESS={runs[pfrac][1]["ESS"]:.0f})',
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("logl")
    fig.suptitle("Dynamic NS trace: sampling density depends on pfrac", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_trace_overlay(runs, out_path):
    """All three traces overlaid, colour-coded by pfrac."""
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = {0.8: "pfrac=0.8 (80/20)", 1.0: "pfrac=1.0 (posterior)",
              0.0: "pfrac=0.0 (evidence)"}
    colors = {0.8: "tab:blue", 1.0: "tab:green", 0.0: "tab:red"}
    for pfrac in [0.0, 0.8, 1.0]:
        res = runs[pfrac][0]
        logl = np.asarray(res["logl"])
        logvol = np.asarray(res["logvol"])
        ax.plot(-logvol, logl, ".", ms=2.5, alpha=0.55,
                color=colors[pfrac], label=labels[pfrac])
    ax.set_xlabel("-logvol")
    ax.set_ylabel("logl")
    ax.set_title("Dynamic NS: trace overlay (3 pfrac settings)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_weight_components(runs, out_path):
    """3x3 panel: rows = pfrac settings; cols = pweight/zweight/combined."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    pfracs = [0.8, 1.0, 0.0]
    for row, pfrac in enumerate(pfracs):
        res = runs[pfrac][0]
        # rebuild a dict in the form weight_function expects
        r = {
            "logl": np.asarray(res["logl"]),
            "logvol": np.asarray(res["logvol"]),
            "logwt": np.asarray(res["logwt"]),
            "logz": np.asarray(res["logz"]),
            "samples_n": np.asarray(res["samples_n"]),
        }
        try:
            bounds, (p, z, w) = weight_function(
                r, args={"pfrac": pfrac, "maxfrac": 0.8, "pad": 1},
                return_weights=True)
        except Exception:
            continue
        logl = r["logl"]
        for col, (weight, name) in enumerate(
                [(p, "pweight (posterior)"),
                 (z, "zweight (evidence)"),
                 (w, f"weight = {pfrac:.1f}*p + {1-pfrac:.1f}*z")]):
            ax = axes[row, col]
            ax.semilogy(logl, weight, ".", ms=2,
                         color=["tab:purple", "tab:orange", "tab:gray"][col],
                         alpha=0.6)
            if col == 0:
                ax.set_ylabel(f"pfrac={pfrac}\nweight")
            if row == 0:
                ax.set_title(name)
            if row == 2:
                ax.set_xlabel("logl")
            ax.grid(alpha=0.3)
            # mark bounds on the combined column
            if col == 2 and np.isfinite(bounds[0]):
                ax.axvline(bounds[0], color="r", ls="--", alpha=0.5)
            if col == 2 and np.isfinite(bounds[1]):
                ax.axvline(bounds[1], color="r", ls="--", alpha=0.5)
    fig.suptitle("Dynamic NS weight components per pfrac", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlive", type=int, default=200)
    p.add_argument("--maxbatch", type=int, default=3)
    args = p.parse_args()

    out = Path(__file__).resolve().parent / "output_05_jnesty"
    out.mkdir(exist_ok=True)

    # Static baseline
    print("=== Static run ===")
    t0 = time.time()
    s_static = NestedSampler(loglikelihood, prior_transform, ndim=2,
                              nlive=args.nlive, bound="none", verbose=False)
    s_static.run_nested(print_progress=False)
    rt_static = time.time() - t0
    ess_static = kish_ess(np.asarray(s_static.results["logwt"])
                           - float(s_static.results["logz"]))
    print(f"static: logZ={float(s_static.results['logz']):.3f} "
          f"± {float(s_static.results['logzerr']):.3f}, "
          f"ESS={ess_static:.0f}, rt={rt_static:.1f}s")

    # Three dynamic runs at different pfrac
    runs = {}
    summary = {
        "static": {
            "logZ": float(s_static.results["logz"]),
            "logZ_err": float(s_static.results["logzerr"]),
            "ESS": float(ess_static),
            "rt": rt_static,
            "nlive": args.nlive,
        },
        "dynamic": [],
    }
    for pfrac in [0.8, 1.0, 0.0]:
        print(f"\n=== Dynamic run pfrac={pfrac} ===")
        res, dyn_summary = run_dynamic(args.nlive, args.maxbatch, pfrac)
        runs[pfrac] = (res, dyn_summary)
        summary["dynamic"].append(dyn_summary)
        print(f"dynamic pfrac={pfrac}: logZ={dyn_summary['logZ']:.3f} "
              f"± {dyn_summary['logZ_err']:.3f}, ESS={dyn_summary['ESS']:.0f}, "
              f"niter={dyn_summary['niter']}, rt={dyn_summary['rt']:.1f}s, "
              f"nbatches={dyn_summary['nbatches']}")

    # Plots
    plot_trace_trio(runs, out / "trace_trio.png")
    plot_trace_overlay(runs, out / "trace_overlay.png")
    plot_weight_components(runs, out / "weight_components.png")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutput: {out}/")
    print(f"  - trace_trio.png (3-panel per pfrac)")
    print(f"  - trace_overlay.png (single-panel overlay)")
    print(f"  - weight_components.png (pweight/zweight/weight per pfrac)")
    print(f"  - summary.json")
    print(f"\nESS summary:")
    print(f"  static           : {ess_static:.0f}")
    for pfrac in [0.8, 1.0, 0.0]:
        s = runs[pfrac][1]
        print(f"  dynamic pfrac={pfrac}: {s['ESS']:.0f} "
              f"(gain vs static: {s['ESS']/max(ess_static,1):.2f}x)")


if __name__ == "__main__":
    main()
