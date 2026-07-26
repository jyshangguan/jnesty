#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling (JNesty)

Runs DynamicNestedSampler on a 2D Gaussian at three pfrac settings
(0.8 = 80/20 posterior/evidence, 1.0 = 100% posterior, 0.0 = 100%
evidence) and produces dynesty-style runplots (one per pfrac) so the
sculpting of sampling density along the logvol axis is visible in the
standard NS diagnostic format.

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 200] [--maxbatch 3]

Output:
    Creates output_05_jnesty/ with:
    - summary.json: Numerical results for static + 3 dynamic runs
    - runplot_pfrac08.png: dynesty runplot for pfrac=0.8
    - runplot_pfrac10.png: dynesty runplot for pfrac=1.0
    - runplot_pfrac00.png: dynesty runplot for pfrac=0.0
    - runplot_trio.png: 3-panel side-by-side runplot
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
from jnesty.dynamic import kish_ess
from dynesty import plotting as dyplot


def loglikelihood(x):
    """2D Gaussian centered at (0.5, 0.5), sigma=0.1 per dim."""
    return -0.5 * np.sum(((x - 0.5) / 0.1) ** 2)


def prior_transform(u):
    return u


def run_dynamic(nlive, maxbatch, pfrac, seed=0):
    """Run one dynamic configuration. Returns (sampler, runtime)."""
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
    summary = {
        "pfrac": pfrac,
        "logZ": logz,
        "logZ_err": logzerr,
        "ESS": float(ess),
        "niter": len(np.asarray(res["logl"])),
        "rt": rt,
        "nbatches": len(s.batch_nlive_log) - 1,
    }
    return s, summary


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
          f"+/- {float(s_static.results['logzerr']):.3f}, "
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
        s, dyn_summary = run_dynamic(args.nlive, args.maxbatch, pfrac)
        runs[pfrac] = (s, dyn_summary)
        summary["dynamic"].append(dyn_summary)
        print(f"dynamic pfrac={pfrac}: logZ={dyn_summary['logZ']:.3f} "
              f"+/- {dyn_summary['logZ_err']:.3f}, ESS={dyn_summary['ESS']:.0f}, "
              f"niter={dyn_summary['niter']}, rt={dyn_summary['rt']:.1f}s, "
              f"nbatches={dyn_summary['nbatches']}")

    # --- Runplots via dynesty.plotting.runplot ---
    # Each runplot is a 4-panel figure: (logvol, logl), (logvol, logwt),
    # (logvol, logZ), (logvol, posterior mass). Standard NS diagnostic.
    tag = {0.8: "pfrac08", 1.0: "pfrac10", 0.0: "pfrac00"}
    title = {0.8: "pfrac=0.8 (80% posterior / 20% evidence)",
             1.0: "pfrac=1.0 (100% posterior)",
             0.0: "pfrac=0.0 (100% evidence)"}
    color = {0.8: "tab:blue", 1.0: "tab:green", 0.0: "tab:red"}

    # Single-panel runplot per pfrac
    for pfrac in [0.8, 1.0, 0.0]:
        sampler = runs[pfrac][0]
        dres = sampler.to_dynesty_results()
        fig, axes = dyplot.runplot(dres, color=color[pfrac],
                                     label_kwargs={"fontsize": 10})
        fig.suptitle(title[pfrac] +
                     f"  | logZ={runs[pfrac][1]['logZ']:.2f}"
                     f" +/- {runs[pfrac][1]['logZ_err']:.2f}"
                     f", ESS={runs[pfrac][1]['ESS']:.0f}",
                     y=1.00, fontsize=11)
        fig.tight_layout()
        out_path = out / f"runplot_{tag[pfrac]}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")

    # 3-panel side-by-side runplot
    fig, axes_row = plt.subplots(4, 3, figsize=(15, 10))
    for col, pfrac in enumerate([0.8, 1.0, 0.0]):
        sampler = runs[pfrac][0]
        dres = sampler.to_dynesty_results()
        # Build a single-figure runplot then graft its axes into the column.
        sub_fig, _ = plt.subplots(4, 1, figsize=(5, 10))
        plt.close(sub_fig)  # we just use it as a template layout reference
        # Easier: call dyplot.runplot with a custom fig that has 4x3 subplots
        # But runplot wants a 4x1 grid; instead we re-implement the runplot
        # per column by calling runplot into its own figure and rendering.
        # Simpler approach: render each pfrac's runplot into a separate figure
        # and combine via imshow on the grid.
    plt.close(fig)  # discard the placeholder

    # Simpler approach for the trio: make 3 separate runplots and stack
    # their rendered PNGs into one image via PIL.
    try:
        from PIL import Image
        images = [Image.open(out / f"runplot_{tag[p]}.png") for p in [0.8, 1.0, 0.0]]
        h = max(im.height for im in images)
        total_w = sum(im.width for im in images) + 20 * 2
        canvas = Image.new("RGBA", (total_w, h), (255, 255, 255, 255))
        x_offset = 0
        for im in images:
            canvas.paste(im, (x_offset, 0))
            x_offset += im.width + 20
        canvas.save(out / "runplot_trio.png")
        print("  -> runplot_trio.png")
    except ImportError:
        print("  (PIL not available, skipping runplot_trio.png)")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutput: {out}/")
    print(f"  - runplot_pfrac08.png, runplot_pfrac10.png, runplot_pfrac00.png")
    print(f"  - runplot_trio.png (3-panel side-by-side)")
    print(f"  - summary.json")
    print(f"\nESS summary:")
    print(f"  static           : {ess_static:.0f}")
    for pfrac in [0.8, 1.0, 0.0]:
        s = runs[pfrac][1]
        print(f"  dynamic pfrac={pfrac}: {s['ESS']:.0f} "
              f"(gain vs static: {s['ESS']/max(ess_static,1):.2f}x)")


if __name__ == "__main__":
    main()
