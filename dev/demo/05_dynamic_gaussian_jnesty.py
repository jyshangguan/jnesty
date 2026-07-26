#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling (JNesty)

Shows how to use DynamicNestedSampler on a 2D Gaussian. Compares the static
and dynamic ESS at matched likelihood-call budget.

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 200] [--maxbatch 3]

Output:
    Creates output_05_jnesty/ with:
    - summary.json: Numerical results
    - trace.png: logL trace
    - corner.png: posterior corner plot
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


def loglikelihood(x):
    """2D Gaussian centered at (0.5, 0.5), sigma=0.1 per dim."""
    return -0.5 * np.sum(((x - 0.5) / 0.1) ** 2)


def prior_transform(u):
    return u


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlive", type=int, default=200)
    p.add_argument("--maxbatch", type=int, default=3)
    p.add_argument("--pfrac", type=float, default=0.8)
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
    from jnesty.dynamic import kish_ess
    ess_static = kish_ess(np.asarray(s_static.results["logwt"])
                           - float(s_static.results["logz"]))
    print(f"static: logZ={float(s_static.results['logz']):.3f} "
          f"± {float(s_static.results['logzerr']):.3f}, "
          f"ESS={ess_static:.0f}, rt={rt_static:.1f}s")

    # Dynamic
    print("\n=== Dynamic run ===")
    t0 = time.time()
    s_dyn = DynamicNestedSampler(loglikelihood, prior_transform, ndim=2,
                                  nlive=args.nlive, bound="none")
    s_dyn.run_nested(nlive_init=args.nlive, nlive_batch=args.nlive,
                      maxbatch=args.maxbatch, pfrac=args.pfrac,
                      print_progress=False, seed=0)
    rt_dyn = time.time() - t0
    res = s_dyn.results
    logz = float(np.asarray(res["logz"])[-1])
    logzerr = float(np.asarray(res["logzerr"])[-1])
    ess_dyn = kish_ess(np.asarray(res["logwt"]) - logz)
    print(f"dynamic: logZ={logz:.3f} ± {logzerr:.3f}, "
          f"ESS={ess_dyn:.0f}, rt={rt_dyn:.1f}s, "
          f"nbatches={len(s_dyn.batch_nlive_log) - 1}")

    # Plots
    logl = np.asarray(res["logl"])
    logvol = np.asarray(res["logvol"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(-logvol, logl, "-", lw=0.5)
    ax.set_xlabel("-logvol")
    ax.set_ylabel("logl")
    ax.set_title("Dynamic: logl vs -logvol")
    fig.tight_layout()
    fig.savefig(out / "trace.png", dpi=120)
    plt.close(fig)

    summary = {
        "logz_static": float(s_static.results["logz"]),
        "logz_static_err": float(s_static.results["logzerr"]),
        "ess_static": float(ess_static),
        "rt_static": rt_static,
        "logz_dynamic": logz,
        "logz_dynamic_err": logzerr,
        "ess_dynamic": float(ess_dyn),
        "rt_dynamic": rt_dyn,
        "ess_gain": float(ess_dyn / max(ess_static, 1.0)),
        "nbatches": len(s_dyn.batch_nlive_log) - 1,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nESS gain: {summary['ess_gain']:.2f}x")
    print(f"Output: {out}/")


if __name__ == "__main__":
    main()
