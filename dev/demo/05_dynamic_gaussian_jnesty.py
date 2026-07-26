#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling (JNesty)

Runs DynamicNestedSampler on a 3-D correlated Gaussian (the canonical
dynesty dynamic NS example).  Produces runplot, trace plot, and summary.

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 500] [--maxbatch 4]

Output:
    Creates output_05_jnesty/ with:
    - summary.json: Numerical results
    - runplot.png: dynesty-style 4-panel runplot
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jnesty import DynamicNestedSampler
from jnesty.dynamic import kish_ess
from dynesty import plotting as dyplot

NDIM = 3
C_np = np.identity(NDIM); C_np[C_np == 0] = 0.95
Cinv_np = np.linalg.inv(C_np)
LNORM = -0.5 * (np.log(2 * np.pi) * NDIM + np.log(np.linalg.det(C_np)))
_Cinv = jnp.asarray(Cinv_np); _lnorm = float(LNORM)


def loglikelihood(x):
    return -0.5 * jnp.dot(x, jnp.dot(_Cinv, x)) + _lnorm


def prior_transform(u):
    return 20.0 * u - 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--maxbatch", type=int, default=4)
    ap.add_argument("--pfrac", type=float, default=0.8)
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "output_05_jnesty"
    out.mkdir(exist_ok=True)
    lnz_truth = NDIM * -np.log(20.0)
    nlive, mb = args.nlive, args.maxbatch

    print(f"Dynamic NS (JNesty): 3-D correlated Gaussian")
    print(f"  nlive={nlive}, maxbatch={mb}, pfrac={args.pfrac}")
    print(f"  analytic lnZ = {lnz_truth:.4f}")

    t0 = time.time()
    s = DynamicNestedSampler(loglikelihood, prior_transform, ndim=NDIM,
                              nlive=nlive, bound="single")
    s.run_nested(nlive_init=nlive, nlive_batch=nlive, maxbatch=mb,
                  pfrac=args.pfrac, n_effective=max(10000, NDIM**2),
                  use_stop=True, print_progress=False, seed=0)
    rt = time.time() - t0

    r = s.results
    logz = float(np.asarray(r["logz"]).flat[-1])
    info = r.get("information", r.get("h", 1.0))
    H = float(np.asarray(info).flat[-1] if np.asarray(info).ndim else info)
    logzerr = float(np.sqrt(max(abs(H), 1e-30) / nlive))
    ess = kish_ess(np.asarray(r["logwt"]) - logz)
    niter = len(np.asarray(r["logl"]))
    nb = len(s.batch_nlive_log) - 1

    print(f"\nResults:")
    print(f"  logZ  = {logz:.4f} +/- {logzerr:.4f}  (truth: {lnz_truth:.4f})")
    print(f"  ESS   = {ess:.0f}")
    print(f"  niter = {niter}, nbatches = {nb}")
    print(f"  time  = {rt:.1f}s")

    # Runplot
    dres = s.to_dynesty_results()
    fig, axes = dyplot.runplot(dres, color="blue", logplot=True,
                                lnz_truth=lnz_truth, truth_color="orange")
    fig.suptitle(f"JNesty Dynamic NS (pfrac={args.pfrac})", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "runplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "logz": logz, "logzerr": logzerr, "ess": float(ess),
        "niter": niter, "nbatches": nb, "runtime": rt,
        "nlive": nlive, "maxbatch": mb, "pfrac": args.pfrac,
        "lnz_truth": lnz_truth,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutput: {out}/")


if __name__ == "__main__":
    main()
