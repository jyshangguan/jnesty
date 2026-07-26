#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling (Dynesty)

Runs dynesty's DynamicNestedSampler on the same 3-D correlated Gaussian
as the JNesty demo.  Produces runplot, trace plot, and summary.

Usage:
    python 05_dynamic_gaussian_dynesty.py [--nlive 500] [--maxbatch 4]

Output:
    Creates output_05_dynesty/ with:
    - summary.json: Numerical results
    - runplot.png: dynesty-style 4-panel runplot
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynesty
from dynesty import plotting as dyplot
from dynesty.utils import get_neff_from_logwt

NDIM = 3
C_np = np.identity(NDIM); C_np[C_np == 0] = 0.95
Cinv_np = np.linalg.inv(C_np)
LNORM = -0.5 * (np.log(2 * np.pi) * NDIM + np.log(np.linalg.det(C_np)))


def loglikelihood(x):
    return -0.5 * np.dot(x, np.dot(Cinv_np, x)) + LNORM


def prior_transform(u):
    return 20.0 * u - 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--maxbatch", type=int, default=4)
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "output_05_dynesty"
    out.mkdir(exist_ok=True)
    lnz_truth = NDIM * -np.log(20.0)
    nlive, mb = args.nlive, args.maxbatch

    print(f"Dynamic NS (Dynesty): 3-D correlated Gaussian")
    print(f"  nlive={nlive}, maxbatch={mb}")
    print(f"  analytic lnZ = {lnz_truth:.4f}")

    t0 = time.time()
    dsampler = dynesty.DynamicNestedSampler(
        loglikelihood, prior_transform, ndim=NDIM, nlive=nlive,
        bound="single", sample="rwalk")
    dsampler.run_nested(nlive_init=nlive, nlive_batch=nlive, maxbatch=mb,
                         n_effective=max(10000, NDIM**2), print_progress=False)
    rt = time.time() - t0

    r = dsampler.results
    logz = float(r.logz[-1])
    logzerr = float(r.logzerr[-1])
    ess = float(get_neff_from_logwt(r.logwt))
    niter = r.niter
    nb = len(r.batch_nlive) - 1 if hasattr(r, 'batch_nlive') else mb

    print(f"\nResults:")
    print(f"  logZ  = {logz:.4f} +/- {logzerr:.4f}  (truth: {lnz_truth:.4f})")
    print(f"  ESS   = {ess:.0f}")
    print(f"  niter = {niter}, nbatches = {nb}")
    print(f"  time  = {rt:.1f}s")

    # Runplot
    fig, axes = dyplot.runplot(r, color="red", logplot=True,
                                lnz_truth=lnz_truth, truth_color="orange")
    fig.suptitle("Dynesty Dynamic NS", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "runplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "logz": logz, "logzerr": logzerr, "ess": ess,
        "niter": niter, "nbatches": nb, "runtime": rt,
        "nlive": nlive, "maxbatch": mb,
        "lnz_truth": lnz_truth,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutput: {out}/")


if __name__ == "__main__":
    main()
