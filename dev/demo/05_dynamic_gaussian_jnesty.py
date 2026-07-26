#!/usr/bin/env python
"""
Demo 5: Dynamic Nested Sampling -- JNesty vs dynesty direct comparison.

Same 3-D correlated Gaussian likelihood, same nlive/dlogz/maxbatch.
Produces overlaid runplots (static + dynamic 80/20) and a quantitative
comparison table against the analytic lnZ.

Usage:
    python 05_dynamic_gaussian_jnesty.py [--nlive 500] [--maxbatch 4]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dynesty import plotting as dyplot
from dynesty import NestedSampler as DynestyNS
from dynesty import DynamicNestedSampler as DynestyDNS
from jnesty import NestedSampler as JNestyNS
from jnesty import DynamicNestedSampler as JNestyDNS

NDIM = 3
C_np = np.identity(NDIM); C_np[C_np == 0] = 0.95
Cinv_np = np.linalg.inv(C_np)
LNORM = -0.5 * (np.log(2*np.pi)*NDIM + np.log(np.linalg.det(C_np)))
import jax.numpy as jnp
_Cinv = jnp.asarray(Cinv_np); _lnorm = float(LNORM)

def loglike_j(x): return -0.5 * jnp.dot(x, jnp.dot(_Cinv, x)) + _lnorm
def loglike_d(x): return -0.5 * np.dot(x, np.dot(Cinv_np, x)) + LNORM
def ptform(u):   return 20.0 * u - 10.0


def safe_logzerr(results_or_res, nlive):
    """Extract logZ error estimate robustly."""
    info = results_or_res.get("information", results_or_res.get("h", 1.0))
    info = np.asarray(info)
    H = float(info.flat[-1] if info.ndim else info)
    return float(np.sqrt(max(abs(H), 1e-30) / nlive))


def overlay_runplot(res_a, res_b, label_a, label_b, color_a, color_b,
                     lnz_truth, out_path, title):
    fig, axes = dyplot.runplot(res_a, color=color_a, mark_final_live=False,
                                logplot=True)
    fig, axes = dyplot.runplot(res_b, color=color_b, logplot=True,
                                lnz_truth=lnz_truth, truth_color="orange",
                                fig=(fig, axes))
    legend = [
        Line2D([0],[0], color=color_a, lw=2, label=label_a),
        Line2D([0],[0], color=color_b, lw=2, label=label_b),
        Line2D([0],[0], color="orange", lw=1, ls="--",
               label="truth (lnZ={:.3f})".format(lnz_truth)),
    ]
    axes[0].legend(handles=legend, fontsize=8, loc="best")
    fig.suptitle(title, y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--maxbatch", type=int, default=4)
    ap.add_argument("--dlogz", type=float, default=0.01)
    ap.add_argument("--maxiter", type=int, default=20000)
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / "output_05_jnesty"
    out.mkdir(exist_ok=True)
    lnz_truth = NDIM * -np.log(20.0)
    nlive, mb = args.nlive, args.maxbatch

    print("truth lnZ = {:.4f}  (nlive={}, maxbatch={})".format(lnz_truth, nlive, mb))

    # --- static ---
    print("dynesty static ...")
    t0 = time.time()
    dns = DynestyNS(loglike_d, ptform, ndim=NDIM, nlive=nlive, bound="single")
    dns.run_nested(dlogz=args.dlogz, maxiter=args.maxiter, print_progress=False)
    drs = dns.results
    rt_ds = time.time() - t0

    print("jneesty static ...")
    t0 = time.time()
    jns = JNestyNS(loglike_j, ptform, ndim=NDIM, nlive=nlive, bound="single", verbose=False)
    jns.run_nested(delta_logZ_threshold=args.dlogz, max_iterations=args.maxiter,
                    print_progress=False)
    jrs_dynres = jns.to_dynesty_results()
    rt_js = time.time() - t0

    # --- dynamic ---
    print("dynesty dynamic 80/20 ...")
    t0 = time.time()
    ddns = DynestyDNS(loglike_d, ptform, ndim=NDIM, nlive=nlive, bound="single")
    ddns.run_nested(nlive_init=nlive, nlive_batch=nlive, maxbatch=mb,
                     n_effective=10009, print_progress=False)
    ddrs = ddns.results
    rt_dd = time.time() - t0

    print("jneesty dynamic 80/20 ...")
    t0 = time.time()
    jdns = JNestyDNS(loglike_j, ptform, ndim=NDIM, nlive=nlive, bound="single")
    jdns.run_nested(nlive_init=nlive, nlive_batch=nlive, maxbatch=mb,
                     pfrac=0.8, n_effective=10009, use_stop=True,
                     print_progress=False, seed=0)
    jdrs_dynres = jdns.to_dynesty_results()
    rt_jd = time.time() - t0

    # --- extract metrics ---
    from dynesty.utils import get_neff_from_logwt
    jns_res = jns.results
    jdns_res = jdns.results
    rows = [
        ("dynesty static",       drs,       rt_ds, float(drs.logzerr[-1])),
        ("jneesty static",       jrs_dynres, rt_js,
         safe_logzerr(jns_res, nlive)),
        ("dynesty dynamic 80/20", ddrs,      rt_dd, float(ddrs.logzerr[-1])),
        ("jneesty dynamic 80/20", jdrs_dynres, rt_jd,
         safe_logzerr(jdns_res, nlive)),
    ]
    entries = []
    for label, res, rt, le in rows:
        lz = float(np.asarray(res.logz).flat[-1])
        ess = float(get_neff_from_logwt(res.logwt))
        niter = int(res.niter)
        delta = lz - lnz_truth
        entries.append(dict(label=label, logZ=lz, logZ_err=le, ESS=ess,
                             niter=niter, runtime=rt, delta=delta))

    # --- report ---
    print()
    hdr = "{:<25s} {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>7s}  {:>7s}".format(
           "Run", "logZ", "+/- err", "Delta", "ESS", "niter", "time")
    print(hdr)
    print("-" * len(hdr))
    for e in entries:
        print("{:<25s} {:>8.4f}  {:>8.4f}  {:+8.4f}  {:>8.0f}  {:>7d}  {:>6.1f}s".format(
              e["label"], e["logZ"], e["logZ_err"], e["delta"], e["ESS"],
              e["niter"], e["runtime"]))

    # --- gates ---
    print()
    g = entries
    dz_static = abs(g[1]["logZ"] - g[0]["logZ"])
    sig_static = np.sqrt(g[1]["logZ_err"]**2 + g[0]["logZ_err"]**2)
    st = "PASS" if dz_static < 3*sig_static else "FAIL"
    print("static   JNE vs DYN:  Delta={:.3f}, 3sigma={:.3f}  [{}]".format(dz_static, 3*sig_static, st))

    dz_dyn = abs(g[3]["logZ"] - g[2]["logZ"])
    sig_dyn = np.sqrt(g[3]["logZ_err"]**2 + g[2]["logZ_err"]**2)
    dt = "PASS" if dz_dyn < 3*sig_dyn else "FAIL"
    print("dynamic  JNE vs DYN:  Delta={:.3f}, 3sigma={:.3f}  [{}]".format(dz_dyn, 3*sig_dyn, dt))

    nsigma = abs(g[3]["delta"]) / max(g[3]["logZ_err"], 1e-30)
    jt = "PASS" if nsigma < 3 else "FAIL"
    print("JNE dyn vs truth:     Delta={:+.3f}, sigma={:.3f}, {:.1f}sigma  [{}]".format(
          g[3]["delta"], g[3]["logZ_err"], nsigma, jt))

    # --- plots ---
    print()
    print("Plotting ...")
    overlay_runplot(
        drs, jrs_dynres,
        "dynesty static (ESS={:.0f})".format(g[0]["ESS"]),
        "jneesty static (ESS={:.0f})".format(g[1]["ESS"]),
        "black", "red", lnz_truth,
        out / "runplot_static_jnesty_vs_dynesty.png",
        "Static NS: dynesty (black) vs jnesty (red)  |  nlive={} dlogz={}".format(nlive, args.dlogz))
    overlay_runplot(
        ddrs, jdrs_dynres,
        "dynesty dyn 80/20 (ESS={:.0f})".format(g[2]["ESS"]),
        "jneesty dyn 80/20 (ESS={:.0f})".format(g[3]["ESS"]),
        "black", "red", lnz_truth,
        out / "runplot_dynamic_jnesty_vs_dynesty.png",
        "Dynamic NS 80/20: dynesty (black) vs jnesty (red)  |  nlive={} maxbatch={}".format(nlive, mb))

    (out / "summary.json").write_text(json.dumps(entries, indent=2))
    print("Done. Output in {}".format(out))
    print("  - runplot_static_jnesty_vs_dynesty.png")
    print("  - runplot_dynamic_jnesty_vs_dynesty.png")
    print("  - summary.json")


if __name__ == "__main__":
    main()
