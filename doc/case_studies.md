# Case Studies

Practical guides distilled from real inference campaigns. Each entry
states a problem, the experiment that characterised it, the mechanism
behind it, and a concrete recommendation — so you can recognise the
situation in your own problem and act on it without re-deriving the
study.

## How to handle more than 100 latent parameters

*Evidence bias of random-walk nested sampling at high latent dimension.*
This entry distils the T1c Phase-1 robustness study: an independent
verification and characterisation of the evidence bias that
random-walk-based nested sampling develops on grid-like models with
many latent amplitudes (e.g. BLR velocity-map cells), measured against
analytic truths.

### The problem

Nested sampling with a random-walk internal sampler develops a
systematic **positive evidence bias** as the number of latent
parameters grows. On the 128-dimensional GMRF test problem "G1"
(analytic $\log Z = -138.552$), the production random-walk
configuration erred by **+13.3 nats** — and upstream
[dynesty](https://dynesty.readthedocs.io/) 2.1.4's own rwalk erred by
+11.7 on the same problem. This is an algorithmic systematic, not a
jnesty defect, and it is large: a +13 bias at logZ ≈ −139 is a factor
of $5\times10^5$ in the Bayes factor. The gradient-free slice sampler
(`hslice`) fails too (−8.0 and −3.3 on two seeds, opposite sign), so
switching walk styles does not rescue it.

```{image} _static/case_study_engine_comparison.png
:alt: Bar chart of logZ errors on the 128-d G1 problem across engines
:width: 600px
:align: center
```

### The test

The study scanned the GMRF problem family from 16 to 192 latent
dimensions (grid shapes $(n_r, n_\phi, n_\eta)$ from $(2,4,2)$ to
$(4,12,4)$; same generator, likelihood $\mathcal{N}(m, 0.5^2 I)$,
$m = 0.7\,\mathrm{rng}(5)$), three seeds per point, with each engine in
its production configuration: rwalk (multi-ellipsoid bound, `nlive=500`,
$\Delta\log Z$ 0.05) and dfnest (`DNestSampler`, 8 lanes × 1 particle,
1500 saves). Every point has an analytic truth.

| latent dims | 16 | 32 | 64 | 96 | 128 | 192 |
|---|---|---|---|---|---|---|
| rwalk err (mean ± sd) | +0.17 ± 0.20 | −0.19 ± 0.53 | −1.27 ± 1.37 | **+3.97 ± 2.92** | **+7.86 ± 5.77** | **+21.42 ± 2.12** |
| dfnest err (mean ± sd) | −0.18 ± 0.21 | +0.11 ± 0.13 | +0.08 ± 0.18 | −0.27 ± 0.59 | −0.75 ± 0.12 | −1.17 ± 0.52 |

rwalk is accurate up to 64 latent dimensions (errors within the quoted
`logzerr` band), the bias **onsets at 96** and then grows roughly
exponentially — doubling the dimension more than doubles the error
(+4 → +8 → +21 as 96 → 128 → 192). dfnest stays accurate throughout
($|\mathrm{err}| \le 1.2$ at every dimension, no trend), and it is also
the cheapest engine on G1: 3.07M likelihood calls in 42 s versus 4.03M
in 148 s (rwalk) and 3.83M in 688 s (upstream dynesty).

```{image} _static/case_study_dimension_scaling.png
:alt: logZ error versus latent dimension for rwalk and dfnest, 16 to 192 dims
:width: 600px
:align: center
```

### The mechanism

Two failure modes compound at high dimension:

1. **Correlated live points.** At 128 dimensions the random walk accepts
   only short moves (walk acceptance collapses to a few percent), so
   each replacement point stays near its parent. The live set is
   effectively far smaller than `nlive`, the shrinkage volume is
   over-estimated, and the evidence is inflated. The systematic is
   robust to sampler-parallelism choices — same-seed reruns at 8/32/64/128
   walk lanes all err between +8 and +15 — so no lane-count tweak
   rescues it.
2. **Multi-ellipsoid degeneracy.** The multi-ellipsoid bound splits
   into 2–5 ellipsoids at 32–96 dimensions, but at $\ge$128 dimensions
   it *never* splits — hundreds of bound refits across the synthetic
   family and the real 137-parameter problem all returned a single
   ellipsoid around ~500 mutually correlated points. `bound='multi'`
   silently degenerates to `bound='single'` exactly where the bias
   explodes, so the bound cannot mitigate it.

Two controls rule out cheap explanations: `hslice` with perfectly
calibrated acceptance (0.40) still fails (it is correlation, not
tuning), and `bound='none'` with 2000-step walks *reverses* the bias to
−44 (it is neither bound truncation nor walk length).

### The recommendation

- **$\ge$96 latent dimensions: use `DNestSampler`.** The diffusive
  engine is accurate at every dimension tested, and cheapest in wall
  clock and likelihood calls. A convenient starting configuration on
  grid-like problems is 8 lanes × 1 particle with ~1500+ saves; the run
  length is a save budget, so monitor `n_eff` (and optionally the
  opt-in `delta_logZ` stop).
- **$\le$64 latent dimensions: rwalk is fine** — errors sit within the
  quoted `logzerr`, and the Dynesty-style engine brings `logzerr` and
  dynamic sampling.
- On the real 137-parameter grid BLR problem the two engines gave
  rwalk $320.71 \pm 0.29$ versus dfnest $326.08$: the $-5.4$ gap is
  within the $\pm 6$ same-seed rwalk scatter — i.e. the bias itself —
  and there is no truth to adjudicate; the analytic dimension scan is
  what picks the engine.

### A practical checklist

1. **Count your latent dimensions** — the number of sampled
   coordinates (e.g. grid-cell amplitudes), not the data size.
2. **Check where you sit against the onset**: $\le$64 → rwalk is fine;
   $\ge$96 → dfnest; 65–95 → run a cheap synthetic problem of the same
   dimension with a known truth, or simply use dfnest.
3. **Pick the engine** (`DNestSampler` vs `NestedSampler`) and budget
   by its own currency: saves and `n_eff` for dfnest, `nlive` and
   $\Delta\log Z$ for rwalk.
4. **Cross-check when you can afford it**: at any dimension, run both
   engines and compare `logZ`. A gap of more than a few nats at
   $\ge$96 dimensions means the rwalk number is the suspect one.
5. **Reproducibility note**: pass an explicit `queue_size` for rwalk
   (`'auto'` picks the lane count by wall-clock timing, so same-seed
   runs are not bit-reproducible across machine load).

The 128-dimensional member of this family, with the exact GMRF
construction and the dfnest configuration above, runs as
`dev/demo/07_dnest_enhancements.py`.

### Figures and data

The two figures above are reproduced from the study; the originals and
all underlying numbers live in the BLR_grid analysis repository under
`task01c_rwalk_evidence_robustness/` — `figures/` (these two PNGs),
`results/t1c_phase1.json` (consolidated per-engine, per-dimension,
per-seed records) and `RESULTS.md` (the narrative report with every
table reproduced here).
