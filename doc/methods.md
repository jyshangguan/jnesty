# Methods

This page describes how JNesty implements nested sampling. The algorithms
closely follow [dynesty](https://dynesty.readthedocs.io/); we document the
key ideas and JNesty-specific details here.

## Nested Sampling Overview

Nested sampling (Skilling 2006) is a Monte Carlo method for computing
the Bayesian evidence (marginal likelihood)

$$Z = \int \mathcal{L}(\theta)\,\pi(\theta)\,d\theta$$

by reparameterizing the integral in terms of the prior mass $X$. At each
iteration the live point with the lowest likelihood is removed ("dead point"),
the evidence contribution $\Delta Z = \mathcal{L}_i\,\Delta X_i$ is accumulated,
and a new live point is drawn above the current likelihood threshold. The
process terminates when the remaining evidence estimate
$\Delta\log Z$ falls below a user-specified threshold.

## Two-Phase Sampling

JNesty uses a two-phase approach matching dynesty:

**Phase 1 — Uniform rejection sampling**
Draws batches of uniform random points from the unit cube and accepts the first
one above the current likelihood threshold. This is fast when the prior volume
is large relative to the posterior (early iterations). Phase 1 runs until
sampling efficiency drops below `min_eff` (default 10%) after at least
`min_ncall` (default $2 \times \text{nlive}$) likelihood calls.

**Phase 2 — Random walk with adaptive bounds**
Uses the `RWalkSampler` with a chosen bounding method. Runs until convergence
($\Delta\log Z < \text{threshold}$) or the iteration limit is reached.

## Random Walk Sampler

The random walk sampler (`rwalk`) follows dynesty's
`generic_random_walk` / `propose_ball_point` implementation:

1. **Proposal**: Generate a point uniformly inside an n-ball via
   $\hat{x} = z \cdot (U^{1/n}/\|z\|)$ where $z \sim \mathcal{N}(0, I)$ and
   $U \sim \text{Uniform}(0,1)$. Transform by the axes matrix from the current
   bound and add to the current position.

2. **Acceptance**: Accept if the proposal is inside the unit cube
   $[0,1]^{\text{ndim}}$ **and** has $\log\mathcal{L} > \log\mathcal{L}^*$
   (the current worst live point). No Metropolis step — pure constraint
   enforcement matching dynesty.

3. **Scale adaptation**: After each walk, update the proposal scale via
   Robbins-Monro:
   $$\sigma \leftarrow \sigma \cdot \exp\!\left(\frac{f_{\text{acc}} - f_{\text{target}}}{n_{\text{cdim}} \cdot f_{\text{target}}}\right)$$
   where $f_{\text{acc}}$ is the measured acceptance rate and $n_{\text{cdim}}$
   is the number of clustered dimensions (default: `ndim`).

4. **Clustered vs non-clustered dims**: The first `ncdim` dimensions are
   perturbed by the axes transform. Remaining dimensions are resampled
   uniformly from $[0,1]$, matching dynesty's `ncdim` split.

5. **Retry loop**: Each iteration retries the walk with fresh randomness until
   a valid replacement ($\log\mathcal{L} > \log\mathcal{L}^*$) is found,
   matching dynesty's `_new_point()` while-True behavior.

### Sampling frequency

Both static-NS engines accept per-coordinate *sampling-frequency* weights so
that parameters with very different roles can be explored at different rates
— e.g. a few physical parameters among many near-linear amplitude latents:

- **`NestedSampler(..., rwalk_direction_weights=[w])`** (diffusion-rate
  semantics): each clustered coordinate *i*'s walk displacement is scaled by
  $\sqrt{w_i}$, so its exploration rate (position-variance growth per step)
  is $w_i$ relative to the $w=1$ anchors — "w = 0.2 samples at 20% of the
  frequency". Weights must be finite with $0 < w \le 1$ and at least one
  entry exactly $1$ (the anchor); they are used as given, with no
  normalization. Zeros are rejected: freezing a coordinate silently corrupts
  its posterior. `None` (default) and all-ones weights are bit-identical to
  the unweighted sampler. Measured on an 8-D two-scale problem, aligned
  weights (physical parameters at the anchor, tight latents at 1/25) cut
  posterior-moment bias ~3–4x at equal likelihood-call count; weight ratios
  beyond ~25:1 starve mixing at finite walk length (ratio 4 is safe, 25 is
  borderline, 100 fails).
- **`DNestSampler(..., sample_frequency=f)`** (proposal-frequency
  semantics): per-coordinate proposal allocation weights, normalised
  internally. Zeros *freeze* the coordinate (with a warning) — the opposite
  of rwalk's choice, matching CDNest.

Literal Bernoulli selection (perturb coordinate *i* with probability $w_i$
at the natural step magnitude — dnest's mechanism) was implemented and
measured and is **refuted for rwalk**: the random walk has a single global
step scale, so a natural-magnitude move on a coordinate narrower than that
scale is a many-sigma jump that the likelihood constraint always rejects —
the coordinate stops mixing and the evidence is biased (measured: logZ
errors of −7..−15 at a 25:1 ratio on the two-scale benchmark). dnest
survives literal frequency because its MCMC has per-coordinate adaptive
proposal widths. $\sqrt{w}$ scaling delivers the same variance rate $w$
with acceptance-preserving move sizes.

See also the convergence stop for diffusive nested sampling:
`DNestSampler.run(..., delta_logZ=0.05)` stops early (opt-in) once level
creation has finished and the estimated remaining evidence is below the
threshold — measured to stop ~9x earlier with |ΔlogZ| ≤ 0.03 versus the
full save budget.

## Bounding Methods

JNesty implements three bounding methods, selected by the `bound` parameter:

### UnitCube (`bound='none'`)

No bounding — proposals are generated within the full unit cube. The axes
matrix is the identity. Simple and robust, but less efficient for
concentrated or elongated posteriors.

### SingleEllipsoid (`bound='single'`)

Fits a single ellipsoid to the live points using covariance eigenvalue
decomposition. The axes matrix is $\mathbf{A} = \mathbf{V}\sqrt{\boldsymbol{\Lambda}}$
where $\mathbf{V}$ are eigenvectors and $\boldsymbol{\Lambda}$ are eigenvalues.
Good for unimodal, elongated posteriors.

### MultiEllipsoidBound (`bound='multi'`)

Fits multiple ellipsoids via recursive k-means splitting with a BIC stopping
criterion, matching dynesty's `_bounding_ellipsoids`:

1. Fit one bounding ellipsoid to all live points
2. Split into 2 clusters via k-means along the major axis
3. Accept the split if the combined volume decreases significantly (BIC)
4. Recurse on each sub-cluster (bounded depth: `max_ellipsoids`)
5. Enlarge all ellipsoids by a factor (default 1.25) for numerical safety

For rwalk proposals, one ellipsoid is selected proportional to volume and its
axes are used to shape the n-ball proposal. This lets the walker focus on one
mode at a time.

Periodic refitting is controlled by `bound_update_interval` (measured in
likelihood calls, default: `rwalk_K * nlive` for multi-ellipsoid).

## GPU Parallelism

JNesty offers two GPU parallelism modes for the random walk phase:

### Batch mode (`queue_size=0`, default for `bound='none'`)

Runs `batch_size` independent walks in parallel via `jax.vmap`, each with
`rwalk_K // batch_size` steps. Each walk starts from a different live point
with a different (randomly selected) ellipsoid. The first valid candidate
becomes the replacement. Batch size is auto-tuned to ~5 steps/walk and capped
by GPU memory.

### Queue mode (`queue_size>1`, default for `bound='multi'`)

Dynesty-style GPU parallelism: generates `queue_size` candidates in parallel,
each with a **full** `rwalk_K` steps. The main loop tests one candidate per
iteration against the current $\log\mathcal{L}^*$. Scale adapts only when the
queue drains, matching dynesty's multiprocessing behavior. Leftover candidates
are preserved across iterations.

## Convergence

The convergence criterion is:

$$\Delta\log Z < \text{threshold}$$

where $\Delta\log Z$ estimates the maximum remaining evidence contribution from
the live points:

$$\Delta\log Z = \log\!\left(1 + e^{\log\mathcal{L}_{\max} + \log X_{\text{current}} - \log Z}\right)$$

The default threshold is 0.01. Use 0.1 for quick tests, 0.001 for
publication-quality results.

## Evidence and Weight Computation

After sampling completes, `format_results()` computes posterior weights:

1. **Dead points**: Prior volume shrinks as $\log X_i = -i / \text{nlive}$.
   Trapezoidal weights combine adjacent likelihoods:
   $\log w_i = \log\text{addexp}(\log\mathcal{L}_i, \log\mathcal{L}_{i-1}) + \log\Delta V_i$

2. **Live points**: Remaining live points are added using dynesty's
   `add_live_points()` formula with accelerating volume shrinkage:
   $\log V_{\text{live},j} = \log(1 - (j+1)/(n_{\text{live}}+1)) + \log V_{\text{last dead}}$

3. **Cumulative evidence**: $\log Z_i = \text{logaddexp.accumulate}(\log w_i)$

The evidence error is estimated as $\sigma_{\log Z} = \sqrt{|H| / n_{\text{live}}}$
where $H$ is the information.

## Dynamic Nested Sampling

JNesty implements dynamic nested sampling following
[Higson et al. (2019)](https://doi.org/10.1007/s11222-018-9844-0) and
[dynesty](https://dynesty.readthedocs.io/en/latest/dynamic.html). The
key idea is to run a baseline static NS survey, then iteratively add
**batches** of live points targeted at the regions that contribute most
to posterior or evidence uncertainty.

### Algorithm

1. **Base run**: A standard static NS run with `nlive_init` live points
   produces the initial set of dead points.

2. **Batch loop** (repeated until `stopping_function` returns `True` or
   `maxbatch` is reached):
   - **Weight function**: computes where additional samples would be
     most valuable, returning `(logl_min, logl_max)` bounds based on a
     weighted combination of posterior and evidence importance:
     $$w_i = p_{\text{frac}} \cdot p_i + (1 - p_{\text{frac}}) \cdot z_i$$
   - **Seed live points**: volume-weighted subset selection from saved
     dead points above `logl_min`, then **fresh point generation** via
     rejection sampling to produce `nlive_batch` independent live points.
   - **Batch run**: constrained static NS between `(logl_min, logl_max)`
     with pre-loaded saved history for correct convergence.
   - **Combine**: merge batch dead points into the saved run, recompute
     `logvol` with combined live-point counts, and recalculate evidence
     via `compute_integrals`.

3. **Stopping**: the run stops when
   $$\text{stop} = p_{\text{frac}} \cdot \frac{n_{\text{target}}}{n_{\text{eff}}}
   + (1 - p_{\text{frac}}) \cdot \frac{\sigma_{\log Z}}{\text{evid\_thresh}} \leq 1$$

### GPU Acceleration

The fresh point generation step uses a **single-shot GPU batch**: samples
`nlive_batch * 8` points from the unit cube, evaluates `logL` via a
JIT-compiled `vmap`, and filters above `logl_min` — all in one JAX
dispatch. The compiled `logL` evaluator is cached across batches.

### API

```python
from jnesty import DynamicNestedSampler

dsampler = DynamicNestedSampler(loglikelihood, prior_transform, ndim,
                                 nlive=500, bound='single')
dsampler.run_nested(nlive_init=500, nlive_batch=500, maxbatch=10,
                     pfrac=0.8, n_effective=10000)
```

Key parameters:
- `pfrac`: posterior/evidence split (0.8 default; 1.0 = posterior-only;
  0.0 = evidence-only)
- `nlive_batch`: live points per batch (default = `nlive`)
- `maxbatch`: maximum number of batches
- `n_effective`: target effective sample size for stopping

## Diffusive Nested Sampling (DNest)

`DNestSampler` exposes diffusive nested sampling (Brewer, Pártay &
Csányi 2009) as our JAX port of CDNest (Yan-Rong Li's C/MPI
implementation, pinned at commit `3aaa039`). The algorithm differs
fundamentally from the static and importance engines above: there is no
worst-live-point deletion and no bounding union.

### Levels and learned compression

The likelihood range is carved into *levels*. While levels are being
built, every particle whose likelihood exceeds the current top threshold
feeds an "above" reservoir; once the reservoir holds
`new_level_interval` records, a new level is cut at the
`1/compression` quantile (descending by likelihood and tiebreaker), and
the reservoir keeps only the records above the new threshold. Level
creation stops when the recent mean level spacing falls below `ptol`
(CDNest's PTol test), at a fixed `max_num_levels`, or at the capacity of
1000 levels.

Each level's prior volume is *learned* from the ensemble's dynamics:
whenever a particle update starts at level $j$, `visits[j]` increments,
and `exceeds[j]` increments if the particle's likelihood also clears
level $j+1$. With regularisation $R$ (CDNest:
`int(new_level_interval * sqrt(lam))`),

$$\log X_j = \log X_{j-1} + \log\frac{\text{exceeds}_{j-1} + R/c}
{\text{visits}_{j-1} + R}.$$

Because `exceeds <= visits` and `c > 1`, every ratio lies in $(0,1)$ and
the volumes strictly decrease. When level creation finishes the counter
histories are renormalised to the regularisation (with CDNest's `+1`
smoothing), which keeps the learning responsive.

### Diffusion across levels

Each MCMC step selects one local particle and, in random order, applies
two Metropolis updates:

1. **Particle update** — perturb one coordinate by CDNest's
   heavy-tailed step $h = 10^{1.5-3|t|}\,\mathcal{N}(0,1)$ (with $t$ a
   Student-$t_2$), wrap into the unit cube, and accept if the proposal
   clears the particle's level threshold (the wrapped one-coordinate
   proposal is symmetric, so the Hastings ratio is exactly one).
2. **Level move** — propose
   $j' = j + \mathrm{int}(10^{2U}\mathcal{N}(0,1))$ (forced to $j\pm1$
   when unchanged, wrapped modulo the level count) with acceptance

$$\log A = \log\frac{X_j}{X_{j'}} + \big(\text{log\_push}(j') -
\text{log\_push}(j)\big) + \beta \log\frac{\text{tries}_j + 1}
{\text{tries}_{j'} + 1},$$

where `log_push` = $(j - j_{\max})/\lambda$ while levels are built (a
drift toward the newest level) and the $\beta$ equal-push term is active
afterwards. The visit/exceed bookkeeping of step 1 is what teaches the
level volumes.

**Per-parameter proposal frequencies.** `sample_frequency` supplies
relative weights for which coordinate each proposal perturbs (scalar
broadcast or length-`ndim` vector, normalised internally; default
uniform, the CDNest behaviour). Use a smaller weight for parameters that
change little between good fits: fewer proposals are spent on them. The
weights only allocate proposals — acceptance ratios, posterior weights
and the evidence are unaffected — and `coordinate_attempts` /
`coordinate_accepts` report the realised frequencies. Zero entries freeze
a coordinate (with a warning).

### Gradient-informed proposals

The opt-in `gradient_scales=True` enhancement (constructor argument,
with `gradient_interval=100` MCMC steps between gradient computations)
modulates the per-coordinate proposal scale using the local shape of
the likelihood. Every `gradient_interval` steps the engine evaluates
$\nabla_u \log L$ at one uniformly chosen live particle — one
forward+backward pass through `jax.grad` — and updates a per-coordinate
running average $\bar g_i$ of $|\partial\log L/\partial u_i|$. The
proposal scale of coordinate $i$ is then multiplied by

$$m_i = \mathrm{clip}\!\left(\frac{\mathrm{gm}(\bar g)}{\bar g_i},\ \frac{1}{100},\ 100\right),$$

with the geometric mean taken over live coordinates: sensitive
coordinates (large average gradient) take *smaller* steps, insensitive
ones *larger* steps, equalising per-coordinate acceptance at an
unchanged overall step budget. Dead coordinates (zero average) receive
the exact cap.

**Why this is safe.** The multiplier is a fixed positive scalar, so the
perturbation stays inside the *symmetric wrapped one-coordinate class*
of the particle update: the Hastings ratio remains exactly one, detailed
balance is preserved, and the evidence is untouched. The gradient is
diagnostic only — it is drawn from a separate fold-in key stream
(`random.fold_in` on the seed key, which does not consume it), so the
proposal randomness is never perturbed, and with `gradient_scales=False`
(the default) the branch is not even traced: results are bit-identical
to a run without the argument. The extra cost is honest and measurable:
`n_like` grows by exactly `n_gradient_events`, an amortised
$1/\texttt{gradient\_interval}$ backward pass per MCMC step (events are
settled at block boundaries, at most 32 per block, with catch-up).

`gradient_scales` composes orthogonally with `sample_frequency`: the
frequency weights allocate *how often* each coordinate is proposed, the
gradient scales *how far* its step reaches. The live state is exposed
as the diagnostics `gradient_magnitudes`, `gradient_scale_multipliers`
and `n_gradient_events` on the sampler (and in `print_summary`).

**Measured efficiency.** On the 137-parameter grid BLR problem this
engine was built for (3000 saves, 8 lanes, seed 42), switching the
scales on raised the entropy effective sample size by **17.7%**
($n_{\mathrm{eff}}$ 333.0 → 391.9) at essentially unchanged acceptance
(0.723 → 0.725), with a deeper level structure (142 vs 120 levels) and
multipliers spanning $[0.01, 7.54]$ — the strongly anisotropic regime
the feature targets. On the 128-dimensional GMRF test problem of the
{doc}`case_studies` dimension scan, the evidence error stays inside the
engine's normal band ($-0.91 \to +0.6$ against an analytic truth) with
an $n_{\mathrm{eff}}$ gain (132.8 → 140.2 in that study; the runnable
`dev/demo/07_dnest_enhancements.py` reproduces the default arm
bit-exactly and lands in the same band on the gradient arm, whose exact
trajectory — unlike the default path — depends on the compilation
environment). The wall-clock cost on that production run was +42% for
65,760 gradient events; on CPU the amortised cost is $\sim$1% of
`n_like`.

### GPU parallelism

`num_lanes` independent workers — the JAX analogue of CDNest's MPI
ranks — each hold `num_particles` local particles. One compiled program
(`vmap` over lanes, `lax.scan` over each lane's sequential steps)
advances `thread_steps = int(ndim · thread_steps_factor ·
num_particles)` updates per lane with a frozen copy of the levels, then
the host aggregates the counters, cuts at most one level, recomputes the
volumes, broadcasts, and saves at most one particle — exactly CDNest's
MPI loop structure. Every step performs exactly one likelihood
evaluation, batched across lanes.

### Postprocessing

Evidence and weights follow CDNest's C postprocessor. Each saved sample
is sandwiched between the levels whose thresholds straddle its
likelihood; within a level, samples sorted by likelihood receive
deterministic order-statistic volumes between the level bounds, and each
sample's prior mass is the trapezoid between its neighbours, normalised
globally:

$$\log Z = \operatorname{logsumexp}(\log p + \log L), \qquad
\log w = \log p + \log L - \log Z.$$

The information is $H = \sum w \log L - \log Z$ and the effective
sample size is the *entropy* ESS $\exp(-\sum w \log w)$. Runs stop at
`max_num_saves` saved particles — a run-length control, not a
convergence criterion — and there is no per-run `logzerr` (CDNest
reports none either; repeat runs for evidence scatter). Preserved CDNest
details include the root-lane-only lagging-particle replacement and the
single-save-per-block cadence.

**The Dynesty-shaped results bridge.** `to_results()` converts the
native posterior into a Dynesty-style `Results` object so every shared
plotting function and the FITS I/O work without gaps. Three fields are
included purely for that compatibility, with diffusive-NS semantics:
`nlive` is the ensemble size `num_lanes * num_particles` (a *proxy* —
diffusive nested sampling has no live-point shrinkage);
`delta_logZ_trajectory` is an all-NaN array with one entry per saved
particle (there is no per-iteration convergence trajectory — the run
length is set by the save budget and the opt-in delta-logZ stop is a
scalar terminal statistic); and `rwalk_K` is NaN (random-walk steps per
iteration are a Dynesty-style-engine concept that does not apply here).
`logvol` keeps the native level-sandwich volumes and `logzerr` is NaN.

### Temperature post-processing

Posterior exploration at a temperature $T \ge 1$ is an opt-in
*post-processing* view — the sampling itself is never tempered:

- **`results_at_temperature(T)`** returns a full plotting-compatible
  `Results` view at $T$, and **`posterior_at_temperature(T)`** the
  corresponding `(points, log_w, log_l)` triple (native save order;
  the sample coordinates and the returned `log_l` are the untempered
  saved values — only the weights carry the temperature).

The convention is CDNest's: likelihoods are divided by $T$ for the
evidence and the weights, $\log w = \log p + \log L / T$, while level
placement and the prior masses $\log p$ stay untempered. The
T-ladder semantics follow directly:

- **$T = 1$** is the standard posterior — `results_at_temperature(1)`
  is bit-identical to `to_results()`, and the default
  `to_results()`/`posterior()` paths are unchanged by the feature;
- **$T \to \infty$** approaches the *prior*: the likelihood factor
  $\exp(\log L/T) \to 1$, so the weights become proportional to the
  prior masses, $\log Z \to \log\int\pi = 0$ and $H \to 0$ (note the
  limiting *weights* are the prior masses, which themselves spread over
  several nats of order-statistic volumes — the max/min weight ratio
  need not approach one).

Because the tempered view only re-runs the deterministic
postprocessor, the engine caches the result per temperature: repeated
calls at the same $T$ are cheap, and a continued run invalidates the
cache. Temperature values must be real numbers $\ge 1$ (the same
validation style as `run()`'s). A typical use is a temperature ladder
($T = 1, 2, 5, 10, \dots$) with one `runplot` per rung to watch the
posterior flatten toward the prior; the runnable script is
`dev/demo/07_dnest_enhancements.py`.

## Importance Nested Sampling (Nautilus)

`NautilusSampler` exposes importance nested sampling. It is our JAX/GPU port
of the Nautilus algorithm, implemented by `jnesty.nautilus.JaxSampler` (see
`jnesty/nautilus/`); the external `nautilus-sampler` package is only a
support dependency (reference fitting routines, the reference emulator,
`Prior` compatibility and parity tests), never the CPU `Sampler` itself. The
port keeps the algorithm and moves the numeric work — likelihood and prior
evaluation, bound construction, containment tests, emulator training and
prediction — onto the JAX device; shell bookkeeping, stopping decisions and
checkpointing stay on the host.

### Nested neural-network bounds

The live set is enclosed by a sequence of nested bounds. Each bound combines a
union of minimum-volume enclosing ellipsoids, fitted to the live points and
their above-threshold samples, with an ensemble of MLP emulators that predicts
whether a proposal clears the current likelihood threshold
(`JaxNeuralBound`). The emulator ensemble size is `n_networks` (default 4);
`n_networks=0` disables the neural rejection step and leaves a purely
ellipsoidal bound. `JaxNautilusBound` gathers the non-overlapping bounds plus
one outer bound used for proposals.

### Shells, evidence and point weights

Bound $i$ defines the shell

$$S_i = B_i \setminus \bigcup_{k>i} B_k,$$

the part of bound $i$ not already covered by any later (inner) bound. Every
likelihood evaluation is kept. When a shell is updated from its stored points,
with $n_i$ points out of $n_{\text{sample},i}$ Monte Carlo volume draws and mean
likelihood $\bar{L}_i$, its statistics are

$$\log V_i = \log V(B_i) + \log\frac{n_i}{n_{\text{sample},i}}, \qquad
\log \bar{L}_i = \operatorname{logsumexp}(\log L) - \log n_i,$$

and the evidence is the importance sum over shells

$$\log Z = \operatorname{logsumexp}_i\left(\log V_i + \log \bar{L}_i\right).$$

Each retained point contributes the weight

$$\log w = \log V_i - \log n_i + \log L,$$

and `posterior()` returns `(points, log_w, log_l)` with those weights
normalised. `log_z`, `n_eff` (Kish effective sample size), `n_like` (total
likelihood evaluations) and `eta` (asymptotic sampling efficiency) expose the
run diagnostics.

### Run control and stopping

`run(f_live=0.01, n_shell=1, n_eff=10000, ...)` runs in two phases.
**Exploration** adds likelihood batches and rebuilds bounds until the estimated
fraction of the evidence still held by the live set falls to `f_live` (default
0.01). **Refinement** then adds samples until every shell holds at least
`n_shell` points and the total effective sample size reaches `n_eff` (default
10 000). There is no `logzerr`: monitor `n_eff` instead.

### Checkpointing

`sampler.write(path)` stores a versioned HDF5 checkpoint (`.h5`/`.hdf5`);
constructing the sampler with `filepath=path` and `resume=True` (the default)
continues from it with the same RNG state, shell volumes and neural-network
weights. Two kinds of run cannot be checkpointed and raise an explicit error:
runs with likelihood blobs (blobs are not stored) and bounds trained with
`neural_backend='reference'` (scikit-learn networks are not serialisable);
re-run with the default `neural_backend='jax'` to enable checkpoints.

### Runtime configuration and required callables

Constructing a `NautilusSampler` enables JAX float64 (x64) mode and an on-disk
compilation cache — process-global settings. The cache defaults to
`~/.cache/nautilus_jax` (`$XDG_CACHE_HOME/nautilus_jax`); set
`NAUTILUS_JAX_CACHE_DIR` to another directory, or to an empty string to disable
it. Plain `import jnesty` and the Dynesty-style samplers touch neither setting.
The engine's dependencies are optional (`pip install "jnesty[nautilus]"`), and
both `prior_transform` and `loglikelihood` must be JAX-traceable: use
`jax.numpy`, avoid in-place NumPy mutation, and let the likelihood return
`log_l` or `(log_l, *blobs)`.

### Choosing an engine

The Dynesty-style samplers shrink the prior volume geometrically, replace one
live point per iteration (discarding the other proposal work) and estimate the
evidence with trapezoidal weights, quoting a `logzerr` derived from the
information. The Nautilus engine instead fits nested bounds, retains every
likelihood call and sums importance-weighted shell integrals, refining until
the requested `n_eff` is reached. The DNest engine carves the likelihood into
levels with learned compression and lets the ensemble diffuse across them, so
it keeps exploring already-conquered territory and supports per-parameter
proposal frequencies; its run length is a save budget, and evidence scatter
comes from repeated runs. Pick Dynesty-style for the familiar dynesty
API, error bars and dynamic allocation; pick Nautilus when likelihood calls are
expensive enough that discarding them is wasteful, large weighted posterior
samples are useful, and `n_eff` is a sufficient convergence diagnostic; pick
DNest for CDNest-style diffusive exploration, nuisance-parameter control via
`sample_frequency`, or when comparing directly against CDNest results.
