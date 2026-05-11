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
