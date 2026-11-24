<p align="center">
  <img src="doc/logo.jpg" alt="JNesty logo" width="400">
</p>

# JNesty

[![PyPI version](https://img.shields.io/pypi/v/jnesty.svg)](https://pypi.org/project/jnesty/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jyshangguan/jnesty/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://jyshangguan.github.io/jnesty/)

GPU-accelerated nested sampling with JAX, designed for expensive likelihood
functions. JNesty parallelizes likelihood evaluations across GPU cores using
JAX's `vmap`, making it well-suited for problems where each likelihood call is
computationally costly (e.g., forward models, simulations, large-data
inferences).

## Features

- **JAX GPU acceleration** — likelihood evaluations are parallelized via
  `jax.vmap`, with automatic batch sizing based on GPU memory.
- **Dynesty-compatible API** — familiar `NestedSampler` interface; results
  include log-evidence, posterior samples, and importance weights.
- **Two sampling strategies** — uniform rejection sampling (fast early phase)
  followed by random walk proposals with adaptive scale tuning (Robbins-Monro).
- **Multi-ellipsoid bounds** — recursive k-means splitting with BIC selection,
  matching dynesty's `bound='multi'` implementation.
- **Queue mode** — dynesty-style GPU parallelism where each candidate gets a
  full random walk and scale adapts at queue drain.
- **Sampling-frequency weights** — `rwalk_direction_weights=[w]` explores
  coordinates at different diffusion rates (max=1 anchored; the rwalk analogue
  of dnest's `sample_frequency`), and `DNestSampler.run(..., delta_logZ=...)`
  adds an opt-in convergence stop to diffusive nested sampling.
- **JAX Nautilus engine** — our GPU port of the Nautilus importance nested
  sampler: neural-network bounds, every likelihood call kept, and large
  weighted posterior samples (optional `jnesty[nautilus]` extra).
- **JAX DNest engine** — our GPU port of the CDNest diffusive nested
  sampler: likelihood levels with learned compression, particles that
  diffuse across levels, per-parameter proposal frequencies, opt-in
  gradient-informed proposal scales (`gradient_scales=True`), and
  CDNest-compatible postprocessing with temperature views
  (`results_at_temperature(T)`) (pure JAX, no extra dependencies).
- **Dynesty result interop** — convert results to dynesty's `Results` object
  for use with its plotting utilities.

## Installation

Install from PyPI:

```bash
pip install jnesty
```

The base install uses CPU JAX. For GPU acceleration:

```bash
pip install "jnesty[cuda]"
```

Optional extras:

```bash
pip install "jnesty[nautilus]"   # JAX Nautilus engine
pip install "jnesty[dynesty]"    # dynesty comparison toolchain
```

For development:

```bash
pip install -e ".[nautilus,test]"
```

## Documentation

Build the docs locally:

```bash
pip install sphinx myst-parser sphinx-rtd-theme
cd doc && sphinx-build -b html . _build/html
```

Then open `doc/_build/html/index.html` in your browser.

**Online documentation:** https://jyshangguan.github.io/jnesty/

## Quick Example

```python
import jax.numpy as jnp
from jnesty import NestedSampler, plotting

# Define the problem: 5D correlated Gaussian
def loglikelihood(x):
    return -0.5 * jnp.sum(x**2)

def prior_transform(u):
    return 20.0 * u - 10.0  # uniform [-10, 10] in each dimension

# Run nested sampling
sampler = NestedSampler(loglikelihood, prior_transform, ndim=5)
sampler.run_nested(max_iterations=50000, delta_logZ_threshold=0.01)

# Access results
r = sampler.results
print(f"logZ = {r['logz']:.4f} +/- {r['logzerr']:.4f}")
print(f"H    = {r['information']:.4f}")
print(f"Converged: {r['converged']}")

# Built-in plots (no dynesty dependency)
fig, axes = plotting.runplot(r)
fig, axes = plotting.traceplot(r)
fig, axes = plotting.cornerplot(r)
```

### Multi-modal problem with multi-ellipsoid bounds

```python
sampler = NestedSampler(
    loglikelihood,
    prior_transform,
    ndim=5,
    nlive=1000,
    bound='multi',           # multi-ellipsoid decomposition
    # queue_size=8 is auto-enabled for bound='multi'
    # bound_update_interval=rwalk_K*nlive calls (auto)
)
sampler.run_nested()
```

### Importance nested sampling with the Nautilus engine

```python
from jnesty import NautilusSampler, plotting

sampler = NautilusSampler(
    loglikelihood,
    prior_transform,
    ndim=5,
    nlive=500,
    n_networks=4,             # 0 disables the neural bounds
    seed=0,
)
sampler.run(f_live=0.01, n_eff=10000)
points, log_w, log_l = sampler.posterior()
print(f"logZ = {sampler.log_z:.4f}  ESS = {sampler.n_eff:.0f}")

# Dynesty-compatible view of the same run (cached: sampler.results)
res = sampler.to_results()
fig, axes = plotting.cornerplot(res)
```

### Diffusive nested sampling with the DNest engine

```python
from jnesty import DNestSampler

sampler = DNestSampler(loglikelihood, prior_transform, ndim=5,
                       num_lanes=8, num_particles=2,
                       sample_frequency=1.0, seed=0)
log_z = sampler.run(max_num_saves=10000)
points, log_w, log_l = sampler.posterior()
print(f"logZ = {log_z:.4f}  ESS = {sampler.n_eff:.0f}")

# Dynesty-compatible view (cached: sampler.results)
res = sampler.to_results()
fig, axes = plotting.runplot(res)
```

`DNestSampler` wraps **our JAX port** of the CDNest diffusive nested
sampler. The likelihood is carved into levels with a fixed compression;
each level's prior volume is *learned* from visit/exceed statistics, and
the particle ensemble diffuses across levels (Brewer's diffusive nested
sampling). `run()` stops after `max_num_saves` saved particles — a
run-length control like CDNest's, not an evidence-convergence criterion —
and `ptol` only stops level creation. There is no per-run `logzerr`;
monitor `n_eff` (the entropy effective sample size) and, as CDNest
recommends, repeat runs for evidence scatter.

The headline feature is `sample_frequency`: per-parameter relative
proposal weights. Parameters that barely move between good fits (nuisance
parameters, slowly varying components) can get a smaller weight so the
sampler proposes elsewhere:

```python
# Propose coordinate 0 three times as often as coordinate 1;
# never touch coordinate 2 with the built-in proposal.
sampler = DNestSampler(loglikelihood, prior_transform, ndim=3,
                       sample_frequency=[3.0, 1.0, 0.0])
```

The weights only allocate proposals — they never alter acceptance
ratios, posterior weights or the evidence — and `coordinate_attempts` /
`coordinate_accepts` diagnostics report the realised frequencies.
`num_lanes` independent workers (the JAX analogue of CDNest's MPI ranks)
each hold `num_particles` particles and run one compiled program per
block, so likelihood calls are batched across the whole ensemble.

`NautilusSampler` wraps **our JAX port** of the Nautilus importance nested
sampler (the external `nautilus-sampler` package is only a support
dependency). It keeps the native Nautilus semantics: nested
neural-network shells, every likelihood evaluation kept, evidence and
posterior weights from shell volumes. There is no `logzerr`; uncertainty is
monitored through the effective sample size (`n_eff`). Runs can be
checkpointed to HDF5 (`sampler.write(path)`) and resumed. See
[doc/methods.md](doc/methods.md#importance-nested-sampling-nautilus) for the
algorithm details.

The `to_results()` bridge reuses the shared plotting: `logzerr` is `nan`
(importance NS uses `n_eff` as the uncertainty proxy), while `logvol` and
`logz_trajectory` are derived quantities for plotting only.

Note: constructing a `NautilusSampler` enables JAX float64 mode and an
on-disk compilation cache (`~/.cache/nautilus_jax`); both are process-wide
settings. Plain `import jnesty` and the Dynesty-style samplers are
unaffected. Constructing a `DNestSampler` enables JAX float64 mode as well
(CDNest computes in double precision) but no compilation cache.

### Engine comparison

| Engine | Class | Run method | Results access | Uncertainty | Stopping |
|--------|-------|------------|----------------|-------------|----------|
| Dynesty-style | `NestedSampler`, `DynamicNestedSampler` | `run_nested()` | `sampler.results` dict (`logz`, `samples`, `logwt`, ...) | `logzerr` | remaining-evidence threshold `delta_logZ`; dynamic runs add batch/ESS targets |
| Nautilus | `NautilusSampler` | `run(f_live=0.01, n_eff=10000)` | `posterior()` → `(points, log_w, log_l)`; `log_z`, `n_eff`, `n_like`, `eta` | `n_eff` (no `logzerr`) | exploration to `f_live`, then refinement to `n_shell` and `n_eff` |
| DNest | `DNestSampler` | `run(max_num_saves=10000)` returns `log_z` | `posterior()` → `(points, log_w, log_l)`; `log_z`, `n_eff`, `n_like`, levels, coordinate diagnostics | `n_eff` (no `logzerr`); repeat runs for evidence scatter | level creation stops at `ptol`; run length is `max_num_saves` |

### Saving and loading results

```python
from jnesty import save_results, load_results

save_results(sampler.results, 'output.fits')
loaded = load_results('output.fits')
print(f"Loaded logZ = {loaded['logz']:.4f}")
```

### Using dynesty's plotting

```python
sampler.plot_run()
sampler.plot_trace()
sampler.plot_corner()
```

### JAX log noise

Importing `jnesty` sets `ABSL_MIN_LOG_LEVEL` and `TF_CPP_MIN_LOG_LEVEL` to
`3` via `os.environ.setdefault` before JAX is imported, so absl/XLA INFO and
WARNING messages are hidden by default. This includes the frequent and
harmless

    W1110 18:45:45.731838 3694514 pjrt_executable.cc:642] Assume version
    compatibility. PjRt-IFRT does not track XLA executable versions.

that XLA prints whenever an executable is loaded from the persistent
compilation cache (the Nautilus engine enables that cache). To see the logs
again, export a lower level before running:

```bash
ABSL_MIN_LOG_LEVEL=0 python your_script.py
```

`0` is INFO, `1` WARNING, `2` ERROR, `3` FATAL. Because `setdefault` is
used, a value you already exported always wins.

**Import order matters:** the defaults only take effect if `import jnasty`
runs before the first `import jax` — the C++ log threshold is read when
jaxlib loads and cannot be changed afterwards. Scripts that must import
`jax` first (e.g. a jax-centric module that imports `jnasty` lazily) should
export `TF_CPP_MIN_LOG_LEVEL=3` (and `ABSL_MIN_LOG_LEVEL=3`) in the shell
instead.

## How It Works

JNesty follows dynesty's separation of concerns: a **bounding method**
defines the proposal region and an **internal sampler** generates candidate
points within that region.

**Sampling strategies** (internal samplers):

| Strategy | Description |
|----------|-------------|
| Uniform rejection | Draw random points from the unit cube; accept the first one above the likelihood threshold. Used automatically in the early phase when efficiency is high. |
| Random walk (`rwalk`) | N-ball proposals with adaptive scale via Robbins-Monro tuning. Matches dynesty's `sample='rwalk'`. |

**Bounding methods**:

| Bound | Description |
|-------|-------------|
| `none` | No bounding — proposals use the full unit cube (default). |
| `single` | Single ellipsoid fitted to live points. |
| `multi` | Multi-ellipsoid decomposition via recursive k-means splitting with BIC criterion. Periodic refitting measured in likelihood calls. |

The sampler runs in two phases:

1. **Phase 1** — Uniform rejection sampling. Fast when the prior volume is
   large relative to the posterior. Automatically switches to Phase 2 when
   sampling efficiency drops below a threshold.
2. **Phase 2** — Random walk with the chosen bounding method and adaptive
   scale tuning. Converges when the remaining evidence estimate
   `delta_logZ` falls below the threshold.

For expensive likelihoods, both phases parallelize evaluations across GPU
cores. The random walk phase supports two GPU parallelism modes:
**batch mode** (split `rwalk_K` steps across multiple independent walks) and
**queue mode** (dynesty-style batch queue where each candidate gets full
`rwalk_K` steps).

## Repository layout

The package directory sits at the repository root — flat like the upstream
`dynesty` and `nautilus-sampler` packages — and is installed editable via
`pip install -e .` (`pyproject.toml`: `[tool.setuptools.packages.find]
where = ["."]`). There is no `src/` directory.

```text
jnesty/
    __init__.py            # public API + JAX log-noise defaults
    dynesty_sampler.py     # public dynesty-style classes   (twins: one public
    nautilus_sampler.py    # public Nautilus adapter          module per engine)
    dnest_sampler.py       # public DNest adapter
    dynesty/               # dynesty-style engine           (peers: one engine
    nautilus/              # JAX Nautilus engine               subpackage each;
    dfnest/                # JAX DNest engine                  dnest = the
    utils.py               # shared helpers (single _convert_to_numpy, ...)
    results.py             # shared Results + engine bridges
    plotting.py            # shared plotting for all engines
```

Each sampler family has the same structure: one public adapter module
(`dynesty_sampler.py`, `nautilus_sampler.py`, `dnest_sampler.py`) plus one
engine subpackage (`dynesty/`, `nautilus/`, `dfnest/`). `utils.py`,
`results.py` and `plotting.py` are engine-agnostic and shared by all
engines.

## More Examples

The `dev/demo/` folder contains complete comparison scripts against dynesty:

| Demo | Problem | Highlights |
|------|---------|------------|
| `01_multimodal_gaussian_mixture` | 5D bimodal Gaussian | Multi-ellipsoid bound, mode detection |
| `02_rosenbrock_banana` | 2D Rosenbrock | Non-Gaussian correlated posterior |
| `03_highd_gaussian` | 20D Gaussian | GPU scaling at high dimensions |
| `04_gaussian_shells` | 2D Gaussian shells | Thin degenerate ring structures |
| `01_multimodal_gaussian_mixture_nautilus` | 5D bimodal Gaussian | Nautilus engine variant: neural bounds, every likelihood call kept |
| `02_rosenbrock_banana_nautilus` | 2D Rosenbrock | Nautilus engine variant of the correlated banana |
| `01_multimodal_gaussian_mixture_dnest` | 5D bimodal Gaussian | DNest engine variant: level diffusion, finite-box analytic reference |
| `02_rosenbrock_banana_dnest` | 2D Rosenbrock | DNest engine variant of the correlated banana |
| `03_highd_gaussian_dnest` | 20D Gaussian | DNest engine variant with the truncated-prior analytic H |
| `04_gaussian_shells_dnest` | 2D Gaussian shells | DNest engine variant with polar-quadrature reference |

Each Dynesty-style demo has a JNesty variant (`*_jnesty.py`) and a dynesty
variant (`*_dynesty.py`) for direct comparison; the `*_nautilus.py` and
`*_dnest.py` scripts run the same problems with the Nautilus and DNest
engines.

## Acknowledgments

JNesty's sampling algorithms and bounding methods closely follow the [dynesty](https://github.com/joshspeagle/dynesty) nested sampling package. The Nautilus engine is our JAX port of the [nautilus](https://github.com/johannesulf/nautilus) importance nested sampler by Johannes U. Lange. The DNest engine is our JAX port of [CDNest](https://github.com/LiyrAstroph/CDNest), Yan-Rong Li's C implementation of Brendon J. Brewer's [diffusive nested sampling](https://arxiv.org/abs/0912.2380). We are also inspired by [JAXNS](https://github.com/Joshuaalbert/jaxns) to use JAX to accelerate nested sampling on GPU. This package was developed with [Claude Code](https://claude.com/claude-code), powered by the [GLM](https://bigmodel.cn/) 5.1 model.

## License

MIT
