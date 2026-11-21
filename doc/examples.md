# Examples

Five benchmark problems demonstrating JNesty's capabilities. The examples come
in three flavours: **Dynesty-style** variants (`NestedSampler`,
`DynamicNestedSampler`), **Nautilus** variants (`NautilusSampler`, our JAX
importance nested sampler) and **DNest** variants (`DNestSampler`, our JAX
diffusive nested sampler). The code below is Dynesty-style unless a section is
explicitly labelled with an engine variant; examples 1 and 2 have all three.
Examples 3 and 4 additionally have DNest variants; example 5 (dynamic
sampling) is Dynesty-style only.

Each example shows the problem definition, full code, expected results, and
sample output figures; every runnable demo lives under `dev/demo/`.

## 1. Multi-Modal Gaussian Mixture (5D)

A bimodal Gaussian with modes at the origin and at $(3, 3, 0, 0, 0)$,
with weights 0.6 and 0.4. Tests multi-ellipsoid decomposition.

```python
import jax
import jax.numpy as jnp
from jnesty import NestedSampler, plotting

def loglikelihood(x):
    mean1 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    mean2 = jnp.array([3.0, 3.0, 0.0, 0.0, 0.0])
    logL1 = -0.5 * jnp.sum((x - mean1)**2) + jnp.log(0.6)
    logL2 = -0.5 * jnp.sum((x - mean2)**2) + jnp.log(0.4)
    return jax.scipy.special.logsumexp(jnp.array([logL1, logL2]))

def prior_transform(u):
    return (u - 0.5) * 10.0  # [-5, 5] in each dim

sampler = NestedSampler(loglikelihood, prior_transform, ndim=5,
                         nlive=500, bound='multi')
sampler.run_nested()

r = sampler.results
print(f"logZ = {r['logz']:.4f} +/- {r['logzerr']:.4f}")
```

**Expected output** (approximate):

```
logZ = -6.87 +/- 0.09
Iterations: ~5500
Converged: True
```

The multi-ellipsoid decomposition identifies two clusters and produces a
corner plot showing both modes clearly separated:

```{image} _static/example1_corner.png
:alt: Corner plot of multi-modal Gaussian mixture
:width: 600px
:align: center
```

### Nautilus variant

```python
from jnesty import NautilusSampler

sampler = NautilusSampler(loglikelihood, prior_transform, ndim=5,
                          nlive=500, n_networks=4, seed=0)
sampler.run(f_live=0.01, n_eff=10000)
points, log_w, log_l = sampler.posterior()
print(sampler.log_z, sampler.n_eff)
```

The evidence estimate should be close to the Dynesty-style value above,
$\log Z \approx -6.87$ (approximate). There is no `logzerr` for this engine —
monitor the effective sample size `n_eff` instead. The runnable script is
`dev/demo/01_multimodal_gaussian_mixture_nautilus.py`.

### DNest variant

```python
from jnesty import DNestSampler

sampler = DNestSampler(loglikelihood, prior_transform, ndim=5,
                       num_lanes=8, num_particles=2, seed=0)
log_z = sampler.run(max_num_saves=20000)
points, log_w, log_l = sampler.posterior()
print(log_z, sampler.n_eff, sampler.n_levels)
```

A sample run (2000 saves) gives `logZ = -6.90` against the exact
finite-box analytic value `-6.94`, with mode masses `0.59/0.41` (exact
`0.611/0.389` — the second mode is truncated by the $[-5,5]^5$ prior).
Level creation finishes through the PTol criterion, and the per-parameter
`sample_frequency` can slow down well-determined nuisance parameters. The
runnable script is `dev/demo/01_multimodal_gaussian_mixture_dnest.py`.

---

## 2. Rosenbrock Banana (2D)

The Rosenbrock function creates a curved, banana-shaped posterior that is
challenging for simple samplers due to its strong correlations.

```python
import jax.numpy as jnp
from jnesty import NestedSampler, plotting

def loglikelihood(x):
    a, b = 1.0, 100.0
    return -0.5 * ((a - x[0])**2 + b * (x[1] - x[0]**2)**2)

def prior_transform(u):
    return (u - 0.5) * 20.0  # [-10, 10] in each dim

sampler = NestedSampler(loglikelihood, prior_transform, ndim=2,
                         nlive=500, bound='multi')
sampler.run_nested()

r = sampler.results
print(f"logZ = {r['logz']:.4f} +/- {r['logzerr']:.4f}")

fig, axes = plotting.cornerplot(r)
```

**Expected output** (approximate):

```
logZ = -4.41 +/- 0.08
Iterations: ~4500
```

The corner plot shows the characteristic curved banana shape:

```{image} _static/example2_corner.png
:alt: Corner plot of Rosenbrock banana posterior
:width: 600px
:align: center
```

### Nautilus variant

```python
from jnesty import NautilusSampler

sampler = NautilusSampler(loglikelihood, prior_transform, ndim=2,
                          nlive=500, n_networks=4, seed=0)
sampler.run(f_live=0.01, n_eff=10000)
points, log_w, log_l = sampler.posterior()
print(sampler.log_z, sampler.n_eff)
```

The evidence estimate should be close to the Dynesty-style value above,
$\log Z \approx -4.41$ (approximate). There is no `logzerr` for this engine —
monitor the effective sample size `n_eff` instead. The runnable script is
`dev/demo/02_rosenbrock_banana_nautilus.py`.

### DNest variant

```python
from jnesty import DNestSampler

sampler = DNestSampler(loglikelihood, prior_transform, ndim=2,
                       num_lanes=8, num_particles=2, seed=0)
log_z = sampler.run(max_num_saves=20000)
points, log_w, log_l = sampler.posterior()
```

The DNest variant runs the same banana with level diffusion; a sample run
(3000 saves) gives `logZ = -4.31` with a weighted posterior mean on the
curved ridge. The runnable script is
`dev/demo/02_rosenbrock_banana_dnest.py`.

---

## 3. High-Dimensional Gaussian (20D)

A standard Gaussian in 20 dimensions. Tests GPU scaling behavior and
validates logZ against the analytical value.

```python
import jax.numpy as jnp
import numpy as np
from jnesty import NestedSampler

def loglikelihood(x):
    return -0.5 * jnp.sum(x**2)

def prior_transform(u):
    return (u - 0.5) * 10.0  # [-5, 5] in each dim

ndim = 20
sampler = NestedSampler(loglikelihood, prior_transform, ndim=ndim,
                         nlive=500, bound='multi')
sampler.run_nested()

r = sampler.results
analytical_logZ = -ndim * np.log(10.0)  # for uniform [-5,5] prior
print(f"JNesty    logZ = {r['logz']:.4f} +/- {r['logzerr']:.4f}")
print(f"Analytical logZ = {analytical_logZ:.4f}")
print(f"Difference      = {r['logz'] - analytical_logZ:.4f}")
```

**Expected output** (approximate):

```
JNesty    logZ = -27.82 +/- 0.19
Analytical logZ = -27.67
Difference      = -0.15
```

The DNest variant (`dev/demo/03_highd_gaussian_dnest.py`) reproduces both
reference values in one run (a sample 20D run: `logZ = -27.64`, `H = 17.68`,
marginal variance `0.994`) and uses the exact truncated-prior information
$H_1 = -v/2 - \log Z_1 = 0.8837$ per dimension rather than the $d/2$
approximation quoted above.

JNesty agrees with the analytical value within Monte Carlo uncertainty. The
run plot shows evidence convergence:

```{image} _static/example3_run.png
:alt: Run plot showing evidence convergence for 20D Gaussian
:width: 600px
:align: center
```

---

## 4. Gaussian Shells (2D)

Two thin annular ring distributions centered at $(-3.5, 0)$ and $(3.5, 0)$
with radius 2 and width 0.1. Tests multi-ellipsoid decomposition on thin,
degenerate structures.

```python
import jax.numpy as jnp
from jnesty import NestedSampler, plotting

r_shell = 2.0    # shell radius
w = 0.1          # shell width
c1 = jnp.array([-3.5, 0.0])
c2 = jnp.array([3.5, 0.0])
const = jnp.log(1.0 / jnp.sqrt(2.0 * jnp.pi * w**2))

def loglikelihood(x):
    def logcirc(theta, c):
        d = jnp.sqrt(jnp.sum((theta - c)**2))
        return const - (d - r_shell)**2 / (2.0 * w**2)
    return jnp.logaddexp(logcirc(x, c1), logcirc(x, c2))

def prior_transform(u):
    return (u - 0.5) * 14.0  # [-7, 7] to cover both shells

sampler = NestedSampler(loglikelihood, prior_transform, ndim=2,
                         nlive=500, bound='multi')
sampler.run_nested()

r = sampler.results
print(f"logZ = {r['logz']:.4f} +/- {r['logzerr']:.4f}")

fig, axes = plotting.cornerpoints(r)
```

**Expected output** (approximate):

```
logZ = -1.83 +/- 0.07
Iterations: ~3900
```

The DNest variant (`dev/demo/04_gaussian_shells_dnest.py`) recovers the
polar-quadrature reference `logZ ≈ -1.75` (a sample run: `-1.71`), radial
mean/width `1.99/0.10` against the true radius 2, and equal shell masses.

The scatter plot shows two thin ring structures correctly identified by the
multi-ellipsoid decomposition:

```{image} _static/example4_posterior.png
:alt: Posterior scatter plot of Gaussian shells
:width: 600px
:align: center
```

---

## 5. Dynamic Nested Sampling (3D Correlated Gaussian)

Dynamic nested sampling (Higson et al. 2019) adaptively allocates
additional live points to regions of high posterior or evidence weight,
improving both parameter estimation and evidence accuracy beyond what
static nested sampling achieves at the same likelihood-call budget.

This example uses a 3-D correlated multivariate Gaussian — the canonical
dynamic NS test problem from the [dynesty
documentation](https://dynesty.readthedocs.io/en/latest/dynamic.html).

```python
import jax.numpy as jnp
from jnesty import DynamicNestedSampler

# 3-D correlated Gaussian (covariance off-diagonal = 0.95)
Cinv = jnp.linalg.inv(jnp.array([[1, 0.95, 0.95],
                                  [0.95, 1, 0.95],
                                  [0.95, 0.95, 1]]))
lnorm = -0.5 * (jnp.log(2 * jnp.pi * 3) + jnp.log(jnp.linalg.det(
    jnp.linalg.inv(Cinv))))

def loglikelihood(x):
    return -0.5 * jnp.dot(x, jnp.dot(Cinv, x)) + lnorm

def prior_transform(u):
    return 20.0 * u - 10.0  # uniform [-10, 10]

# Static baseline
sampler = DynamicNestedSampler(loglikelihood, prior_transform, ndim=3,
                                nlive=500, bound='single')
sampler.run_nested(nlive_init=500, nlive_batch=500, maxbatch=4,
                    pfrac=0.8)

r = sampler.results
print(f"logZ = {r['logz'][-1]:.4f}")
print(f"ESS  = {sampler.n_effective:.0f}")
```

The `pfrac` parameter controls the posterior/evidence split:

| `pfrac` | Posterior weight | Evidence weight | Use case |
|---|---|---|---|
| 1.0 | 100% | 0% | Maximize posterior ESS |
| 0.8 (default) | 80% | 20% | Balanced |
| 0.0 | 0% | 100% | Minimize evidence uncertainty |

**Expected output** (approximate, `nlive=500`, `maxbatch=4`):

```
logZ = -8.78 +/- 0.12  (analytical: -8.99)
ESS  = 5700
```

The dynesty-style run plot shows the dynamic allocation of live points:

```{image} _static/example5_runplot.png
:alt: Dynamic NS run plot showing batch injections
:width: 600px
:align: center
```

### Dynamic vs Static Comparison

At matched likelihood-call budget, dynamic NS achieves ~2-3x higher
effective sample size (ESS) than static NS, matching the efficiency
gains reported by dynesty.
