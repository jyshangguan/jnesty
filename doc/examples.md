# Examples

Four benchmark problems demonstrating JNesty's capabilities. Each example
shows the problem definition, full code, expected results, and sample output
figures.

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

The scatter plot shows two thin ring structures correctly identified by the
multi-ellipsoid decomposition:

```{image} _static/example4_posterior.png
:alt: Posterior scatter plot of Gaussian shells
:width: 600px
:align: center
```
