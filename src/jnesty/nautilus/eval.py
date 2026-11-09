"""Batched JAX evaluation of priors and likelihoods.

The evaluator wraps user callables into one compiled program that maps a batch
of unit-cube points to likelihood values (and optional blobs).  Two calling
conventions are supported, matching :class:`nautilus.Sampler`:

* ``vectorized=False`` - the likelihood acts on a single parameter point and is
  vectorized with ``jax.vmap``;
* ``vectorized=True`` - the likelihood already accepts an ``(n_points, ...)``
  array.

Prior transforms run on the device when the prior is a JAX callable.  When a
``nautilus.Prior`` instance is supplied the (cheap) transform runs on the host
with SciPy, because frozen SciPy distributions are not JAX traceable; the
likelihood, which is the expensive part, still runs on the device.
"""

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ['JaxEvaluator']


class JaxEvaluator:
    """Compile and run batched prior/likelihood evaluations on the device."""

    def __init__(self, prior, likelihood, n_dim, pass_dict=False,
                 vectorized=False, context=None, device=None, dtype=None):
        self.prior = prior
        self.likelihood = likelihood
        self.n_dim = int(n_dim)
        self.pass_dict = pass_dict
        self.vectorized = vectorized
        self.context = context
        self.dtype = jnp.float64 if dtype is None else dtype

        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = jax.devices(device)[0]

        self._compiled = None
        self._prior_program = None
        self._has_blobs = None
        self._is_prior_object = not callable(prior)

    # ------------------------------------------------------------------
    # construction of the compiled program
    # ------------------------------------------------------------------
    def _call_likelihood(self, x):
        if self.context is None:
            return self.likelihood(x)
        return self.likelihood(x, self.context)

    def _call_prior(self, u):
        if self.context is None:
            return self.prior(u)
        return self.prior(u, self.context)

    def _build(self):
        evaluator = self

        def evaluate_batch(unit_points):
            if evaluator._is_prior_object:
                physical = unit_points
            else:
                physical = evaluator._call_prior(unit_points)
            return evaluator._call_likelihood(physical)

        if self.vectorized:
            program = evaluate_batch
        else:
            program = jax.vmap(evaluate_batch)

        self._compiled = jax.jit(program)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def _pad_batch(self, unit):
        """Pad a batch to a power-of-two length so the program is reused.

        Every distinct array length would otherwise compile one executable.
        Padded rows duplicate existing points and are dropped after the call.
        """
        n = unit.shape[0] if hasattr(unit, 'shape') else len(
            jax.tree_util.tree_leaves(unit)[0])
        n_bucket = max(32, 1 << (max(n, 1) - 1).bit_length())
        if n == 0 or n_bucket == n:
            return unit, n

        def pad(array):
            extra = n_bucket - n
            reps = int(np.ceil(n_bucket / max(n, 1)))
            tiled = jnp.concatenate([array] * reps, axis=0)[:n_bucket]
            del extra
            return tiled

        return jax.tree_util.tree_map(pad, unit), n

    def _prepare_unit_points(self, points):
        points = np.asarray(points, dtype=np.float64)
        if self._is_prior_object:
            if self.pass_dict:
                physical = self.prior.unit_to_dictionary(points)
                return jax.tree_util.tree_map(
                    lambda a: jax.device_put(jnp.asarray(a, self.dtype)),
                    physical)
            physical = self.prior.unit_to_physical(points)
            return jax.device_put(jnp.asarray(physical, self.dtype))
        return jax.device_put(jnp.asarray(points, self.dtype))

    def evaluate(self, points):
        """Evaluate the likelihood on unit-cube points.

        Returns
        -------
        log_l : numpy.ndarray, shape (n_points,)
        blobs : list of numpy.ndarray or None
            One array per blob column, following the reference implementation.
        """
        if self._compiled is None:
            self._build()

        unit = self._prepare_unit_points(points)
        unit, n_real = self._pad_batch(unit)
        result = self._compiled(unit)
        result = jax.tree_util.tree_map(lambda a: a[:n_real], result)

        if isinstance(result, tuple):
            log_l = np.asarray(result[0])
            blobs = [np.asarray(r) for r in result[1:]]
            self._has_blobs = True
        else:
            log_l = np.asarray(result)
            blobs = None
            self._has_blobs = False

        log_l = np.asarray(log_l).reshape(-1)

        if np.any(np.isnan(log_l)):
            raise ValueError(
                'The likelihood returned NaN. nautilus cannot handle NaN '
                'likelihood values; return -inf for forbidden regions instead.')

        # ``vmap`` of a scalar function returns one value per point.  A
        # vectorized likelihood that ignores the batch dimension would silently
        # broadcast, so check the shape explicitly.
        if len(log_l) != len(points):
            raise ValueError(
                f'The likelihood returned {len(log_l)} values for '
                f'{len(points)} points. With vectorized=True the function must '
                'return one value per input row.')

        return log_l, blobs

    def prior_to_physical(self, points, as_dict=False):
        """Map unit-cube points to physical parameters on the host.

        Used when returning posterior samples, where the result must be a plain
        NumPy array or dictionary rather than a traced value.
        """
        points = np.asarray(points)
        if self._is_prior_object:
            if as_dict:
                return self.prior.unit_to_dictionary(points)
            return self.prior.unit_to_physical(points)
        if as_dict:
            raise ValueError(
                'Cannot return points as a dictionary: the prior is a plain '
                'callable, not a nautilus.Prior instance.')
        if self._prior_program is None:
            self._prior_program = jax.jit(jax.vmap(self._call_prior))
        return np.asarray(self._prior_program(jnp.asarray(points)))

    def blob_names(self, n_blobs):
        """Default structured-array names for blobs."""
        if n_blobs == 1:
            return None
        return [f'blob_{i}' for i in range(n_blobs)]