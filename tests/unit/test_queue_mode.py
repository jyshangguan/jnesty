"""Tests for Dynesty-style queue mode (vmap_queue_refill)."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jnesty.internal_samplers import vmap_queue_refill


def _loglik_gaussian(x):
    return -0.5 * jnp.sum(x ** 2)


class TestVmapQueueRefill:
    """Test the parallel queue refill function."""

    def test_output_shapes(self):
        """Queue refill returns arrays with leading dim = queue_size."""
        ndim = 3
        queue_size = 4
        nlive = 50
        key = random.PRNGKey(0)
        live_x = random.uniform(key, (nlive, ndim))
        live_logL = jnp.array([-0.5 * jnp.sum((x - 0.5) ** 2) for x in live_x])
        axes = jnp.eye(ndim)

        qx, ql, qna, qnt = vmap_queue_refill(
            random.PRNGKey(1), live_x, live_logL, jnp.min(live_logL),
            _loglik_gaussian,
            jnp.zeros((1, ndim, ndim)), jnp.array([-jnp.inf]),  # me_axes, me_logvol
            axes, 1.0, 20, ndim, ndim, queue_size, False, None,
        )

        assert qx.shape == (queue_size, ndim)
        assert ql.shape == (queue_size,)
        assert qna.shape == (queue_size,)
        assert qnt.shape == (queue_size,)

    def test_all_valid_with_low_constraint(self):
        """With -inf constraint, all candidates should be valid."""
        ndim = 2
        queue_size = 8
        nlive = 100
        key = random.PRNGKey(42)
        live_x = random.uniform(key, (nlive, ndim))
        live_logL = jnp.array([-0.5 * jnp.sum((x - 0.5) ** 2) for x in live_x])
        axes = jnp.eye(ndim)

        qx, ql, qna, qnt = vmap_queue_refill(
            random.PRNGKey(1), live_x, live_logL, -jnp.inf,
            _loglik_gaussian,
            jnp.zeros((1, ndim, ndim)), jnp.array([-jnp.inf]),
            axes, 1.0, 20, ndim, ndim, queue_size, False, None,
        )

        # All should be finite (valid points)
        assert jnp.all(jnp.isfinite(ql))
        # All should have n_tot > 0 (took some steps)
        assert jnp.all(qnt > 0)

    def test_multi_ellipsoid_axes(self):
        """Queue refill works with multi-ellipsoid axes selection."""
        ndim = 2
        queue_size = 4
        nlive = 50
        key = random.PRNGKey(7)
        live_x = random.uniform(key, (nlive, ndim))
        live_logL = jnp.array([-0.5 * jnp.sum((x - 0.5) ** 2) for x in live_x])

        # Create 2 dummy ellipsoids
        me_axes = jnp.stack([jnp.eye(ndim), 2.0 * jnp.eye(ndim)])
        me_logvol = jnp.array([0.0, -1.0])

        qx, ql, qna, qnt = vmap_queue_refill(
            random.PRNGKey(1), live_x, live_logL, -jnp.inf,
            _loglik_gaussian,
            me_axes, me_logvol,
            jnp.eye(ndim), 1.0, 20, ndim, ndim, queue_size, True, None,
        )

        assert qx.shape == (queue_size, ndim)
        assert jnp.all(jnp.isfinite(ql))

    def test_ntot_equals_rwalk_K(self):
        """Each walk should report n_tot == rwalk_K steps."""
        ndim = 2
        rwalk_K = 30
        queue_size = 4
        nlive = 50
        key = random.PRNGKey(0)
        live_x = random.uniform(key, (nlive, ndim))
        live_logL = jnp.array([-0.5 * jnp.sum((x - 0.5) ** 2) for x in live_x])
        axes = jnp.eye(ndim)

        _, _, _, qnt = vmap_queue_refill(
            random.PRNGKey(1), live_x, live_logL, -jnp.inf,
            _loglik_gaussian,
            jnp.zeros((1, ndim, ndim)), jnp.array([-jnp.inf]),
            axes, 1.0, rwalk_K, ndim, ndim, queue_size, False, None,
        )

        np.testing.assert_array_equal(np.array(qnt), rwalk_K)
