import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jnesty.multi_ellipsoid import (
    fit_bounding_ellipsoid, fit_multi_ellipsoid, MultiEllipsoidState,
    get_axes_for_rwalk, sample_from_union, contains, contains_all,
)


class TestFitBoundingEllipsoid:
    def test_returns_five(self):
        pts = random.uniform(random.PRNGKey(0), shape=(50, 2))
        center, cov, axes, precision, logvol = fit_bounding_ellipsoid(pts)
        assert center.shape == (2,)
        assert cov.shape == (2, 2)
        assert axes.shape == (2, 2)
        assert precision.shape == (2, 2)
        assert logvol.shape == ()

    def test_contains_all_input_points(self):
        pts = 0.5 + 0.1 * random.normal(random.PRNGKey(0), shape=(50, 2))
        center, cov, axes, precision, logvol = fit_bounding_ellipsoid(pts)
        delta = pts - center
        maha = jnp.einsum('ij,ijk,ik->i', delta, precision[None], delta)
        assert jnp.all(maha < 1.5)  # generous bound for expansion factor

    def test_logvol_positive(self):
        pts = random.uniform(random.PRNGKey(0), shape=(100, 3))
        _, _, _, _, logvol = fit_bounding_ellipsoid(pts)
        assert float(logvol) > 0


class TestFitMultiEllipsoid:
    def test_returns_state(self):
        pts = random.uniform(random.PRNGKey(0), shape=(100, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=5)
        assert isinstance(state, MultiEllipsoidState)
        assert state.n_active >= 1

    def test_single_cluster(self):
        pts = 0.5 + 0.01 * random.normal(random.PRNGKey(0), shape=(50, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=5)
        assert state.n_active == 1


class TestGetAxesForRwalk:
    def test_shape(self):
        pts = random.uniform(random.PRNGKey(0), shape=(50, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=3)
        axes = get_axes_for_rwalk(random.PRNGKey(0), state)
        assert axes.shape == (2, 2)


class TestSampleFromUnion:
    def test_shape(self):
        pts = random.uniform(random.PRNGKey(0), shape=(50, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=3)
        pt, ell_idx = sample_from_union(random.PRNGKey(1), state)
        assert pt.shape == (2,)
        assert int(ell_idx) < state.n_active


class TestContains:
    def test_true_for_member(self):
        pts = random.uniform(random.PRNGKey(0), shape=(50, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=3)
        for i in range(min(10, len(pts))):
            assert bool(contains(state, pts[i]))

    def test_false_for_distant(self):
        pts = random.uniform(random.PRNGKey(0), shape=(50, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=3)
        far = jnp.array([100.0, 100.0])
        assert not bool(contains(state, far))

    def test_contains_all_vectorized(self):
        pts = random.uniform(random.PRNGKey(0), shape=(50, 2))
        state = fit_multi_ellipsoid(pts, max_ellipsoids=3)
        assert bool(contains_all(state, pts))
