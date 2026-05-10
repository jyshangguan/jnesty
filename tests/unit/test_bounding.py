import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jnesty.bounding import (
    UnitCube, SingleEllipsoid, MultiEllipsoidBound, get_bound,
)


class TestUnitCube:
    def test_fit_returns_self(self):
        b = UnitCube(ndim=3)
        pts = random.uniform(random.PRNGKey(0), shape=(50, 3))
        assert b.fit(pts) is b

    def test_sample_shape(self):
        b = UnitCube(ndim=3)
        pts = random.uniform(random.PRNGKey(0), shape=(50, 3))
        b.fit(pts)
        result = b.sample(random.PRNGKey(1), n=10)
        assert result.shape == (10, 3)

    def test_sample_in_bounds(self):
        b = UnitCube(ndim=3)
        b.fit(random.uniform(random.PRNGKey(0), shape=(50, 3)))
        pts = b.sample(random.PRNGKey(1), n=1000)
        assert jnp.all(pts >= 0.0) and jnp.all(pts <= 1.0)

    def test_get_axes_identity(self):
        b = UnitCube(ndim=3)
        np.testing.assert_allclose(b.get_axes(), jnp.eye(3))

    def test_contains_interior(self):
        b = UnitCube(ndim=3)
        assert bool(b.contains(jnp.array([0.5, 0.5, 0.5])))

    def test_contains_exterior(self):
        b = UnitCube(ndim=3)
        assert not bool(b.contains(jnp.array([1.5, 0.5, 0.5])))


class TestSingleEllipsoid:
    def test_fit_updates_state(self, gaussian_points_2d):
        e = SingleEllipsoid(ndim=2)
        e.fit(gaussian_points_2d)
        assert e.center.shape == (2,)
        assert float(jnp.linalg.norm(e.center - jnp.array([0.5, 0.5]))) < 0.1

    def test_sample_shape(self, gaussian_points_2d):
        e = SingleEllipsoid(ndim=2)
        e.fit(gaussian_points_2d)
        pt = e.sample(random.PRNGKey(0))
        assert pt.shape == (2,)

    def test_contains_center(self, gaussian_points_2d):
        e = SingleEllipsoid(ndim=2)
        e.fit(gaussian_points_2d)
        assert bool(e.contains(e.center))

    def test_get_axes_shape(self, gaussian_points_2d):
        e = SingleEllipsoid(ndim=2)
        e.fit(gaussian_points_2d)
        assert e.get_axes().shape == (2, 2)


class TestMultiEllipsoidBound:
    def test_fit_state_not_none(self, unit_points_2d):
        m = MultiEllipsoidBound(ndim=2)
        m.fit(unit_points_2d)
        assert m.state is not None
        assert m.state.n_active >= 1

    def test_sample_shape(self, unit_points_2d):
        m = MultiEllipsoidBound(ndim=2)
        m.fit(unit_points_2d)
        pts = m.sample(random.PRNGKey(0), n=5)
        assert pts.shape == (5, 2)

    def test_get_axes_shape(self, unit_points_2d):
        m = MultiEllipsoidBound(ndim=2)
        m.fit(unit_points_2d)
        assert m.get_axes().shape == (2, 2)

    def test_walk_schedule_length(self, unit_points_2d):
        m = MultiEllipsoidBound(ndim=2)
        m.fit(unit_points_2d)
        sched = m.get_walk_schedule(rwalk_K=25)
        assert len(sched) == 25
        assert jnp.all(sched >= 0)
        assert jnp.all(sched < m.state.n_active)

    def test_contains_fitted_points(self, unit_points_2d):
        m = MultiEllipsoidBound(ndim=2)
        m.fit(unit_points_2d)
        for i in range(len(unit_points_2d)):
            assert bool(m.contains(unit_points_2d[i]))


class TestFactory:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown bound"):
            get_bound('nonexistent', ndim=3)
