"""Geometry tests: parity with the reference bounds and analytic checks."""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from nautilus.bounds.basic import Ellipsoid as RefEllipsoid
from nautilus.bounds.basic import (
    minimum_volume_enclosing_ellipsoid as ref_mvee)
from jnesty.nautilus.bounds import JaxEllipsoid, JaxUnitCubeEllipsoidMixture
from jnesty.nautilus.geometry import minimum_volume_enclosing_ellipsoid


@pytest.fixture(scope='module', autouse=True)
def _x64():
    from jnesty.nautilus import __name__ as _  # noqa: F401  (enables x64)


@pytest.mark.parametrize('n_points,n_dim', [(200, 2), (500, 3), (800, 5)])
def test_mvee_matches_reference(n_points, n_dim):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(n_points, n_dim)) * 2 + 1
    c, a, a_inv = minimum_volume_enclosing_ellipsoid(jnp.asarray(points))
    c_ref, a_ref, a_inv_ref = ref_mvee(points)
    assert np.allclose(np.asarray(c), c_ref, atol=1e-10)
    assert np.allclose(np.asarray(a), a_ref, atol=1e-10 * abs(a_ref).max())
    assert np.allclose(np.asarray(a_inv), a_inv_ref,
                       atol=1e-10 * abs(a_inv_ref).max())


def test_ellipsoid_volume_matches_reference():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(600, 3))
    ours = JaxEllipsoid.compute(points)
    ref = RefEllipsoid.compute(points)
    assert float(ours.log_v) == pytest.approx(ref.log_v, rel=0, abs=1e-10)


def test_sphere_mvee_recovers_unit_ball():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(4000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    ellipsoid = JaxEllipsoid.compute(points, enlarge_per_dim=1.0)
    assert float(ellipsoid.log_v) == pytest.approx(
        np.log(4.0 / 3.0 * np.pi), abs=1e-6)
    assert bool(jnp.all(ellipsoid.contains(points)))


def test_ellipsoid_transform_round_trip():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(50, 4))
    ellipsoid = JaxEllipsoid.compute(points)
    back = ellipsoid.transform(ellipsoid.transform(points), inverse=True)
    assert np.allclose(np.asarray(back), points, atol=1e-10)


def test_ellipsoid_sampling_is_uniform():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(1000, 2))
    ellipsoid = JaxEllipsoid.compute(points, enlarge_per_dim=1.0)
    samples = np.asarray(ellipsoid.sample(20000, key=jax.random.PRNGKey(0)))
    # Uniform inside the ellipsoid: the transformed radius^2 is Beta(d/2, 1)
    # with mean d/(d+2) = 0.5 for two dimensions.
    radius2 = np.asarray(ellipsoid.transform(samples))
    radius2 = np.sum(radius2 ** 2, axis=1)
    assert np.all(radius2 < 1.0)
    assert radius2.mean() == pytest.approx(0.5, abs=0.02)


def test_mixture_selects_cube_dimensions():
    rng = np.random.default_rng(5)
    # Wide and uniform along the first axis, narrow along the second.
    points = np.column_stack([rng.uniform(size=400),
                              rng.normal(0.5, 0.01, size=400)])
    mixture = JaxUnitCubeEllipsoidMixture.compute(points)
    assert bool(mixture.dim_cube[0])
    assert not bool(mixture.dim_cube[1])
    assert bool(jnp.all(mixture.contains(points)))
    # The mixture must not be larger than a bare ellipsoid in this case.
    assert float(mixture.log_v) <= float(
        JaxEllipsoid.compute(points).log_v) + 1e-9


def test_union_contains_and_volume():
    from jnesty.nautilus.bounds import JaxUnion

    rng = np.random.default_rng(6)
    points = np.concatenate([rng.normal(-0.3, 0.05, size=(300, 2)),
                             rng.normal(0.3, 0.05, size=(300, 2))])
    # These test points deliberately extend outside the unit cube, so the
    # union must be built without the optional unit-cube restriction.
    union = JaxUnion.compute(points, n_points_min=10, unit=False)
    while union.split():
        pass
    assert len(union.bounds) >= 2
    assert bool(jnp.all(union.contains(points)))
    samples = np.asarray(union.sample(5000, key=jax.random.PRNGKey(1)))
    inside = np.asarray(union.contains(samples))
    assert inside.mean() > 0.99

def test_mixture_sampling_reproduces_its_own_volume():
    """The Monte-Carlo volume of a mixture must match a direct measurement.

    The proposal distribution used to normalise its random direction over all
    dimensions, including the ones sampled from the unit cube.  When the cube
    dimensions exist (always, in more than a couple of dimensions) the ellipsoid
    coordinate collapsed towards its centre, so proposals never left the cube
    and the sampler reported a bound volume far larger than the region it was
    actually sampling -- which biased the evidence and slowed the climb.
    """
    from nautilus.bounds.basic import UnitCubeEllipsoidMixture as RefMixture
    from jnesty.nautilus.bounds import JaxUnitCubeEllipsoidMixture

    rng = np.random.default_rng(0)
    # Wide in most dimensions, narrow in one: the mixture will keep the wide
    # ones as cube dimensions, which is the configuration the bug needed.
    points = rng.uniform(size=(400, 5))
    points[:, 0] = rng.normal(0.5, 0.02, size=400)

    port = JaxUnitCubeEllipsoidMixture.compute(points)
    ref = RefMixture.compute(points)
    assert np.array_equal(np.asarray(port.dim_cube), ref.dim_cube)
    assert float(port.log_v) == pytest.approx(float(ref.log_v), abs=1e-9)

    # Direct measurement of the same region, over 200 000 uniform points.
    test = rng.uniform(size=(200000, 5))
    measured = np.log(np.mean(np.asarray(port.contains(test))))
    assert measured == pytest.approx(float(port.log_v), abs=0.05)

    # And the sampler must land in the same distribution as the reference one.
    from_port = np.asarray(port.sample(20000, key=jax.random.PRNGKey(0)))
    from_ref = np.asarray(ref.sample(20000))
    both_inside = np.mean(np.asarray(port.contains(from_port)))
    assert both_inside > 0.999
    # Mean position along the narrow dimension must agree (the collapsed
    # sampler sat at the centre instead of covering the ellipsoid).
    assert from_port[:, 0].mean() == pytest.approx(from_ref[:, 0].mean(), abs=0.01)
    assert from_port[:, 0].std() == pytest.approx(from_ref[:, 0].std(), rel=0.1)


def test_union_contains_respects_cube_restriction():
    """A union with unit=True must exclude points outside [0, 1)^d."""
    from jnesty.nautilus.bounds import JaxUnion

    rng = np.random.default_rng(7)
    points = rng.uniform(size=(100, 2))
    union = JaxUnion.compute(points)
    test = np.array([[-0.1, 0.5], [1.2, 0.5], [0.5, 0.5]])
    assert np.array_equal(np.asarray(union.contains(test)),
                          np.array([False, False, True]))

    free = JaxUnion.compute(points, unit=False)
    # With unit=False the same points are governed only by the ellipsoid.
    assert np.array_equal(np.asarray(free.contains(test)),
                          np.asarray(free.bounds[0].contains(test)))


def test_unkeyed_sampling_advances_the_random_stream():
    """Volume-estimation calls must not replay the same proposals forever."""
    from jnesty.nautilus.bounds import JaxEllipsoid

    rng = np.random.default_rng(11)
    bound = JaxEllipsoid.compute(rng.normal(size=(200, 3)))
    first = np.asarray(bound.sample(32))
    second = np.asarray(bound.sample(32))
    assert not np.array_equal(first, second)

    # A keyed call and an unkeyed call must also continue rather than restart.
    keyed = np.asarray(bound.sample(32, key=jax.random.PRNGKey(99)))
    unkeyed = np.asarray(bound.sample(32))
    assert not np.array_equal(keyed, unkeyed)


def test_union_samples_never_cross_prior_boundary():
    """The cube mask must be applied after shuffling, not before it."""
    from jnesty.nautilus.bounds import JaxUnion

    # A cluster near the edge makes the fitted ellipsoid extend well outside
    # [0, 1]^d.  A mask computed before the shuffle leaks most of those rows.
    rng = np.random.default_rng(12)
    points = np.column_stack([
        np.clip(rng.normal(0.93, 0.06, size=300), 0.0, 1.0),
        rng.uniform(size=(300, 4)),
    ])
    union = JaxUnion.compute(points, n_points_min=10)
    samples = np.asarray(union.sample(10000, key=jax.random.PRNGKey(3)))
    assert np.all((samples >= 0.0) & (samples < 1.0))
    assert np.all(np.asarray(union.contains(samples)))
