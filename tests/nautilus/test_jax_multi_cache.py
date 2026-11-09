"""Exact-semantics tests for the packed multi-bound argument cache.

The cache must change only *how often* the immutable padded device arguments
are rebuilt.  Every mask it feeds the kernel has to stay exactly equal to the
per-bound ``contains`` results, including for bounds that are not
``JaxNautilusBound`` instances, for empty bound lists, and for fresh points.
"""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from jnesty.nautilus import JaxNautilusBound, JaxUnitCube
from jnesty.nautilus import bounds as jb

# Small but real networks: the tests exercise the same code path as production
# without paying for the default (100, 50, 20) training budget.
_NETWORK_KWARGS = dict(hidden=(16, 8), max_epochs=200, patience=5)


def _make_bound(seed, center=0.5):
    """Compute a small real :class:`JaxNautilusBound` on Gaussian data."""
    rng = np.random.default_rng(seed)
    points = rng.normal(center, 0.12, size=(512, 2))
    log_l = -0.5 * np.sum(((points - center) / 0.12) ** 2, axis=1)
    log_l_min = float(np.quantile(log_l, 0.5))
    return JaxNautilusBound.compute(
        points, log_l, log_l_min, np.log(0.05), n_networks=2,
        neural_network_kwargs=_NETWORK_KWARGS,
        key=jax.random.PRNGKey(seed))


@pytest.fixture(scope='module')
def real_bounds():
    """Two independently computed bounds, so a changed tuple is meaningful."""
    return _make_bound(0), _make_bound(1, center=0.55)


def _masks(bounds, points):
    return np.asarray(jb._multi_bound_contains(bounds, points))


def _expected(bounds, points):
    return np.stack([np.asarray(b.contains(points)) for b in bounds], axis=-1)


def test_repeated_calls_return_exactly_equal_masks(real_bounds):
    first, second = real_bounds
    bounds = [JaxUnitCube.compute(2), first, second]
    rng = np.random.default_rng(11)
    points = rng.uniform(size=(257, 2))

    mask_1 = _masks(bounds, points)
    mask_2 = _masks(bounds, points)

    assert mask_1.dtype == np.bool_
    assert mask_1.shape == (len(points), len(bounds))
    assert np.array_equal(mask_1, mask_2)
    # Column i is exactly bounds[i].contains(points).
    assert np.array_equal(mask_1, _expected(bounds, points))


def test_pack_is_reused_for_identical_bound_tuple(real_bounds):
    first, second = real_bounds
    jb._clear_multi_bound_pack_cache()
    chunk = [(0, first), (1, second)]
    rng = np.random.default_rng(12)
    points = rng.uniform(size=(200, 2))

    mask_1 = np.asarray(jb._multi_bound_chunk(chunk, points))
    stats = dict(jb._MULTI_BOUND_PACK_STATS)
    assert stats['builds'] == 1
    assert stats['misses'] == 1
    assert stats['hits'] == 0

    mask_2 = np.asarray(jb._multi_bound_chunk(chunk, points))
    stats = dict(jb._MULTI_BOUND_PACK_STATS)
    assert stats['builds'] == 1
    assert stats['misses'] == 1
    assert stats['hits'] == 1
    assert np.array_equal(mask_1, mask_2)

    # Different point values with the same bound tuple still reuse the pack.
    mask_3 = np.asarray(jb._multi_bound_chunk(chunk, points[:63]))
    stats = dict(jb._MULTI_BOUND_PACK_STATS)
    assert stats['builds'] == 1
    assert stats['hits'] == 2
    assert mask_3.shape == (63, len(chunk))


def test_changed_bound_tuple_gets_a_new_pack(real_bounds):
    first, second = real_bounds
    jb._clear_multi_bound_pack_cache()
    rng = np.random.default_rng(13)
    points = rng.uniform(size=(100, 2))

    jb._multi_bound_chunk([(0, first)], points)
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 1

    jb._multi_bound_chunk([(0, second)], points)
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 2

    # Order is part of the packed argument layout, so it is a new key too.
    jb._multi_bound_chunk([(0, second), (1, first)], points)
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 3
    assert jb._MULTI_BOUND_PACK_STATS['misses'] == 3


def test_point_values_are_not_cached(real_bounds):
    first, second = real_bounds
    jb._clear_multi_bound_pack_cache()
    bounds = [JaxUnitCube.compute(2), first, second]
    inside = np.asarray(first.sample(64, key=jax.random.PRNGKey(123)))
    outside = np.full((64, 2), 2.0)

    mask_inside = _masks(bounds, inside)
    mask_outside = _masks(bounds, outside)

    assert np.array_equal(mask_inside, _expected(bounds, inside))
    assert np.array_equal(mask_outside, _expected(bounds, outside))
    # Same cached pack, different point-dependent masks.
    assert np.all(mask_inside[:, 1])
    assert not np.any(mask_outside)
    assert not np.array_equal(mask_inside, mask_outside)
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 1
    assert jb._MULTI_BOUND_PACK_STATS['hits'] >= 1


def test_empty_and_non_nautilus_bounds():
    rng = np.random.default_rng(14)
    points = rng.uniform(size=(32, 2))
    cube = JaxUnitCube.compute(2)

    empty = np.asarray(jb._multi_bound_contains([], points))
    assert empty.shape == (len(points), 0)
    assert empty.dtype == np.bool_

    only_cube = _masks([cube], points)
    assert np.array_equal(only_cube[:, 0], np.asarray(cube.contains(points)))

    mixed = _masks([cube, JaxUnitCube.compute(2)], points)
    assert np.array_equal(mixed, _expected([cube, cube], points))


def test_outer_geometry_mutation_invalidates_the_pack(real_bounds):
    first, _ = real_bounds
    # Use a directly computed multi-ellipsoid union so the split is a real
    # geometry change (the fixture's Gaussian outer bound need not be
    # splittable).
    rng = np.random.default_rng(21)
    cloud = np.concatenate([rng.normal(0.25, 0.05, size=(300, 2)),
                            rng.normal(0.75, 0.05, size=(300, 2))])
    union = jb.JaxUnion.compute(cloud, n_points_min=20, unit=True)
    first.outer_bound = union
    jb._clear_multi_bound_pack_cache()
    chunk = [(0, first)]
    points = np.random.default_rng(22).uniform(size=(256, 2))

    mask_before = np.asarray(jb._multi_bound_chunk(chunk, points))
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 1
    token_before = union._geometry_token

    split = False
    for attempt in range(3):
        if union.split(key=jax.random.PRNGKey(attempt)):
            split = True
            break
    assert split
    assert union._geometry_token is not token_before

    mask_after = np.asarray(jb._multi_bound_chunk(chunk, points))
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 2
    assert not np.array_equal(mask_before, mask_after)
    # The rebuilt pack must still agree exactly with the per-bound result.
    assert np.array_equal(mask_after[:, 0], np.asarray(first.contains(points)))


def test_neural_tensor_replacement_invalidates_the_pack(real_bounds):
    first, _ = real_bounds
    jb._clear_multi_bound_pack_cache()
    chunk = [(0, first)]
    points = np.random.default_rng(23).uniform(size=(128, 2))

    mask_before = np.asarray(jb._multi_bound_chunk(chunk, points))
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 1
    emulator = first.neural_bounds[0].emulator
    original_params = emulator.params
    try:
        # Replacing the arrays is what a deferred calibration/update would do;
        # adding zero keeps the predictions exactly unchanged.
        emulator.params = jax.tree_util.tree_map(
            lambda x: x + 0.0, original_params)
        mask_updated = np.asarray(jb._multi_bound_chunk(chunk, points))
        assert jb._MULTI_BOUND_PACK_STATS['builds'] == 2
        assert np.array_equal(mask_before, mask_updated)
    finally:
        emulator.params = original_params

    mask_restored = np.asarray(jb._multi_bound_chunk(chunk, points))
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == 3
    assert np.array_equal(mask_before, mask_restored)


def test_lru_is_bounded_and_evicts_oldest(real_bounds):
    first, second = real_bounds
    jb._clear_multi_bound_pack_cache()
    points = np.random.default_rng(24).uniform(size=(64, 2))
    chunks = [
        [(0, first)],
        [(0, second)],
        [(0, first), (1, second)],
        [(0, second), (1, first)],
        [(0, first), (1, first)],
        [(0, second), (1, second)],
    ]
    for chunk in chunks:
        jb._multi_bound_chunk(chunk, points)
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == len(chunks)
    assert len(jb._MULTI_BOUND_PACK_CACHE) == jb._MULTI_BOUND_PACK_CACHE_SIZE
    assert jb._MULTI_BOUND_PACK_CACHE_SIZE == 4

    # The first key has been evicted, so this is a build, not a hit.
    jb._multi_bound_chunk(chunks[0], points)
    assert jb._MULTI_BOUND_PACK_STATS['builds'] == len(chunks) + 1
    assert jb._MULTI_BOUND_PACK_STATS['misses'] == len(chunks) + 1
    assert len(jb._MULTI_BOUND_PACK_CACHE) == jb._MULTI_BOUND_PACK_CACHE_SIZE
