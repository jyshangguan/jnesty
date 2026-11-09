"""Exact-semantics tests for MVEE reuse in ``JaxNautilusBound`` rebuilds.

The optimization may only change *how many* identical minimum-volume fits are
performed.  A reused outer ellipsoid must be bit-for-bit the geometry the old
path fitted, and the sampler stream must be unchanged.  A point set that does
not match the union's stored ``points_bounds`` must take the old path.
"""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


from contextlib import contextmanager
import json

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from jnesty.nautilus import JaxNautilusBound, JaxSampler
from jnesty.nautilus import bounds as jb

# Small but real ensembles: the production code path without the default
# (100, 50, 20) training budget.
_NETWORK_KWARGS = dict(hidden=(16, 8), max_epochs=100, patience=5)


def _gaussian_data(seed=0):
    """A compact 2-D problem with a well-defined above-threshold core."""
    rng = np.random.default_rng(seed)
    points = rng.normal(0.5, 0.12, size=(512, 2))
    log_l = -0.5 * np.sum(((points - 0.5) / 0.12) ** 2, axis=1)
    log_l_min = float(np.quantile(log_l, 0.5))
    return points, log_l, log_l_min


def _build_bound():
    points, log_l, log_l_min = _gaussian_data()
    return JaxNautilusBound.compute(
        points, log_l, log_l_min, np.log(0.05), n_networks=2,
        neural_network_kwargs=_NETWORK_KWARGS,
        key=jax.random.PRNGKey(0))


@contextmanager
def _no_reuse():
    """Explicitly restore the pre-optimization code path for one build."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jb, '_region_ellipsoid_reusable',
                      lambda *args, **kwargs: False)
        original = jb.JaxUnitCubeEllipsoidMixture.compute

        @classmethod
        def without_initial(cls, points, enlarge_per_dim=1.1, rng=None,
                            backend='numpy', ellipsoid=None):
            return original(points, enlarge_per_dim=enlarge_per_dim,
                            rng=rng, backend=backend)

        patch.setattr(jb.JaxUnitCubeEllipsoidMixture, 'compute',
                      without_initial)
        yield


def _count_mvee(monkeypatch):
    """Record every ``fit_mvee`` call made while the fixture is active."""
    calls = []
    original = jb.fit_mvee

    def counted(points, backend='numpy'):
        calls.append((np.asarray(points).shape, backend))
        return original(points, backend=backend)

    monkeypatch.setattr(jb, 'fit_mvee', counted)
    return calls


def _assert_ellipsoid_equal(actual, expected, label):
    for name in ('c', 'A', 'A_inv', 'B', 'B_inv'):
        assert np.array_equal(np.asarray(getattr(actual, name)),
                              np.asarray(getattr(expected, name))), (
            label, name)
    assert float(actual.log_v) == float(expected.log_v), (label, 'log_v')
    assert actual.backend == expected.backend, (label, 'backend')


def _assert_emulator_equal(actual, expected, label):
    assert actual.n_networks == expected.n_networks, label
    assert actual.neural_backend == expected.neural_backend, label
    assert actual.score_predict_min == expected.score_predict_min, label
    for layer, (actual_layer, expected_layer) in enumerate(
            zip(actual.emulator.params, expected.emulator.params)):
        for part, (actual_part, expected_part) in enumerate(
                zip(actual_layer, expected_layer)):
            assert np.array_equal(np.asarray(actual_part),
                                  np.asarray(expected_part)), (
                label, 'params', layer, part)
    assert np.array_equal(np.asarray(actual.emulator.mean),
                          np.asarray(expected.emulator.mean)), (label, 'mean')
    assert np.array_equal(np.asarray(actual.emulator.scale),
                          np.asarray(expected.emulator.scale)), (label, 'scale')


def _assert_unified_equal(actual, expected, label):
    actual_u = actual.unified()
    expected_u = expected.unified()
    assert len(actual_u) == len(expected_u), label
    for index in range(len(actual_u)):
        if isinstance(actual_u[index], float):
            assert actual_u[index] == expected_u[index], (label, index)
        else:
            assert np.array_equal(np.asarray(actual_u[index]),
                                  np.asarray(expected_u[index])), (
                label, index)


def _assert_bound_equal(actual, expected):
    """Every geometric, neural, threshold and key-order-visible value."""
    assert actual.n_dim == expected.n_dim
    assert (actual.shift is None) == (expected.shift is None)
    assert actual.n_ell == expected.n_ell
    assert actual.n_net == expected.n_net
    assert len(actual.neural_bounds) == len(expected.neural_bounds)
    assert len(actual.outer_bound.bounds) == len(expected.outer_bound.bounds)

    for index, (actual_neural, expected_neural) in enumerate(
            zip(actual.neural_bounds, expected.neural_bounds)):
        label = f'neural[{index}]'
        _assert_ellipsoid_equal(actual_neural.outer_bound,
                                expected_neural.outer_bound, label)
        _assert_emulator_equal(actual_neural, expected_neural, label)

    for index, (actual_bound, expected_bound) in enumerate(
            zip(actual.outer_bound.bounds, expected.outer_bound.bounds)):
        label = f'outer[{index}]'
        assert np.array_equal(np.asarray(actual_bound.dim_cube),
                              np.asarray(expected_bound.dim_cube)), label
        _assert_unified_equal(actual_bound, expected_bound, label)

    for index, (actual_points, expected_points) in enumerate(
            zip(actual.outer_bound.points_bounds,
                expected.outer_bound.points_bounds)):
        assert np.array_equal(np.asarray(actual_points),
                              np.asarray(expected_points)), (
            'outer points', index)
    assert np.array_equal(np.asarray(actual.outer_bound.log_v_all),
                          np.asarray(expected.outer_bound.log_v_all))
    assert np.array_equal(np.asarray(actual.outer_bound.block),
                          np.asarray(expected.outer_bound.block))


def test_reuse_removes_only_the_duplicate_mvee_fits(monkeypatch):
    calls = _count_mvee(monkeypatch)
    matches = []
    original = jb._region_ellipsoid_reusable

    def spy(*args):
        result = original(*args)
        matches.append(bool(result))
        return result

    monkeypatch.setattr(jb, '_region_ellipsoid_reusable', spy)

    bound = _build_bound()
    reuse_calls = len(calls)
    # This unimodal cloud has one neural region and all its above-threshold
    # rows are exactly the region's stored point set.
    assert len(matches) == len(bound.neural_bounds)
    assert sum(matches) == len(bound.neural_bounds)
    # The shared object is what proves the fit was reused, not copied.
    assert (bound.neural_bounds[0].outer_bound
            is bound.outer_bound.bounds[0].ellipsoid)

    calls.clear()
    with _no_reuse():
        baseline = _build_bound()
    baseline_calls = len(calls)
    assert len(baseline.neural_bounds) == len(bound.neural_bounds)
    # One avoided fit per neural region plus the mixture's initial fit.
    assert baseline_calls == reuse_calls + len(bound.neural_bounds) + 1


def test_reused_bound_is_exactly_the_nonreusing_bound(monkeypatch):
    reused = _build_bound()
    with _no_reuse():
        baseline = _build_bound()

    _assert_bound_equal(reused, baseline)

    points = np.random.default_rng(31).uniform(size=(777, 2))
    assert np.array_equal(np.asarray(reused.contains(points)),
                          np.asarray(baseline.contains(points)))
    assert np.array_equal(np.asarray(reused.outer_bound.contains(points)),
                          np.asarray(baseline.outer_bound.contains(points)))
    assert float(reused.log_v) == float(baseline.log_v)

    key = jax.random.PRNGKey(71)
    assert np.array_equal(np.asarray(reused.sample(512, key=key)),
                          np.asarray(baseline.sample(512, key=key)))


def test_seeded_sampler_history_and_evidence_are_identical(monkeypatch):
    matches = []
    original = jb._region_ellipsoid_reusable

    def spy(*args):
        result = original(*args)
        matches.append(bool(result))
        return result

    monkeypatch.setattr(jb, '_region_ellipsoid_reusable', spy)

    def run():
        def prior(u):
            return u

        def likelihood(x):
            return -0.5 * jnp.sum((x - 0.5) ** 2) / 0.1 ** 2

        sampler = JaxSampler(
            prior, likelihood, n_dim=2, n_live=100, n_batch=100,
            n_networks=2, seed=0, neural_network_kwargs=_NETWORK_KWARGS)
        finished = sampler.run(f_live=0.2, n_eff=0, n_like_max=1200)
        return sampler, finished

    reused, reused_finished = run()
    assert any(matches)
    with _no_reuse():
        baseline, baseline_finished = run()

    assert reused_finished == baseline_finished
    assert reused.n_like == baseline.n_like
    assert reused.log_z == baseline.log_z
    assert reused.n_eff == baseline.n_eff
    assert (json.dumps(reused.history, sort_keys=True)
            == json.dumps(baseline.history, sort_keys=True))


def test_mixture_and_union_accept_an_exact_initial_ellipsoid(monkeypatch):
    rng = np.random.default_rng(41)
    points = rng.normal(0.5, 0.12, size=(512, 2))
    ellipsoid = jb.JaxEllipsoid.compute(points, enlarge_per_dim=1.1)
    snapshot = {name: np.array(np.asarray(getattr(ellipsoid, name)),
                               copy=True)
                for name in ('c', 'A', 'A_inv', 'B', 'B_inv')}
    log_v = float(ellipsoid.log_v)

    calls = _count_mvee(monkeypatch)
    fresh = jb.JaxUnitCubeEllipsoidMixture.compute(points)
    fresh_calls = len(calls)
    calls.clear()
    reused = jb.JaxUnitCubeEllipsoidMixture.compute(
        points, ellipsoid=ellipsoid)
    assert len(calls) == fresh_calls - 1
    _assert_unified_equal(reused, fresh, 'mixture')
    assert reused.ellipsoid is ellipsoid

    # The supplied object must not have been modified in any way.
    for name, value in snapshot.items():
        assert np.array_equal(np.asarray(getattr(ellipsoid, name)), value)
    assert float(ellipsoid.log_v) == log_v

    union_fresh = jb.JaxUnion.compute(
        points, n_points_min=20, bound_class=jb.JaxUnitCubeEllipsoidMixture)
    union_reused = jb.JaxUnion.compute(
        points, n_points_min=20, bound_class=jb.JaxUnitCubeEllipsoidMixture,
        initial_ellipsoid=ellipsoid)
    assert union_reused.bounds[0].ellipsoid is ellipsoid
    assert np.array_equal(np.asarray(union_reused.bounds[0].dim_cube),
                          np.asarray(union_fresh.bounds[0].dim_cube))
    _assert_unified_equal(union_reused.bounds[0], union_fresh.bounds[0],
                          'union mixture')

    test = rng.uniform(size=(333, 2))
    assert np.array_equal(np.asarray(reused.contains(test)),
                          np.asarray(fresh.contains(test)))


def test_content_check_is_exact_and_fallback_uses_normal_path(monkeypatch):
    points, log_l, log_l_min = _gaussian_data()
    select = np.ones(len(points), dtype=bool)
    region = points[log_l >= log_l_min]
    assert jb._region_ellipsoid_reusable(
        points, log_l, log_l_min, select, region)

    # A moved value, an extra row, a reordered row and a dropped row are all
    # content differences, so reuse must be refused.
    moved = region.copy()
    moved[0, 0] = np.nextafter(moved[0, 0], np.inf)
    assert not jb._region_ellipsoid_reusable(
        points, log_l, log_l_min, select, moved)
    assert not jb._region_ellipsoid_reusable(
        points, log_l, log_l_min, select, np.vstack([region, region[:1]]))
    assert not jb._region_ellipsoid_reusable(
        points, log_l, log_l_min, select, region[::-1])
    assert not jb._region_ellipsoid_reusable(
        points, log_l, log_l_min, select, region[:-1])
    dropped = select.copy()
    dropped[np.flatnonzero(log_l >= log_l_min)[0]] = False
    assert not jb._region_ellipsoid_reusable(
        points, log_l, log_l_min, dropped, region)

    # With the check forced false, the neural bounds must receive no
    # precomputed outer bound, and the result must still be the old geometry.
    seen = []
    original = jb.JaxNeuralBound.compute

    def spy(cls, *args, **kwargs):
        seen.append(kwargs.get('outer_bound'))
        return original(*args, **kwargs)

    monkeypatch.setattr(jb.JaxNeuralBound, 'compute', classmethod(spy))
    with _no_reuse():
        fallback = _build_bound()
    assert seen and all(item is None for item in seen)
    _assert_bound_equal(fallback, _build_bound())