"""Tests for the optional rank-1 MVEE backends."""
import pytest

# The engine tests need the optional 'jnesty[nautilus]' dependencies.
pytest.importorskip('jnesty.nautilus')

# Everything here JIT-compiles JAX programs or runs end-to-end.
pytestmark = pytest.mark.slow


import numpy as np
import pytest

jax = pytest.importorskip('jax')

from nautilus.bounds.basic import (
    minimum_volume_enclosing_ellipsoid as ref_mvee)
from jnesty.nautilus import fitting
from jnesty.nautilus.fitting import fit_gmm, fit_mvee, normalise_backend

RANK1_BACKENDS = ('numpy_rank1', 'cpu_rank1')


def _points(n_points, n_dim, seed):
    """Deterministic, anisotropic point cloud."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, n_dim))
    points *= np.linspace(0.5, 2.0, n_dim)
    points += np.linspace(-0.3, 0.4, n_dim)
    return points


@pytest.mark.parametrize('backend', RANK1_BACKENDS)
@pytest.mark.parametrize('n_points,n_dim,seed', [(300, 2, 0), (100, 27, 1)])
def test_rank1_backends_match_reference(backend, n_points, n_dim, seed):
    points = _points(n_points, n_dim, seed)
    c_ref, a_ref, _ = ref_mvee(points)
    c, a, a_inv = (
        np.asarray(value) for value in fit_mvee(points, backend=backend))

    # The probe over 147 real MVEE inputs measured a maximum centre difference
    # of 1.15e-12 and a maximum relative A difference of 2.22e-11.  Keep a
    # margin for the different point clouds and platforms, but do not require
    # bitwise equality: the two inverse updates round differently.
    assert np.max(np.abs(c - c_ref)) <= 1e-8
    a_scale = max(float(np.max(np.abs(a_ref))), 1e-300)
    assert np.max(np.abs(a - a_ref)) / a_scale <= 1e-7

    # A and A_inv must remain mutually consistent and define the ellipsoid.
    assert a.shape == (n_dim, n_dim)
    assert a_inv.shape == (n_dim, n_dim)
    assert np.allclose(a, a.T, atol=1e-10 * a_scale)
    assert np.allclose(a_inv, a_inv.T, atol=1e-10 * np.max(np.abs(a_inv)))
    assert np.allclose(a @ a_inv, np.eye(n_dim), atol=1e-8)
    assert np.allclose(a_inv, np.linalg.inv(a), rtol=0,
                       atol=1e-8 * np.max(np.abs(a_inv)))

    radii = np.einsum('...i,ij,...j', points - c, a, points - c)
    assert np.max(radii) <= 1.0 + 1e-8
    assert np.max(radii) == pytest.approx(1.0, abs=1e-8)


def test_default_backend_is_still_numpy():
    points = _points(120, 3, 2)
    default = tuple(np.asarray(value) for value in fit_mvee(points))
    explicit = tuple(
        np.asarray(value) for value in fit_mvee(points, backend='numpy'))
    reference = ref_mvee(points)
    for got, want in zip(default, explicit):
        assert np.array_equal(got, want)
    for got, want in zip(default, reference):
        assert np.array_equal(got, want)


def test_cpu_rank1_program_cache_reuse():
    points = _points(60, 3, 3)
    key = (60, 3, np.dtype(points.dtype).str)
    fitting._CPU_RANK1_PROGRAMS.pop(key, None)
    program = fitting._cpu_rank1_program(60, 3, points.dtype)
    assert fitting._CPU_RANK1_PROGRAMS[key] is program
    assert fitting._cpu_rank1_program(60, 3, points.dtype) is program

    c, a, _ = (np.asarray(value)
               for value in fit_mvee(points, backend='cpu_rank1'))
    assert fitting._CPU_RANK1_PROGRAMS[key] is program

    # The same shape and dtype must reuse the cached program for new values.
    other = _points(60, 3, 4)
    c_other, a_other, _ = (
        np.asarray(value) for value in fit_mvee(other, backend='cpu_rank1'))
    assert fitting._CPU_RANK1_PROGRAMS[key] is program
    assert not np.array_equal(c, c_other)

    for values, (got_c, got_a) in ((points, (c, a)),
                                   (other, (c_other, a_other))):
        c_ref, a_ref, _ = ref_mvee(values)
        assert np.max(np.abs(got_c - c_ref)) <= 1e-8
        a_scale = max(float(np.max(np.abs(a_ref))), 1e-300)
        assert np.max(np.abs(got_a - a_ref)) / a_scale <= 1e-7

    # A different dtype is a different program.
    key32 = (60, 3, np.dtype(np.float32).str)
    fitting._CPU_RANK1_PROGRAMS.pop(key32, None)
    program32 = fitting._cpu_rank1_program(60, 3, np.float32)
    assert program32 is not program
    assert fitting._CPU_RANK1_PROGRAMS[key32] is program32


def test_cpu_rank1_ellipsoid_uses_host_postprocessing():
    from jnesty.nautilus.bounds import JaxEllipsoid

    points = _points(100, 4, 12)
    bound = JaxEllipsoid.compute(points, backend='cpu_rank1')
    reference = JaxEllipsoid.compute(points, backend='numpy')
    assert bound._log_v_host is not None
    np.testing.assert_allclose(bound.B, reference.B, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(bound.B_inv, reference.B_inv,
                               rtol=1e-6, atol=1e-8)
    assert bound.log_v == pytest.approx(reference.log_v, abs=1e-8)


def test_cpu_rank1_returns_default_device_arrays():
    points = _points(40, 2, 5)
    default_device = jax.devices()[0]
    for value in fit_mvee(points, backend='cpu_rank1'):
        assert set(value.devices()) == {default_device}


@pytest.mark.parametrize('backend,private_name', [
    ('numpy_rank1', '_minimum_volume_enclosing_ellipsoid_rank1'),
    ('cpu_rank1', '_minimum_volume_enclosing_ellipsoid_cpu_rank1')])
@pytest.mark.parametrize('failure', ['nonfinite', 'zeros', 'raises'])
def test_rank1_backend_falls_back(monkeypatch, backend, private_name, failure):
    points = _points(50, 3, 8)
    c_ref, a_ref, a_inv_ref = ref_mvee(points)
    n_dim = points.shape[1]

    def broken(_points):
        if failure == 'raises':
            raise RuntimeError('deliberate failure')
        if failure == 'zeros':
            return (np.zeros(n_dim), np.zeros((n_dim, n_dim)),
                    np.zeros((n_dim, n_dim)))
        return c_ref, a_ref * np.nan, a_inv_ref

    monkeypatch.setattr(fitting, private_name, broken)
    c, a, a_inv = (np.asarray(value)
                   for value in fit_mvee(points, backend=backend))
    assert np.array_equal(c, c_ref)
    assert np.array_equal(a, a_ref)
    assert np.array_equal(a_inv, a_inv_ref)


def test_backend_normalisation_accepts_rank1():
    assert normalise_backend(None) == 'numpy'
    assert normalise_backend('numpy') == 'numpy'
    assert normalise_backend('numpy_rank1') == 'numpy_rank1'
    assert normalise_backend('cpu_rank1') == 'cpu_rank1'
    assert normalise_backend('jax') == 'jax'


@pytest.mark.parametrize('backend', ['rank1', 'numpy_rank', 'cpu_rank', ''])
def test_invalid_backend_still_errors(backend):
    points = _points(20, 2, 6)
    with pytest.raises(ValueError, match='Unknown fit backend'):
        fit_mvee(points, backend=backend)
    with pytest.raises(ValueError, match='Unknown fit backend'):
        normalise_backend(backend)


@pytest.mark.parametrize('backend', RANK1_BACKENDS)
def test_rank1_backends_use_reference_gmm(backend):
    """The rank-1 choice affects MVEE only; GMM splits stay on NumPy."""
    points = _points(100, 2, 7)
    key = jax.random.PRNGKey(0)
    expected = fit_gmm(points, key=key, backend='numpy')
    got = fit_gmm(points, key=key, backend=backend)
    for got_value, expected_value in zip(got, expected):
        np.testing.assert_allclose(got_value, expected_value,
                                   rtol=0, atol=0)