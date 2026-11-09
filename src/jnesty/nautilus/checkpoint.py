"""HDF5 checkpointing for the JAX nautilus sampler.

The on-disk layout is versioned and independent of the reference
implementation's format.  It stores everything needed to continue a run
bit-for-bit on the same backend: shell data, bound geometry, neural-network
weights, counters and the host PRNG state.
"""

import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

__all__ = ['read_sampler', 'write_sampler']

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# generic (de)serialisation helpers
# ----------------------------------------------------------------------
def _to_numpy(value):
    if hasattr(value, '__array__'):
        return np.asarray(value)
    return value


def _write_value(group, key, value):
    value = _to_numpy(value)
    if isinstance(value, dict):
        _write_dict(group.create_group(key), value)
    elif isinstance(value, (list, tuple)):
        sub = group.create_group(key)
        sub.attrs['_type'] = 'list'
        sub.attrs['_len'] = len(value)
        for i, item in enumerate(value):
            _write_value(sub, str(i), item)
    elif isinstance(value, np.ndarray):
        if value.dtype == object:
            group.create_dataset(key, data=np.array(json.dumps(value.tolist())))
            group[key].attrs['_type'] = 'json'
        else:
            group.create_dataset(key, data=value)
    elif value is None:
        group.attrs[key] = '_none_'
    elif isinstance(value, (bool, np.bool_)):
        group.attrs[key] = bool(value)
    elif isinstance(value, (int, np.integer)):
        group.attrs[key] = int(value)
    elif isinstance(value, (float, np.floating)):
        group.attrs[key] = float(value)
    elif isinstance(value, str):
        group.attrs[key] = value
    else:
        group.attrs[key] = json.dumps(value)


def _write_dict(group, data):
    for key, value in data.items():
        _write_value(group, key, value)


def _read_value(group, key):
    if key in group.attrs:
        value = group.attrs[key]
        if isinstance(value, str) and value == '_none_':
            return None
        return value
    item = group[key]
    if isinstance(item, h5py.Group):
        if item.attrs.get('_type') == 'list':
            n = int(item.attrs['_len'])
            return [_read_value(item, str(i)) for i in range(n)]
        return _read_dict(item)
    if item.attrs.get('_type') == 'json':
        return json.loads(item[()])
    return np.asarray(item)


def _read_dict(group):
    data = {key: _read_value(group, key) for key in group.attrs}
    for key in group:
        if key not in data:
            data[key] = _read_value(group, key)
    return data


# ----------------------------------------------------------------------
# bound serialisation
# ----------------------------------------------------------------------
def _bound_to_dict(bound):
    from .bounds import (JaxEllipsoid, JaxNautilusBound, JaxNeuralBound,
                         JaxUnion, JaxUnitCube, JaxUnitCubeEllipsoidMixture)

    if isinstance(bound, JaxUnitCube):
        return {'kind': 'unit_cube', 'n_dim': bound.n_dim,
                'sample_key': getattr(bound, '_sample_key', None)}
    if isinstance(bound, JaxEllipsoid):
        return {'kind': 'ellipsoid', 'n_dim': bound.n_dim, 'c': bound.c,
                'A': bound.A, 'A_inv': bound.A_inv, 'B': bound.B,
                'B_inv': bound.B_inv,
                'sample_key': getattr(bound, '_sample_key', None)}
    if isinstance(bound, JaxUnitCubeEllipsoidMixture):
        return {'kind': 'mixture', 'n_dim': bound.n_dim,
                'dim_cube': np.asarray(bound.dim_cube),
                'sample_key': getattr(bound, '_sample_key', None),
                'ellipsoid': (None if bound.ellipsoid is None
                              else _bound_to_dict(bound.ellipsoid))}
    if isinstance(bound, JaxNeuralBound):
        if getattr(bound, 'neural_backend', 'jax') == 'reference':
            raise ValueError(
                "Cannot checkpoint a bound trained with "
                "neural_backend='reference': the reference emulator keeps "
                "scikit-learn networks that this format cannot serialise. "
                "Re-run with neural_backend='jax' to use checkpointing.")
        emulator = None
        if bound.emulator is not None:
            emulator = {
                'mean': np.asarray(bound.emulator.mean),
                'scale': np.asarray(bound.emulator.scale),
                'params': [[np.asarray(w), np.asarray(b)]
                           for w, b in bound.emulator.params],
            }
        return {'kind': 'neural', 'n_dim': bound.n_dim,
                'neural_backend': 'jax',
                'score_predict_min': bound.score_predict_min,
                'outer_bound': _bound_to_dict(bound.outer_bound),
                'emulator': emulator}
    if isinstance(bound, JaxNautilusBound):
        shift = None
        if bound.shift is not None:
            shift = {'periodic': np.asarray(bound.shift.periodic),
                     'centers': np.asarray(bound.shift.centers)}
        return {'kind': 'nautilus', 'n_dim': bound.n_dim, 'shift': shift,
                'neural_bounds': [_bound_to_dict(b)
                                  for b in bound.neural_bounds],
                'outer_bound': _bound_to_dict(bound.outer_bound),
                'points': np.asarray(bound.points),
                'sample_key': getattr(bound, '_sample_key', None),
                'n_sample': bound.n_sample, 'n_reject': bound.n_reject}
    if isinstance(bound, JaxUnion):
        return {
            'kind': 'union', 'n_dim': bound.n_dim,
            'enlarge_per_dim': bound.enlarge_per_dim,
            'n_points_min': bound.n_points_min,
            'block': np.asarray(bound.block),
            'log_v_all': np.asarray(bound.log_v_all),
            'bounds': [_bound_to_dict(b) for b in bound.bounds],
            'points_bounds': [np.asarray(p) for p in bound.points_bounds],
            'points': np.asarray(bound.points),
            'sample_key': getattr(bound, '_sample_key', None),
            'n_sample': bound.n_sample, 'n_reject': bound.n_reject,
            'backend': getattr(bound, 'backend', 'numpy'),
            'bound_class': ('ellipsoid' if bound.bound_class is JaxEllipsoid
                            else 'mixture'),
        }
    raise TypeError(f'Cannot serialise bound of type {type(bound)!r}')


def _dict_to_bound(data):
    from .bounds import (JaxEllipsoid, JaxNautilusBound, JaxNeuralBound,
                         JaxUnion, JaxUnitCube, JaxUnitCubeEllipsoidMixture,
                         PhaseShift)
    from .neural import Emulator

    kind = data['kind']
    if kind == 'unit_cube':
        bound = JaxUnitCube.compute(int(data['n_dim']))
        bound._sample_key = data.get('sample_key')
        return bound
    if kind == 'ellipsoid':
        bound = JaxEllipsoid()
        bound.n_dim = int(data['n_dim'])
        for key in ['c', 'A', 'A_inv', 'B', 'B_inv']:
            setattr(bound, key, data[key])
        bound._sample_key = data.get('sample_key')
        return bound
    if kind == 'mixture':
        bound = JaxUnitCubeEllipsoidMixture()
        bound.n_dim = int(data['n_dim'])
        bound.dim_cube = data['dim_cube']
        bound._sample_key = data.get('sample_key')
        bound.ellipsoid = (None if data.get('ellipsoid') is None
                           else _dict_to_bound(data['ellipsoid']))
        bound.cube = (JaxUnitCube.compute(int(np.sum(bound.dim_cube)))
                      if np.any(bound.dim_cube) else None)
        return bound
    if kind == 'neural':
        bound = JaxNeuralBound()
        bound.n_dim = int(data['n_dim'])
        bound.neural_backend = data.get('neural_backend', 'jax')
        bound.score_predict_min = float(data['score_predict_min'])
        bound.outer_bound = _dict_to_bound(data['outer_bound'])
        emulator = data.get('emulator')
        if emulator is None:
            bound.emulator = None
        else:
            params = [(np.asarray(w), np.asarray(b))
                      for w, b in emulator['params']]
            bound.emulator = Emulator(params, np.asarray(emulator['mean']),
                                      np.asarray(emulator['scale']))
        return bound
    if kind == 'nautilus':
        bound = JaxNautilusBound()
        bound.n_dim = int(data['n_dim'])
        shift = data.get('shift')
        if shift is None:
            bound.shift = None
        else:
            bound.shift = PhaseShift()
            bound.shift.periodic = np.asarray(shift['periodic'], dtype=int)
            bound.shift.centers = np.asarray(shift['centers'])
        bound.neural_bounds = [_dict_to_bound(b)
                               for b in data['neural_bounds']]
        bound.outer_bound = _dict_to_bound(data['outer_bound'])
        bound.n_sample = int(data['n_sample'])
        bound.n_reject = int(data['n_reject'])
        bound._sample_key = data.get('sample_key')
        bound.points = (np.asarray(data['points']) if data.get('points')
                        is not None else np.zeros((0, bound.n_dim)))
        bound.rng = None
        return bound
    if kind == 'union':
        bound = JaxUnion()
        bound.n_dim = int(data['n_dim'])
        bound.backend = data.get('backend', 'numpy')
        bound.enlarge_per_dim = float(data['enlarge_per_dim'])
        bound.n_points_min = int(data['n_points_min'])
        bound.block = np.atleast_1d(np.asarray(data['block'], dtype=bool))
        bound.log_v_all = np.asarray(data['log_v_all'], dtype=float)
        bound.bounds = [_dict_to_bound(b) for b in data['bounds']]
        bound.points_bounds = [np.asarray(p) for p in data['points_bounds']]
        bound.n_sample = int(data['n_sample'])
        bound.n_reject = int(data['n_reject'])
        bound._sample_key = data.get('sample_key')
        bound.cube = JaxUnitCube.compute(bound.n_dim)
        bound.bound_class = (JaxEllipsoid
                             if data['bound_class'] == 'ellipsoid'
                             else JaxUnitCubeEllipsoidMixture)
        bound.points = (np.asarray(data['points']) if data.get('points')
                        is not None else np.zeros((0, bound.n_dim)))
        bound._invalidate()
        return bound
    raise ValueError(f'Unknown bound kind {kind!r}')


# ----------------------------------------------------------------------
# sampler checkpoint
# ----------------------------------------------------------------------
_SAMPLER_SCALARS = [
    'n_dim', 'n_live', 'n_update', 'enlarge_per_dim', 'n_points_min',
    'split_threshold', 'n_networks', 'n_batch', 'n_like_new_bound',
    'vectorized', 'pass_dict', 'n_like', 'explored', 'fit_backend',
    'neural_backend', '_discard_exploration', 'n_update_iter',
    'n_like_iter',
]
_SAMPLER_ARRAYS = [
    'shell_n', 'shell_n_sample', 'shell_n_eff', 'shell_log_l_min',
    'shell_log_l', 'shell_log_v', 'shell_n_sample_exp', 'shell_end_exp',
    'points_t', 'shell_t', 'log_l_t',
]


def write_sampler(sampler, filepath, overwrite=False):
    """Write the sampler state to an HDF5 checkpoint."""
    if h5py is None:
        raise ImportError('Checkpointing requires h5py.')
    if sampler.blobs is not None:
        raise ValueError(
            'Checkpointing a run with likelihood blobs is not supported: '
            'the blobs are not part of the checkpoint format and would be '
            'lost. Resume is only implemented for blob-free runs.')
    filepath = Path(filepath)
    if filepath.suffix not in ['.h5', '.hdf5']:
        raise ValueError("File ending must '.h5' or '.hdf5'.")
    if filepath.exists():
        if not overwrite:
            raise RuntimeError(f'File {filepath!s} already exists.')
        filepath.unlink()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'x') as fstream:
        group = fstream.create_group('sampler')
        group.attrs['schema_version'] = SCHEMA_VERSION
        group.attrs['backend'] = 'jax'

        for key in _SAMPLER_SCALARS:
            _write_value(group, key, getattr(sampler, key))
        for key in _SAMPLER_ARRAYS:
            _write_value(group, key, getattr(sampler, key))
        _write_value(group, 'periodic',
                     None if sampler.periodic is None
                     else np.asarray(sampler.periodic))
        _write_value(group, 'neural_network_kwargs',
                     {k: v for k, v in sampler.neural_network_kwargs.items()
                      if isinstance(v, (int, float, str, bool))})

        state = sampler.rng.bit_generator.state
        group.attrs['rng_state'] = str(state['state']['state'])
        group.attrs['rng_inc'] = str(state['state']['inc'])
        group.attrs['rng_has_uint32'] = state['has_uint32']
        group.attrs['rng_uinteger'] = state['uinteger']

        points_group = group.create_group('shell_points')
        logl_group = group.create_group('shell_log_l_shells')
        for i in range(len(sampler.points)):
            _write_value(points_group, str(i), np.asarray(sampler.points[i]))
            _write_value(logl_group, str(i), np.asarray(sampler.log_l[i]))

        bounds_group = group.create_group('bounds')
        for i, bound in enumerate(sampler.bounds):
            _write_dict(bounds_group.create_group(str(i)),
                        _bound_to_dict(bound))


def read_sampler(sampler, filepath):
    """Restore sampler state from an HDF5 checkpoint.

    The sampler must already have been constructed with the same likelihood,
    prior and dimensionality; only the run state is restored.
    """
    if h5py is None:
        raise ImportError('Checkpointing requires h5py.')

    with h5py.File(Path(filepath), 'r') as fstream:
        group = fstream['sampler']
        version = int(group.attrs.get('schema_version', 0))
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f'Checkpoint schema {version} does not match the current '
                f'schema {SCHEMA_VERSION}.')

        for key in _SAMPLER_SCALARS:
            if key not in group.attrs and key not in group:
                # Optional field absent in an older checkpoint: keep
                # the value the constructor set.
                continue
            value = _read_value(group, key)
            if key in ('explored', '_discard_exploration'):
                value = bool(value)
            setattr(sampler, key, value)
        for key in _SAMPLER_ARRAYS:
            setattr(sampler, key, _read_value(group, key))

        sampler.n_dim = int(sampler.n_dim)
        sampler.n_live = int(sampler.n_live)
        sampler.n_batch = int(sampler.n_batch)

        state = sampler.rng.bit_generator.state
        state['state']['state'] = int(group.attrs['rng_state'])
        state['state']['inc'] = int(group.attrs['rng_inc'])
        state['has_uint32'] = int(group.attrs['rng_has_uint32'])
        state['uinteger'] = int(group.attrs['rng_uinteger'])
        sampler.rng.bit_generator.state = state

        points_group = group['shell_points']
        logl_group = group['shell_log_l_shells']
        sampler.points = [np.asarray(points_group[str(i)])
                          for i in range(len(points_group))]
        sampler.log_l = [np.asarray(logl_group[str(i)])
                         for i in range(len(logl_group))]

        bounds_group = group['bounds']
        sampler.bounds = [_dict_to_bound(_read_dict(bounds_group[str(i)]))
                          for i in range(len(bounds_group))]

    # Buffered proposals are restored (``bound.points`` together with the
    # Monte Carlo counters n_sample/n_reject), so a resumed run continues
    # with the same volume estimates as the interrupted one.  Blobs and
    # reference-emulator bounds are the state this format cannot cover;
    # writing such runs is refused in ``write_sampler``.
    return sampler
