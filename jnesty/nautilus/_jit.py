"""Shape-bucketed JIT helpers.

JAX compiles a specialised executable for every array shape it sees.  A
nautilus run feeds arrays whose length changes constantly (shell sizes, point
counts), so eager dispatch would recompile on every call.  Rounding the leading
dimension up to a power of two and padding with zeros gives a handful of
compiled programs instead; the padded rows are sliced off afterwards.
"""

import jax
import jax.numpy as jnp

__all__ = ['bucket_size', 'clear_cache', 'padded_eval', 'stack_eval']

_CACHE = {}


def bucket_size(n_points, minimum=64):
    """Round ``n_points`` up to a power of two, at least ``minimum``."""
    n = max(int(n_points), int(minimum))
    return 1 << (n - 1).bit_length()


def clear_cache():
    """Drop every cached compiled program (mostly useful in tests)."""
    _CACHE.clear()


def _compiled(name, shapes, fn):
    key = (name, shapes)
    program = _CACHE.get(key)
    if program is None:
        program = jax.jit(fn)
        _CACHE[key] = program
    return program


def _arg_shapes(args):
    return str(jax.tree_util.tree_map(
        lambda x: tuple(getattr(x, 'shape', ())), args))


def padded_eval(name, fn, points, *args, minimum=64):
    """Call ``fn(*args, padded_points)`` through a bucketed compiled program.

    ``fn`` must accept the points array as its last positional argument.
    """
    points = jnp.asarray(points)
    if points.ndim == 1:
        return fn(*args, points)

    n = points.shape[0]
    n_bucket = bucket_size(n, minimum)
    # The key must use the *padded* length: the program is compiled for that
    # shape, so keying on the original length would compile a new executable
    # for every distinct point count (which is what a sampler produces
    # constantly).
    shapes = (_arg_shapes(args), (n_bucket,) + tuple(points.shape[1:]))
    program = _compiled(name, shapes, fn)

    if n_bucket != n:
        # ``jnp.pad`` allocates once; zeros-plus-concatenate allocates twice and
        # measured ~10 % of a 2-D run on its own (the padded rows are discarded,
        # so their value is irrelevant).
        widths = ((0, n_bucket - n),) + ((0, 0),) * (points.ndim - 1)
        points = jnp.pad(points, widths)
    return program(*args, points)[:n]


def stack_eval(name, fn, *args, **kwargs):
    """Compile ``fn`` once per combination of argument shapes."""
    shapes = _arg_shapes(args)
    return _compiled(name, shapes, fn)(*args, **kwargs)