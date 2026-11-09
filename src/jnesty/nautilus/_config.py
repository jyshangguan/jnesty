"""Runtime configuration for the JAX Nautilus engine.

The engine needs two process-global JAX settings that plain ``import jnesty``
must not change:

* float64 (x64) mode — volumes, covariances and the evidence are accumulated
  in float64; truncating them to float32 silently loses the precision the
  algorithm relies on;
* the on-disk compilation cache — the first run of a sampler compiles a
  handful of programs, which can cost more than the sampling itself, so
  compiled kernels are cached on disk per machine/backend.

:func:`initialize_runtime` applies both settings, is idempotent, and runs when
:mod:`jnesty.nautilus` is imported (explicit engine use) and when a
:class:`~jnesty.nautilus.sampler.JaxSampler` is constructed. Dynesty-style-only
sessions never import this module.

The cache directory defaults to ``$XDG_CACHE_HOME/nautilus_jax``; set
``NAUTILUS_JAX_CACHE_DIR`` to choose another location, or to an empty string
to disable the cache. (The name keeps compatibility with the development
tree this engine was ported from.)
"""

import os

import jax

__all__ = ['COMPILATION_CACHE_DIR', 'initialize_runtime']

_CONFIGURED = False
COMPILATION_CACHE_DIR = None


def _enable_compilation_cache():
    cache_dir = os.environ.get('NAUTILUS_JAX_CACHE_DIR')
    if cache_dir is None:
        base = os.environ.get('XDG_CACHE_HOME',
                               os.path.join(os.path.expanduser('~'), '.cache'))
        cache_dir = os.path.join(base, 'nautilus_jax')
    if not cache_dir:
        return None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update('jax_compilation_cache_dir', cache_dir)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.5)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    except Exception:  # pragma: no cover - older JAX without the flag
        return None
    return cache_dir


def initialize_runtime():
    """Enable x64 and the compilation cache; idempotent.

    Returns the compilation cache directory, or ``None`` when the cache is
    disabled or unsupported by the installed JAX.
    """
    global _CONFIGURED, COMPILATION_CACHE_DIR
    if _CONFIGURED:
        return COMPILATION_CACHE_DIR
    # Idempotent in itself; safe when x64 is already enabled.
    jax.config.update('jax_enable_x64', True)
    COMPILATION_CACHE_DIR = _enable_compilation_cache()
    _CONFIGURED = True
    return COMPILATION_CACHE_DIR