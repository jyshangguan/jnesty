"""Default absl/TF C++ log-level suppression for JAX's noisy backend.

The messages JAX prints from its C++ side (XLA, absl) are controlled only by
the ``ABSL_MIN_LOG_LEVEL`` / ``TF_CPP_MIN_LOG_LEVEL`` environment variables.
``import jnesty`` sets both with ``setdefault`` *before* it imports jax, so a
plain import is quiet while an explicitly exported value still wins.

Every check runs in a subprocess (same pattern as
``tests/nautilus/test_import_side_effects.py``) so it observes the real
import-time environment instead of the already-initialised pytest process,
which imports jnesty and therefore sets these variables itself. Marked slow
because each check pays one JAX import.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

_LOGGING_VARS = ('ABSL_MIN_LOG_LEVEL', 'TF_CPP_MIN_LOG_LEVEL')


def _run_child_result(tmp_path, body, env_extra=None):
    """Run ``body`` in a fresh interpreter with an isolated cache home.

    Returns the completed process so a test can inspect stderr (e.g. for
    C++-side XLA warnings). The log-level variables are stripped from the
    inherited environment so the child really starts unset (the parent pytest
    process has them set by its own ``import jnesty``); ``env_extra`` puts the
    ones a test wants pre-set back in place.
    """
    script = tmp_path / 'child.py'
    script.write_text(body)
    env = dict(os.environ)
    env['XDG_CACHE_HOME'] = str(tmp_path / 'cachehome')
    for name in _LOGGING_VARS:
        env.pop(name, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(script)], env=env,
                          capture_output=True, text=True)


def _run_child(tmp_path, body, env_extra=None):
    """Run ``body`` like :func:`_run_child_result`, returning only stdout."""
    result = _run_child_result(tmp_path, body, env_extra)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_plain_import_sets_default_log_levels(tmp_path):
    out = _run_child(tmp_path, """\
import os

# The defaults must be applied by the package itself, not inherited.
assert 'ABSL_MIN_LOG_LEVEL' not in os.environ
assert 'TF_CPP_MIN_LOG_LEVEL' not in os.environ

import jnesty
import jax

assert os.environ['ABSL_MIN_LOG_LEVEL'] == '3', os.environ
assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '3', os.environ
# The env block must not disturb JAX itself (x64 stays off on plain import).
assert jax.config.jax_enable_x64 is False
print('DEFAULTS_OK')
""")
    assert 'DEFAULTS_OK' in out


def test_user_environment_overrides_defaults(tmp_path):
    out = _run_child(tmp_path, """\
import os

import jnesty

# setdefault: a value exported by the user must not be replaced.
assert os.environ['ABSL_MIN_LOG_LEVEL'] == '0', os.environ
assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '0', os.environ
print('OVERRIDE_OK')
""", env_extra={'ABSL_MIN_LOG_LEVEL': '0', 'TF_CPP_MIN_LOG_LEVEL': '0'})
    assert 'OVERRIDE_OK' in out


def test_first_submodule_import_still_sets_defaults(tmp_path):
    out = _run_child(tmp_path, """\
import os

# No top-level ``import jnesty`` first: the package __init__ runs anyway.
import jnesty.dynesty.core

assert os.environ['ABSL_MIN_LOG_LEVEL'] == '3', os.environ
assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '3', os.environ
print('SUBMODULE_OK')
""")
    assert 'SUBMODULE_OK' in out


def test_persistent_cache_load_is_quiet(tmp_path):
    """Loading an executable from the persistent cache must be silent.

    Contract: importing the package before jax sets the C++ log threshold to
    FATAL while jaxlib loads, and that threshold cannot be changed afterwards,
    so the ``W1110 ... pjrt_executable.cc`` line XLA prints whenever an
    executable is loaded from the persistent compilation cache stays hidden.
    The first child populates a shared cache without importing the package;
    the second child imports jnasty first, then hits that same cache.
    """
    cache = tmp_path / 'jaxcache'
    populate = f"""\
import jax, jax.numpy as jnp

jax.config.update('jax_compilation_cache_dir', {str(cache)!r})
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
f = jax.jit(lambda x: (x * 2).sum())
f(jnp.arange(10.0))
"""
    hit = f"""\
import jnesty
import jax, jax.numpy as jnp

jax.config.update('jax_compilation_cache_dir', {str(cache)!r})
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
f = jax.jit(lambda x: (x * 2).sum())
assert float(f(jnp.arange(10.0))) == 90.0
print('HIT_OK')
"""
    populated = _run_child_result(tmp_path, populate)
    assert populated.returncode == 0, populated.stdout + populated.stderr
    assert any(cache.iterdir()), 'populating child wrote no cache entry'

    result = _run_child_result(tmp_path, hit)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'HIT_OK' in result.stdout
    assert 'pjrt_executable' not in result.stderr, result.stderr
