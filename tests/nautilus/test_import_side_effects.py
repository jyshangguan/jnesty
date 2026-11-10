"""Import-order and side-effect guarantees for the merged package.

Every check runs in a subprocess so the parent pytest process keeps its own
JAX configuration. Marked slow because each check pays one JAX import.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def _run_child(tmp_path, body, needs_engine=False):
    """Run ``body`` in a fresh interpreter with an isolated cache home."""
    if needs_engine:
        # Proxy for "the jnesty[nautilus] extra is installed": the child
        # cannot skip, so skip the whole check here instead.
        pytest.importorskip('optax')
    script = tmp_path / 'child.py'
    script.write_text(body)
    env = dict(os.environ)
    env['XDG_CACHE_HOME'] = str(tmp_path / 'cachehome')
    result = subprocess.run([sys.executable, str(script)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _cache_path(tmp_path):
    return tmp_path / 'cachehome' / 'nautilus_jax'


def test_plain_import_is_side_effect_free(tmp_path):
    out = _run_child(tmp_path, """\
import os

import jax
import jnesty
from jnesty import (NestedSampler, DynamicNestedSampler, NautilusSampler,
                   Results, save_results, load_results)

# Canonical deep imports.
from jnesty.dynesty_sampler import NestedSampler as legacy_static
from jnesty.dynesty_sampler import DynamicNestedSampler as legacy_dynamic
from jnesty.dynesty._dynamic import (kish_ess,
                                     estimate_fresh_batch_size_from_memory,
                                     generate_fresh_batch_points)
from jnesty.dynesty.core import run_nested_sampling, WhileLoopNSConfig
import jnesty.dynesty.core as _core
assert _core.run_nested_sampling is run_nested_sampling
from jnesty.dynesty_sampler import NestedSampler as canonical

assert jax.config.jax_enable_x64 is False
cache = os.path.join(os.environ['XDG_CACHE_HOME'], 'nautilus_jax')
assert not os.path.exists(cache)
assert NautilusSampler is not None
print('PLAIN_OK')
""")
    assert 'PLAIN_OK' in out


def test_adapter_class_import_is_side_effect_free(tmp_path):
    out = _run_child(tmp_path, """\
import os

import jax
from jnesty import NautilusSampler

assert jax.config.jax_enable_x64 is False
cache = os.path.join(os.environ['XDG_CACHE_HOME'], 'nautilus_jax')
assert not os.path.exists(cache)
print('ADAPTER_OK')
""")
    assert 'ADAPTER_OK' in out


def test_engine_import_enables_x64_and_cache(tmp_path):
    out = _run_child(tmp_path, """\
import os

import jax
import jnesty.nautilus

assert jax.config.jax_enable_x64 is True
cache = os.path.join(os.environ['XDG_CACHE_HOME'], 'nautilus_jax')
assert os.path.isdir(cache)
print('ENGINE_OK')
""", needs_engine=True)
    assert 'ENGINE_OK' in out
    assert _cache_path(tmp_path).is_dir()


def test_construction_enables_x64_after_dynesty_use(tmp_path):
    out = _run_child(tmp_path, """\
import jax
import jax.numpy as jnp
import jnesty
from jnesty import NautilusSampler, NestedSampler

assert jax.config.jax_enable_x64 is False

# Dynesty-style sampler first: construction works in float32 mode.
static = NestedSampler(lambda x: -0.5 * x[0] ** 2, lambda u: u, ndim=2,
                       verbose=False)
assert jax.config.jax_enable_x64 is False

# Constructing the Nautilus sampler flips the process-global setting.
sampler = NautilusSampler(lambda x: -0.5 * jnp.sum(x ** 2), lambda u: u,
                          ndim=2, nlive=20, n_networks=0, device='cpu')
assert jax.config.jax_enable_x64 is True
print('MIXED_OK')
""", needs_engine=True)
    assert 'MIXED_OK' in out


def test_missing_optional_extra_gives_install_hint(tmp_path):
    out = _run_child(tmp_path, """\
import sys

# Simulate an environment without the optional engine dependency.
sys.modules['optax'] = None

import jax.numpy as jnp
from jnesty import NautilusSampler

try:
    NautilusSampler(lambda x: -0.5 * jnp.sum(x ** 2), lambda u: u,
                    ndim=2, nlive=20)
except ImportError as exc:
    assert 'jnesty[nautilus]' in str(exc), str(exc)
    assert 'optax' in str(exc), str(exc)
else:
    raise SystemExit('expected an ImportError with the install hint')
print('HINT_OK')
""")
    assert 'HINT_OK' in out


def test_top_level_nautilus_still_resolves_upstream(tmp_path):
    out = _run_child(tmp_path, """\
import nautilus
import jnesty.nautilus

# The external CPU reference package must not be shadowed.
assert not hasattr(nautilus, 'JaxSampler')
assert nautilus.__file__ != jnesty.nautilus.__file__
print('UPSTREAM_OK')
""", needs_engine=True)
    assert 'UPSTREAM_OK' in out
