"""JAX/GPU implementation of the Nautilus importance nested sampler.

This subpackage is *our* JAX port of the Nautilus algorithm. The CPU
reference implementation lives in the external ``nautilus`` package
(nautilus-sampler), which this engine uses for a handful of support routines
(NumPy minimum-volume-ellipsoid fitting, the optional reference emulator and
``nautilus.Prior`` compatibility) and as a parity oracle in the test suite;
it is not re-exported here.

Importing this module configures JAX for the engine (float64 mode and the
on-disk compilation cache) via
:func:`jnesty.nautilus._config.initialize_runtime`. Plain ``import jnesty``
never imports this package, so Dynesty-style-only sessions are unaffected.

Examples
--------
>>> from jnesty.nautilus import JaxSampler
>>> sampler = JaxSampler(prior, likelihood, n_dim=3, n_live=500, seed=0)
>>> sampler.run(verbose=True)
>>> points, log_w, log_l = sampler.posterior()
"""

from . import _config
from .bounds import (JaxEllipsoid, JaxNautilusBound, JaxNeuralBound,  # noqa: F401
                     JaxUnion, JaxUnitCube, JaxUnitCubeEllipsoidMixture,
                     PhaseShift)
from .eval import JaxEvaluator  # noqa: F401
from .neural import Emulator, train_emulator  # noqa: F401
from .sampler import JaxSampler  # noqa: F401

# Explicit engine use configures the JAX runtime (idempotent).
_config.initialize_runtime()

__all__ = ['Emulator', 'JaxEllipsoid', 'JaxEvaluator', 'JaxNautilusBound',
           'JaxNeuralBound', 'JaxSampler', 'JaxUnion', 'JaxUnitCube',
           'JaxUnitCubeEllipsoidMixture', 'PhaseShift', 'train_emulator']