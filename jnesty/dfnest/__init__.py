"""JAX/GPU engine for CDNest-style diffusive nested sampling.

This subpackage is a device-accelerated port of the *algorithm* of CDNest
(LiyrAstroph/CDNest, the C implementation of Brendon Brewer's diffusive
nested sampling), pinned at commit ``3aaa039f9a0a76919c0ed65b355a50ecb497e6a2``:

* likelihood levels are built from above-threshold records with a fixed
  compression, and each level's prior volume ``logX`` is *learned* from
  the visit/exceed statistics of the particle ensemble;
* particles carry level assignments and diffuse across levels with a
  heavy-tailed level proposal (Brewer et al. 2011);
* the run stops after a fixed number of saved particles (``max_num_saves``),
  not on an evidence-uncertainty criterion;
* evidence, information and posterior weights follow the C postprocessor
  in CDNest's ``src/dnestpostprocess.c``.

What changes relative to the C/MPI code is where the work happens: one
compiled JAX program advances ``num_lanes`` independent workers (the
analogue of MPI ranks) in parallel, each holding ``num_particles`` local
particles, while level creation, volume recomputation and saves stay on
the host.

The public entry point is :class:`jnesty.DNestSampler`; this module only
needs to be imported directly for advanced/native use.
"""

from .engine import JaxDNestSampler
from .postprocess import DNestResult, dnest_postprocess

__all__ = ['JaxDNestSampler', 'DNestResult', 'dnest_postprocess']
