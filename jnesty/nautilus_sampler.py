"""
JNesty public API — Nautilus importance nested sampler.

Dedicated adapter around *our* JAX Nautilus engine (:mod:`jnesty.nautilus`,
the port of the Nautilus importance nested sampling algorithm to JAX/GPU),
exposed with JNesty's constructor convention:

>>> from jnesty import NautilusSampler
>>> sampler = NautilusSampler(loglikelihood, prior_transform, ndim=2,
...                           nlive=500, seed=0)
>>> sampler.run(f_live=0.01, n_eff=10000)
>>> points, log_w, log_l = sampler.posterior()
>>> sampler.log_z

The engine keeps its native semantics: nested neural-network shells, every
likelihood evaluation kept, and evidence / posterior weights from shell
volumes. There is no Dynesty-style ``logzerr``; run-to-run uncertainty is
monitored through the effective sample size ``n_eff``. The underlying
:class:`~jnesty.nautilus.sampler.JaxSampler` is available as
:attr:`NautilusSampler.native_sampler` for everything this adapter does
not forward explicitly.

The engine's dependencies (``nautilus-sampler``, ``optax``,
``scikit-learn``, ``threadpoolctl``, ``h5py``) are optional: plain
``import jnesty`` never imports them. Install them with
``pip install 'jnesty[nautilus]'``. Constructing a sampler enables JAX
float64 mode and the engine's compilation cache — process-global JAX
settings, see :mod:`jnesty.nautilus._config`.
"""

import numpy as np

__all__ = ['NautilusSampler']

_INSTALL_HINT = (
    "The JAX Nautilus engine requires optional dependencies "
    "(nautilus-sampler, optax, scikit-learn, threadpoolctl, h5py). "
    "Install them with:  pip install 'jnesty[nautilus]'")


def _load_engine():
    """Import the JAX Nautilus engine on first use."""
    try:
        from . import nautilus
    except ImportError as exc:
        raise ImportError(f'{_INSTALL_HINT}  (import failed: {exc})') from exc
    return nautilus


class NautilusSampler:
    """Importance nested sampler with JAX/GPU acceleration.

    Thin composition wrapper around
    :class:`~jnesty.nautilus.sampler.JaxSampler` following JNesty's
    constructor convention (likelihood first, prior transform second,
    ``ndim``/``nlive`` spelled as in :class:`~jnesty.NestedSampler`).
    Run control, posterior extraction, checkpointing and shell diagnostics
    keep the native engine's semantics.

    Parameters
    ----------
    loglikelihood : callable
        JAX-compatible log-likelihood of one parameter point (in physical
        space). May return ``log_l`` or ``(log_l, *blobs)``.
    prior_transform : callable
        JAX-compatible map from the unit hypercube to physical parameters.
    ndim : int
        Number of dimensions (at least 2).
    nlive : int, optional
        Number of live points. Default is 500, matching JNesty's other
        samplers; the native engine's own default is 2000.
    seed : int or None, optional
        Seed for reproducible runs.
    device : str or None, optional
        JAX device platform, e.g. ``'gpu'`` or ``'cpu'``. Default None uses
        the default device.
    context : pytree or None, optional
        Passed as a second argument to the likelihood and prior transform,
        keeping large datasets out of the compiled kernels.
    **kwargs
        Forwarded to :class:`~jnesty.nautilus.sampler.JaxSampler`; the
        important ones: ``n_update``, ``enlarge_per_dim``,
        ``n_points_min``, ``split_threshold``, ``periodic``,
        ``n_networks`` (0 disables the neural bounds), 
        ``neural_network_kwargs``, ``n_batch``, ``n_like_new_bound``,
        ``vectorized``, ``pass_dict``, ``blobs_dtype``, ``filepath``,
        ``resume``, ``dtype``, ``fit_backend``, ``neural_backend``.

    Notes
    -----
    Constructing a sampler enables JAX float64 mode and the engine's
    compilation cache; both are process-global JAX settings, so a session
    that also uses the Dynesty-style samplers should treat float64 as
    enabled from the first :class:`NautilusSampler` construction onwards.

    Examples
    --------
    >>> from jnesty import NautilusSampler
    >>> sampler = NautilusSampler(loglikelihood, prior_transform, ndim=3)
    >>> sampler.run(f_live=0.01, n_eff=10000, verbose=True)
    True
    >>> points, log_w, log_l = sampler.posterior()
    """

    def __init__(self, loglikelihood, prior_transform, ndim, nlive=500,
                 seed=None, device=None, context=None, **kwargs):
        engine = _load_engine()
        # JNesty argument order: the likelihood comes first and the prior
        # transform second; the native engine takes them the other way
        # round.
        self._native = engine.JaxSampler(
            prior_transform, loglikelihood, n_dim=int(ndim),
            n_live=int(nlive), seed=seed, device=device, context=context,
            **kwargs)

    # ------------------------------------------------------------------
    # native API
    # ------------------------------------------------------------------
    def run(self, f_live=0.01, n_shell=1, n_eff=10000, n_like_max=np.inf,
            discard_exploration=False, timeout=np.inf, verbose=False):
        """Run the sampler until convergence.

        Same semantics as :meth:`jnesty.nautilus.sampler.JaxSampler.run`:
        exploration shrinks the live-set fraction to ``f_live``, then
        refinement fills every shell to ``n_shell`` samples and the total
        effective sample size to ``n_eff``. Returns ``True`` when the run
        finished normally, ``False`` when the likelihood-call or time limit
        stopped it early.
        """
        return self._native.run(
            f_live=f_live, n_shell=n_shell, n_eff=n_eff,
            n_like_max=n_like_max,
            discard_exploration=discard_exploration,
            timeout=timeout, verbose=verbose)

    def posterior(self, return_as_dict=None, equal_weight=False,
                  equal_weight_boost=1.0, return_blobs=False):
        """Return the posterior sample estimate.

        Same semantics as
        :meth:`jnesty.nautilus.sampler.JaxSampler.posterior`: returns
        ``(points, log_w, log_l)`` with normalised log-weights, or a 4-tuple
        with blobs when ``return_blobs=True``.
        """
        return self._native.posterior(
            return_as_dict=return_as_dict, equal_weight=equal_weight,
            equal_weight_boost=equal_weight_boost,
            return_blobs=return_blobs)

    def write(self, filepath, overwrite=False):
        """Write an HDF5 checkpoint of the sampler state."""
        return self._native.write(filepath, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Results bridge (so jnesty.plotting works for this engine too)
    # ------------------------------------------------------------------
    def to_results(self):
        """Build a Dynesty-style :class:`~jnesty.results.Results` view.

        The shared :mod:`jnesty.plotting` helpers expect Dynesty-style
        result arrays, so this converts the native posterior into that
        shape: physical-space ``samples``, per-point ``logl``/``logwt``,
        the evidence in ``logz`` and derived ``logvol`` /
        ``logz_trajectory`` arrays. The native engine's public API
        (``posterior()``, ``log_z``, ``n_like``, ``n_eff``) is unchanged,
        and no evidence is re-estimated.

        Result semantics differ from the Dynesty-style engine:
        importance nested sampling has no ``logzerr`` (the run-to-run
        uncertainty proxy is the effective sample size ``n_eff``), no
        information estimate, and ``logvol`` / ``logz_trajectory`` are
        derived quantities for plotting rather than native outputs.

        Returns
        -------
        Results
            A fresh snapshot of the current sampler state, with
            per-sample arrays sorted by ascending ``logl``.

        Examples
        --------
        >>> fig, axes = jnesty.plotting.cornerplot(sampler.to_results())
        """
        # Imported lazily so importing this module stays numpy-only and
        # never pulls in jax through jnesty.results/jnesty.utils.
        from .results import format_nautilus_results

        return format_nautilus_results(self._native)

    @property
    def results(self):
        """Cached :class:`~jnesty.results.Results` snapshot.

        Built with :meth:`to_results` on first access and reused while the
        native sampler's likelihood-evaluation count (``n_like``) is
        unchanged; a further run invalidates it and the next access
        rebuilds the snapshot. Treat the returned object as a read-only
        view of the sampler state at snapshot time.

        The same caveats as :meth:`to_results` apply: no ``logzerr``, and
        ``logvol`` / ``logz_trajectory`` are derived plotting quantities.

        Returns
        -------
        Results
            Dynesty-compatible results object for :mod:`jnesty.plotting`.
        """
        snapshot = self.__dict__.get('_results_snapshot')
        n_like = int(self._native.n_like)
        if snapshot is None or snapshot[0] != n_like:
            self._results_snapshot = (n_like, self.to_results())
        return self._results_snapshot[1]

    # ------------------------------------------------------------------
    # key diagnostics (everything else forwards via __getattr__)
    # ------------------------------------------------------------------
    @property
    def native_sampler(self):
        """The underlying :class:`jnesty.nautilus.JaxSampler`."""
        return self._native

    @property
    def log_z(self):
        """Estimated log evidence."""
        return self._native.log_z

    @property
    def n_eff(self):
        """Estimated total effective sample size."""
        return self._native.n_eff

    @property
    def n_like(self):
        """Total number of likelihood evaluations."""
        return self._native.n_like

    @property
    def eta(self):
        """Estimated asymptotic sampling efficiency."""
        return self._native.eta

    @property
    def discard_exploration(self):
        """Whether exploration-phase points are excluded from the results."""
        return self._native.discard_exploration

    @discard_exploration.setter
    def discard_exploration(self, discard_exploration):
        self._native.discard_exploration = discard_exploration

    def __getattr__(self, name):
        native = self.__dict__.get('_native')
        if native is None:
            raise AttributeError(name)
        return getattr(native, name)

    def __repr__(self):
        native = self.__dict__.get('_native')
        if native is None:
            return 'NautilusSampler(not constructed)'
        return (f'NautilusSampler(n_dim={native.n_dim}, '
                f'n_live={native.n_live}, n_like={native.n_like})')
