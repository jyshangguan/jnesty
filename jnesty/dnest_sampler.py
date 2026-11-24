"""JNesty public API — DNest diffusive nested sampler.

Dedicated adapter around the JAX DNest engine (:mod:`jnesty.dfnest`, a
device-accelerated port of CDNest, Yan-Rong Li's C implementation of
Brendon Brewer's diffusive nested sampling), exposed with JNesty's
constructor convention:

>>> from jnesty import DNestSampler
>>> sampler = DNestSampler(loglikelihood, prior_transform, ndim=2,
...                        num_particles=1, seed=0)
>>> log_z = sampler.run(max_num_saves=10000)
>>> points, log_w, log_l = sampler.posterior()

The engine keeps its native semantics: likelihood levels are built from
above-threshold records with a fixed compression, level prior volumes are
*learned* from visit/exceed statistics, particles diffuse across levels
with a heavy-tailed level proposal, and the run stops after
``max_num_saves`` saved particles (a run-length control, not an evidence
convergence criterion; an opt-in Dynesty-style ``delta_logZ`` convergence
stop is available as a :meth:`DNestSampler.run` argument and is off by
default).  Evidence, information and posterior weights
follow CDNest's C postprocessor.  There is no Dynesty-style ``logzerr``
for a single run; the uncertainty proxy is the entropy effective sample
size ``n_eff``, and CDNest's own recommendation for evidence scatter is
repeated independent runs.

The engine's only dependency is JAX itself (already a core dependency of
jnesty), so unlike the Nautilus engine there is no optional extra to
install.  Constructing a ``DNestSampler`` enables JAX float64 mode
process-wide; plain ``import jnesty`` does not import the engine.

Post-processing at a temperature ``T >= 1`` (CDNest convention:
likelihoods divided by ``T`` for the evidence and weights, level
placement untempered) is available as an opt-in API:
:meth:`DNestSampler.results_at_temperature` returns a full
plotting-compatible :class:`~jnesty.results.Results` view at ``T``, and
:meth:`DNestSampler.posterior_at_temperature` the corresponding
``(points, log_w, log_l)`` triple.  ``T = 1`` reproduces the standard
posterior exactly; the default ``to_results()``/``posterior()`` paths
are unchanged.
"""

import math

__all__ = ['DNestSampler']


def _load_engine():
    """Import the JAX DNest engine on first use."""
    from . import dfnest
    return dfnest


class DNestSampler:
    """Diffusive nested sampler (CDNest algorithm) with JAX acceleration.

    Thin composition wrapper around
    :class:`~jnesty.dfnest.engine.JaxDNestSampler` following JNesty's
    constructor convention (likelihood first, prior transform second,
    ``ndim`` spelled as in :class:`~jnesty.NestedSampler`).

    Parameters
    ----------
    loglikelihood : callable
        JAX-compatible log-likelihood of one parameter point (in physical
        space); may return ``-inf`` for forbidden regions.
    prior_transform : callable
        JAX-compatible map from the unit hypercube to physical parameters.
    ndim : int
        Number of dimensions (at least 1).
    num_lanes : int, optional
        Independent workers, the JAX analogue of CDNest's MPI ranks.
        Default 1.  Larger values widen the per-block likelihood batch;
        the total ensemble is ``num_lanes * num_particles`` particles.
    num_particles : int, optional
        Local particles per lane, exactly as CDNest counts
        ``NumberParticles`` (default 1).  This is *not* a static-NS
        ``nlive``; diffusive nested sampling has no live-point shrinkage.
    sample_frequency : scalar or array-like, optional
        Per-parameter proposal frequencies: a positive scalar (broadcast
        to all dimensions) or a length-``ndim`` sequence of non-negative
        relative weights, normalised internally so that larger values mean
        the parameter is proposed more often.  Default is uniform, the
        CDNest behaviour.  This is useful for parameters that change
        little between good fits: give them a smaller weight so the
        sampler spends its proposals elsewhere.  Zero entries freeze the
        coordinate (a warning is emitted).  The frequencies only allocate
        proposals; they never modify acceptance ratios, posterior weights
        or the evidence.
    thread_steps_factor : float, optional
        CDNest ``ThreadStepsFactor`` (default 10): sequential updates per
        lane per block are ``int(ndim * thread_steps_factor *
        num_particles)``.
    new_level_interval_factor : float, optional
        CDNest ``NewLevelIntervalFactor`` (default 2): a new level is cut
        every ``int(num_lanes * thread_steps * factor)`` global updates.
    save_interval_factor : float or None, optional
        CDNest ``SaveIntervalFactor``; ``None`` (default) inherits
        ``new_level_interval_factor``.
    max_num_levels : int, optional
        0 (default) creates levels until the PTol criterion fires; a
        positive value fixes the level count.
    lam : float, optional
        CDNest ``BacktrackingLength`` (default 10): the log_push scale
        while levels are being built.
    beta : float, optional
        CDNest ``StrengthEqualPush`` (default 100): equal-push strength
        once level creation has finished.
    compression : float, optional
        Level compression in ``(1, e]``; default ``e``.
    ptol : float, optional
        CDNest ``PTol`` (default 0.1): mean level log-likelihood spacing
        that stops level creation.
    seed : int or None, optional
        Seed for reproducible runs.
    device : str, jax.Device or None, optional
        Device for the compiled programs (e.g. ``'gpu'`` or ``'cpu'``);
        None uses the default device.  The global platform setting is
        never modified.
    context : pytree or None, optional
        Passed as a second argument to every likelihood and prior call,
        keeping large datasets out of the compiled kernels.

    Notes
    -----
    Constructing the sampler enables JAX float64 mode (CDNest computes in
    double precision); this is a process-global JAX setting, so a session
    that also uses the Dynesty-style samplers should treat float64 as
    enabled from the first ``DNestSampler`` construction onwards.

    Examples
    --------
    >>> from jnesty import DNestSampler
    >>> sampler = DNestSampler(loglikelihood, prior_transform, ndim=3)
    >>> log_z = sampler.run(max_num_saves=5000)
    >>> points, log_w, log_l = sampler.posterior()
    """

    def __init__(self, loglikelihood, prior_transform, ndim,
                 num_lanes=1, num_particles=1, sample_frequency=None,
                 thread_steps_factor=10.0, new_level_interval_factor=2.0,
                 save_interval_factor=None, max_num_levels=0, lam=10.0,
                 beta=100.0, compression=math.e, ptol=0.1, seed=None,
                 device=None, context=None):
        engine = _load_engine()
        self._native = engine.JaxDNestSampler(
            loglikelihood, prior_transform, ndim,
            num_lanes=num_lanes, num_particles=num_particles,
            sample_frequency=sample_frequency,
            thread_steps_factor=thread_steps_factor,
            new_level_interval_factor=new_level_interval_factor,
            save_interval_factor=save_interval_factor,
            max_num_levels=max_num_levels, lam=lam, beta=beta,
            compression=compression, ptol=ptol, seed=seed, device=device,
            context=context)

    # ------------------------------------------------------------------
    # native API
    # ------------------------------------------------------------------
    def run(self, max_num_saves=10000, ptol=None, verbose=False,
            delta_logZ=None):
        """Run the sampler until ``max_num_saves`` particles are saved.

        A larger target than already saved continues the native state
        (levels, particles, counters); an equal or smaller target is a
        no-op returning the existing evidence.

        Returns
        -------
        float
            The natural-log evidence ``log_z``.

        Parameters
        ----------
        max_num_saves : int, optional
            Total save target (default 10000).
        ptol : float or None, optional
            Overrides the level-creation PTol for subsequent bookkeeping.
        verbose : bool, optional
            Print level-creation and progress lines.
        delta_logZ : float or None, optional
            Opt-in Dynesty-style stopping criterion (default ``None``).
            When set (e.g. ``0.05``), the run may stop early once level
            creation has finished, a minimum save floor was reached and
            the estimated remaining-evidence ratio
            ``delta = logaddexp(0, logl_max + logX_tail - logZ_running)``
            dropped below ``delta_logZ``; ``logX_tail`` is the learned
            prior volume of the last level and ``logZ_running`` a cheap
            conservative estimate from the level structure (see
            :func:`jnesty.dfnest.levels.delta_log_z_estimate`).  The save
            budget still caps the run if the criterion never fires.  A
            run stopped this way reports ``delta_logz_stopped`` and can be
            continued with a later ``run()``.

        Notes
        -----
        ``max_num_saves`` is a run-length control in the spirit of CDNest,
        not an evidence-convergence criterion: diffusive nested sampling
        stops when its save budget is spent.  ``ptol`` only controls when
        level *creation* finishes; after that the sampler diffuses across
        the finished level set for the remaining budget.  ``delta_logZ``
        adds the opt-in convergence stop on top; with ``None`` (the
        default) the sampling path is bit-identical to a run without the
        argument.
        """
        return self._native.run(max_num_saves=max_num_saves, ptol=ptol,
                                verbose=verbose, delta_logz=delta_logZ)

    def posterior(self):
        """Return the posterior sample estimate.

        Returns ``(points, log_w, log_l)``: physical-space samples in
        native save order, posterior log-weights normalised so that
        ``logsumexp(log_w) == 0``, and the saved log-likelihoods.
        """
        return self._native.posterior()

    def print_summary(self):
        """Print a short native summary of the run."""
        native = self._native
        if native.native_result is None:
            print('DNestSampler: no results yet. Call run() first.')
            return
        res = native.native_result
        print('=' * 70)
        print('DNestSampler summary')
        print('=' * 70)
        print(f"Log evidence: logZ = {res.logz:.4f}  (no per-run logzerr; "
              f"n_eff = {res.n_eff:.1f})")
        print(f"Information: H = {res.information:.4f} nats")
        print(f"Levels: {native.n_levels} (finalised: "
              f"{native.levels_finalized})")
        print(f"Saves: {native.n_saves}  MCMC updates: {native.n_steps}"
              f"  Likelihood calls: {native.n_like}")
        print(f"Acceptance rate: {native.acceptance_rate:.1%}")
        att = native.coordinate_attempts
        if att.sum() > 0:
            rate = native.coordinate_acceptance_rate
            print('Coordinate proposals/accepts: '
                  + ', '.join(f'{a}/{c} ({r:.0%})' for a, c, r in
                              zip(att, native.coordinate_accepts, rate)))
        print(f"Save target reached: {native.completed}")
        if native.delta_logz_stopped:
            print(f"Delta-logZ stop: yes (delta logZ = "
                  f"{native.last_delta_logz:.3g} at {native.n_saves} saves)")
        print('=' * 70)

    # ------------------------------------------------------------------
    # results bridge (shared plotting / FITS I/O)
    # ------------------------------------------------------------------
    def to_results(self):
        """Build a Dynesty-style :class:`~jnesty.results.Results` view.

        The shared :mod:`jnesty.plotting` helpers expect Dynesty-style
        arrays, so this converts the native posterior into that shape:
        physical-space ``samples`` and unit-cube ``samples_u`` ordered by
        ascending likelihood, per-point ``logl``/``logwt``, the native
        postprocessed ``logvol``, a derived ``logz_trajectory`` and the
        per-sample level assignments.  No evidence is re-estimated.

        Result semantics differ from the Dynesty-style engine: there is no
        ``logzerr`` (CDNest reports none for a single run; the uncertainty
        proxy is the effective sample size ``n_eff``), and ``logvol`` is
        the native DNest prior volume rather than a ``-i/nlive`` sequence.
        """
        from .results import format_dnest_results
        return format_dnest_results(self._native)

    def results_at_temperature(self, temperature):
        """Build a Dynesty-style :class:`~jnesty.results.Results` view
        at a posterior temperature.

        Same bridge as :meth:`to_results` (including the
        ``nlive``/``delta_logZ_trajectory``/``rwalk_K`` compatibility
        fields), but with the evidence, weights, ``logz_trajectory``,
        information and effective sample size computed at temperature
        ``T``: the CDNest postprocessor convention divides likelihoods
        by ``T`` for the evidence and weights while level placement and
        prior masses stay untempered.  The engine caches the underlying
        postprocess result per temperature, so repeated calls at the
        same ``T`` are cheap.

        Parameters
        ----------
        temperature : float
            Posterior temperature ``T >= 1``.  ``T = 1`` returns exactly
            the :meth:`to_results` view (bit-identical); ``T -> infinity``
            approaches the prior (weights proportional to the prior
            masses).

        Returns
        -------
        Results
            Plotting-compatible results at temperature ``T``; ``logl``
            and ``samples`` remain the untempered saved values (the
            temperature enters through ``logwt``/``logz``).

        Examples
        --------
        >>> res_hot = sampler.results_at_temperature(2.0)
        >>> _ = plotting.runplot(res_hot)
        """
        from .results import format_dnest_results
        return format_dnest_results(self._native, temperature)

    def posterior_at_temperature(self, temperature):
        """Return the posterior sample estimate at a temperature.

        Returns ``(points, log_w, log_l)``: physical-space samples in
        native save order (identical to :meth:`posterior` — the sample
        coordinates do not depend on ``T``), temperature-``T`` posterior
        log-weights normalised so that ``logsumexp(log_w) == 0``, and
        the *untempered* saved log-likelihoods (CDNest applies
        ``logl / T`` only to the weights).

        Parameters
        ----------
        temperature : float
            Posterior temperature ``T >= 1``; see
            :meth:`results_at_temperature`.

        Returns
        -------
        tuple
            ``(points, log_w, log_l)`` with the temperature-``T``
            weights.
        """
        result = self._native.native_result_at_temperature(temperature)
        return result.samples, result.logwt, result.logl

    @property
    def results(self):
        """Cached :class:`~jnesty.results.Results` snapshot.

        Built with :meth:`to_results` on first access and reused while the
        native sampler's revision counter is unchanged; a further run
        invalidates the cache.  Repeated access never consumes randomness
        or recomputes the (deterministic) postprocessing.
        """
        native = self._native
        snapshot = self.__dict__.get('_results_snapshot')
        revision = native.revision
        if snapshot is None or snapshot[0] != revision:
            self._results_snapshot = (revision, self.to_results())
        return self._results_snapshot[1]

    # ------------------------------------------------------------------
    # key diagnostics (everything else forwards via __getattr__)
    # ------------------------------------------------------------------
    @property
    def native_sampler(self):
        """The underlying :class:`jnesty.dfnest.JaxDNestSampler`."""
        return self._native

    @property
    def log_z(self):
        """Estimated log evidence (None before the first run)."""
        return self._native.log_z

    @property
    def information(self):
        """Information H in nats (None before the first run)."""
        return self._native.information

    @property
    def n_eff(self):
        """Entropy effective sample size ``exp(-sum(w log w))``."""
        return self._native.n_eff

    @property
    def n_like(self):
        """Total likelihood evaluations."""
        return self._native.n_like

    @property
    def n_saves(self):
        """Number of saved particles."""
        return self._native.n_saves

    @property
    def n_levels(self):
        """Number of likelihood levels."""
        return self._native.n_levels

    @property
    def sample_frequency(self):
        """Normalised per-coordinate proposal weights."""
        return self._native.sample_frequency

    @property
    def coordinate_attempts(self):
        """Per-coordinate proposal counts."""
        return self._native.coordinate_attempts

    @property
    def coordinate_accepts(self):
        """Per-coordinate accepted-proposal counts."""
        return self._native.coordinate_accepts

    @property
    def coordinate_acceptance_rate(self):
        """Per-coordinate acceptance rates (NaN where never proposed)."""
        return self._native.coordinate_acceptance_rate

    def __getattr__(self, name):
        native = self.__dict__.get('_native')
        if native is None:
            raise AttributeError(name)
        return getattr(native, name)

    def __repr__(self):
        native = self.__dict__.get('_native')
        if native is None:
            return 'DNestSampler(not constructed)'
        return (f'DNestSampler(ndim={native.config.ndim}, '
                f'num_lanes={native.num_lanes}, '
                f'num_particles={native.num_particles}, '
                f'n_saves={native.n_saves}, n_levels={native.n_levels})')
