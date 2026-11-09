"""JAX/GPU version of the nautilus sampler.

This module is a device-accelerated port of :mod:`nautilus.sampler`.  The
*algorithm* is the same importance nested sampling scheme: live-set guided
exploration builds a sequence of nested bounds, every likelihood evaluation is
kept, and the final evidence/posterior estimates use shell volumes and
importance weights rather than Dynesty-style volume shrinkage.

What changes is where the work happens:

* likelihood and prior evaluations are compiled JAX programs (``vmap``/``jit``),
* bound construction, sampling and containment tests run on the device,
* neural emulator training and prediction run on the device,
* the shell bookkeeping, stopping decisions and checkpointing stay on the host,
  because the number and shape of shells change during a run.

Randomness is threaded through a single ``numpy.random.Generator`` on the host,
which seeds device keys per batch.  That keeps runs reproducible from a seed
and makes the RNG state checkpointable, exactly like the reference
implementation.
"""

from pathlib import Path
from time import time
from warnings import warn

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp

from .bounds import JaxNautilusBound, JaxUnitCube, _multi_bound_contains
from .eval import JaxEvaluator
from . import _config

__all__ = ['JaxSampler']


class JaxSampler:
    """Importance nested sampling with JAX/GPU accelerated kernels.

    Parameters
    ----------
    prior : callable or nautilus.Prior
        Prior mapping the unit hypercube to physical parameters.  For the GPU
        fast path the callable must be compatible with JAX (use ``jnp``
        operations rather than in-place NumPy).
    likelihood : callable
        Natural logarithm of the likelihood.  Must be JAX-compatible.  It may
        return ``(log_l,)`` or ``(log_l, *blobs)``.
    n_dim : int, optional
        Number of dimensions.  Inferred when ``prior`` is a ``nautilus.Prior``.
    n_live : int, optional
        Number of live points.  Default is 2000.
    n_update : int or None, optional
        Maximum number of live-set additions before a new bound is built.
        Default is ``n_live``.
    enlarge_per_dim : float, optional
        Per-dimension enlargement of the bounding ellipsoids.  Default is 1.1.
    n_points_min : int or None, optional
        Minimum number of points per ellipsoid.  Default is ``n_dim + 50``.
    split_threshold : float, optional
        Split bounds while the proposal volume exceeds this multiple of the
        target volume.  Default is 100.
    periodic : array or None, optional
        Indices of periodic parameters.  Default is None.
    n_networks : int, optional
        Size of the emulator ensemble.  Default is 4.  ``0`` disables neural
        rejection and gives a purely ellipsoidal bound.
    neural_network_kwargs : dict or None, optional
        Overrides for the emulator (``hidden``, ``learning_rate``,
        ``batch_size``, ``max_epochs``, ``patience``, ``dtype``).
    n_batch : int, optional
        Number of likelihood evaluations per step.  Default is 100.
    n_like_new_bound : int or None, optional
        Maximum likelihood calls before a new bound is built.  Default is
        ``10 * n_live``.
    vectorized : bool, optional
        Whether ``likelihood`` already accepts an ``(n_points, n_dim)`` array.
        Default is False (the function is vectorized with ``vmap``).
    pass_dict : bool or None, optional
        Whether the prior returns dictionaries.  Default is True for
        ``nautilus.Prior`` and False for callables.
    seed : int or None, optional
        Seed for reproducible runs.  Default is None.
    device : str or None, optional
        JAX device platform to place arrays on, e.g. ``'gpu'`` or ``'cpu'``.
        Default is None (current default device).
    filepath : str or Path or None, optional
        HDF5 checkpoint path.  Default is None (no checkpointing).
    resume : bool, optional
        Resume from ``filepath`` when it exists.  Default is True.
    blobs_dtype : object or None, optional
        Data type of likelihood blobs.  Default is None (inferred).
    """

    def __init__(self, prior, likelihood, n_dim=None, n_live=2000,
                 n_update=None, enlarge_per_dim=1.1, n_points_min=None,
                 split_threshold=100, periodic=None, n_networks=4,
                 neural_network_kwargs=None, n_batch=100,
                 n_like_new_bound=None, vectorized=False, pass_dict=None,
                 seed=None, blobs_dtype=None, filepath=None, resume=True,
                 device=None, context=None, dtype=None, fit_backend='numpy',
                 neural_backend='jax', pool=None):
        _config.initialize_runtime()

        if callable(prior):
            self.prior = prior
            if n_dim is None:
                raise ValueError("When passing a function as the 'prior' "
                                 "argument, 'n_dim' cannot be None.")
            self.n_dim = int(n_dim)
            if pass_dict is None:
                pass_dict = False
        else:
            self.prior = prior
            self.n_dim = int(prior.dimensionality())
            if pass_dict is None:
                pass_dict = True

        if self.n_dim <= 1:
            raise ValueError('Cannot run nautilus with less than 2 parameters.')

        self.likelihood = likelihood
        self.pass_dict = pass_dict
        self.vectorized = vectorized
        self.n_live = int(n_live)
        self.n_update = int(n_live if n_update is None else n_update)
        self.enlarge_per_dim = enlarge_per_dim
        self.n_points_min = int(self.n_dim + 50 if n_points_min is None
                                else n_points_min)
        self.split_threshold = split_threshold
        self.periodic = periodic
        self.n_networks = int(n_networks)
        self.neural_network_kwargs = dict(neural_network_kwargs or {})
        self.n_batch = int(n_batch)
        self.n_like_new_bound = int(10 * n_live if n_like_new_bound is None
                                    else n_like_new_bound)
        self.blobs_dtype = blobs_dtype
        self.fit_backend = fit_backend
        if neural_backend not in ('jax', 'reference'):
            raise ValueError("neural_backend must be 'jax' or 'reference', "
                             f"got {neural_backend!r}")
        self.neural_backend = neural_backend
        if pool not in (None, 1):
            raise ValueError(
                'JaxSampler evaluates the likelihood in batched device '
                'programs, not in a process pool; run the CPU reference '
                'Sampler if you need multiprocessing.')
        self.context = context
        self.dtype = dtype
        self.device = device

        self.rng = np.random.default_rng(seed)

        #: One entry per sampling batch: enough to see how a run progresses
        #: (calls, evidence, remaining evidence fraction, bounds) without
        #: re-instrumenting the sampler from outside.
        self.history = []

        # Run state -----------------------------------------------------
        self.n_like = 0
        self.explored = False
        self.bounds = []
        self.points = []
        self.log_l = []
        self.blobs = None
        self._discard_exploration = False
        self.shell_n = np.zeros(0, dtype=int)
        self.shell_n_sample = np.zeros(0, dtype=int)
        self.shell_n_eff = np.zeros(0, dtype=float)
        self.shell_log_l_min = np.zeros(0, dtype=float)
        self.shell_log_l = np.zeros(0, dtype=float)
        self.shell_log_v = np.zeros(0, dtype=float)
        self.shell_n_sample_exp = np.zeros(0, dtype=int)
        self.shell_end_exp = np.zeros(0, dtype=int)
        self.points_t = np.zeros((0, self.n_dim))
        self.shell_t = np.zeros(0, dtype=int)
        self.log_l_t = np.zeros(0)
        self.blobs_t = None

        self.n_update_iter = 0
        self.n_like_iter = 0

        self.evaluator = JaxEvaluator(
            self.prior, self.likelihood, self.n_dim, pass_dict=self.pass_dict,
            vectorized=self.vectorized, context=context, device=device,
            dtype=dtype)

        self.filepath = None if filepath is None else Path(filepath)
        if (resume and self.filepath is not None
                and self.filepath.exists()):
            from .checkpoint import read_sampler
            read_sampler(self, self.filepath)

    # ------------------------------------------------------------------
    # run loop
    # ------------------------------------------------------------------
    def run(self, f_live=0.01, n_shell=1, n_eff=10000, n_like_max=np.inf,
            discard_exploration=False, timeout=np.inf, verbose=False):
        """Run the sampler until convergence.

        Returns ``True`` when the run finished normally, and ``False`` when a
        likelihood-call or time limit stopped it early.

        With ``verbose=True`` a one-line status is refreshed about every two
        seconds, labelled with the current phase: ``sampling`` while the
        nested shells are explored, then ``refine`` while the shells are
        filled up to the ``n_shell`` and ``n_eff`` targets.  The wall time
        of each phase is collected in :attr:`phase_times` (always, not only
        when verbose).
        """
        t_start = time()

        # Phase bookkeeping: pure wall-clock attribution, never touches the
        # sampling stream.
        self.phase_times = {'sampling': 0.0, 'refine': 0.0}
        phase = 'refine' if self.explored else 'sampling'
        t_phase = t_start
        t_print = t_start

        def _switch(new_phase):
            nonlocal phase, t_phase
            if new_phase == phase:
                return
            now = time()
            self.phase_times[phase] += now - t_phase
            phase = new_phase
            t_phase = now
            if verbose:
                print(f'\n=== phase: {phase} ===', flush=True)

        if verbose:
            if self.n_like == 0:
                print('Starting the JAX nautilus sampler...')
            else:
                print('Resuming JAX nautilus run...')
            print(f'=== phase: {phase} ===')
            self.print_status(header=True)

        if len(self.bounds) == 0:
            self.add_bound()
            self.n_update_iter = -self.n_live
            self.n_like_iter = 0

        success = (self.explored and np.all(self.shell_n >= n_shell)
                   and self.n_eff >= n_eff)

        while (self.n_like < n_like_max and time() - t_start < timeout
               and not success):
            if not self.explored:
                if ((self.n_update_iter >= self.n_update
                     or self.n_like_iter >= self.n_like_new_bound)
                        and np.sum(self.shell_n) > self.n_live):
                    self.add_bound()
                    self.n_update_iter = 0
                    self.n_like_iter = 0
                    if self.filepath is not None:
                        self.write(self.filepath, overwrite=True)

                self.n_update_iter += self.add_samples(-1)
                self.n_like_iter += self.n_batch
                self._record_history()
                if self.filepath is not None:
                    self.write(self.filepath, overwrite=True)

                if self.f_live <= f_live:
                    if np.any(self.shell_n == 0):
                        for shell in np.flatnonzero(self.shell_n == 0)[::-1]:
                            self.bounds.pop(shell)
                            self.points.pop(shell)
                            self.log_l.pop(shell)
                            if self.blobs is not None:
                                self.blobs.pop(shell)
                            for key in ['shell_n', 'shell_n_sample',
                                        'shell_n_eff', 'shell_log_l_min',
                                        'shell_log_l', 'shell_log_v']:
                                setattr(self, key, np.delete(
                                    getattr(self, key), shell))

                    self.shell_n_sample_exp = np.copy(self.shell_n_sample)
                    self.shell_end_exp = np.array(
                        [len(p) for p in self.points])

                    self.explored = True
                    self.discard_exploration = discard_exploration
                    if self.filepath is not None:
                        self.write(self.filepath, overwrite=True)

            elif np.any(self.shell_n < n_shell):
                _switch('refine')
                shell = int(np.flatnonzero(self.shell_n < n_shell)[0])
                self.add_samples(shell)
                if self.filepath is not None:
                    self.write(self.filepath, overwrite=True)

            elif self.n_eff < n_eff:
                _switch('refine')
                shell = int(np.argmax(
                    self.shell_log_l + self.shell_log_v
                    - 0.5 * np.log(self.shell_n)
                    - 0.5 * np.log(np.maximum(self.shell_n_eff, 1))))
                self.add_samples(shell)
                if self.filepath is not None:
                    self.write(self.filepath, overwrite=True)

            if verbose and time() - t_print >= 2.0:
                self.print_status(phase, end='\r')
                t_print = time()

            success = (self.explored and np.all(self.shell_n >= n_shell)
                       and self.n_eff >= n_eff)

        self.phase_times[phase] += time() - t_phase
        if verbose:
            self.print_status('Finished' if success else 'Stopped')
            print(f"phase wall times: sampling "
                  f"{self.phase_times['sampling']:.1f} s, refine "
                  f"{self.phase_times['refine']:.1f} s", flush=True)
        return success

    # ------------------------------------------------------------------
    # exploration / sampling
    # ------------------------------------------------------------------
    def _record_history(self):
        """Append one progress record; cheap, and never raises."""
        try:
            f_live = self.f_live
        except ValueError:
            f_live = None
        self.history.append({
            'n_like': int(self.n_like),
            'log_z': (None if self.log_z is None else float(self.log_z)),
            'f_live': None if f_live is None else float(f_live),
            'bounds': len(self.bounds),
            'explored': bool(self.explored),
        })

    def sample_shell(self, index, shell_t=None):
        """Sample a batch of points uniformly from a shell."""
        if shell_t is not None and index not in [-1, len(self.bounds) - 1]:
            raise ValueError("'shell_t' must be empty list if not sampling "
                             "from the last bound/shell.")

        n_bound = 0
        n_sample = 0
        idx_t = np.zeros(0, dtype=int)
        points_all = []

        while n_sample < self.n_batch:
            key = self._next_key()
            points = self.bounds[index].sample(
                self.n_batch - n_sample, key=key)
            n_bound += self.n_batch - n_sample

            # Exclusion is a pure AND-chain of deterministic per-point tests,
            # so every enclosing bound can be evaluated in ONE batched
            # containment call (one dispatch regardless of how many bounds
            # exist) and compacted with a single host round-trip.  Nothing is
            # uploaded when there is no enclosing bound (the common
            # last-shell case).  Compaction happens on the host: device
            # masking of a data-dependent output length would recompile once
            # per distinct length.
            points_np = np.asarray(points)
            enclosing = self.bounds[index:][1:]
            if not enclosing:
                points = points_np
            else:
                masks = _multi_bound_contains(enclosing, points_np)
                in_shell = ~np.asarray(jnp.any(masks, axis=1))
                points = points_np[in_shell]

            if shell_t is not None and len(shell_t) > 0 and len(points) > 0:
                shell_p = self.shell_association(points, n_max=len(self.bounds) - 1)
                replace = np.zeros(len(points), dtype=bool)
                for shell in range(len(self.bounds) - 1):
                    idx_1 = np.flatnonzero(shell_t == shell)
                    idx_2 = np.flatnonzero(shell_p == shell)
                    n = min(len(idx_1), len(idx_2))
                    if n > 0:
                        chosen = self.rng.choice(idx_1, size=n, replace=False)
                        idx_t = np.append(idx_t, chosen)
                        shell_t[chosen] = -1
                        replace[self.rng.choice(idx_2, size=n, replace=False)] = True
                points = points[~replace]

            if len(points) > 0:
                points_all.append(points)
                n_sample += len(points)

        points = np.concatenate(points_all) if points_all else np.zeros((0, self.n_dim))

        if shell_t is None:
            return points, n_bound
        return points, n_bound, idx_t

    def add_samples(self, shell, verbose=False):
        """Add one batch of samples to a shell; returns live-set additions."""
        if verbose:
            self.print_status('Sampling', end='\r')

        if shell == -1 and len(self.shell_t) > 0:
            points, n_bound, idx_t = self.sample_shell(-1, self.shell_t)
            if len(idx_t) > 0:
                self.points[-1] = np.concatenate(
                    (self.points[-1], self.points_t[idx_t]))
                self.log_l[-1] = np.concatenate(
                    (self.log_l[-1], self.log_l_t[idx_t]))
                if self.blobs is not None:
                    self.blobs[-1] = np.concatenate(
                        (self.blobs[-1], self.blobs_t[idx_t]))
        else:
            points, n_bound = self.sample_shell(shell)

        if verbose:
            self.print_status('Computing', end='\r')

        self.shell_n_sample[shell] += n_bound
        log_l, blobs = self.evaluate_likelihood(points)
        self.points[shell] = np.append(self.points[shell], points, axis=0)
        self.log_l[shell] = np.append(self.log_l[shell], log_l, axis=0)
        if blobs is not None:
            if self.blobs is None:
                self.blobs = [blobs]
            else:
                self.blobs[shell] = np.append(self.blobs[shell], blobs, axis=0)
        self.update_shell_info(shell)
        return int(np.sum(log_l >= self.shell_log_l_min[shell]))

    def add_bound(self, verbose=False):
        """Try building a new bound from the existing points."""
        if len(self.bounds) == 0:
            log_l_min = -np.inf
            self.bounds.append(JaxUnitCube.compute(self.n_dim))
            success = True
        else:
            if verbose:
                self.print_status('Bounding', end='\r')
            log_l = np.concatenate(self.log_l)
            points = np.concatenate(self.points)[np.argsort(log_l)]
            log_l = np.sort(log_l)
            log_l_min = log_l[-self.n_live]

            if (np.sum(log_l == log_l_min) > 1
                    and np.sum(log_l > log_l_min) >= self.n_points_min):
                log_l_min = np.amin(log_l[log_l > log_l_min])

            if np.all(log_l >= log_l_min):
                success = False
            else:
                self._warn_skewed(points)
                key = self._next_key()
                bound = JaxNautilusBound.compute(
                    points, log_l, log_l_min, self.log_v_live,
                    enlarge_per_dim=self.enlarge_per_dim,
                    n_points_min=self.n_points_min,
                    split_threshold=self.split_threshold,
                    periodic=self.periodic,
                    n_networks=self.n_networks,
                    neural_network_kwargs=self.neural_network_kwargs,
                    key=key, backend=self.fit_backend,
                    neural_backend=self.neural_backend)
                bound.sample(1000, key=self._next_key())
                if bound.log_v < self.bounds[-1].log_v:
                    self.bounds.append(bound)
                    success = True
                else:
                    success = False

        if success:
            self.shell_n = np.append(self.shell_n, 0)
            self.shell_n_sample = np.append(self.shell_n_sample, 0)
            self.shell_n_eff = np.append(self.shell_n_eff, 0)
            self.shell_log_l = np.append(self.shell_log_l, np.nan)
            self.shell_log_v = np.append(self.shell_log_v, np.nan)
            self.shell_log_l_min = np.append(self.shell_log_l_min, log_l_min)
            self.points.append(np.zeros((0, self.n_dim)))
            self.log_l.append(np.zeros(0))
            if self.blobs is not None:
                self.blobs.append(
                    np.zeros(self.blobs[-1][:0].shape, dtype=self.blobs_dtype))
        else:
            self.shell_log_l_min[-1] = log_l_min
            return False

        if len(self.bounds) > 1:
            self.shell_t = []
            self.points_t = []
            self.log_l_t = []
            if self.blobs is not None:
                self.blobs_t = []

            # One batched containment test instead of one per shell: ``contains``
            # is per-point deterministic, so re-associating a concatenated
            # block and splitting the mask back by the known shell lengths is
            # exactly equivalent (~120 dispatches late in a run -> 1).
            shell_lengths = np.array(
                [len(p) for p in self.points[:-1]], dtype=int)
            in_bound_all = np.asarray(self.bounds[-1].contains(
                np.concatenate(self.points[:-1])))
            start = 0
            for shell, length in enumerate(shell_lengths):
                end = start + length
                in_bound = in_bound_all[start:end]
                start = end
                self.shell_t.append(np.repeat(shell, np.sum(in_bound)))
                self.points_t.append(self.points[shell][in_bound])
                self.points[shell] = self.points[shell][~in_bound]
                self.log_l_t.append(self.log_l[shell][in_bound])
                self.log_l[shell] = self.log_l[shell][~in_bound]
                if self.blobs is not None:
                    self.blobs_t.append(self.blobs[shell][in_bound])
                    self.blobs[shell] = self.blobs[shell][~in_bound]
                self.shell_n[shell] -= int(np.sum(in_bound))
                self.update_shell_info(shell)

            self.shell_t = np.concatenate(self.shell_t)
            self.points_t = np.concatenate(self.points_t)
            self.log_l_t = np.concatenate(self.log_l_t)
            if self.blobs is not None:
                self.blobs_t = np.concatenate(self.blobs_t)

        return True

    def _warn_skewed(self, points):
        fraction_skewed = 0.1
        distance = 1e-5
        for j in range(points.shape[-1]):
            skewed = ((points[:, j] <= distance)
                      | (points[:, j] >= 1.0 - distance))
            if np.mean(skewed) > fraction_skewed:
                warn(f"For parameter {j + 1}, the live points are converging "
                     "towards the prior boundary. This may lead to numerical "
                     "issues. Consider adjusting your prior.", RuntimeWarning,
                     stacklevel=3)

    # ------------------------------------------------------------------
    # likelihood evaluation
    # ------------------------------------------------------------------
    def evaluate_likelihood(self, points):
        """Evaluate the likelihood on a batch of unit-cube points."""
        log_l, blobs = self.evaluator.evaluate(np.asarray(points))
        self.n_like += len(log_l)

        if blobs is not None:
            if self.blobs_dtype is None:
                if len(blobs) > 1:
                    self.blobs_dtype = [(f'blob_{i}', b.dtype)
                                        for i, b in enumerate(blobs)]
                else:
                    self.blobs_dtype = blobs[0].dtype
            if len(blobs) == 1:
                blobs = blobs[0].astype(self.blobs_dtype)
            else:
                blobs = np.array(list(zip(*blobs)), dtype=self.blobs_dtype)
        return log_l, blobs

    # ------------------------------------------------------------------
    # shell bookkeeping
    # ------------------------------------------------------------------
    def update_shell_info(self, index):
        """Recompute the summary statistics of one shell."""
        shell_n_sample = self.shell_n_sample[index]

        if self._discard_exploration and self.explored:
            start = self.shell_end_exp[index]
            shell_n_sample -= self.shell_n_sample_exp[index]
        else:
            start = 0

        log_l = self.log_l[index][start:]
        shell_n = len(log_l)
        self.shell_n[index] = shell_n

        if self.shell_n[index] > 0:
            self.shell_log_v[index] = (self.bounds[index].log_v
                                       + np.log(shell_n / shell_n_sample))
            self.shell_log_l[index] = logsumexp(log_l) - np.log(shell_n)
            if not np.all(log_l == -np.inf):
                self.shell_n_eff[index] = np.exp(
                    2 * logsumexp(log_l) - logsumexp(2 * log_l))
            else:
                self.shell_n_eff[index] = len(log_l)
        else:
            self.shell_log_v[index] = -np.inf
            self.shell_log_l[index] = np.nan
            self.shell_n_eff[index] = 0

    def shell_association(self, points, n_max=None):
        """Determine which shell each point belongs to."""
        if n_max is None:
            n_max = len(self.bounds)
        points = np.asarray(points)
        if len(points) == 0:
            return np.repeat(-1, 0)
        # One batched containment call over all candidate bounds.  The highest
        # bound containing a point wins: ``argmax`` returns the first True, so
        # scanning the reversed bound order and flipping the index reproduces
        # the per-bound loop (which tested undecided points bound by bound,
        # from the end) with identical results.
        bounds = self.bounds[:n_max]
        masks = np.asarray(_multi_bound_contains(bounds, points))
        assigned = masks.any(axis=1)
        highest = len(bounds) - 1 - np.argmax(masks[:, ::-1], axis=1)
        return np.where(assigned, highest, -1)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def print_status(self, status='', header=False, end='\n'):
        """Print a one-line progress summary."""
        if header:
            data = ['Status', 'Bounds', 'Ellipses', 'Networks', 'Calls',
                    'f_live', 'N_eff', 'log Z']
        else:
            data = [status, len(self.bounds)]
            if len(self.bounds) > 1:
                data.extend([self.bounds[-1].n_ell, self.bounds[-1].n_net])
            else:
                data.extend([0, 0])
            # ``f_live`` assumes shell counts and stored points agree, which
            # holds everywhere except between appending a batch and refreshing
            # the shell summary -- exactly when a progress line can be printed.
            # Show 'N/A' there instead of failing the run.
            try:
                f_live = self.f_live
            except ValueError:
                f_live = None
            data.extend([self.n_like, f_live, self.n_eff, self.log_z])
            fmt = ['{}', '{:d}', '{:d}', '{:d}', '{:d}', '{:.4f}', '{:.0f}',
                   '{:+.2f}']
            for i in range(len(data)):
                data[i] = 'N/A' if data[i] is None else fmt[i].format(data[i])
        for i, length in enumerate([9, 6, 8, 8, 8, 6, 5, 7]):
            data[i] = '{:<{}}'.format(data[i], length)
        print(' | '.join(data), end=end, flush=True)

    @property
    def f_live(self):
        """Estimated fraction of the evidence still contained in the live set."""
        if self.explored:
            return None
        if np.sum(self.shell_n) == 0:
            return 1.0
        log_v = np.repeat(self.shell_log_v
                          - np.log(np.maximum(self.shell_n, 1)),
                          self.shell_n)
        log_l = np.concatenate(self.log_l)
        log_w = log_v + log_l
        log_w_live = log_w[np.argsort(log_l)][-self.n_live:]
        return float(np.exp(logsumexp(log_w_live) - logsumexp(log_w)))

    @property
    def log_v_live(self):
        """Log volume currently contained in the live set."""
        if len(self.bounds) == 0:
            return 1.0
        log_l = np.concatenate(self.log_l)
        log_v = np.repeat(self.shell_log_v
                          - np.log(np.maximum(self.shell_n, 1)), self.shell_n)
        log_v_live = log_v[np.argsort(log_l)][-self.n_live:]
        return float(logsumexp(log_v_live))

    @property
    def n_eff(self):
        """Estimated total effective sample size."""
        if np.all(self.shell_n_eff == 0):
            return 0.0
        select = self.shell_n_eff > 0
        sum_w = np.exp(self.shell_log_l + self.shell_log_v
                       - np.nanmax(self.shell_log_l + self.shell_log_v))[select]
        sum_w_sq = sum_w ** 2 / self.shell_n_eff[select]
        return float(np.sum(sum_w) ** 2 / np.sum(sum_w_sq))

    @property
    def log_z(self):
        """Estimated log evidence."""
        if np.sum(self.shell_n) == 0:
            return None
        select = ~np.isnan(self.shell_log_l)
        return float(logsumexp(self.shell_log_l[select]
                               + self.shell_log_v[select]))

    @property
    def eta(self):
        """Estimated asymptotic sampling efficiency."""
        shell_log_z = self.shell_log_l + self.shell_log_v
        shell_eta = self.shell_n_eff / np.maximum(self.shell_n, 1)
        select = ~np.isnan(self.shell_log_l)
        return float(np.exp(
            2 * logsumexp(shell_log_z[select])
            - 2 * logsumexp(shell_log_z[select]
                            - 0.5 * np.log(shell_eta[select]))))

    def evidence(self):
        """Deprecated alias of :attr:`log_z` (kept for API parity)."""
        warn("The function 'evidence' is deprecated. Please use the property "
             "'log_z', instead.", DeprecationWarning, stacklevel=2)
        return self.log_z

    def effective_sample_size(self):
        """Deprecated alias of :attr:`n_eff` (kept for API parity)."""
        warn("The function 'effective_sample_size' is deprecated. Please use "
             "the property 'n_eff', instead.", DeprecationWarning, stacklevel=2)
        return self.n_eff

    def asymptotic_sampling_efficiency(self):
        """Deprecated alias of :attr:`eta` (kept for API parity)."""
        warn("The function 'asymptotic_sampling_efficiency' is deprecated. "
             "Please use the property 'eta', instead.", DeprecationWarning,
             stacklevel=2)
        return self.eta

    @property
    def discard_exploration(self):
        """Whether exploration-phase points are excluded from the results."""
        return self._discard_exploration

    @discard_exploration.setter
    def discard_exploration(self, discard_exploration):
        if not isinstance(discard_exploration, bool):
            raise TypeError("'discard_exploration' must be a bool.")
        self._discard_exploration = discard_exploration
        for index in range(len(self.log_l)):
            self.update_shell_info(index)

    def posterior(self, return_as_dict=None, equal_weight=False,
                  equal_weight_boost=1.0, return_blobs=False):
        """Return the posterior sample estimate."""
        if return_as_dict is None:
            return_as_dict = callable(self.prior) and self.pass_dict

        if self._discard_exploration and self.explored:
            start = self.shell_end_exp
        else:
            start = np.zeros(len(self.points), dtype=int)

        points = np.concatenate([p[s:] for p, s in zip(self.points, start)])
        log_v = np.repeat(self.shell_log_v
                          - np.log(np.maximum(self.shell_n, 1)), self.shell_n)
        log_l = np.concatenate([ll[s:] for ll, s in zip(self.log_l, start)])
        log_w = log_v + log_l
        if return_blobs:
            if self.blobs is None:
                raise ValueError('No blobs have been calculated.')
            blobs = np.concatenate([b[s:] for b, s in zip(self.blobs, start)])

        if equal_weight:
            repeats = np.exp(log_w - np.amax(log_w)) * equal_weight_boost
            repeats = np.floor(repeats).astype(int) + (
                self.rng.random(len(repeats)) < repeats - np.floor(repeats)
            ).astype(int)
            points = np.repeat(points, repeats, axis=0)
            log_w = np.zeros(np.sum(repeats))
            log_l = np.repeat(log_l, repeats, axis=0)
            if return_blobs:
                blobs = np.repeat(blobs, repeats, axis=0)

        points = self.evaluator.prior_to_physical(points, as_dict=return_as_dict)
        if not return_as_dict and callable(self.prior) and self.pass_dict:
            raise ValueError('Cannot return points as numpy array. The prior '
                             'function only returns dictionaries.')

        log_w = log_w - logsumexp(log_w)
        if return_blobs:
            return points, log_w, log_l, blobs
        return points, log_w, log_l

    def shell_bound_occupation(self, fractional=True):
        """Fraction of points of each shell that are part of each bound."""
        m = np.zeros((len(self.bounds), len(self.bounds)), dtype=int)
        for i, points in enumerate(self.points):
            for k, bound in enumerate(self.bounds):
                m[i, k] = int(np.sum(np.asarray(bound.contains(points))))
        if fractional:
            diag = np.diag(m).copy()
            diag[diag == 0] = 1
            m = m / diag[:, np.newaxis]
        return m

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _next_key(self):
        return jax.random.PRNGKey(
            int(self.rng.integers(0, 2 ** 31 - 1)))

    def _points_dtype(self):
        return jnp.float64

    def write(self, filepath, overwrite=False):
        """Write a checkpoint to disk."""
        from .checkpoint import write_sampler
        write_sampler(self, filepath, overwrite=overwrite)