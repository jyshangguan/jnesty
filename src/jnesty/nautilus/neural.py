"""Pure-JAX neural network emulator used by the neural bounds.

Reference implementation: :mod:`nautilus.neural`, which trains an ensemble of
``sklearn.neural_network.MLPRegressor`` models with hidden layers
``(100, 50, 20)``, ReLU activations, Adam, learning rate ``1e-2``, mini-batches
of at most 200 points and early stopping after 10 epochs without improvement.

The port keeps those defaults but stores the ensemble as a stacked pytree and
trains it with ``optax`` on the device.  Weights are float32 by default (the
emulator only needs to rank points well) while all sampler arithmetic stays in
float64.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

__all__ = ['Emulator', 'ReferenceEmulator', 'train_emulator']

_DEFAULT_HIDDEN = (100, 50, 20)
_DEFAULT_LR = 1e-2
_DEFAULT_BATCH = 200
_DEFAULT_MAX_EPOCHS = 10000
_DEFAULT_PATIENCE = 10

# Compiled training programs keyed by shape bucket so that repeated bound
# builds reuse one XLA executable instead of recompiling for every point count.
_COMPILED_TRAIN = {}

# Compiled prediction programs, bucketed like the training programs.
_COMPILED_PREDICT = {}

# Compiled normalization/padding programs, keyed by bucket, dimension and
# dtype.  Normalizing the *unbucketed* input eagerly used to compile a fresh
# collection of tiny programs for almost every bound's new point count.
_COMPILED_PREPARE = {}


def _bucket_size(n_points, minimum=64):
    """Round ``n_points`` up to a power of two (at least ``minimum``)."""
    n = max(int(n_points), minimum)
    return 1 << (n - 1).bit_length()


def _pad_to(x, y, n_target):
    """Pad the training set to ``n_target`` rows by repeating rows."""
    n = x.shape[0]
    if n == n_target:
        return x, y
    reps = int(np.ceil(n_target / n))
    x = jnp.concatenate([x] * reps, axis=0)[:n_target]
    y = jnp.concatenate([y] * reps, axis=0)[:n_target]
    return x, y


def _prepare_kernel(x_zeros, y_zeros, n_real):
    """Normalize real rows, then tile them to the compiled bucket shape.

    ``x_zeros``/``y_zeros`` are zero-padded on the host.  Statistics use only
    the first ``n_real`` rows, so they do not depend on the bucket.  The tiled
    rows are identical to :func:`_pad_to` applied after normalization.
    """
    n_bucket = x_zeros.shape[0]
    valid = jnp.arange(n_bucket) < n_real
    count = n_real.astype(x_zeros.dtype)
    mean = jnp.sum(x_zeros, axis=0) / count
    centered = jnp.where(valid[:, None], x_zeros - mean, 0.0)
    scale = jnp.sqrt(jnp.sum(centered ** 2, axis=0) / count)
    scale = jnp.where(scale > 0, scale, 1.0)
    indices = jnp.arange(n_bucket) % n_real
    x_tiled = x_zeros[indices]
    y_tiled = y_zeros[indices]
    return (x_tiled - mean) / scale, y_tiled, mean, scale


def _prepare_training_arrays(x, y, dtype):
    """Return bucketed ``(x, y, mean, scale)`` without per-row-count compiles.

    Padding and conversion are cheap host operations on the few hundred rows
    used by a neural bound.  One compiled program per bucket then performs the
    same normalization and repeated-row padding as the previous eager path.
    """
    dtype = np.dtype(dtype)
    x_np = np.asarray(x, dtype=dtype)
    y_np = np.asarray(y, dtype=dtype).reshape(-1)
    n_real, n_dim = x_np.shape
    n_target = _bucket_size(n_real)
    pad = n_target - n_real
    if pad:
        x_np = np.pad(x_np, ((0, pad), (0, 0)))
        y_np = np.pad(y_np, (0, pad))
    key = (n_target, n_dim, dtype.name)
    program = _COMPILED_PREPARE.get(key)
    if program is None:
        program = jax.jit(_prepare_kernel)
        _COMPILED_PREPARE[key] = program
    return program(jnp.asarray(x_np), jnp.asarray(y_np),
                   jnp.asarray(n_real, dtype=jnp.int32))


def _init_params(key, sizes, dtype):
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        limit = jnp.sqrt(6.0 / (n_in + n_out))
        w = jax.random.uniform(sub, (n_in, n_out), dtype,
                               minval=-limit, maxval=limit)
        params.append((w, jnp.zeros((n_out,), dtype)))
    return params


def _forward(params, x):
    """Forward pass of one network; ``params`` is a list of ``(W, b)``."""
    h = x
    n_layers = len(params)
    for i, (w, b) in enumerate(params):
        h = h @ w + b
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h[:, 0]


def _forward_ensemble(params, x):
    """Forward pass of the stacked ensemble; returns ``(n_networks, n_points)``."""
    n_networks = params[0][0].shape[0]
    h = jnp.broadcast_to(x, (n_networks,) + x.shape)
    n_layers = len(params)
    for i, (w, b) in enumerate(params):
        h = jnp.einsum('nio,npi->npo', w, h) + b[:, None, :]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h[:, :, 0]


def _mse(params, x, y):
    return jnp.mean((_forward(params, x) - y) ** 2)


def _train_one(x, y, key, hidden, learning_rate, batch_size,
               max_epochs, patience, dtype):
    """Train a single MLP with Adam and early stopping on the training loss."""
    n_points = x.shape[0]
    sizes = (x.shape[1],) + tuple(hidden) + (1,)

    split_key, key = jax.random.split(key)
    params0 = _init_params(split_key, sizes, dtype)

    # Mini-batch size.  With fewer points than one batch, train on all of them
    # in a single full batch: ``permutation(n_used)`` with n_used > n_points
    # would index past the end of the array, which JAX clamps silently to the
    # last row -- the batches then consist mostly of one repeated point and the
    # emulator never learns the contours.
    if n_points <= batch_size:
        n_batches = 1
        per_batch = n_points
    else:
        per_batch = batch_size
        n_batches = n_points // batch_size
    n_used = n_batches * per_batch
    x_trim = x[:n_used]
    y_trim = y[:n_used]

    optimizer = optax.adam(learning_rate)
    opt_state0 = optimizer.init(params0)

    def epoch(carry, key_epoch):
        params, opt_state = carry
        perm = jax.random.permutation(key_epoch, n_used)
        idx = perm.reshape(n_batches, per_batch)

        def batch(carry_b, ids):
            params, opt_state = carry_b
            _, grads = jax.value_and_grad(_mse)(
                params, x_trim[ids], y_trim[ids])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        (params, opt_state), _ = jax.lax.scan(batch, (params, opt_state), idx)
        loss = _mse(params, x, y)
        return (params, opt_state), loss

    def cond(carry):
        _, _, _, wait, epoch_idx, _ = carry
        return jnp.logical_and(wait < patience, epoch_idx < max_epochs)

    def body(carry):
        params, opt_state, best, wait, epoch_idx, key_state = carry
        key_state, sub = jax.random.split(key_state)
        (params, opt_state), loss = epoch((params, opt_state), sub)
        improved = loss < best
        best = jnp.minimum(best, loss)
        wait = jnp.where(improved, 0, wait + 1)
        return params, opt_state, best, wait, epoch_idx + 1, key_state

    params, _, _, _, _, _ = jax.lax.while_loop(
        cond, body, (params0, opt_state0, jnp.inf, 0, 0, key))
    return params


def _mse_networks(params, x, y):
    """Per-network training loss for stacked ``params``; ``(n_networks,)``.

    Same arithmetic as :func:`_mse`, but reduced over the points axis instead
    of over the whole array so that every ensemble member keeps its own loss.
    """
    return jnp.mean((_forward_ensemble(params, x) - y) ** 2, axis=1)


def _mse_networks_batched(params, x, y):
    """Per-network loss on one mini-batch.

    ``x`` carries a leading network axis -- ``(n_networks, per_batch, n_dim)``
    -- because each member draws its own permutation.  ``y`` is
    ``(n_networks, per_batch)``.
    """
    h = x
    n_layers = len(params)
    for i, (w, b) in enumerate(params):
        h = jnp.einsum('nio,npi->npo', w, h) + b[:, None, :]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return jnp.mean((h[:, :, 0] - y) ** 2, axis=1)


def _freeze(new, old, active):
    """Keep ``old`` for the leaves of members whose ``active`` flag is False.

    Used by the batched training loop to freeze members that have early
    stopped, so each one ends on exactly the parameters it would have reached
    if it had stopped alone.  Scalar leaves (Adam's step counter) keep the new
    value: they are shared by every member and are only needed by the members
    that are still training.
    """
    def pick(n, o):
        if jnp.ndim(o) == 0:
            return n
        return jnp.where(active.reshape((-1,) + (1,) * (o.ndim - 1)), n, o)

    return jax.tree_util.tree_map(pick, new, old)


def _train_many(x, y, key, n_networks, hidden, learning_rate, batch_size,
                max_epochs, patience, dtype):
    """Train every ensemble member in one program (one device launch).

    The per-member arithmetic is the same as :func:`_train_one`; only the
    epoch step is batched over the network axis.  Early stopping stays
    per network: a member that stops is frozen by :func:`_freeze`, and the
    shared loop runs until the last member stops.
    """
    n_points = x.shape[0]
    sizes = (x.shape[1],) + tuple(hidden) + (1,)

    keys = jax.random.split(key, n_networks)
    # Match ``_train_one``: the first split output initialises, the second
    # drives the epoch keys.  Splitting the batch of keys gives every member
    # the same pair of keys the sequential path hands it.
    split_keys = jax.vmap(lambda k: jax.random.split(k))(keys)
    params_keys = split_keys[:, 0]
    loop_keys = split_keys[:, 1]

    per_net = [_init_params(params_keys[i], sizes, dtype)
               for i in range(n_networks)]
    params0 = [(jnp.stack([p[layer][0] for p in per_net], axis=0),
                jnp.stack([p[layer][1] for p in per_net], axis=0))
               for layer in range(len(per_net[0]))]

    # Same mini-batch layout as ``_train_one``.
    if n_points <= batch_size:
        n_batches = 1
        per_batch = n_points
    else:
        per_batch = batch_size
        n_batches = n_points // batch_size
    n_used = n_batches * per_batch
    x_trim = x[:n_used]
    y_trim = y[:n_used]

    optimizer = optax.adam(learning_rate)
    opt_state0 = optimizer.init(params0)

    def epoch(carry, key_epoch):
        params, opt_state = carry
        perm = jax.vmap(
            lambda k: jax.random.permutation(k, n_used))(key_epoch)
        idx = jnp.swapaxes(
            perm.reshape(n_networks, n_batches, per_batch), 0, 1)

        def batch(carry_b, ids):
            params, opt_state = carry_b
            xb = jnp.take(x_trim, ids, axis=0)
            yb = jnp.take(y_trim, ids, axis=0)
            _, grads = jax.value_and_grad(
                lambda p: jnp.sum(_mse_networks_batched(p, xb, yb)))(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        (params, opt_state), _ = jax.lax.scan(batch, (params, opt_state), idx)
        loss = _mse_networks(params, x, y)
        return (params, opt_state), loss

    def cond(carry):
        _, _, _, wait, epoch_idx, _ = carry
        return jnp.logical_and(jnp.any(wait < patience),
                               epoch_idx < max_epochs)

    def body(carry):
        params, opt_state, best, wait, epoch_idx, key_state = carry
        active = wait < patience
        keys_split = jax.vmap(lambda k: jax.random.split(k))(key_state)
        key_state = keys_split[:, 0]
        sub = keys_split[:, 1]
        (params_new, opt_state_new), loss = epoch((params, opt_state), sub)
        improved = loss < best
        best = jnp.where(active, jnp.minimum(best, loss), best)
        wait = jnp.where(active, jnp.where(improved, 0, wait + 1), wait)
        params = _freeze(params_new, params, active)
        opt_state = _freeze(opt_state_new, opt_state, active)
        return params, opt_state, best, wait, epoch_idx + 1, key_state

    params, _, _, _, _, _ = jax.lax.while_loop(
        cond, body,
        (params0, opt_state0, jnp.full((n_networks,), jnp.inf, dtype),
         jnp.zeros((n_networks,), jnp.int32), 0, loop_keys))
    return params


def train_emulator(x, y, key, n_networks=4, hidden=_DEFAULT_HIDDEN,
                   learning_rate=_DEFAULT_LR, batch_size=_DEFAULT_BATCH,
                   max_epochs=_DEFAULT_MAX_EPOCHS, patience=_DEFAULT_PATIENCE,
                   dtype=jnp.float32, parallel=False):
    """Train an ensemble of MLP regressors on ``(x, y)``.

    Parameters
    ----------
    x : array (n_points, n_dim)
        Input coordinates (already transformed into ellipsoid coordinates).
    y : array (n_points,)
        Target scores.
    key : PRNG key
    n_networks : int, optional
        Ensemble size. Default is 4.
    parallel : bool, optional
        Train all ensemble members in one compiled program instead of one
        program per member (default False, bit-identical to the sequential
        path).  Early stopping remains per member: members that have stopped
        are frozen, and the shared loop runs until the last one stops.

    Returns
    -------
    Emulator
        Object with a ``predict(x)`` method.
    """
    n_target = _bucket_size(int(x.shape[0]))
    x_pad, y_pad, mean, scale = _prepare_training_arrays(x, y, dtype)

    cache_key = (n_target, int(x_pad.shape[1]), tuple(hidden),
                 float(learning_rate), int(batch_size), int(max_epochs),
                 int(patience), jnp.dtype(dtype).name, bool(parallel))
    if parallel:
        # The batched program is compiled for a fixed number of members.
        cache_key += (int(n_networks),)
    compiled = _COMPILED_TRAIN.get(cache_key)
    if compiled is None:
        if parallel:
            compiled = jax.jit(
                lambda xb, yb, k: _train_many(
                    xb, yb, k, n_networks, hidden, learning_rate, batch_size,
                    max_epochs, patience, dtype))
        else:
            compiled = jax.jit(
                lambda xb, yb, k: _train_one(
                    xb, yb, k, hidden, learning_rate, batch_size, max_epochs,
                    patience, dtype))
        _COMPILED_TRAIN[cache_key] = compiled

    if parallel:
        return Emulator(compiled(x_pad, y_pad, key), mean, scale)

    keys = jax.random.split(key, n_networks)
    trained = [compiled(x_pad, y_pad, keys[i]) for i in range(n_networks)]

    stacked = []
    for layer in range(len(trained[0])):
        w = jnp.stack([p[layer][0] for p in trained], axis=0)
        b = jnp.stack([p[layer][1] for p in trained], axis=0)
        stacked.append((w, b))

    return Emulator(stacked, mean, scale)


def _predict_kernel(mean, scale, params, x):
    """One compiled prediction program, mirroring :meth:`Emulator.predict`."""
    x_norm = (x - mean) / scale
    pred = _forward_ensemble(params, x_norm)
    return jnp.mean(pred, axis=0)


def _predict_program(bucket, ndim, dtype_name, n_networks, layer_shapes):
    """Compiled predict program for one bucketed input shape.

    The ensemble itself is passed as an argument (not closed over), so two
    emulators with the same layer shapes reuse one executable without ever
    seeing each other's weights.
    """
    key = (int(bucket), int(ndim), dtype_name, int(n_networks), layer_shapes)
    program = _COMPILED_PREDICT.get(key)
    if program is None:
        program = jax.jit(_predict_kernel)
        _COMPILED_PREDICT[key] = program
    return program


class Emulator:
    """Trained MLP ensemble with a ``predict`` method."""

    def __init__(self, params, mean, scale):
        self.params = params
        self.mean = mean
        self.scale = scale
        # Layer shapes identify the compiled prediction program (``params`` are
        # dynamic arguments, so the weights never end up baked into a program).
        self._layer_shapes = tuple(
            (tuple(w.shape[1:]), tuple(b.shape[1:])) for w, b in params)

    def predict(self, x):
        """Mean prediction of the ensemble, in the target units."""
        dtype = self.params[0][0].dtype
        x = jnp.asarray(x, dtype=dtype)
        if x.ndim != 2:
            x_norm = (x - self.mean) / self.scale
            return jnp.mean(_forward_ensemble(self.params, x_norm), axis=0)

        n = int(x.shape[0])
        n_bucket = _bucket_size(n)
        if n_bucket != n:
            # Padded rows are sliced off; only their shape matters, and
            # ``jnp.pad`` allocates once (as in ``_jit.padded_eval``).
            x = jnp.pad(x, ((0, n_bucket - n), (0, 0)))
        program = _predict_program(n_bucket, x.shape[1], dtype.name,
                                   self.n_networks, self._layer_shapes)
        return program(self.mean, self.scale, self.params, x)[:n]

    @property
    def n_networks(self):
        """Number of networks in the ensemble."""
        return self.params[0][0].shape[0]

class ReferenceEmulator:
    """The reference (scikit-learn) emulator, used as a parity oracle.

    ``neural_backend='reference'`` trains upstream's
    :class:`nautilus.neural.NeuralNetworkEmulator` instead of the device
    network.  Bound geometry is then identical to upstream and only the random
    stream differs, which is what makes "is our sampler behaving like the
    original?" a question with an answer.  Prediction runs on the host; the
    purpose of this path is fidelity, not speed.
    """

    def __init__(self, emulator):
        self._emulator = emulator

    @classmethod
    def train(cls, x, y, n_networks=4, neural_network_kwargs=None):
        """Train the reference ensemble on ``(x, y)`` (NumPy in, NumPy out)."""
        from nautilus.neural import NeuralNetworkEmulator

        kwargs = dict(neural_network_kwargs or {})
        hidden = kwargs.pop('hidden', None)
        if hidden is not None:
            kwargs.setdefault('hidden_layer_sizes', tuple(hidden))
        for unsupported in ('learning_rate', 'batch_size', 'max_epochs',
                            'patience', 'dtype'):
            kwargs.pop(unsupported, None)
        return cls(NeuralNetworkEmulator.train(
            np.asarray(x), np.asarray(y), n_networks=n_networks,
            neural_network_kwargs=kwargs))

    def predict(self, x):
        """Mean ensemble prediction, on the host."""
        return self._emulator.predict(np.asarray(x))

    @property
    def n_networks(self):
        """Number of networks in the ensemble."""
        return len(self._emulator.neural_networks)
