"""Random-key state shared by the bound classes.

Nautilus calls ``sample()`` in two ways: with an explicit key while a sampler
is drawing a batch, and without one while a bound estimates its own volume.
A constant default key would make the second kind repeat the same proposals
on every call.  These helpers keep a per-object JAX key and advance it for
both calling conventions.
"""

import jax


def take_key(owner, key=None):
    """Return a fresh key and store the stream position on ``owner``.

    Passing a key explicitly uses that key and records it as the new stream
    position; passing ``None`` splits the owner's current key.  Consequently
    a sequence of no-key calls never repeats, and mixing keyed and unkeyed
    calls also continues rather than restarting the stream.
    """
    if key is not None:
        owner._sample_key = key
        return key

    current = getattr(owner, '_sample_key', None)
    if current is None:
        current = jax.random.PRNGKey(0)
    owner._sample_key, sub = jax.random.split(current)
    return sub


def finish_key(owner, key):
    """Record the position after a sampling operation."""
    owner._sample_key = key
    return key