"""
Global random seed of physXAI.

`set_seed` seeds Python, NumPy and Keras (TensorFlow) once for a whole script and acts as the
default `random_state` of all physXAI objects. A `random_state` passed to an object explicitly
takes precedence. Without any seed, a fresh one is drawn and stored in the configs, so every run
can be reproduced afterwards.
"""

from __future__ import annotations

import random

import numpy as np

_global_seed: int | None = None


def set_seed(seed: int | None) -> None:
    """
    Sets the global seed of physXAI.

    Args:
        seed: The seed. None clears the global seed, so objects draw fresh seeds again.
    """
    global _global_seed
    if seed is None:
        _global_seed = None
        return
    _global_seed = int(seed)
    random.seed(_global_seed)
    np.random.seed(_global_seed % 2**32)  # noqa: NPY002 - seeds legacy code relying on the global state
    import keras

    keras.utils.set_random_seed(_global_seed % 2**32)


def get_seed() -> int | None:
    """Returns the global seed, or None if it is not set."""
    return _global_seed


def resolve_seed(random_state: int | None = None) -> int:
    """
    Determines the seed of an object.

    Args:
        random_state: Explicit seed of the object.

    Returns:
        int: `random_state` if given, else the global seed, else a freshly drawn seed.
    """
    if random_state is not None:
        return int(random_state)
    if _global_seed is not None:
        return _global_seed
    return int(np.random.SeedSequence().generate_state(1)[0])
