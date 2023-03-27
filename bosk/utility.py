import numpy as np
from numpy.random import Generator
from typing import Optional


def get_random_generator(seed: Optional[int | Generator]) -> Generator:
    """Utility function to obtain random generator from a seed number."""
    if isinstance(seed, Generator):
        return seed
    else:
        return np.random.default_rng(seed)


def get_rand_int(generator: Generator) -> int:
    """Utility function to obtain single non-negative 32 bit integer value."""
    return generator.integers(0, np.iinfo(np.int32).max)
