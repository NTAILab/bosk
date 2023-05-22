"""Utility functions.

"""
import numpy as np
from numpy.random import Generator
from typing import Callable, Optional, Tuple, TypeVar, Union
from time import process_time


def get_random_generator(seed: Optional[Union[int, Generator]]) -> Generator:
    """Utility function to obtain random generator from a seed number.

    Args:
        seed: Random seed.

    Returns:
        Random number generator.

    """
    if isinstance(seed, Generator):
        return seed
    else:
        return np.random.default_rng(seed)


def get_rand_int(generator: Generator) -> int:
    """Utility function to obtain pseudo-random non-negative 32 bit integer value.

    Args:
        generator: Random number generator.

    Returns:
        Non-negative 32 bit integer value.

    """
    return generator.integers(0, np.iinfo(np.int32).max)


RT = TypeVar('RT')


def timer_wrap(func: Callable[..., RT]) -> Callable[..., Tuple[RT, float]]:
    """Decorator that returns execution CPU-time for a callable object.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function.
        It will return a tuple of wrapped function result and execution time.

    """
    def wrapper(*args, **kwargs):
        time_stamp = process_time()
        func_res = func(*args, **kwargs)
        exec_time = process_time() - time_stamp
        return func_res, exec_time
    return wrapper
