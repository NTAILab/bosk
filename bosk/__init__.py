__version__ = '0.1'


from .data import Data, BaseData, CPUData, GPUData
from .stages import Stages, Stage
from .utility import get_random_generator, get_rand_int, timer_wrap


__all__ = [
    # modules
    "auto",
    "block",
    "comparison",
    "executor",
    "painter",
    "pipeline",
    "visitor",
    # objects
    "Data",
    "BaseData",
    "CPUData",
    "GPUData",
    "Stages",
    "Stage",
    "get_random_generator",
    "get_rand_int",
    "timer_wrap"
]
