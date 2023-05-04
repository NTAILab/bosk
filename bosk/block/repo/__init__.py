"""Block repository module.

Contains repository classes that can be used to obtain block classes.

The block classes are located in :py:mod:`bosk.block.zoo`.

"""
from .base import BaseBlockClassRepository
from .scope import ScopeBlockClassRepository
from .zoo import ZooBlockClassRepository


DEFAULT_BLOCK_CLASS_REPOSITORY = ZooBlockClassRepository()


__all__ = [
    "BaseBlockClassRepository",
    "ScopeBlockClassRepository",
    "ZooBlockClassRepository",
    "DEFAULT_BLOCK_CLASS_REPOSITORY",
]
