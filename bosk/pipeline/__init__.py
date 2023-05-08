"""Pipeline that defines a Deep Forest structure.

Pipelines are represented as sets of nodes (blocks) and connections between them.

There are two ways to define a pipeline:

1. Manually, by defining a list of nodes and connections.
2. Using the builder in functional style.

"""

from .base import BasePipeline
from .dynamic import BaseDynamicPipeline
from .connection import Connection


__all__ = [
    "BasePipeline",
    "BaseDynamicPipeline",
    "Connection",
]
