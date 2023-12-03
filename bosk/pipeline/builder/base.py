from ..base import BasePipeline
from ...block.functional import FunctionalBlockWrapper
from ...block.base import BlockInputSlot, BlockOutputSlot
from ...state import BoskState
from abc import ABC, abstractmethod
from typing import Mapping, Union


class BasePipelineBuilder(ABC):
    """Base pipeline builder, the parent of every pipeline builder.

    """
    @abstractmethod
    def build(self) -> BasePipeline:
        """Get pipeline (optionally after building).

        Returns:
            Build pipeline.

        """

    def __enter__(self):
        """Enter the builder scope.
        """
        BoskState().active_builders.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the builder scope.
        """
        assert BoskState().active_builders.peek() is self, "BoskState().active_builders corrupted"
        BoskState().active_builders.pop()
        return False

