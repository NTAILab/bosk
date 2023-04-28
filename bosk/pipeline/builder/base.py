from ..base import BasePipeline
from abc import ABC, abstractmethod
from ...block.functional import FunctionalBlockWrapper
from typing import Mapping, Union
from ...block.base import BlockInputSlot, BlockOutputSlot


class BasePipelineBuilder(ABC):
    """Base pipeline builder, the parent of every pipeline builder.

    """
    @abstractmethod
    def build(self) -> BasePipeline:
        """Get pipeline (optionally after building).

        Returns:
            Build pipeline.

        """
