from ..base import BasePipeline
from abc import ABC, abstractmethod


class BasePipelineBuilder(ABC):
    """Base pipeline builder, the parent of every pipeline builder.

    Attributes:
        pipeline: Built pipeline.

    """
    @property
    @abstractmethod
    def pipeline(self) -> BasePipeline:
        """Get pipeline (optionally after building).

        Returns:
            Build pipeline.

        """
        ...
