from abc import ABC, abstractmethod
from ..base import BasePipeline
from ...block.base import BaseBlock


class BaseBlockSerializer(ABC):
    """Base Block Serializer.

    Can serialize blocks, and could be used as a component of pipeline serialization.
    """
    @abstractmethod
    def dump(self, block: BaseBlock, out_file):
        """Serialize and dump the pipeline to the file.

        Args:
            pipeline: The pipeline to serialize.
            out_file: Output file or stream.

        """
        ...

    @abstractmethod
    def load(self, in_file) -> BaseBlock:
        """Load and deserialize a block from the file.

        Args:
            in_file: Input file or stream.

        """
        ...


class BasePipelineSerializer(ABC):
    """Base Pipeline Serializer.

    Can be used for the whole pipeline serialization.
    """
    @abstractmethod
    def dump(self, pipeline: BasePipeline, out_file):
        """Serialize and dump the pipeline to the file.

        Args:
            pipeline: The pipeline to serialize.
            out_file: Output file or stream.

        """
        ...

    @abstractmethod
    def load(self, in_file) -> BasePipeline:
        """Load and deserialize a pipeline from the file.

        Args:
            in_file: Input file or stream.

        """
        ...
