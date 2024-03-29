from abc import ABC, abstractmethod
from typing import Optional, Sequence, TypeVar
from ..pipeline.base import BasePipeline
from ..executor.base import BaseExecutor


class BasePainter(ABC):
    """Abstract base class for all pipeline painters.
    """

    @abstractmethod
    def from_pipeline(self, pipeline: BasePipeline) -> 'BasePainter':
        """Method for drawing the `pipeline` and saving the
        result into internal representation.
        """

    @abstractmethod
    def from_executor(self, executor: BaseExecutor) -> 'BasePainter':
        """Method for drawing the pipeline connected with the `executor`
        and saving the result into internal representation. The drawn
        graph should represent executor's behaviour.
        """

    @abstractmethod
    def render(self, output_filename: str, format: Optional[str] = None) -> None:
        """Method for saving drawn computational graph.

        Args:
            output_filename: Path (containing the filename) where the output graphics file will be saved.
            format: Format of the output file. If `None`, then the `output_filename` determines the format.
        """

    @property
    @abstractmethod
    def available_formats(self) -> Sequence[str]:
        """List of the available formats to render the
        computational graph into.
        """
