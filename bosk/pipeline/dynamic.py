from dataclasses import dataclass
from typing import Optional, Sequence

from .base import BasePipeline
from .connection import Connection


class BaseDynamicPipeline(BasePipeline):
    """Dynamic pipeline.

    It is a pipeline that can be extended with other pipeline
    and connections between self and the other pipelien nodes.

    """
    def extend(self, other: BasePipeline,
               connections_extension: Optional[Sequence[Connection]] = None):
        """Extend current pipeline with other pipeline inplace.

        Args:
            other: Other pipeline.
            connections_extension: List of connections between
                                   current and `other` pipeline.

        """
        self.nodes.extend(other.nodes)
        self.connections.extend(other.connections)
        if connections_extension is not None:
            self.connections.extend(connections_extension)
