from dataclasses import dataclass
from typing import Sequence
from .connection import Connection
from ..slot import BlockInputSlot, BlockOutputSlot
from ..block.base import BaseBlock


@dataclass
class BasePipeline:
    nodes: Sequence[BaseBlock]
    connections: Sequence[Connection]

    def find_connections(self, *,
                        src: None | BlockOutputSlot = None,
                        dst: None | BlockInputSlot = None) -> Sequence[Connection]:
        """Find connections by src or dst.

        Args:
            src: Source slot.
            dst: Destination slot.

        Returns:
            List of connections.

        """
        assert src is not None or dst is not None, "Either src or dst should be specified"
        results = []
        for conn in self.connections:
            if src is not None and conn.src == src:
                results.append(conn)
            if dst is not None and conn.dst == dst:
                results.append(conn)
        return results
