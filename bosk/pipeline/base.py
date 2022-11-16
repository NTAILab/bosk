from dataclasses import dataclass
from typing import List, Dict, Sequence
from .connection import Connection
from ..slot import BlockInputSlot, BlockOutputSlot
from ..block.base import BaseBlock


class BasePipeline:
    """The computational graph itself.

    Attributes:
        nodes (List[BaseBlock]): Nodes of the computational graph,
            which are blocks of the deep forest.
        connections (List[Connection]): Edges of the computational graph,
            head is the BlockOutputSlot and tail is BlockInputSlot.
        inputs (Dict[str, BlockInputSlot]): Inputs of the computational graph.
            Executors, proceeding the pipeline, can use some of them.
        outputs (Dict[str, BlockInputSlot]): Outputs of the computational graph.
            Executors, proceeding the pipeline, can use some of them.
    """


    def __init__(self, nodes: List[BaseBlock], connections: List[Connection],
        inputs: Dict[str, BlockInputSlot], outputs: Dict[str, BlockOutputSlot]) -> None:
        self.nodes = nodes
        self.connections = connections
        self.inputs = inputs
        self.outputs = outputs

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
