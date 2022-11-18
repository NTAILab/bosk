from dataclasses import dataclass
from typing import List, Dict
from .connection import Connection
from ..slot import BlockInputSlot, BlockOutputSlot
from ..block.base import BaseBlock


@dataclass(frozen=True)
class BasePipeline:
    """The computational graph itself.

    Attributes:
        nodes: Nodes of the computational graph,
            which are blocks of the deep forest.
        connections: Edges of the computational graph,
            head is the BlockOutputSlot and tail is BlockInputSlot.
        inputs: Inputs of the computational graph.
            Executors, proceeding the pipeline, can use some of them.
        outputs: Outputs of the computational graph.
            Executors, proceeding the pipeline, can use some of them.
    """
    nodes: List[BaseBlock]
    connections: List[Connection]
    inputs: Dict[str, BlockInputSlot]
    outputs: Dict[str, BlockOutputSlot]
