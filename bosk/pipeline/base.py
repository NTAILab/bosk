from dataclasses import dataclass
from numpy.random import Generator
from typing import List, Dict
from .connection import Connection
from ..block.slot import BlockInputSlot, BlockOutputSlot
from ..block.base import BaseBlock
from ..utility import get_random_generator
from ..visitor.base import BaseVisitor


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

    def accept(self, visitor: BaseVisitor):
        """Accept the visitor.

        Args:
            visitor: The visitor which can visit pipelines, nodes and connections.

        """
        for node in self.nodes:
            node.accept(visitor)

        for conn in self.connections:
            conn.accept(visitor)

        visitor.visit(self)

    def set_random_state(self, seed: int | Generator) -> None:
        """Set random seed for each block in the pipeline."""
        gen = get_random_generator(seed)
        for block in self.nodes:
            block.set_random_state(gen)
