from dataclasses import dataclass
from ..block.base import BlockInputSlot, BlockOutputSlot
from ..visitor.base import BaseVisitor


@dataclass(eq=True, frozen=True)
class Connection:
    """Connection between two blocks in a pipeline.

    It connects the output slot of the source block
    to the input slot of the destination block.

    """
    src: BlockOutputSlot
    """The source block (output slot)."""
    dst: BlockInputSlot
    """The destination block (input slot)."""

    def accept(self, visitor: BaseVisitor):
        """Accept the visitor.

        Args:
            visitor: The visitor which can visit connections.

        """
        visitor.visit(self)
