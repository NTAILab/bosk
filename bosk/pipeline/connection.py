from dataclasses import dataclass
from ..block.slot import BlockInputSlot, BlockOutputSlot
from ..visitor.base import BaseVisitor


@dataclass(eq=True, frozen=True)
class Connection:
    src: BlockOutputSlot
    dst: BlockInputSlot

    def accept(self, visitor: BaseVisitor):
        """Accept the visitor.

        Args:
            visitor: The visitor which can visit connections.

        """
        visitor.visit(self)
