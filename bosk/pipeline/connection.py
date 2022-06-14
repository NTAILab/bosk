from dataclasses import dataclass
from ..slot import BlockInputSlot, BlockOutputSlot


@dataclass(eq=True, frozen=True)
class Connection:
    src: BlockOutputSlot
    dst: BlockInputSlot
