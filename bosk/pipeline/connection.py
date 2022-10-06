from dataclasses import dataclass
from ..slot import BlockInputSlot, BlockOutputSlot, OutputSlotMeta, InputSlotMeta


@dataclass(eq=True, frozen=True)
class Connection:
    src: OutputSlotMeta
    dst: InputSlotMeta
