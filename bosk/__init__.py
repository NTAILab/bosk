__version__ = '0.1'


from .data import Data
from .stages import Stages
# from .block.slot import BaseSlot, BlockInputSlot, BlockOutputSlot, SlotT
from .block.base import BaseSlot, BlockInputSlot, BlockOutputSlot


__all__ = [
    "Data",
    "Stages",
    "BaseSlot",
    "BlockInputSlot",
    "BlockOutputSlot",
    "SlotT",
]
