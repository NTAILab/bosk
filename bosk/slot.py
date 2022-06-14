from dataclasses import dataclass
from typing import TypeVar
from .stages import Stages


@dataclass(eq=True, frozen=True)
class BaseSlot:
    """Base slot.

    Slot is a named placeholder for data.

    Attributes:
        name: Slot name.
        stages: At which stages slot value is needed.
    """
    name: str
    stages: Stages = Stages()


@dataclass(eq=True, frozen=True)
class BlockInputSlot(BaseSlot):
    """Block input slot.

    Contains the information required for the input data processing, and input-output matching.
    """


@dataclass(eq=True, frozen=True)
class BlockOutputSlot(BaseSlot):
    """Block output slot.

    Contains the information about the output data for input-output matching.
    """


SlotT = TypeVar('SlotT', bound=BaseSlot)
"""Slot generic typevar.
"""
