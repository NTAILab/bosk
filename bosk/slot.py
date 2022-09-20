from dataclasses import dataclass
from typing import List, Mapping, TypeVar
from .stages import Stages


@dataclass(eq=False, frozen=False)
class BaseSlot:
    """Base slot.

    Slot is a named placeholder for data.

    Attributes:
        name: Slot name.
        stages: At which stages slot value is needed.
        debug_info: Debugging info.

    """
    name: str
    stages: Stages = Stages()
    debug_info: str = ""

    def __hash__(self) -> int:
        return id(self)


@dataclass(eq=False, frozen=False)
class BlockInputSlot(BaseSlot):
    """Block input slot.

    Contains the information required for the input data processing, and input-output matching.
    """


@dataclass(eq=False, frozen=False)
class BlockOutputSlot(BaseSlot):
    """Block output slot.

    Contains the information about the output data for input-output matching.
    """


SlotT = TypeVar('SlotT', bound=BaseSlot)
"""Slot generic typevar.
"""


def list_of_slots_to_mapping(slots_list: List[SlotT]) -> Mapping[str, SlotT]:
    """Convert list of slots to mapping (name -> slot).

    Args:
        slots_list: List of slots.

    Returns:
        Mapping dict (name -> slot).

    """
    return {
        slot.name: slot
        for slot in slots_list
    }
