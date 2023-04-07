from dataclasses import dataclass, field
from typing import List, Mapping, Set, TypeVar
from ..stages import Stages


@dataclass(eq=True, frozen=True)
class BaseSlotMeta:
    """Base slot meta information.

    Slot meta is unique for a block slot.

    Attributes:
        name: Slot name.
        stages: At which stages slot value is needed.

    """
    name: str
    stages: Stages = Stages()


@dataclass(eq=True, frozen=True)
class InputSlotMeta(BaseSlotMeta):
    """Block input slot meta.
    """


@dataclass(eq=True, frozen=True)
class OutputSlotMeta(BaseSlotMeta):
    """Block output slot meta.
    """


BaseBlock = TypeVar('BaseBlock') # will be removed later

@dataclass(eq=False, frozen=False)
class BaseSlot:
    """Base slot.

    Slot is a named placeholder for data.

    Attributes:
        name: Slot name.
        stages: At which stages slot value is needed.
        debug_info: Debugging info.

    """
    meta: BaseSlotMeta
    parent_block: BaseBlock
    debug_info: str = ""

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return self.meta.name


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

SlotMetaT = TypeVar('SlotMetaT', bound=BaseSlotMeta)
"""Slot Meta generic typevar.
"""


def list_of_slots_to_mapping(slots_list: List[SlotT]) -> Mapping[str, SlotT]:
    """Convert list of slots to mapping (name -> slot).

    Args:
        slots_list: List of slots.

    Returns:
        Mapping dict (name -> slot).

    """
    raise NotImplementedError()
    # return {
    #     slot.name: slot
    #     for slot in slots_list
    # }

def list_of_slots_meta_to_mapping(slots_meta_list: List[SlotMetaT]) -> Mapping[str, SlotMetaT]:
    """Convert list of slots meta to mapping (name -> slot meta).

    Args:
        slots_meta_list: List of slots meta.

    Returns:
        Mapping dict (name -> slot meta).

    """
    return {
        slot_meta.name: slot_meta
        for slot_meta in slots_meta_list
    }


@dataclass(frozen=True)
class BlockGroup:
    name: str

    def add(self, block: BaseBlock):
        """Add the block to the group.

        Args:
            block: Block to add.

        """
        block.slots.groups.add(self)

    def remove(self, block: BaseBlock):
        """Remove the block from the group.

        Args:
            block: Block to remove.

        """
        block.slots.groups.remove(self)


@dataclass
class BlockSlots:
    inputs: Mapping[str, BlockInputSlot]
    outputs: Mapping[str, BlockOutputSlot]
    groups: Set[BlockGroup] = field(default_factory=lambda: set())
