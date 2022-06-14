from dataclasses import dataclass
from typing import Mapping, List
from ..slot import BlockInputSlot, BlockOutputSlot, SlotT


@dataclass(init=False)
class BlockMeta:
    inputs: Mapping[str, BlockInputSlot]
    outputs: Mapping[str, BlockOutputSlot]

    def __list_of_slots_to_mapping(self, slots_list: List[SlotT]) -> Mapping[str, SlotT]:
        return {
            slot.name: slot
            for slot in slots_list
        }

    def __init__(self, *, inputs=None, outputs=None):
        assert inputs is not None, 'Meta inputs description is required'
        assert outputs is not None, 'Meta outputs description is required'
        self.inputs = self.__list_of_slots_to_mapping(inputs)
        self.outputs = self.__list_of_slots_to_mapping(outputs)
