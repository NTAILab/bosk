from dataclasses import dataclass
from typing import Mapping, List
from ..slot import BlockInputSlot, BlockOutputSlot, SlotT, list_of_slots_to_mapping


@dataclass(init=False)
class BlockMeta:
    inputs: Mapping[str, BlockInputSlot]
    outputs: Mapping[str, BlockOutputSlot]

    def __init__(self, *, inputs=None, outputs=None):
        assert inputs is not None, 'Meta inputs description is required'
        assert outputs is not None, 'Meta outputs description is required'
        self.inputs = list_of_slots_to_mapping(inputs)
        self.outputs = list_of_slots_to_mapping(outputs)
