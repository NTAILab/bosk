from dataclasses import dataclass
from typing import Mapping, List
from ..slot import InputSlotMeta, OutputSlotMeta, list_of_slots_meta_to_mapping


@dataclass(init=False)
class BlockMeta:
    inputs: Mapping[str, InputSlotMeta]
    outputs: Mapping[str, OutputSlotMeta]

    def __init__(self, *, inputs=None, outputs=None):
        assert inputs is not None, 'Meta inputs description is required'
        assert outputs is not None, 'Meta outputs description is required'
        self.inputs = list_of_slots_meta_to_mapping(inputs)
        self.outputs = list_of_slots_meta_to_mapping(outputs)
