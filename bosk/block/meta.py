from dataclasses import dataclass
from typing import Mapping, List
from ..slot import InputSlotMeta, OutputSlotMeta, list_of_slots_meta_to_mapping


@dataclass(init=False)
class BlockMeta:
    """Block meta, containing input and output slots description.

    Attributes:
        inputs: Mapping from input slots names to their meta.
        outputs: Mapping from output slots names to their meta.

    """
    inputs: Mapping[str, InputSlotMeta]
    outputs: Mapping[str, OutputSlotMeta]

    def __init__(self, *, inputs=None, outputs=None):
        assert inputs is not None, 'Meta inputs description is required'
        assert outputs is not None, 'Meta outputs description is required'
        self.inputs = list_of_slots_meta_to_mapping(inputs)
        self.outputs = list_of_slots_meta_to_mapping(outputs)


def make_simple_meta(input_names: List[str], output_names: List[str]) -> BlockMeta:
    """Make simple block meta from input and output slot names.

    Args:
        input_names: List of input slot names.
        output_names: List of output slot names.

    Returns:
        Block meta with given inputs and outputs.

    """
    return BlockMeta(
        inputs=[
            InputSlotMeta(name=name)
            for name in input_names
        ],
        outputs=[
            OutputSlotMeta(name=name)
            for name in output_names
        ]
    )
