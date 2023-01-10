from dataclasses import dataclass
from typing import Mapping, List
from .slot import InputSlotMeta, OutputSlotMeta, list_of_slots_meta_to_mapping


@dataclass
class BlockExecutionProperties:  # TODO: move to a separate file
    """Block execution properties.

    Attributes:
        cpu: Has CPU implementation that handles CPU data.
        gpu: Has GPU implementation that handles GPU data.
        threadsafe: CPU/GPU implementation is thread safe, can be executed in parallel.
        plain: Block implementation is straightforward and not computantionally expensive,
               it should not be parallelized (parallelization costs are larger than computation).

    """
    cpu: bool = True
    gpu: bool = False
    threadsafe: bool = False
    plain: bool = False


@dataclass(init=False)
class BlockMeta:
    """Block meta, containing input and output slots description.

    Attributes:
        inputs: Mapping from input slots names to their meta.
        outputs: Mapping from output slots names to their meta.

    """
    inputs: Mapping[str, InputSlotMeta]
    outputs: Mapping[str, OutputSlotMeta]
    execution_props: BlockExecutionProperties

    def __init__(self, *, inputs=None, outputs=None, execution_props=None):
        assert inputs is not None, 'Meta inputs description is required'
        assert outputs is not None, 'Meta outputs description is required'
        self.inputs = list_of_slots_meta_to_mapping(inputs)
        self.outputs = list_of_slots_meta_to_mapping(outputs)
        if execution_props is None:  # use default properties
            execution_props = BlockExecutionProperties()
        self.execution_props = execution_props


def make_simple_meta(input_names: List[str], output_names: List[str], **additional_params) -> BlockMeta:
    """Make simple block meta from input and output slot names.

    Args:
        input_names: List of input slot names.
        output_names: List of output slot names.
        **additional_params: Additional meta params.

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
        ],
        **additional_params
    )
