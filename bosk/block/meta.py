from dataclasses import dataclass
from typing import Mapping, List
from ..stages import Stages


@dataclass(eq=True, frozen=True)
class BaseSlotMeta:
    """Base slot meta information.

    Slot meta is unique for a block slot.

    Attributes:
        name: Slot name.
    """
    name: str


@dataclass(eq=True, frozen=True)
class InputSlotMeta(BaseSlotMeta):
    """Block input slot meta.

    Attributes:
        stages: At which stages slot value is needed.
    """
    stages: Stages = Stages()


@dataclass(eq=True, frozen=True)
class OutputSlotMeta(BaseSlotMeta):
    """Block output slot meta.
    """


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
        self.inputs = BlockMeta.list_of_slots_meta_to_mapping(inputs)
        self.outputs = BlockMeta.list_of_slots_meta_to_mapping(outputs)
        if execution_props is None:  # use default properties
            execution_props = BlockExecutionProperties()
        self.execution_props = execution_props

    @staticmethod
    def list_of_slots_meta_to_mapping(slots_meta_list: List[BaseSlotMeta]) -> Mapping[str, BaseSlotMeta]:
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


class DynamicBlockMetaStub(BlockMeta):
    """Meta stub for blocks with dynamic meta.

    Usually blocks define static meta, but sometimes meta can be defined at
    block initialization time.

    In this case, the stub can be used to express that the meta is dynamic.
    """

    def __init__(self, *, inputs=None, outputs=None, execution_props=None):
        ...


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
