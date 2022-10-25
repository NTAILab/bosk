from abc import ABC, abstractmethod, abstractproperty
from typing import Mapping, TypeVar, List
from dataclasses import dataclass

# from .meta import BlockMeta
from ..data import Data
# from .slot import BlockOutputSlot, BlockInputSlot
from ..stages import Stages




BlockT = TypeVar('BlockT', bound='BaseBlock')
"""Block generic typevar.

Required to constrain a block to return itself in `fit(...)`.
"""

BlockInputData = Mapping[str, Data]
"""Block input values container data type.

It is indexed by input slot names.
"""

TransformOutputData = Mapping[str, Data]
"""Block transform output values container data type.

It is indexed by output slot names.
"""

# BlockOutputData = Mapping[BlockOutputSlot, Data]
# """Block output values container data type.

# It is indexed by output slots, not their names.
# """


class BaseBlock(ABC):
    """Base block, the parent of every computation block.

    Attributes:
        meta: Meta information of the block.

    """
    @property
    @abstractmethod
    def meta(self):
        """Meta information property getter.

        Children classes must specify meta.
        It can be implemented as an attribute without redefining a property, for example::

            class StubBlock(BaseBlock):
                meta = BlockMeta(...)

        """

    @abstractmethod
    def fit(self: BlockT, inputs: BlockInputData) -> BlockT:
        """Fit the block on the given input data.

        Args:
            inputs: Block input data for the fitting stage.

        Returns:
            Self.

        """

    @abstractmethod
    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        """Transform the given input data, i.e. compute values for each output slot.

        Args:
            inputs: Block input data for the transforming stage.

        Returns:
            Outputs calculated for the given inputs.

        """

    def wrap(self, output_values: Mapping[str, Data]):# -> BlockOutputData: # change was made!
        """Wrap outputs dictionary into ``BlockOutputs`` object.

        Args:
            output_values: Dictionary of values indexed by slot names.

        Returns:
            Block outputs object indexed by slots.

        """
        return {
            self.meta.outputs[slot_name]: value
            for slot_name, value in output_values.items()
        }

    def make_simple_meta(self, input_names: List[str], output_names: List[str]):
        return BlockMeta(
            inputs=[
                BlockInputSlot(name=name, parent_block=self)
                for name in input_names
            ],
            outputs=[
                BlockOutputSlot(name=name, parent_block=self)
                for name in output_names
            ]
        )


@dataclass(eq=False, frozen=False)
class BaseSlot:
    """Base slot.

    Slot is a named placeholder for data.

    Attributes:
        name: Slot name.
        stages: At which stages slot value is needed.
        debug_info: Debugging info.

    """
    # suggestion
    parent_block: BaseBlock # I had to refactor files because of the circular dependency problem

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


@dataclass(init=False)
class BlockMeta:
    inputs: Mapping[str, BlockInputSlot]
    outputs: Mapping[str, BlockOutputSlot]

    def __init__(self, *, inputs=None, outputs=None):
        assert inputs is not None, 'Meta inputs description is required'
        assert outputs is not None, 'Meta outputs description is required'
        self.inputs = list_of_slots_to_mapping(inputs)
        self.outputs = list_of_slots_to_mapping(outputs)