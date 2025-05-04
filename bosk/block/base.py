from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from numpy.random import Generator
from typing import Dict, Mapping, Set, TypeVar, Optional, Union

from .meta import BaseSlotMeta, BlockMeta, InputSlotMeta, OutputSlotMeta
from ..data import BaseData
from ..visitor.base import BaseVisitor
from ..exceptions import MultipleBlockInputsError, MultipleBlockOutputsError, NoDefaultBlockOutputError


BlockT = TypeVar('BlockT', bound='BaseBlock')
"""Block generic typevar.

Required to constrain a block to return itself in `fit(...)`.
"""

SlotT = TypeVar('SlotT', bound='BaseSlot')
"""Slot generic typevar.
"""

SlotMetaT = TypeVar('SlotMetaT', bound='BaseSlotMeta')
"""Slot Meta generic typevar.
"""


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
    parent_block: 'BaseBlock'
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
    meta: InputSlotMeta


@dataclass(eq=False, frozen=False)
class BlockOutputSlot(BaseSlot):
    """Block output slot.

    Contains the information about the output data for input-output matching.
    """
    meta: OutputSlotMeta


@dataclass(frozen=True)
class BlockGroup:
    """Block group is an assiciated collection of blocks.

    Practically, each block has collection of groups to which the block belongs.

    Block groups can be used to express that some blocks belong to the same Deep Forest layer,
    and it helps to draw more understandable pipeline diagrams.

    """
    name: str
    """Block group name.
    """

    def add(self, block: BlockT):
        """Add the block to the group.

        Args:
            block: Block to add.

        """
        block.slots.groups.add(self)

    def remove(self, block: BlockT):
        """Remove the block from the group.

        Args:
            block: Block to remove.

        """
        block.slots.groups.remove(self)

    def __repr__(self) -> str:
        return self.name


@dataclass
class BlockSlots:
    """Collection of block input and output slots, as well as block groups.

    Attributes:
        inputs: Input slots.
        outputs: Output slots.
        groups: Block groups.

    """
    inputs: Mapping[str, BlockInputSlot]
    outputs: Mapping[str, BlockOutputSlot]
    groups: Set[BlockGroup] = field(default_factory=lambda: set())


BlockInputData = Dict[str, BaseData]
"""Block input values container data type.

It is indexed by input slot names.
"""

TransformOutputData = Dict[str, BaseData]
"""Block transform output values container data type.

It is indexed by output slot names.
"""

BlockOutputData = Mapping[BlockOutputSlot, BaseData]
"""Block output values container data type.

It is indexed by output slots, not their names.
"""


class BaseBlock(ABC):
    """Base block, the parent of every computation block.

    Block has meta information, that defines the inputs and outputs as well as block execution properties,
    and slots which are unique for each block instance and can be used to connect different
    block between each other.

    Main block methods are: `fit(...)` and `transform(...)`.
    Both accept dictionaries as input, `transform(...)` returns a dictionary with output data.

    Attributes:
        meta: Meta information of the block.
              May be shaped between different blocks.
              If the meta information cannot be specified at the class definition step,
              :py:class:`DynamicBlockMetaStub` should be used as a stub, and
              then redefined at the initialization step.
        slots: Block slots of type :py:class:`BlockSlots`, made dynamically at the initialization.
               Slots are unique for the block instance.

    """
    def __init__(self):
        super().__init__()
        self.slots = self._make_slots()

    def _make_slots(self):
        """Make slots"""
        return BlockSlots(
            inputs={
                name: BlockInputSlot(meta=input_slot_meta, parent_block=self)
                for name, input_slot_meta in self.meta.inputs.items()
            },
            outputs={
                name: BlockOutputSlot(meta=output_slot_meta, parent_block=self)
                for name, output_slot_meta in self.meta.outputs.items()
            },
        )

    @property
    @abstractmethod
    def meta(self) -> BlockMeta:
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

    def wrap(self, output_values: Mapping[str, BaseData]) -> BlockOutputData:
        """Wrap outputs dictionary into ``BlockOutputs`` object.

        Args:
            output_values: Dictionary of values indexed by slot names.

        Returns:
            Block outputs object indexed by slots.

        """
        return {
            self.slots.outputs[slot_name]: value
            for slot_name, value in output_values.items()
        }

    def __repr__(self) -> str:
        return self.__class__.__name__

    def accept(self, visitor: BaseVisitor):
        """Accept the visitor.

        Args:
            visitor: The visitor which can visit blocks.

        """
        visitor.visit(self)

    def set_random_state(self, seed: Optional[Union[int, Generator]]) -> None:
        """Set random seed for the block using numpy
        random generator or integer value.
        """

    @property
    def default_output(self) -> Optional[str]:
        """Get default output name.

        If the block has a single output, it will be used as a default.
        Otherwise, the block can override this property to set a specific default output.
        If the block can't have a single default output (its outputs have equal importance),
        this method should return `None`.
        """
        if len(self.meta.outputs) == 1:
            return next(iter(self.meta.outputs.keys()))
        else:
            return None

    def get_default_output(self) -> BlockOutputSlot:
        """Get the default block output slot.

        Returns:
            The block output slot.
        """
        default_output = self.default_output
        if default_output is None:
            raise NoDefaultBlockOutputError(self)
        return self.slots.outputs[default_output]

    def get_single_input(self) -> BlockInputSlot:
        """Get the single block input slot.

        Returns:
            The block input slot.
        """
        if len(self.slots.inputs) != 1:
            raise MultipleBlockInputsError(self)
        return next(iter(self.slots.inputs.values()))


class BaseInputBlock(BaseBlock, metaclass=ABCMeta):
    """Base input block. It is guaranteed that is has a single input and some name.

    An input block can help to automatically determine pipeline inputs.
    The name can be None, in this case the block is not considered as one of pipeline inputs.
    """
    def _make_slots(self):
        """Make slots"""
        result = super()._make_slots()
        assert len(result.inputs) == 1, f'The input block {self!r} must have exactly one input'
        return result

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """Get the input name.

        Returns:
            The block instance name.
        """


class BaseOutputBlock(BaseBlock, metaclass=ABCMeta):
    """Base output block. It is guaranteed that is has a single output and some name.

    An output block can help to automatically determine pipeline outputs.
    The name can be None, in this case the block is not considered as one of pipeline outputs.
    """
    def _make_slots(self):
        """Make slots"""
        result = super()._make_slots()
        assert len(result.outputs) == 1, f'The output block {self!r} must have exactly one output'
        return result

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """Get the input name.

        Returns:
            The block instance name or None if the block does not have name.
        """

    def get_single_output(self) -> BlockOutputSlot:
        """Get the single block output slot.

        Returns:
            The block output slot.
        """
        if len(self.slots.outputs) != 1:
            raise MultipleBlockOutputsError(self)
        return next(iter(self.slots.outputs.values()))
