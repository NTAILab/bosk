from abc import ABC, ABCMeta, abstractmethod
from numpy.random import Generator
from typing import Mapping, TypeVar, Optional

from .meta import BlockMeta
from ..data import Data
from .slot import BlockInputSlot, BlockOutputSlot, BlockSlots
from ..visitor.base import BaseVisitor


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

BlockOutputData = Mapping[BlockOutputSlot, Data]
"""Block output values container data type.

It is indexed by output slots, not their names.
"""


class BaseBlock(ABC):
    """Base block, the parent of every computation block.

    Attributes:
        meta: Meta information of the block.

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

    def wrap(self, output_values: Mapping[str, Data]) -> BlockOutputData:
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

    def set_random_state(self, seed: Optional[int | Generator]) -> None:
        """Set random seed for the block using numpy
        random generator or integer value.
        """


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

    def get_single_input(self) -> BlockInputSlot:
        """Get the single block input slot.

        Returns:
            The block input slot.
        """
        assert len(self.slots.inputs) == 1
        return next(iter(self.slots.inputs.values()))


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
        assert len(self.slots.outputs) == 1
        return next(iter(self.slots.outputs.values()))
