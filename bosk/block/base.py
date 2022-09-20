from abc import ABC, abstractmethod, abstractproperty
from typing import Mapping, TypeVar

from .meta import BlockMeta
from ..data import Data
from ..slot import BlockOutputSlot
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

BlockOutputData = Mapping[BlockOutputSlot, Data]
"""Block output values container data type.

It is indexed by output slots, not their names.
"""


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

    def wrap(self, output_values: Mapping[str, Data]) -> BlockOutputData:
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
