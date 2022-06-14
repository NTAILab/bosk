from abc import ABC, abstractmethod
from typing import Mapping, TypeVar, Type
from .meta import BlockMeta
from ..slot import BlockInputSlot, BlockOutputSlot
from ..data import Data
from ..stages import Stages


BlockT = TypeVar('BlockT', bound='BaseBlock')
"""Block generic typevar.
"""


BlockInputData = Mapping[BlockInputSlot, Data]
"""Block input values container data type.

It is indexed by input slots, not their names.
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
    def meta(self):
        ...

    @meta.getter
    @abstractmethod
    def meta(self):
        """Meta information property getter.

        Children classes must specify meta.
        It can be implemented as an attribute without redefining a property, for example::

            class StubBlock(BaseBlock):
                meta = BlockMeta(...)

        """

    @abstractmethod
    def fit(self, inputs: BlockInputData) -> BlockT:
        """Fit the block on the given input data.

        Args:
            inputs: Block input data for the fitting stage.

        Returns:
            Self.

        """

    @abstractmethod
    def transform(self, inputs: BlockInputData) -> BlockOutputData:
        """Transform the given input data, i.e. compute values for each output slot.

        Args:
            inputs: Block input data for the transforming stage.

        Returns:
            Outputs calculated for the given inputs.

        """

    def get(self, inputs: BlockInputData, slot_name: str) -> Data:
        """Get input value by name.

        Args:
            inputs: Block inputs, passed to the instance.
            slot_name: Input slot name.

        Returns:
            Input data value.

        """
        return inputs[self.meta.inputs[slot_name]]

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


def auto_block(cls: Type[BaseBlock]):
    """Convert any fit-transform class into a block.
    
    Args:
        cls: Class with fit-transform interface.

    Returns:
        Block class.
    
    """
    fit_argnames = set(cls.fit.__code__.co_varnames[1:cls.fit.__code__.co_argcount])
    transform_argnames = set(cls.transform.__code__.co_varnames[1:cls.transform.__code__.co_argcount])

    class AutoBlock(BaseBlock):
        meta = BlockMeta(
            inputs=[
                BlockInputSlot(
                    name=name,
                    stages=Stages(
                        fit=(name in fit_argnames),
                        transform=(name in transform_argnames)
                    )
                )
                for name in fit_argnames | transform_argnames
            ],
            outputs=[
                BlockOutputSlot(name='output')
            ],
        )

        def __init__(self, *args, **kwargs):
            self.__instance = cls(*args, **kwargs)

        def __prepare_kwargs(self, inputs: BlockInputData) -> Mapping[BlockInputSlot, Data]:
            kwargs = {
                slot_name: self.get(inputs, slot_name)
                for slot_name, slot in self.meta.inputs.items()
                if slot in inputs
            }
            return kwargs

        def fit(self, inputs: BlockInputData) -> 'AutoBlock':
            self.__instance.fit(**self.__prepare_kwargs(inputs))
            return self
        
        def transform(self, inputs: BlockInputData) -> BlockOutputData:
            transformed = self.__instance.transform(**self.__prepare_kwargs(inputs))
            return self.wrap({'output': transformed})

    return AutoBlock
