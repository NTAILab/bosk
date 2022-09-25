from typing import Optional, Type, Mapping
from .base import BaseBlock, BlockInputData, BlockOutputData, TransformOutputData
from .meta import BlockMeta
from ..slot import BlockInputSlot, BlockOutputSlot
from ..data import Data
from ..stages import Stages


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
        meta: Optional[BlockMeta] = None

        def __init__(self, *args, **kwargs):
            self.__instance = cls(*args, **kwargs)
            self.meta = BlockMeta(
                inputs=[
                    BlockInputSlot(
                        name=name,
                        stages=Stages(
                            fit=(name in fit_argnames),
                            transform=(name in transform_argnames)
                        ),
                    )
                    for name in fit_argnames | transform_argnames
                ],
                outputs=[
                    BlockOutputSlot(name='output')
                ],
            )

        def __prepare_kwargs(self, inputs: BlockInputData) -> Mapping[str, Data]:
            assert self.meta is not None
            kwargs = {
                slot_name: inputs[slot_name]
                for slot_name, _slot in self.meta.inputs.items()
                if slot_name in inputs
            }
            return kwargs

        def fit(self, inputs: BlockInputData) -> 'AutoBlock':
            self.__instance.fit(**self.__prepare_kwargs(inputs))
            return self

        def transform(self, inputs: BlockInputData) -> TransformOutputData:
            transformed = self.__instance.transform(**self.__prepare_kwargs(inputs))
            return {'output': transformed}

    return AutoBlock
