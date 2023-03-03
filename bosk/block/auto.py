from typing import Optional, Type, Mapping
from .base import BaseBlock, BlockInputData, BlockOutputData, TransformOutputData
from .meta import BlockMeta, BlockExecutionProperties
from .slot import BlockInputSlot, BlockOutputSlot, InputSlotMeta, OutputSlotMeta
from ..data import Data
from ..stages import Stages
from functools import wraps


def auto_block(_implicit_cls=None,
               execution_props: Optional[BlockExecutionProperties] = None,
               auto_state: bool = False):
    """Decorator for conversion from a fit-transform class into a block.

    Args:
        _implicit_cls: Class with fit-transform interface.
                       It is automatically passed when `@auto_block` without
                       brackets is used. Otherwise it should be `None`.
        execution_props: Custom block execution properties.
        auto_state: Automatically implement `__getstate__` and `__setstate__` methods.
                    These methods are required for serialization.

    Returns:
        Block wrapping function.

    """
    def _auto_block_impl(cls: Type[BaseBlock]):
        """Make auto block wrapper instance.

        Args:
            cls: Class with fit-transform interface.

        Returns:
            Block class.

        """
        fit_argnames = set(cls.fit.__code__.co_varnames[1:cls.fit.__code__.co_argcount])
        transform_argnames = set(cls.transform.__code__.co_varnames[1:cls.transform.__code__.co_argcount])

        @wraps(cls, updated=())
        class AutoBlock(BaseBlock):
            meta: Optional[BlockMeta] = BlockMeta(
                inputs=[
                    InputSlotMeta(
                        name=name,
                        stages=Stages(
                            fit=(name in fit_argnames),
                            transform=(name in transform_argnames)
                        ),
                    )
                    for name in fit_argnames | transform_argnames
                ],
                outputs=[
                    OutputSlotMeta(name='output')
                ],
                execution_props=execution_props
            )

            def __init__(self, *args, **kwargs):
                super().__init__()
                self.__instance = cls(*args, **kwargs)
                self.__args = args
                self.__kwargs = kwargs

            def __prepare_kwargs(self, inputs: BlockInputData) -> Mapping[str, Data]:
                kwargs = {
                    slot_name: inputs[slot_name].data
                    for slot_name, _slot in self.slots.inputs.items()
                    if slot_name in inputs
                }
                return kwargs

            def fit(self, inputs: BlockInputData) -> 'AutoBlock':
                self.__instance.fit(**self.__prepare_kwargs(inputs))
                return self

            def transform(self, inputs: BlockInputData) -> TransformOutputData:
                transformed = self.__instance.transform(**self.__prepare_kwargs(inputs))
                return {'output': transformed}

            def __get_instance_state(self):
                if hasattr(self.__instance, '__getstate__'):
                    return self.__instance.__getstate__()
                if auto_state:
                    return self.__instance.__dict__
                else:
                    raise NotImplementedError(
                        f"{type(self.__instance)!r} has no '__getstate__' implementation. "
                        "Please, implement this method or use `@auto_block(auto_state=True)` decorator."
                    )

            def __set_instance_state(self, instance_state):
                if hasattr(self.__instance, '__setstate__'):
                    return self.__instance.__setstate__(instance_state)
                if auto_state:
                    self.__instance.__dict__ = instance_state
                else:
                    raise NotImplementedError(
                        f"{type(self.__instance)!r} has no '__setstate__' implementation. "
                        "Please, implement this method or use `@auto_block(auto_state=True)` decorator."
                    )

            def __getstate__(self):
                """Get state. Required for serialization.

                Returns:
                    State dictionary.

                """
                return {
                    '__instance': self.__get_instance_state(),
                    '__args': self.__args,
                    '__kwargs': self.__kwargs,
                    'slots': self.slots,
                }

            def __setstate__(self, state: dict):
                """Set state. Required for deserialization.

                Args:
                    state: State dictionary.

                """
                self.__args = state['__args']
                self.__kwargs = state['__kwargs']
                self.__instance = cls(*self.__args, **self.__kwargs)
                self.__set_instance_state(state['__instance'])
                self.slots = state['slots']

        return AutoBlock

    if _implicit_cls is not None:
        return _auto_block_impl(_implicit_cls)

    return _auto_block_impl
