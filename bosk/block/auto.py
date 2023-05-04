"""Auto block module allows to automatically convert any fit-transform class into a block.

For example, wrapping simple fit-transform class into a block can be done with decorator :py:func:`auto_block`:

.. code-block::

    @auto_block
    class MyFitTransformClassBlock:
        def fit(self, arg1, arg2):
            return self

        def transform(self, arg1):
            return arg1

The resulting class will be derived from :py:class:`BaseBlock` and can be used as a block in the pipeline.

"""
import numpy as np
from numpy.random import Generator
from typing import Any, Optional, Type, Mapping, Protocol, Union
from .base import BaseBlock, BlockInputData, TransformOutputData
from .meta import BlockMeta, BlockExecutionProperties, InputSlotMeta, OutputSlotMeta
from ..data import BaseData, CPUData
from ..stages import Stages
from ..utility import get_random_generator, get_rand_int
from functools import wraps
import warnings


class FitTransformClass(Protocol):
    """Protocol for fit-transform estimator classes.

    Classes decorated with `@auto_block` should implement this protocol.
    """

    def fit(self):
        """Fit the estimator.

        Wrapped class can have multiple arguments.

        """
        ...

    def transform(self) -> Union[np.ndarray, BaseData]:
        """Transform data with the estimator.

        Wrapped class can have multiple arguments.

        Returns:
            Numpy array or data object (e.g. `CPUData`).

        """
        ...


def auto_block(_implicit_cls=None,  # noqa: C901
               execution_props: Optional[BlockExecutionProperties] = None,
               random_state_field: str | None = 'random_state',
               auto_state: bool = False) -> Type[BaseBlock]:
    """Decorator for conversion from a fit-transform class into a block.

    Args:
        _implicit_cls: Class with fit-transform interface.
                       It is automatically passed when `@auto_block` without
                       brackets is used. Otherwise it should be `None`.
        execution_props: Custom block execution properties.
        random_state_field: Field name in the class that corresponds to object's random seed.
            Pass `None` if the class doesn't have any. If the class already has
            the `set_random_state` method, it won't be redefined.
        auto_state: Automatically implement `__getstate__` and `__setstate__` methods.
                    These methods are required for serialization.

    Returns:
        Block wrapping function.

    """
    def _auto_block_impl(cls: Type[FitTransformClass]) -> Type[BaseBlock]:
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
            meta: BlockMeta = BlockMeta(
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

            def __prepare_kwargs(self, inputs: BlockInputData) -> Mapping[str, Any]:
                kwargs = {
                    slot_name: inputs[slot_name].data
                    for slot_name, _slot in self.slots.inputs.items()
                    if slot_name in inputs
                }
                return kwargs

            def fit(self, inputs: BlockInputData) -> 'AutoBlock':
                """Fit method wrapper.

                Calls the wrapped class `fit` method on converted arguments.

                Args:
                    inputs: Block input data (mapping from input names to data).

                Returns:
                    Self.

                """
                self.__instance.fit(**self.__prepare_kwargs(inputs))
                return self

            def transform(self, inputs: BlockInputData) -> TransformOutputData:
                """Transform method wrapper.

                Calls the wrapped class `transform` method on converted arguments.

                Args:
                    inputs: Block input data (mapping from input names to data).

                Returns:
                    Self.

                """
                transformed = self.__instance.transform(**self.__prepare_kwargs(inputs))
                if isinstance(transformed, BaseData):
                    output = transformed
                elif isinstance(transformed, np.ndarray):
                    output = CPUData(transformed)
                else:
                    raise TypeError(
                        f"Unexpected type {type(transformed)} for transformed output. "
                        f"Expected `BaseData` or `np.ndarray`."
                    )
                return {'output': output}

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

            def set_random_state(self, seed: Optional[int | Generator]) -> None:
                """Set random state (seed).

                If `random_state_field` is defined and the wrapped class has corresponding
                property, this field value will be set.

                Otherwise, for more complex behavior, it can be redefined in wrapped class
                by implementing `set_random_state` method.

                Args:
                    seed: Seed or random number generator.

                """
                if hasattr(self.__instance, 'set_random_state'):
                    return self.__instance.set_random_state(seed)
                if random_state_field is None:
                    return super().set_random_state(seed)
                if hasattr(self.__instance, random_state_field):
                    gen = get_random_generator(seed)
                    setattr(self.__instance, random_state_field, get_rand_int(gen))
                else:
                    warnings.warn("%s doesn't have random_state_field '%s'" % (cls.__name__, random_state_field))

        return AutoBlock

    if _implicit_cls is not None:
        return _auto_block_impl(_implicit_cls)

    return _auto_block_impl
