from .base import BaseBlock
from .functional import FunctionalBlockWrapper
from ..state import BoskState


class PlaceholderMixin:
    """Placeholder function mixin.

    Adds self to the active builder when called with `__call__`.

    A placeholder function can be used to build a pipeline in a `with` scope.
    Example:

        Let `SomePlaceholderFunction` be a block, derived from `PlaceholderMixin`

        >>> with FunctionalPipelineBuilder() as b:
        >>>     y = SomePlaceholderFunction(x)

        This code implicitly adds the underlying block of `SomePlaceholderFunction` to the builder.

    To make a placeholder function from a block just derive from the mixin first.

    Example:

        Make placeholder from the block `SomeNewBlock` with name `SomeNew`:

        >>> class SomeNew(PlaceholderMixin, SomeNewBlock):
        >>>     ...

        Make a new block with inherent placeholder functionality:

        >>> class SomeNew(PlaceholderMixin, BaseBlock):
        >>>     # implement fit, transform methods here
        >>>     ...

    """
    def __call__(self, *pfn_args, **pfn_kwargs) -> FunctionalBlockWrapper:
        assert isinstance(self, BaseBlock), 'Placeholder can be mixed only to a block'
        builder = BoskState().active_builders.peek()
        assert builder is not None, \
            'No active builder found. ' \
            'Please, enter a builder scope by `with FunctionalPipelineBuilder() as b:`'
        return builder.wrap(self)(*pfn_args, **pfn_kwargs)

