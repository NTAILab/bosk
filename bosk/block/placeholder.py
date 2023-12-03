from ..state import BoskState


class PlaceholderFunction:
    """Placeholder function.

    Adds self to the active builder.
    """
    def __init__(self, block):
        if block is self:
            self.__block = None
            self.__block_is_self = True
        else:
            self.__block = block
            self.__block_is_self = False

    def __call__(self, *pfn_args, **pfn_kwargs) -> 'FunctionalBlockWrapper':
        builder = BoskState().active_builders.peek()
        assert builder is not None, \
            'No active builder found. ' \
            'Please, enter a builder scope by `with FunctionalPipelineBuilder() as b:`'
        block = self if self.__block_is_self else self.__block
        return builder.wrap(block)(*pfn_args, **pfn_kwargs)

