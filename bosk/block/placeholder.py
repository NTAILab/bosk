from ..state import BoskState


class PlaceholderFunction:
    """Placeholder function.

    Adds self to the active builder.
    """
    def __init__(self, block):
        self.__block = block

    def __call__(self, *pfn_args, **pfn_kwargs) -> 'FunctionalBlockWrapper':
        builder = BoskState().active_builders.peek()
        assert builder is not None, \
            'No active builder found. ' \
            'Please, enter a builder scope by `with FunctionalPipelineBuilder() as b:`'
        return builder.wrap(self.__block)(*pfn_args, **pfn_kwargs)

