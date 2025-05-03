"""All custom exceptions for the module.
"""


__all__ = [
    "NoDefaultBlockOutputError",
    "MultipleBlockSlotsError",
    "MultipleBlockInputsError",
    "MultipleBlockOutputsError",
]


from typing import Literal


class NoDefaultBlockOutputError(ValueError):
    """Raised when no default block output is found for a block, but one is requested.
    """
    def __init__(self, block):
        super().__init__(f"No default block output found for block {block!r}")


class MultipleBlockSlotsError(ValueError):
    """Raised when multiple block slots are found for a block, while single is required.
    """
    def __init__(self, block, slot: Literal['input', 'output']):
        super().__init__(f"Block {block!r} has more than one {slot} slot, but single is required")


class MultipleBlockInputsError(MultipleBlockSlotsError):
    """Raised when multiple block inputs are found for a block, while single is required.
    """
    def __init__(self, block):
        super().__init__(block, slot='input')


class MultipleBlockOutputsError(MultipleBlockSlotsError):
    """Raised when multiple block outputs are found for a block, while single is required.
    """
    def __init__(self, block):
        super().__init__(block, slot='output')


class BlockReuseError(Exception):
    """Raised when the same block is tried to be reused (used twice in the pipeline).

    For example, when using the `FunctionalPipelineBuilder`, "calling" the same block
    twice leads to this exception.
    """
    def __init__(self, block):
        super().__init__(
            f'The block {block!r} is tried to be reused, which is prohibited. ' \
            'If it is intended, please implement separate blocks, one for each use. '
            'If the blocks have to share some state, return it from the first block and consume in the second block.'
        )


class BlockInputMissingError(Exception):
    """Raised when the block is tried to be executed without one of required inputs.
    """
    def __init__(self, block, input):
        super().__init__(
            f'The block {block!r} is tried to be executed without the input {input!r}.'
        )
