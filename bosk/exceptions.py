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
