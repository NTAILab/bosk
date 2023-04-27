"""All custom exceptions for the module.
"""


__all__ = [
    "NoDefaultBlockOutputError",
]


from typing import Literal


class NoDefaultBlockOutputError(ValueError):
    def __init__(self, block):
        super().__init__(f"No default block output found for block {block!r}")


class MultipleBlockSlotsError(ValueError):
    def __init__(self, block, slot: Literal['input', 'output']):
        super().__init__(f"Block {block!r} has more than one {slot} slot, but single is required")


class MultipleBlockInputsError(MultipleBlockSlotsError):
    def __init__(self, block):
        super().__init__(block, slot='input')


class MultipleBlockOutputsError(MultipleBlockSlotsError):
    def __init__(self, block):
        super().__init__(block, slot='output')
