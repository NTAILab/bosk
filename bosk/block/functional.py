from .base import BaseBlock, BlockInputSlot, BlockOutputSlot
from typing import Optional


class FunctionalBlockWrapper:
    """Block wrapper with functional interface.

    It helps to implement passing of block result into another block
    in functional style.

    The block wrapper is based on block and its output name.
    If a block has multiple output slots, each output can be used to
    create distinct wrappers.

    Example:

        Assuming block `test_block` has multiple outputs:
        "out_1" and "out_2", its wrapper can be used to create
        new wrappers for each output separately:

        >>> wrapper = FunctionalBlockWrapper(test_block)
        >>> wrapper.get_output_slot()
            RuntimeException('Block has more than one output')
        >>> first_wrapper = wrapper['out_1']
        >>> first_wrapper.get_output_slot()
            BlockOutputSlot...  # 'out_1'
        >>> second_wrapper = wrapper['out_2']
        >>> second_wrapper.get_output_slot()
            BlockOutputSlot...  # 'out_2'

    """
    def __init__(self, block: BaseBlock, output_name: Optional[str] = None):
        """Initialize functional block wrapper.

        Args:
            block: Block to be wrapped.
            output_name: Block output slot name (function has one output).
                         If there are multiple output slots, this argument
                         is required to specify which output should be used.
                         If there is only one output slot, this argument
                         is not required and can be filled with None value.

        """
        self.block = block
        self.output_name = output_name

    def set_block(self, block):
        self.block = block

    def get_input_slot(self, slot_name: Optional[str] = None) -> BlockInputSlot:
        """Get block input slot by name.

        Args:
            slot_name: Input slot name.
                       If block has only one input, the argument can be omitted.

        Returns:
            Corresponding input slot.

        Raises:
            RuntimeError: If `slot_name` is None, but the block has multiple inputs.

        """
        if slot_name is None:
            if len(self.block.slots.inputs) == 1:
                return list(self.block.slots.inputs.values())[0]
            else:
                raise RuntimeError('Block has more than one input (please, specify it)')
        return self.block.slots.inputs[slot_name]

    def get_output_slot(self) -> BlockOutputSlot:
        """Get output slot.

        If block has one output slot, it will be used even if
        `output_name` was not specified at initialization.
        If block has multiple output slots, the output slot name
        should be specified at initialization by `output_name` argument,
        or the block wrapper corresponding to concrete output slot name
        can be obtained with `__getitem__` method.

        Returns:
            The specified output slot.

        Raises:
            RuntimeError: If `output_name` was not specified at initialization,
                          and the block has more than one output.
                          To avoid it, use `wrapper[demanded_output_name]` to
                          get wrapper with specified slot output name.

        """
        if self.output_name is None:
            # if self.block.default_output is None:
            #     raise RuntimeError('Block has more than one output and the default is not specified')
            # else:
            #     return self.block.slots.outputs[self.block.default_output]
            return self.block.get_default_output()
        return self.block.slots.outputs[self.output_name]

    def __getitem__(self, output_name: str) -> 'FunctionalBlockWrapper':
        """Make functional block wrapper corresponding to the specified output slot name.

        Args:
            output_name: Output slot name.

        Returns:
            New functional block wrapper for the given output slot name.

        """
        return FunctionalBlockWrapper(self.block, output_name=output_name)
