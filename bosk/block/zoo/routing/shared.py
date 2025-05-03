import numpy as np
from typing import Optional, Union

from ...base import BaseBlock, BlockInputData, BlockOutputSlot, TransformOutputData
from ...placeholder import PlaceholderMixin
from ...meta import (
    BlockMeta,
    DynamicBlockMetaStub,
    InputSlotMeta,
    OutputSlotMeta,
)
from ....stages import Stages
from ....exceptions import BlockInputMissingError


class NaiveShared(PlaceholderMixin, BaseBlock):
    """Naive Shared Block.

    **Warning: the wrapped block has to be trained either before the initialization
    or at least before transform. It is complicated to do so in concurrent environments.**
    Please consider `SharedProducer` and `SharedConsumer` instead.

    It wraps a block that has already been trained
    and allows to apply transform to the other input data.

    Dynamically specifies the meta information.

    Args:
        block: Wrapped block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        The same as in `block`.

    Output slots
    ------------

        The same as in `block`.

    Attributes:
        block: The wrapped block.

    """

    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, block: BaseBlock):
        self.meta = BlockMeta(
            inputs=[
                inp for inp in block.meta.inputs.values()
                if inp.stages.transform or inp.stages.transform_on_fit
            ],
            outputs=[
                out for out in block.meta.outputs.values()
            ],
            execution_props=block.meta.execution_props
        )
        self.block = block
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'NaiveShared':
        """The block (wrapper) bypasses the fit step."""
        # do nothing because the warpped block is already fitted
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return self.block.transform(inputs)

    def set_random_state(self, seed: Optional[Union[int, np.random.Generator]]) -> None:
        return self.block.set_random_state(seed)


class SharedProducer(PlaceholderMixin, BaseBlock):
    """Shared Producer Block.

    It wraps a block and returns it at transform.
    **The wrapped block cannot be used elsewhere in pipeline.**

    Dynamically specifies the meta information.

    Args:
        block: Wrapped block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        The same as in `block`.

    Output slots
    ------------

        The same as in `block`.

    Attributes:
        block: The wrapped block.

    """

    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, block: BaseBlock, output_block_name: str = 'block'):
        self.meta = BlockMeta(
            inputs=[
                inp for inp in block.meta.inputs.values()
            ],
            outputs=[
                out for out in block.meta.outputs.values()
            ] + [
                OutputSlotMeta(output_block_name),
            ],
            execution_props=block.meta.execution_props
        )
        self.block = block
        self.output_block_name = output_block_name
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'SharedProducer':
        self.block.fit(inputs)
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return self.block.transform(inputs) | {
            self.output_block_name: self.block
        }

    def get_default_output(self):
        return self.slots.outputs[self.block.get_default_output().meta.name]

    def set_random_state(self, seed: Optional[Union[int, np.random.Generator]]) -> None:
        return self.block.set_random_state(seed)


class SharedConsumer(PlaceholderMixin, BaseBlock):
    """Shared Consumer Block.

    It receives a block as input, as well as the block inputs and returns block's output at transform.

    Dynamically specifies the meta information.

    Args:
        block: Wrapped block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        The same as in `block`.

    Output slots
    ------------

        The same as in `block`.

    Attributes:
        block: The wrapped block.

    """

    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, producer_meta: BlockMeta, input_block_name: str = 'block',
                 default_output_name: str = ''):
        self.meta = BlockMeta(
            inputs=[
                inp for inp in producer_meta.inputs.values()
                if inp.stages.transform or inp.stages.transform_on_fit
            ] + [
                InputSlotMeta(name=input_block_name, stages=Stages(transform_on_fit=True, fit=True)),
            ],
            outputs=[
                out for out in producer_meta.outputs.values()
                if out.name != input_block_name
            ],
            execution_props=producer_meta.execution_props
        )
        if default_output_name not in producer_meta.outputs:
            raise ValueError(
                f'{default_output_name=!r} of SharedConsumer should be chosen from `producer_meta.outputs`: ' \
                f'{list(producer_meta.outputs.keys())!r}'
            )
        self.input_block_name = input_block_name
        self.default_output_name = default_output_name
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'SharedConsumer':
        """The block (wrapper) bypasses the fit step."""
        # do nothing because the warpped block is already fitted
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        if self.input_block_name not in inputs:
            raise BlockInputMissingError(self, self.input_block_name)
        block = inputs.pop(self.input_block_name)
        return block.transform(inputs)

    def get_default_output(self):
        return self.slots.outputs[self.default_output_name]

    def set_random_state(self, seed: Optional[Union[int, np.random.Generator]]) -> None:
        # SharedProducer owns the block, so it changes the random state
        return


NaiveSharedBlock = NaiveShared
