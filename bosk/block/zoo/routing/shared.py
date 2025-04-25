import numpy as np
from typing import Optional, Union

from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...placeholder import PlaceholderMixin
from ...meta import (
    BlockMeta,
    DynamicBlockMetaStub,
)


class Shared(PlaceholderMixin, BaseBlock):
    """Shared Block.

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

    def fit(self, inputs: BlockInputData) -> 'Shared':
        """The block (wrapper) bypasses the fit step."""
        # do nothing because the warpped block is already fitted
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return self.block.transform(inputs)

    def set_random_state(self, seed: Optional[Union[int, np.random.Generator]]) -> None:
        return self.block.set_random_state(seed)
