from typing import Optional
from ...base import BaseOutputBlock, BlockInputData, TransformOutputData
from ...meta import BlockExecutionProperties, BlockMeta, DynamicBlockMetaStub, make_simple_meta


class OutputBlock(BaseOutputBlock):
    """Output block.

    Bypasses its input. Can be used to mark outputs when building pipeline in functional style.

    Dynamically specifies the meta information.

    Args:
        name: The output slot name. If None, the default name is used.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `name` (default="out"): Output data array.

    Output slots
    ------------

        - `name` (default="out"): Output data array.

    """
    DEFAULT_OUTPUT_NAME = 'out'
    name = None
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, name: Optional[str] = None):
        slot_name = name if name is not None else self.DEFAULT_OUTPUT_NAME
        self.meta = make_simple_meta(
            [slot_name],
            [slot_name],
            execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True)
        )
        super().__init__()
        self.name = name

    def fit(self, _inputs: BlockInputData) -> 'OutputBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs
