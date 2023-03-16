from typing import Callable, Sequence
from bosk.block.meta import BlockExecutionProperties
from bosk.block.slot import InputSlotMeta, OutputSlotMeta
from bosk.stages import Stages
from ...base import BaseBlock, BlockInputData, BlockMeta, TransformOutputData


class FitLambdaBlock(BaseBlock):
    """Fit Lambda Block that executes some function at the fit stage
    and bypasses input ad the transform stage.
    """

    meta = None

    def __init__(self, function: Callable, inputs: Sequence[str]):
        self.meta = BlockMeta(
            inputs=[
                InputSlotMeta(
                    name=inp,
                    stages=Stages(transform=False, transform_on_fit=True),
                )
                for inp in inputs
            ],
            outputs=[
                OutputSlotMeta(
                    name=inp,
                )
                for inp in inputs
            ],
            execution_props=BlockExecutionProperties(plain=True),
        )
        self.function = function
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'FitLambdaBlock':
        self.function(inputs)
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs