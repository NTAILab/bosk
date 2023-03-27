from typing import Callable, Sequence
from ...meta import BlockExecutionProperties
from ...slot import InputSlotMeta, OutputSlotMeta
from ....stages import Stages
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
                    stages=Stages(transform=True, transform_on_fit=True),
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


class TransformLambdaBlock(BaseBlock):
    """Transform Lambda Block that executes some function at the fit stage
    and bypasses input ad the transform stage.
    """

    meta = None

    def __init__(self, function: Callable, inputs: Sequence[str]):
        self.meta = BlockMeta(
            inputs=[
                InputSlotMeta(
                    name=inp,
                    stages=Stages(transform=True, transform_on_fit=True),
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

    def fit(self, _inputs: BlockInputData) -> 'FitLambdaBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        self.function(inputs)
        return inputs

