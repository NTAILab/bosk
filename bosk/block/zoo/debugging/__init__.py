"""Blocks for pipeline debugging.
"""

from typing import Callable, Sequence
from ...meta import BlockExecutionProperties, DynamicBlockMetaStub, InputSlotMeta, OutputSlotMeta
from ....stages import Stages
from ...base import BaseBlock, BlockInputData, BlockMeta, TransformOutputData
from ...placeholder import PlaceholderMixin


__all__ = [
    'FitLambda',
    'TransformLambda',
    # for backward compatibility:
    'FitLambdaBlock',
    'TransformLambdaBlock',
]


class FitLambda(PlaceholderMixin, BaseBlock):
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, function: Callable, inputs: Sequence[str]):
        """Fit-Lambda Block that executes some function at the fit stage
        and bypasses input at the transform stage.

        Dynamically specifies the meta information.

        Args:
            function: Function to execute at the fit stage.
            inputs: List of input slot names.

        Input slots
        -----------

        Fit inputs
        ~~~~~~~~~~

            - `inputs[0]`: Data array 0.
            - ...
            - `inputs[n]`: Data array n.

        Transform inputs
        ~~~~~~~~~~~~~~~~

            - `inputs[0]`: Data array 0.
            - ...
            - `inputs[n]`: Data array n.

        Output slots
        ------------

            - `inputs[0]`: Data array 0.
            - ...
            - `inputs[n]`: Data array n.

        """
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


class TransformLambda(PlaceholderMixin, BaseBlock):
    """Transform-Lambda Block that executes some function at the fit stage
    and bypasses input ad the transform stage.

    Dynamically specifies the meta information.

    Args:
        function: Function to execute at the transform stage.
        inputs: List of input slot names.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `inputs[0]`: Data array 0.
        - ...
        - `inputs[n]`: Data array n.

    Output slots
    ------------

        - `inputs[0]`: Data array 0.
        - ...
        - `inputs[n]`: Data array n.

    """
    meta: BlockMeta = DynamicBlockMetaStub()

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

    def fit(self, _inputs: BlockInputData) -> 'TransformLambdaBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        self.function(inputs)
        return inputs


FitLambdaBlock = FitLambda
TransformLambdaBlock = TransformLambda

