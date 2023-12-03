from typing import Union

from ...base import BaseBlock, TransformOutputData, BlockInputData
from ...placeholder import PlaceholderMixin
from ...meta import make_simple_meta, BlockExecutionProperties
from ....data import CPUData, GPUData, BaseData


__all__ = [
    "MoveTo",
    # for backward compatibility:
    "MoveToBlock",
]


class MoveTo(PlaceholderMixin, BaseBlock):
    """Move-to block.

    Moves the input data to the specified device.

    Args:
        to: Device to move to.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Feature data array.

    Output slots
    ------------

        - X: The same array on the specified device.

    """

    meta = make_simple_meta(['X'], ['X'], execution_props=BlockExecutionProperties(plain=True))

    def __init__(self, to: Union[str, None] = None):
        super().__init__()
        assert to == "CPU" or to == "GPU"
        self.to = to

    def fit(self, inputs: BlockInputData) -> 'MoveToBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        input_data = inputs['X']
        input_type = type(input_data)
        assert issubclass(input_type, BaseData)
        if self.to == 'CPU':
            return {'X': input_data.to_cpu()}
        elif self.to == 'GPU':
            return {'X': input_data.to_gpu()}
        else:
            return inputs


MoveToBlock = MoveTo

