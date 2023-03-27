from typing import Union

from ...base import BaseBlock, TransformOutputData, BlockInputData
from ...meta import make_simple_meta, BlockExecutionProperties
from ....data import CPUData, GPUData, BaseData


class MoveToBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['X'], execution_props=BlockExecutionProperties(plain=True))

    def __init__(self, to: Union[str, None] = None):
        super().__init__()
        assert to == "CPU" or to == "GPU"
        self.to = to

    def fit(self, inputs: BlockInputData) -> 'MoveToBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        input_data = inputs['X']
        input_type = type(input_data)
        if input_type not in [BaseData, CPUData, GPUData]:
            return {'X': CPUData(input_data)}
        if self.to == 'CPU':
            return {'X': input_data.to_cpu()}
        elif self.to == 'GPU':
            return {'X': input_data.to_gpu()}
        else:
            return inputs
