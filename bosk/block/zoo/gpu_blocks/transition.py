from typing import Union

from bosk.block import BaseBlock, TransformOutputData, BlockInputData
from bosk.block.meta import make_simple_meta, BlockExecutionProperties


class MoveToBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['X'], execution_props=BlockExecutionProperties(plain=True))

    def __init__(self, to: Union[str, None] = None):
        super().__init__()
        assert to == "CPU" or to == "GPU"
        self.to = to

    def fit_gpu(self, inputs: BlockInputData) -> 'MoveToBlock':
        return self

    def transform_gpu(self, inputs: BlockInputData) -> TransformOutputData:
        return self.transform(inputs)

    def fit(self, inputs: BlockInputData) -> 'MoveToBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        input_data = inputs['X']
        if self.to == 'CPU':
            return {'X': input_data.to_cpu()}
        elif self.to == 'GPU':
            return {'X': input_data.to_gpu()}
        else:
            return inputs
