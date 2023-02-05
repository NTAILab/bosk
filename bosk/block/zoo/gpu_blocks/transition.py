from bosk.block import BaseBlock, TransformOutputData
from bosk.block.meta import make_simple_meta
from bosk.data import Data, CPUData, GPUData


class MoveToBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['X'])

    def __init__(self, device: str):
        super().__init__()
        assert device == "cpu" or device == "cuda"
        self._device = device

    def fit(self, inputs: Data) -> 'MoveToBlock':
        return self

    def transform(self, inputs: Data) -> TransformOutputData:
        if isinstance(inputs['X'], CPUData) and self._device == 'cuda':
            return {'X': GPUData(inputs['X'])}
        elif isinstance(inputs['X'], GPUData) and self._device == 'cpu':
            return {'X': CPUData(inputs['X'])}
        return {'X': inputs['X']}
