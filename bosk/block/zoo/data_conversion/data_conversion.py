from typing import List

import jax.numpy as jnp
import numpy as np

from bosk.block import BaseBlock, BlockInputData, TransformOutputData
from bosk.block.meta import make_simple_meta
from bosk.data import GPUData, CPUData


class ConcatBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'])
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit_gpu(self, inputs: BlockInputData) -> 'ConcatBlock':
        return self.fit(inputs)

    def transform_gpu(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], CPUData):
                ordered_inputs.append(inputs[name].to_gpu().data)  # or raise Exception?
            else:
                ordered_inputs.append(inputs[name].data)
        ordered_inputs = tuple(ordered_inputs)
        concatenated = jnp.concatenate(ordered_inputs, axis=self.axis)
        return {'output': GPUData(concatenated)}

    def fit(self, inputs: BlockInputData) -> 'ConcatBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], GPUData):
                ordered_inputs.append(inputs[name].to_cpu().data)  # or raise Exception?
            else:
                ordered_inputs.append(inputs[name].data)
        ordered_inputs = tuple(ordered_inputs)
        concatenated = np.concatenate(ordered_inputs, axis=self.axis)
        return {'output': concatenated}


class StackBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'])
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit_gpu(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform_gpu(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], CPUData):
                ordered_inputs.append(inputs[name].to_gpu().data)  # or raise Exception?
            else:
                ordered_inputs.append(inputs[name].data)
        ordered_inputs = tuple(ordered_inputs)
        stacked = jnp.stack(ordered_inputs, axis=self.axis)
        return {'output': GPUData(stacked)}

    def fit(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], GPUData):
                ordered_inputs.append(inputs[name].to_cpu().data)  # or raise Exception?
            else:
                ordered_inputs.append(inputs[name].data)
        ordered_inputs = tuple(ordered_inputs)
        stacked = np.stack(ordered_inputs, axis=self.axis)
        return {'output': CPUData(stacked)}


class AverageBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit_gpu(self, inputs: BlockInputData) -> 'AverageBlock':
        return self

    def transform_gpu(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        if isinstance(inputs['X'], CPUData):
            averaged = inputs['X'].to_gpu().data.mean(axis=self.axis)
        else:
            averaged = inputs['X'].data.mean(axis=self.axis)
        return {'output': GPUData(averaged)}

    def fit(self, _inputs: BlockInputData) -> 'AverageBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        if isinstance(inputs['X'], GPUData):
            averaged = inputs['X'].to_cpu().data.mean(axis=self.axis)
        else:
            averaged = inputs['X'].data.mean(axis=self.axis)
        return {'output': CPUData(averaged)}


class ArgmaxBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit_gpu(self, inputs: BlockInputData) -> 'ArgmaxBlock':
        return self

    def transform_gpu(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        if isinstance(inputs['X'], CPUData):
            ids = inputs['X'].to_gpu().data.argmax(axis=self.axis)
        else:
            ids = inputs['X'].data.argmax(axis=self.axis)
        return {'output': GPUData(ids)}

    def fit(self, _inputs: BlockInputData) -> 'ArgmaxBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        if isinstance(inputs['X'], GPUData):
            ids = inputs['X'].to_cpu().data.argmax(axis=self.axis)
        else:
            ids = inputs['X'].data.argmax(axis=self.axis)
        return {'output': CPUData(ids)}
