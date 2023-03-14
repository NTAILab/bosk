from typing import List

import jax.numpy as jnp
import numpy as np

from bosk.block import BaseBlock, BlockInputData, TransformOutputData
from bosk.block.meta import make_simple_meta, BlockExecutionProperties
from bosk.data import GPUData, CPUData


class ConcatBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'],
                                     execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit(self, inputs: BlockInputData) -> 'ConcatBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        input_type = type(next(iter(inputs.values())))
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], input_type):
                ordered_inputs.append(inputs[name].data)
            else:
                raise ValueError("All inputs must be of the same type (CPUData or GPUData)")
        ordered_inputs = tuple(ordered_inputs)
        if input_type == CPUData:
            concatenated = CPUData(np.concatenate(ordered_inputs, axis=self.axis))
        elif input_type == GPUData:
            concatenated = GPUData(jnp.concatenate(ordered_inputs, axis=self.axis))
        else:
            raise ValueError(f"Unexpected input type: {input_type}")

        return {'output': concatenated}


class StackBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'],
                                     execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        input_type = type(next(iter(inputs.values())))
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], input_type):
                ordered_inputs.append(inputs[name].data)
            else:
                raise ValueError("All inputs must be of the same type (CPUData or GPUData)")
        ordered_inputs = tuple(ordered_inputs)
        if input_type == CPUData:
            stacked = CPUData(np.stack(ordered_inputs, axis=self.axis))
        elif input_type == GPUData:
            stacked = GPUData(jnp.stack(ordered_inputs, axis=self.axis))
        else:
            raise ValueError(f"Unexpected input type: {input_type}")
        return {'output': stacked}


class AverageBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'], execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'AverageBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        input_type = type(inputs['X'])
        if input_type not in [CPUData, GPUData]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        averaged = inputs['X'].data.mean(axis=self.axis)
        if input_type == CPUData:
            return {'output': CPUData(averaged)}
        else:
            return {'output': GPUData(averaged)}


class ArgmaxBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'], execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'ArgmaxBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        input_type = type(inputs['X'])
        if input_type not in [CPUData, GPUData]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        ids = inputs['X'].data.argmax(axis=self.axis)
        if input_type == CPUData:
            return {'output': CPUData(ids)}
        else:
            return {'output': GPUData(ids)}
