from typing import List

import numpy as np

from bosk.block import BaseBlock, BlockInputData, TransformOutputData
from bosk.util.util import make_simple_meta


class ConcatBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'])
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit(self, inputs: BlockInputData) -> 'ConcatBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = tuple(
            inputs[name]
            for name in self.ordered_input_names
        )
        concatenated = np.concatenate(ordered_inputs, axis=self.axis)
        return {'output': concatenated}


class StackBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'])
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = tuple(
            inputs[name]
            for name in self.ordered_input_names
        )
        stacked = np.stack(ordered_inputs, axis=self.axis)
        return {'output': stacked}


class AverageBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'AverageBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        averaged = inputs['X'].mean(axis=self.axis)
        return {'output': averaged}


class ArgmaxBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'ArgmaxBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        ids = inputs['X'].argmax(axis=self.axis)
        return {'output': ids}
