from typing import List

import numpy as np

from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import BlockExecutionProperties, BlockMeta, DynamicBlockMetaStub, make_simple_meta
from ....data import CPUData


class CSBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['mask', 'best'],
                            execution_props=BlockExecutionProperties(plain=True))

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def fit(self, inputs: BlockInputData) -> 'CSBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        X = inputs['X'].data
        best_mask = X.max(axis=1) > self.eps
        best = X[best_mask]
        return {
            'mask': CPUData(~best_mask),
            'best': CPUData(best),
        }


class CSFilterBlock(BaseBlock):
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, input_names: List[str]):
        output_names = input_names
        self.input_names = input_names
        self.meta = make_simple_meta(input_names + ['mask'], output_names,
                                     execution_props=BlockExecutionProperties(plain=True))
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'CSFilterBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        mask = inputs['mask'].data
        return {
            name: CPUData(inputs[name].data[mask])
            for name in self.input_names
        }


class CSJoinBlock(BaseBlock):
    meta = make_simple_meta(['best', 'refined', 'mask'], ['output'],
                            execution_props=BlockExecutionProperties(plain=True))

    def __init__(self):
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'CSJoinBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        best = inputs['best'].data
        refined = inputs['refined'].data
        mask = inputs['mask'].data
        n_samples = mask.shape[0]
        rest_dims = best.shape[1:]
        result = np.empty((n_samples, *rest_dims), dtype=best.dtype)
        result[~mask] = best
        result[mask] = refined
        return {'output': CPUData(result)}
