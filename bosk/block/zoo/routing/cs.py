from typing import List

import numpy as np

from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import BlockExecutionProperties, BlockMeta, DynamicBlockMetaStub, make_simple_meta
from ....data import CPUData


class CSBlock(BaseBlock):
    """Confidence screening block.

    Args:
        eps: Confidence threshold.

    Attributes:
        eps: Confidence threshold.

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

        - best: Best data subsample.
        - mask: Mask for the rest of the data array.


    """

    meta = make_simple_meta(['X'], ['mask', 'best'],
                            execution_props=BlockExecutionProperties(plain=True))

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def fit(self, inputs: BlockInputData) -> 'CSBlock':
        """The block bypasses the fit step."""
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
    """Confidence screening filtering block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - mask: Mask array.
        - `input_names[0]`: Features data array 0.
        - ...
        - `input_names[n]`: Features data array n.

    Output slots
    ------------

        - `input_names[0]`: Features data array 0 subset.
        - ...
        - `input_names[n]`: Features data array n subset.

    Attributes:
        input_names: List of input names.

    """

    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, input_names: List[str]):
        """Initialize the confidence screening filtering block.

        Dynamically specifies the meta information.

        Args:
            input_names: The input slot names.

        """
        output_names = input_names
        self.input_names = input_names
        self.meta = make_simple_meta(input_names + ['mask'], output_names,
                                     execution_props=BlockExecutionProperties(plain=True))
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'CSFilterBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        mask = inputs['mask'].data
        return {
            name: CPUData(inputs[name].data[mask])
            for name in self.input_names
        }


class CSJoinBlock(BaseBlock):
    """Confidence screening joining (merging) block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - best: Best data subsample.
        - refined: Refined data subsample (corresponding to `mask == True`).
        - mask: Mask of the refined part.

    Output slots
    ------------

        - output: Output merged data array.

    """
    meta = make_simple_meta(['best', 'refined', 'mask'], ['output'],
                            execution_props=BlockExecutionProperties(plain=True))

    def __init__(self):
        """Initialize the confidence screening joining (merging) block.
        """
        super().__init__()

    def fit(self, inputs: BlockInputData) -> 'CSJoinBlock':
        """The block bypasses the fit step."""
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
