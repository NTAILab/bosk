from ....stages import Stages
import numpy as np
from typing import List, Optional, Tuple, Union, NamedTuple
from functools import partial
from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import BlockMeta, make_simple_meta, BlockExecutionProperties, InputSlotMeta, OutputSlotMeta
from ....data import CPUData, GPUData
from ._convolution_helpers import (
    _PoolingIndices,
    _ConvolutionParams,
    _ConvolutionHelper,
)


class MultiGrainedScanningNDBlock(BaseBlock):
    """N-dimensional Multi Grained Scanning.

    It takes `X` of shape (n_samples, n_channels, n_features_1, ..., n_features_k)
    as an input and returns the pooled tensor of shape (n_samples, n_channels, t_1, ..., t_k).
    """

    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='X',
                stages=Stages(
                    fit=True,
                    transform=True,
                ),
            ),
            InputSlotMeta(
                name='y',
                stages=Stages(
                    fit=True,
                    transform=False
                ),
            ),
        ],
        outputs=[
            OutputSlotMeta(name='output')
        ],
        execution_props=BlockExecutionProperties(cpu=True, gpu=False, plain=False)
    )

    def __init__(self, model,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[None, int, Tuple[int]] = None,
                 dilation: int = 1,
                 padding: Optional[int] = None,
                 chunk_size: int = -1):
        """Initialize N-dimensional Multi Grained Scanning Block.

        Args:
            model: A fit-transform base model.
            kernel_size: Kernel size (int or tuple).
            stride: Stride.
            dilation: Dilation (kernel stride).
            padding: Padding size (see `numpy.pad`);
                     if None padding is disabled.
            chunk_size: Chunk size. Affects performance.

        """
        super().__init__()
        self.model = model
        self.params = _ConvolutionParams(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.chunk_size = chunk_size
        self.pooling_indices_: Optional[_PoolingIndices] = None
        self.helper_ = _ConvolutionHelper(self.params)

    def fit(self, inputs: BlockInputData) -> 'MultiGrainedScanningNDBlock':
        assert 'X' in inputs
        assert 'y' in inputs
        assert isinstance(inputs['X'], CPUData)
        assert isinstance(inputs['y'], CPUData)
        xs = inputs['X'].data
        y = inputs['y'].data
        if self.params.padding is not None:
            xs = self.helper_.pad(xs)
        pooling_indices = self.__prepare_pooling_indices(xs.shape)
        sliced, n_repeats = self.helper_.slice(xs, pooling_indices)
        repeated_y = np.tile(y[:, np.newaxis], (1, n_repeats)).reshape((-1, *y.shape[1:]))
        if isinstance(self.model, BaseBlock):
            self.model.fit({'X': CPUData(sliced), 'y': CPUData(repeated_y)})
        else:
            self.model.fit(sliced, repeated_y)
        return self

    def __prepare_pooling_indices(self, xs_shape):
        if self.pooling_indices_ is not None and xs_shape == self.pooling_indices_.xs_shape:
            return self.pooling_indices_

        self.pooling_indices_ = self.helper_.prepare_pooling_indices(xs_shape)
        return self.pooling_indices_

    def _model_transform(self, sliced: np.ndarray) -> np.ndarray:
        if isinstance(self.model, BaseBlock):
            assert self.model.default_output is not None,\
                'Cannot use base blocks with more than one output and no default chosen'
            outputs = self.model.transform({'X': CPUData(sliced)})
            block_default_out_data = outputs[self.model.default_output].data
            assert isinstance(block_default_out_data, CPUData)
            return block_default_out_data.data
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(sliced)
        if hasattr(self.model, 'transform'):
            return self.model.transform(sliced)
        if hasattr(self.model, 'predict'):
            return self.model.predict(sliced)
        raise RuntimeError(f'Model has no transform-like method: {self.model=}')

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        """Apply Pooling to input 'X'.

        Args:
            inputs: Input data that consists of one element with key 'X'.
                    `inputs['X']` should have shape (n_samples, n_channels, n_features_1, ..., n_features_k).

        """
        assert 'X' in inputs
        assert isinstance(inputs['X'], CPUData)
        xs = inputs['X'].data
        n_samples = xs.shape[0]
        # shape: (n_samples, n_channels, n_features_1, ..., n_features_k)
        assert self.pooling_indices_ is not None
        sliced, n_corners = self.helper_.slice(xs, self.pooling_indices_)
        # sliced shape: (n_samples * pooling_indices.n_corners, n_channels * pooling_indices.n_kernel_points)
        result = self._model_transform(sliced)
        # result shape: (n_samples * n_corners, n_out_channels)
        result = result.reshape((n_samples, n_corners, -1))
        # result shape: (n_samples, n_corners, n_out_channels)
        result = np.swapaxes(result, 1, 2)
        # result shape: (n_samples, n_out_channels, n_corners)
        n_out_channels = result.shape[1]
        result = result.reshape((n_samples, n_out_channels, *self.pooling_indices_.pooled_shape))
        return {'output': CPUData(result)}
