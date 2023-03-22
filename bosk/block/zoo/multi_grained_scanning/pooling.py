import numpy as np
from typing import Optional, Tuple, Union, NamedTuple
from functools import partial
from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import make_simple_meta, BlockExecutionProperties
from ....data import CPUData, GPUData
from ._convolution_helpers import (
    _PoolingIndices,
    _ConvolutionParams,
    _ConvolutionHelper,
)


AGGREGATION_FUNCTIONS = {
    'max': partial(np.max, axis=-1),
    'mean': partial(np.mean, axis=-1),
}


class PoolingBlock(BaseBlock):
    """Pooling Block implements n-dimensional downsampling with an aggregation operation.

    It takes `X` of shape (n_samples, n_channels, n_features_1, ..., n_features_k)
    as an input and returns the pooled tensor of shape (n_samples, n_channels, t_1, ..., t_k).
    """

    meta = make_simple_meta(
        ['X'],
        ['output'],
        execution_props=BlockExecutionProperties(cpu=True, gpu=False, plain=False)
    )

    def __init__(self, kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[None, int, Tuple[int]] = None,
                 dilation: int = 1,
                 padding: Optional[int] = None,
                 aggregation: str = 'max',
                 chunk_size: int = -1):
        """Initialize Pooling Block.

        Args:
            kernel_size: Kernel size (int or tuple).
            stride: Stride.
            dilation: Dilation (kernel stride).
            padding: Padding size (see `numpy.pad`);
                     if None padding is disabled.
            aggregation: Aggregation operation name.
            chunk_size: Chunk size. Affects performance.

        """
        super().__init__()
        self.params = _ConvolutionParams(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.aggregation = aggregation
        self.chunk_size = chunk_size
        self.pooling_indices_: Optional[_PoolingIndices] = None
        self.helper_ = _ConvolutionHelper(self.params)

    def fit(self, _inputs: BlockInputData) -> 'PoolingBlock':
        return self

    def __aggregate(self, grouped_data: np.ndarray):
        """Apply an aggregation function along the last axis.

        Args:
            grouped_data: Array of shape (n_groups, n_kernel_points)

        """
        if callable(self.aggregation):
            return self.aggregation(grouped_data)
        elif self.aggregation in AGGREGATION_FUNCTIONS:
            return AGGREGATION_FUNCTIONS[self.aggregation](grouped_data)
        raise ValueError(f'Wrong aggregation function: {self.aggregation!r}')

    def __prepare_pooling_indices(self, xs_shape):
        if self.pooling_indices_ is not None and xs_shape == self.pooling_indices_.xs_shape:
            return self.pooling_indices_

        self.pooling_indices_ = self.helper_.prepare_pooling_indices(xs_shape)
        return self.pooling_indices_

    def __chunk_pooling(self, xs: np.ndarray) -> np.ndarray:
        n_samples, n_channels, *_ = xs.shape
        if self.params.padding is not None:
            xs = self.helper_.pad(xs)
        pooling_indices = self.__prepare_pooling_indices(xs.shape)
        result = []
        for i in range(n_samples):
            grouped_data = xs[i][pooling_indices.full_index_tuple].reshape((
                -1, pooling_indices.n_corners, pooling_indices.n_kernel_points
            ))
            aggregated = self.__aggregate(grouped_data)
            aggregated = aggregated.reshape((n_channels, *pooling_indices.pooled_shape))
            result.append(aggregated)
        result = np.stack(result, axis=0)
        return result

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        """Apply Pooling to input 'X'.

        Args:
            inputs: Input data that consists of one element with key 'X'.
                    `inputs['X']` should have shape (n_samples, n_channels, n_features_1, ..., n_features_k).

        """
        assert 'X' in inputs
        assert isinstance(inputs['X'], CPUData)
        xs = inputs['X'].data
        # shape: (n_samples, n_channels, n_features_1, ..., n_features_k)
        if self.chunk_size <= 0:
            result = self.__chunk_pooling(xs)
        else:
            n_samples = xs.shape[0]
            n_chunks = round(np.ceil(n_samples / self.chunk_size))
            result = np.concatenate([
                self.__chunk_pooling(xs_chunk)
                for xs_chunk in np.split(xs, n_chunks)
            ], axis=0)
        return {'output': CPUData(result)}
