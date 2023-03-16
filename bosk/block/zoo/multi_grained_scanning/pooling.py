import numpy as np
from typing import Optional, Tuple, Union, NamedTuple
from functools import partial
from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import make_simple_meta, BlockExecutionProperties
from ....data import CPUData, GPUData


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

    AGGREGATION_FUNCTIONS = {
        'max': partial(np.max, axis=-1),
        'mean': partial(np.mean, axis=-1),
    }

    class PoolingIndices(NamedTuple):
        """
        Attributes:
            xs_shape: Input shape.
            full_index_tuple: Tuple of indices of each pixel for each group (corner position).
            n_corners: Number of corners.
            n_kernel_points: Number of points inside kernel.
            pooled_shape: Pooling result shape (excluding the n_sample dimension).
        """
        xs_shape: Tuple[int]
        full_index_tuple: Tuple[np.ndarray]
        n_corners: int
        n_kernel_points: int
        pooled_shape: Tuple[int]


    def __init__(self, kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[None, int, Tuple[int]] = None,
                 dilation: int = 1,
                 padding: bool = False,
                 aggregation: str = 'max',
                 chunk_size: int = -1):
        """Initialize Pooling Block.

        Args:
            kernel_size: Kernel size (int or tuple).
            stride: Stride.
            dilation: Dilation (kernel stride).
            padding: Use padding or crop.
            aggregation: Aggregation operation name.
            chunk_size: Chunk size. Affects performance.

        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.aggregation = aggregation
        self.chunk_size = chunk_size
        self.pooling_indices_: Optional[PoolingBlock.PoolingIndices] = None

    def fit(self, _inputs: BlockInputData) -> 'PoolingBlock':
        return self

    def __aggregate(self, grouped_data: np.ndarray):
        """Apply an aggregation function along the last axis.

        Args:
            grouped_data: Array of shape (n_groups, n_kernel_points)

        """
        if callable(self.aggregation):
            return self.aggregation(grouped_data)
        elif self.aggregation in self.AGGREGATION_FUNCTIONS:
            return self.AGGREGATION_FUNCTIONS[self.aggregation](grouped_data)
        raise ValueError(f'Wrong aggregation function: {self.aggregation!r}')

    def __prepare_kernel(self, n_spatial_dims):
        if isinstance(self.kernel_size, tuple):
            kernel_size = self.kernel_size
        else:
            kernel_size = (self.kernel_size,) * n_spatial_dims
        kernel_ids = np.stack(np.meshgrid(*[np.arange(k) for k in kernel_size], indexing='ij'), axis=0)
        # kernel_ids shape: (n_spatial_dims, k_1, ..., k_nsd)
        if self.dilation > 1:
            kernel_ids = kernel_ids[:, np.all(kernel_ids % self.dilation == 0, axis=0)]
        kernel_ids = kernel_ids.reshape((kernel_ids.shape[0], -1))
        return kernel_size, kernel_ids

    def __prepare_corner(self, spatial_dims, kernel_size):
        n_spatial_dims = len(spatial_dims)
        if self.stride is None:
            stride = kernel_size
        elif isinstance(self.stride, tuple):
            stride = self.stride
        else:
            stride = (self.stride,) * n_spatial_dims
        corner_ids = np.stack(
            np.meshgrid(*[
                np.arange(
                    0,
                    (
                        (spatial_dims[i] // s) * s if s >= kernel_size[i]
                        else ((spatial_dims[i] - kernel_size[i] + 1) // s) * s
                    ),
                    s
                )
                for i, s in enumerate(stride)
            ], indexing='ij'),
            axis=0
        )
        return corner_ids

    def __prepare_pooling_indices(self, xs_shape):
        if self.pooling_indices_ is not None and xs_shape == self.pooling_indices_.xs_shape:
            return self.pooling_indices_

        _, n_channels, *spatial_dims = xs_shape
        n_spatial_dims = len(spatial_dims)
        kernel_size, kernel_ids = self.__prepare_kernel(n_spatial_dims)
        corner_ids = self.__prepare_corner(spatial_dims, kernel_size)
        pooled_shape = corner_ids.shape[1:]
        corner_ids = corner_ids.reshape((corner_ids.shape[0], -1))
        # corner_ids shape: (n_spatial_dims, s_1 * ... * s_nsd)
        all_ids = corner_ids[:, :, np.newaxis] + kernel_ids[:, np.newaxis, :]
        all_ids = all_ids.reshape((n_spatial_dims, -1))
        # all_ids shape: (n_spatial_dims, n_positions)
        full_index_tuple = tuple(
            np.concatenate((
                np.arange(all_ids.shape[1] * n_channels)[np.newaxis] // all_ids.shape[1],
                np.repeat(all_ids, n_channels, axis=1)
            ), axis=0)
        )
        self.pooling_indices_ = self.PoolingIndices(
            xs_shape,
            full_index_tuple,
            corner_ids.shape[1],
            kernel_ids.shape[1],
            pooled_shape
        )
        return self.pooling_indices_

    def __chunk_pooling(self, xs: np.ndarray) -> np.ndarray:
        n_samples, n_channels, *_ = xs.shape
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
