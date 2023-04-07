from operator import mul
import numpy as np
from jax import numpy as jnp
from jax import lax
from typing import Optional, Sequence, Tuple, Union, NamedTuple
from functools import partial, reduce
from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import make_simple_meta, BlockExecutionProperties
from ....data import CPUData, GPUData
from ._convolution_helpers import (
    _PoolingIndices,
    _ConvolutionParams,
    _ConvolutionHelper,
)
from ._pooling_impl import (
    _njit_max_pooling_1d,
    _njit_mean_pooling_1d,
    _njit_max_pooling_2d,
    _njit_mean_pooling_2d,
)


AGGREGATION_FUNCTIONS = {
    'max': partial(np.max, axis=-1),
    'min': partial(np.min, axis=-1),
    'mean': partial(np.mean, axis=-1),
    'sum': partial(np.sum, axis=-1),
}


class PoolingBlock(BaseBlock):
    """Pooling Block implements n-dimensional downsampling with an aggregation operation.

    It takes `X` of shape (n_samples, n_channels, n_features_1, ..., n_features_k)
    as an input and returns the pooled tensor of shape (n_samples, n_channels, t_1, ..., t_k).
    """

    meta = make_simple_meta(
        ['X'],
        ['output'],
        execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=False)
    )

    def __init__(self, kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[None, int, Tuple[int]] = None,
                 dilation: int = 1,
                 padding: Optional[int] = None,
                 aggregation: str = 'max',
                 chunk_size: int = -1,
                 impl_type: str = 'index'):
        """Initialize Pooling Block.

        Args:
            kernel_size: Kernel size (int or tuple).
            stride: Stride.
            dilation: Dilation (kernel stride).
            padding: Padding size (see `numpy.pad`);
                     if None padding is disabled.
            aggregation: Aggregation operation name ('max', 'min', 'mean', 'sum') or function.
                         If the function is given, it should aggregate by the last axis;
                         also the function can be passed only if `impl_type` is `index`
                         and input data are at CPU.
            chunk_size: Chunk size. Affects performance.
            impl_type: Implementation type ('index', 'njit', 'jax').
                       Note that 'jax' impl type will move data to GPU.
                       For the GPU input data only 'jax' implementation is available.

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
        self.impl_type = impl_type
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

    def __index_based_chunk_pooling(self, xs: np.ndarray) -> np.ndarray:
        n_samples, n_channels, *_ = xs.shape
        if self.params.padding is not None:
            xs = self.helper_.pad(xs)
        pooling_indices = self.__prepare_pooling_indices(xs.shape)
        result = []
        for i in range(n_samples):
            grouped_data = xs[i][pooling_indices.full_index_tuple].reshape((
                n_channels, pooling_indices.n_corners, pooling_indices.n_kernel_points
            ))
            aggregated = self.__aggregate(grouped_data)
            aggregated = aggregated.reshape((n_channels, *pooling_indices.pooled_shape))
            result.append(aggregated)
        result = np.stack(result, axis=0)
        return result

    def __njit_based_chunk_pooling(self, xs: np.ndarray) -> np.ndarray:
        if self.params.padding is not None:
            xs = self.helper_.pad(xs)
        n_samples, n_channels, *spatial_dims = xs.shape
        kernel_size = self.helper_.check_kernel_size(len(spatial_dims))
        stride = self.helper_.check_stride(spatial_dims, kernel_size)
        pooled_shape = self.helper_.get_pooled_shape(spatial_dims, kernel_size, stride)
        result = np.zeros((n_samples, n_channels, *pooled_shape), dtype=xs.dtype)
        if len(spatial_dims) == 2:
            if self.aggregation == 'max':
                _njit_max_pooling_2d(xs, result, kernel_size, stride, self.params.dilation)
            elif self.aggregation == 'mean':
                _njit_mean_pooling_2d(xs, result, kernel_size, stride, self.params.dilation)
            else:
                raise NotImplementedError(f'{self.aggregation=}')
        elif len(spatial_dims) == 1:
            if self.aggregation == 'max':
                _njit_max_pooling_1d(xs, result, kernel_size, stride, self.params.dilation)
            elif self.aggregation == 'mean':
                _njit_mean_pooling_1d(xs, result, kernel_size, stride, self.params.dilation)
            else:
                raise NotImplementedError(f'{self.aggregation=}')
        else:
            raise ValueError(
                f'Cannot run njit implementation on {len(spatial_dims)}-dimensional input, '
                'please check the number of dimensions and if it is >= 3, use impl_type="index"'
            )
        return result

    def __jax_based_pooling(self, xs: jnp.ndarray) -> jnp.ndarray:
        JAX_AGG = {
            'max': (-jnp.inf, lax.max),
            'min': (jnp.inf, lax.min),
            'mean': (0.0, lax.add),
            'sum': (0.0, lax.add),
        }
        _n_samples, _n_channels, *spatial_dims = xs.shape
        assert len(spatial_dims) > 0, f'Need at least one spatial dimension to apply pooling. Input shape: {xs.shape!r}'
        assert self.aggregation in JAX_AGG, f'Aggregation ({self.aggregation!r}) must be one of {JAX_AGG.keys()!r}'
        # prepare conv parameters
        kernel_size = self.helper_.check_kernel_size(len(spatial_dims))
        window_dimensions = (1, 1, *kernel_size)  # (N, C, n_1, ..., n_d)
        strides = self.helper_.check_stride(spatial_dims, kernel_size)
        window_strides = (1, 1, *strides)
        # prepare padding
        if self.params.padding is None:
            padding = 'valid'
        elif isinstance(self.params.padding, str):
            padding = self.params.padding
        elif isinstance(self.params.padding, Sequence):
            assert len(self.params.padding) == len(spatial_dims), 'Padding sequence should correspond to spatial dims'
            padding = ((0, 0), (0, 0), *self.params.padding)
        elif type(self.params.padding) == int:
            # symmetric padding
            p = self.params.padding
            padding = ((0, 0), (0, 0)) + ((p, p),) * len(spatial_dims)
        else:
            raise ValueError(f'Unsupported padding: {self.params.padding!r}')

        window_dilation = (1, 1, *(self.params.dilation,) * len(spatial_dims))
        # prepare reduction ops
        init_value, computation = JAX_AGG[self.aggregation]
        result = lax.reduce_window(
            xs,
            init_value,
            computation,
            window_dimensions,
            window_strides,
            padding,
            window_dilation=window_dilation,
        )
        if self.aggregation == 'mean':
            # need to divide by window size
            window_size = reduce(mul, kernel_size, 1)
            result = result / window_size
        return result

    def __chunk_pooling(self, xs: np.ndarray) -> np.ndarray:
        if self.impl_type == 'index':
            return self.__index_based_chunk_pooling(xs)
        elif self.impl_type == 'njit':
            return self.__njit_based_chunk_pooling(xs)
        raise ValueError(f'Wrong {self.impl_type=}')

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        """Apply Pooling to input 'X'.

        Args:
            inputs: Input data that consists of one element with key 'X'.
                    `inputs['X']` should have shape (n_samples, n_channels, n_features_1, ..., n_features_k).

        """
        assert 'X' in inputs
        xs = inputs['X'].data
        # shape: (n_samples, n_channels, n_features_1, ..., n_features_k)
        if isinstance(inputs['X'], GPUData) or self.impl_type == 'jax':
            # can process even CPU data, since JAX natively works with NumPy arrays
            result = self.__jax_based_pooling(xs)
            return {'output': GPUData(result)}
        elif isinstance(inputs['X'], CPUData):
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
        else:
            raise NotImplementedError(f'Not implemented for type {type(inputs["X"])!r}')


class GlobalAveragePoolingBlock(BaseBlock):
    """Global Average Pooling averages across all dimensions except sample and channels.

    For example, given an input of shape `(n_samples, n_channels, n_1, ..., n_d)`,
    the output will be of shape `(n_samples, n_channels)`.
    """
    meta = make_simple_meta(['X'], ['output'], execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def _check_dims(inputs):
        assert 'X' in inputs
        assert inputs['X'].data.ndim > 2, 'Global Average Pooling supports only data with at least one spatial dimension'

    def fit(self, inputs: BlockInputData) -> 'GlobalAveragePoolingBlock':
        self._check_dims(inputs)
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        self._check_dims(inputs)
        input_type = type(inputs['X'])
        if input_type not in [CPUData, GPUData]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        spatial_axes = tuple(range(2, inputs['X'].data.ndim))
        averaged = inputs['X'].data.mean(axis=spatial_axes)
        if input_type == CPUData:
            return {'output': CPUData(averaged)}
        else:
            return {'output': GPUData(averaged)}
