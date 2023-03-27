import numpy as np
from typing import Optional, Tuple, Union, NamedTuple


class _PoolingIndices(NamedTuple):
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


class _ConvolutionParams(NamedTuple):
    kernel_size: Union[int, Tuple[int]] = 3
    stride: Union[None, int, Tuple[int]] = None
    dilation: int = 1
    padding: Optional[int] = None
    chunk_size: int = -1


class _ConvolutionHelper:
    def __init__(self, params: _ConvolutionParams):
        self.params = params

    def check_stride(self, spatial_dims: Tuple[int], kernel_size: Tuple[int]):
        """Check the stride and return a correct stride tuple.

        Args:
            spatial_dims: Spatial dimensions.
            kernel_size: Kernel dimensions.

        Returns:
            Stride tuple.

        """
        n_spatial_dims = len(spatial_dims)
        if self.params.stride is None:
            stride = kernel_size
        elif isinstance(self.params.stride, tuple):
            stride = self.params.stride
        else:
            stride = (self.params.stride,) * n_spatial_dims
        return stride

    def get_pooled_shape(self, spatial_dims: Tuple[int], kernel_size: Tuple[int], stride: Tuple[int]):
        pooled_shape = tuple([
            (
                (spatial_dims[i] // s) if s >= kernel_size[i]
                else ((spatial_dims[i] - kernel_size[i] + 1) // s)
            )
            for i, s in enumerate(stride)
        ])
        return pooled_shape

    def prepare_corner(self, spatial_dims: Tuple[int], kernel_size: Tuple[int]):
        """Prepare sliding window corner ids.

        Args:
            spatial_dims: Spatial dimensions.
            kernel_size: Kernel dimensions.

        Returns:
            Sliding window corner indices.

        """
        stride = self.check_stride(spatial_dims, kernel_size)
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

    def check_kernel_size(self, n_spatial_dims: int):
        """Check the kernel size and return a correct kernel size tuple.

        Args:
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Kernel size tuple.

        """
        if isinstance(self.params.kernel_size, tuple):
            kernel_size = self.params.kernel_size
        else:
            kernel_size = (self.params.kernel_size,) * n_spatial_dims
        return kernel_size

    def prepare_kernel(self, n_spatial_dims: int):
        """Prepare kernel indices.

        Args:
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Tuple (kernel size, kernel indices).

        """
        kernel_size = self.check_kernel_size(n_spatial_dims)
        kernel_ids = np.stack(np.meshgrid(*[np.arange(k) for k in kernel_size], indexing='ij'), axis=0)
        # kernel_ids shape: (n_spatial_dims, k_1, ..., k_nsd)
        if self.params.dilation > 1:
            kernel_ids = kernel_ids[:, np.all(kernel_ids % self.params.dilation == 0, axis=0)]
        kernel_ids = kernel_ids.reshape((kernel_ids.shape[0], -1))
        return kernel_size, kernel_ids

    def pad(self, xs: np.ndarray) -> np.ndarray:
        """Add symmetric edge padding to the input array.

        Args:
            xs: Input data.

        Returns:
            Padded input.

        """
        n_spatial_dims = xs.ndim - 2
        if isinstance(self.params.padding, tuple):
            padding_size = self.params.padding
        else:
            padding_size = ((self.params.padding, self.params.padding),) * n_spatial_dims
        return np.pad(
            xs,
            pad_width=((0, 0), (0, 0), *padding_size),
            mode='edge',
        )

    def prepare_pooling_indices(self, xs_shape: Tuple[int]):
        """Prepare pooling indices using convolution parameters and the input shape.

        Args:
            xs_shape: Input shape.

        Returns:
            Pooling indices.

        """
        _, n_channels, *spatial_dims = xs_shape
        n_spatial_dims = len(spatial_dims)
        kernel_size, kernel_ids = self.prepare_kernel(n_spatial_dims)
        corner_ids = self.prepare_corner(spatial_dims, kernel_size)
        pooled_shape = corner_ids.shape[1:]
        corner_ids = corner_ids.reshape((corner_ids.shape[0], -1))
        # corner_ids shape: (n_spatial_dims, s_1 * ... * s_nsd)
        all_ids = corner_ids[:, :, np.newaxis] + kernel_ids[:, np.newaxis, :]
        all_ids = all_ids.reshape((n_spatial_dims, -1))
        # all_ids shape: (n_spatial_dims, n_positions)
        full_index_tuple = tuple(
            np.concatenate((
                np.arange(all_ids.shape[1] * n_channels)[np.newaxis] // all_ids.shape[1],
                np.tile(all_ids, (1, n_channels))
            ), axis=0)
        )
        pooling_indices = _PoolingIndices(
            xs_shape,
            full_index_tuple,
            corner_ids.shape[1],
            kernel_ids.shape[1],
            pooled_shape
        )
        return pooling_indices

    def slice(self, xs: np.ndarray, pooling_indices: _PoolingIndices) -> np.ndarray:
        """Cut or slice the data into pieces.

        Args:
            xs: Input n-dimensional data of shape (N, C, F1, ..., Fk),
                where N is the number of samples,
                C is the number of channels,
                F1, ..., Fk are the spatial dimensions.

        Returns:
            Tuple (sliced data, number of slices).

        """
        n_samples, n_channels, *_ = xs.shape
        sliced = np.zeros(
            (n_samples, n_channels, pooling_indices.n_corners, pooling_indices.n_kernel_points),
            dtype=xs.dtype
        )
        for i in range(n_samples):
            grouped_data = xs[i][pooling_indices.full_index_tuple].reshape((
                -1, pooling_indices.n_corners, pooling_indices.n_kernel_points
            ))
            sliced[i] = grouped_data
        # out_shape = (n_channels, *pooling_indices.pooled_shape)
        # shape: (n_samples, n_channels, pooling_indices.n_corners, pooling_indices.n_kernel_points)
        sliced = np.swapaxes(sliced, 1, 2)
        # shape: (n_samples, pooling_indices.n_corners, n_channels, pooling_indices.n_kernel_points)
        sliced = sliced.reshape((-1, *sliced.shape[2:]))
        # shape: (n_samples * pooling_indices.n_corners, n_channels, pooling_indices.n_kernel_points)
        sliced = sliced.reshape((sliced.shape[0], -1))
        # shape: (n_samples * pooling_indices.n_corners, n_channels * pooling_indices.n_kernel_points)
        n_corners = pooling_indices.n_corners
        return sliced, n_corners
