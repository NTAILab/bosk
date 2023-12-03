from typing import Tuple, Optional

import itertools
import numpy as np

from ...auto import auto_block
from .base import MultiGrainedScanningBlock


@auto_block(auto_state=True, random_state_field=None)
class MultiGrainedScanning2D(MultiGrainedScanningBlock):
    """2-dimensional Multi Grained Scanning Block.

    Main use case is to process data with two spatial dimensions, like images.

    It takes `X` of shape `(n_samples, n_channels)`
    as an input and returns the tensor of shape `(n_samples, n_out_channels)`.

    Args:
        models: Tuple of underlying models.
        window_size: Size of the sliding window.
        stride: Stride of the sliding window.
        shape_sample: Input data spatial dimensions (shape).

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Data tensor of shape `(n_samples, n_channels)`.
        - y: Target variable values of shape `(n_samples, [n_outputs])`.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Data tensor of shape `(n_samples, n_channels)`.

    Output slots
    ------------

        - output: Prediction tensor of shape `(n_samples, n_out_channels)`.

    """

    def _window_slicing_data(self, X, y=None) -> 'Tuple[np.ndarray, Optional[np.ndarray]]':
        shape = self._shape_sample
        if shape is None:
            raise ValueError("Please set dimension of sample in 'MultiGrainedScanning2DBlock'")
        if any(s < self._window_size for s in shape):
            raise ValueError('window must be smaller than both dimensions for an image')

        len_iter_x = np.floor_divide((shape[1] - self._window_size), self._stride) + 1
        len_iter_y = np.floor_divide((shape[0] - self._window_size), self._stride) + 1
        iterx_array = np.arange(0, self._stride * len_iter_x, self._stride)
        itery_array = np.arange(0, self._stride * len_iter_y, self._stride)

        ref_row = np.arange(0, self._window_size)
        ref_ind = np.ravel([ref_row + shape[1] * i for i in range(self._window_size)])
        inds_to_take = [ref_ind + ix + shape[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]

        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, self._window_size ** 2)
        sliced_target = None
        if y is not None:
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
        return sliced_imgs, sliced_target


MultiGrainedScanning2DBlock = MultiGrainedScanning2D

