from typing import Tuple, Optional

import numpy as np

from .base import MultiGrainedScanningBlock
from ...auto import auto_block


@auto_block(auto_state=True, random_state_field=None)
class MultiGrainedScanning1D(MultiGrainedScanningBlock):
    """1-dimensional Multi Grained Scanning Block.

    Main use case is to process data with one spatial dimension, like sequences.

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
        shape = X.shape[1]
        if shape < self._window_size:
            raise ValueError('window must be smaller than the sequence dimension')

        len_iter = np.floor_divide((shape - self._window_size), self._stride) + 1
        iter_array = np.arange(0, self._stride * len_iter, self._stride)

        ind_1X = np.arange(np.prod(shape))
        inds_to_take = [ind_1X[i:i + self._window_size] for i in iter_array]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, self._window_size)

        sliced_target = None
        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        return sliced_sqce, sliced_target


MultiGrainedScanning1DBlock = MultiGrainedScanning1D

