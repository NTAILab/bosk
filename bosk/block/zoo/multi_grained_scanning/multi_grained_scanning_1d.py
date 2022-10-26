from typing import Tuple, Optional

import numpy as np

from bosk.block.zoo.multi_grained_scanning.multi_grained_scanning import MultiGrainedScanningBlock
from bosk.block import auto_block


@auto_block
class MultiGrainedScanning1DBlock(MultiGrainedScanningBlock):
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
