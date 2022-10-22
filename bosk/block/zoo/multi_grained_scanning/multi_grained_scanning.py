from typing import Any, List, Tuple

import numpy as np
import itertools
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

# from bosk.block import BaseBlock, BlockInputData, TransformOutputData
# from bosk.util.util import make_simple_meta
# from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block import auto_block


@auto_block
class MultiGrainedScanningBlock:
    def __init__(self, windows: List[int], stride: np.ndarray, random_state: int):
        super().__init__()
        self._windows: List[int] = windows
        self._stride: np.ndarray = stride
        self._models: List[Tuple[Any, Any]] = [(RandomForestClassifier(random_state), ExtraTreesClassifier(random_state))
                                               for _ in self._windows]

    def fit(self, X, y) -> 'MultiGrainedScanningBlock':
        for ind, wdw_size in enumerate(self._windows):
            self._window_slicing_fit(X, y, wdw_size, ind)
        return self

    def _window_slicing_fit(self, X, y, window, ind):
        # if X.shape[1] > 1:
        #     print('Slicing Images...')
        #     sliced_X, sliced_y = self._window_slicing_img(X, y, window, X.shape, stride=self._stride)
        # else:
        #     print('Slicing Sequence...')
        sliced_X, sliced_y = self._window_slicing_sequence(X, y, window, X.shape, stride=self._stride)
        prf = self._models[ind][0]
        crf = self._models[ind][1]
        print('Training MGS Random Forests...')
        prf.fit(sliced_X, sliced_y)
        crf.fit(sliced_X, sliced_y)

    def _window_slicing_img(self, X, y, window, shape, stride=1):
        if any(s < window for s in shape):
            raise ValueError('window must be smaller than both dimensions for an image')

        len_iter_x = np.floor_divide((shape[1] - window), stride) + 1
        len_iter_y = np.floor_divide((shape[0] - window), stride) + 1
        iterx_array = np.arange(0, stride * len_iter_x, stride)
        itery_array = np.arange(0, stride * len_iter_y, stride)

        ref_row = np.arange(0, window)
        ref_ind = np.ravel([ref_row + shape[1] * i for i in range(window)])
        inds_to_take = [ref_ind + ix + shape[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]

        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, window ** 2)
        sliced_target = None
        if y is not None:
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
        return sliced_imgs, sliced_target

    def _window_slicing_sequence(self, X, y=None, window=None, shape=None, stride=1):
        if shape[1] < window:
            raise ValueError('window must be smaller than the sequence dimension')

        len_iter = np.floor_divide((shape[1] - window), stride) + 1
        iter_array = np.arange(0, stride * len_iter, stride)

        ind_1X = np.arange(np.prod(shape))
        inds_to_take = [ind_1X[i:i + window] for i in iter_array]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)

        sliced_target = None
        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        return sliced_sqce, sliced_target

    def _window_slicing_predict(self, X, window, ind, shape):
        """ Performs a window slicing of the input data and send them through Random Forests.
        If target values 'y' are provided sliced data are then used to train the Random Forests.
        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.
        :param window: int
            Size of the window to use for slicing.
        :param shape_1X: list or np.array
            Shape of a single sample.
        :param y: np.array (default=None)
            Target values. If 'None' no training is done.
        :return: np.array
            Array of size [n_samples, ..] containing the Random Forest.
            prediction probability for each input sample.
        """
        # if shape[0] > 1:
        #     print('Slicing Images...')
        #     sliced_X, _ = self._window_slicing_img(X, window, shape, stride=self._stride)
        # else:
        print('Slicing Sequence...')
        sliced_X, _ = self._window_slicing_sequence(X, window=window, shape=shape, stride=self._stride)
        pred_prob_prf = self._models[ind][0].predict_proba(sliced_X)
        pred_prob_crf = self._models[ind][1].predict_proba(sliced_X)
        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]
        return pred_prob.reshape([shape[0], -1])

    def transform(self, X) -> 'np.ndarray':
        mgs_pred_prob = []
        for ind, wdw_size in enumerate(self._windows):
            mgs_pred_prob.append(self._window_slicing_predict(X, wdw_size, ind, X.shape))
        return np.concatenate(mgs_pred_prob, axis=1)
