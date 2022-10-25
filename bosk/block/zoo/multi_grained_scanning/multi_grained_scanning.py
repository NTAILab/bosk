from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, List, Tuple, Optional

import numpy as np


class ModelTypeIndex(IntEnum):
    RANDOM = 0,
    COMPLETELY_RANDOM = 1


class MultiGrainedScanningBlock(ABC):
    def __init__(self, models: Tuple[Any], window_size: int, stride: int, shape_sample: Optional[Tuple[int]] = None):
        super().__init__()
        self._shape_sample = shape_sample
        self._window_size: int = window_size
        self._stride: int = stride
        self._models: Tuple[Any, Any] = (models[ModelTypeIndex.RANDOM],
                                         models[ModelTypeIndex.COMPLETELY_RANDOM])

    def fit(self, X, y) -> 'MultiGrainedScanningBlock':
        self._window_slicing_fit(X, y)
        return self

    def _window_slicing_fit(self, X, y) -> 'None':
        sliced_X, sliced_y = self._window_slicing_data(X, y)
        prf = self._models[ModelTypeIndex.RANDOM]
        crf = self._models[ModelTypeIndex.COMPLETELY_RANDOM]
        if hasattr(prf, 'fit') and hasattr(prf, 'fit'):
            prf.fit(sliced_X, sliced_y)
            crf.fit(sliced_X, sliced_y)
        else:
            raise Exception("Please check that the models used in 'MultiGrainedScanningBlock' have the method 'fit'")

    @abstractmethod
    def _window_slicing_data(self, X, y=None) -> 'Tuple[np.ndarray, Optional[np.ndarray]]':
        pass

    def _window_slicing_predict(self, X) -> 'np.ndarray':
        sliced_X, _ = self._window_slicing_data(X)
        pred_prob_prf = self._models[ModelTypeIndex.RANDOM].predict_proba(sliced_X)
        pred_prob_crf = self._models[ModelTypeIndex.COMPLETELY_RANDOM].predict_proba(sliced_X)
        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]
        return pred_prob.reshape([X.shape[0], -1])

    def transform(self, X) -> 'np.ndarray':
        res = self._window_slicing_predict(X)
        return res
