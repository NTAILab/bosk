from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

import numpy as np
from numpy.random import Generator
import warnings

from ....data import CPUData
from ....utility import get_random_generator, get_rand_int


class MultiGrainedScanningBlock(ABC):
    def __init__(self, models: Tuple[Any, Any],
                 window_size: int,
                 stride: int,
                 shape_sample: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self._shape_sample = shape_sample
        self._window_size: int = window_size
        self._stride: int = stride
        self._models: Tuple[Any, Any] = models

    def fit(self, X, y) -> 'MultiGrainedScanningBlock':
        self._window_slicing_fit(X, y)
        return self

    def _window_slicing_fit(self, X, y) -> 'None':
        sliced_X, sliced_y = self._window_slicing_data(X.data, y.data)
        for model in self._models:
            if hasattr(model, 'fit'):
                model.fit(sliced_X, sliced_y)
            else:
                raise Exception("Please check that the model used in 'MultiGrainedScanningBlock' "
                                "have the method 'fit'")

    @abstractmethod
    def _window_slicing_data(self, X, y=None) -> 'Tuple[np.ndarray, Optional[np.ndarray]]':
        pass

    def _window_slicing_predict(self, X) -> 'np.ndarray':
        sliced_X, _ = self._window_slicing_data(X.data)
        predict_prob: np.ndarray = np.array([])
        for model in self._models:
            if hasattr(model, 'predict_proba'):
                predict_prob = np.hstack([predict_prob, model.predict_proba(sliced_X)]) \
                    if predict_prob.size else model.predict_proba(sliced_X)
            else:
                raise Exception(
                    "Please check that the model used in 'MultiGrainedScanningBlock'"
                    " have the method 'predict_proba'")
        return predict_prob.reshape([X.data.shape[0], -1])

    def transform(self, X) -> CPUData:
        return CPUData(self._window_slicing_predict(X))

    def set_random_state(self, seed: Optional[int | Generator]) -> None:
        gen = get_random_generator(seed)
        for model in self._models:
            if hasattr(model, 'random_state'):
                model.random_state = get_rand_int(gen)
            else:
                warnings.warn(f"Model {model.__class__.__name__!r} doesn't have 'random_state' field")
