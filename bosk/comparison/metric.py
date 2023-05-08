from abc import ABC, abstractmethod
from bosk.data import BaseData, GPUData
from typing import Dict, Optional, Callable

import numpy as np


class BaseMetric(ABC):
    """Base class for all metrics, taking part in the
    models' comparison process.
    """
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def get_score(self, data_true: Dict[str, BaseData], data_pred: Dict[str, BaseData]) -> float:
        """Method to obtain a metric score."""


class MetricWrapper(BaseMetric):
    """Wrapper class for classic metric functions that
    take `y_true` as the first argument and `y_pred` as the second.

    Args:
        func: Function that accepts `y_true` and `y_pred` as arguments and returns a score.
        y_true_name: Name of the `y_true`.
        y_pred_name: Name of the `y_pred`.

    """
    def __init__(self, func: Callable[[np.ndarray, np.ndarray], float], y_true_name: str = 'y',
                 y_pred_name: str = 'output', name: Optional[str] = None) -> None:
        super().__init__(name)
        self.fn = func
        self.y_true_name = y_true_name
        self.y_pred_name = y_pred_name

    def get_score(self, data_true: Dict[str, BaseData], data_pred: Dict[str, BaseData]) -> float:
        """Calculate the metric score given `y_true` and `y_pred`.

        Args:
            data_true: Dictionary containing the `y_true`.
            data_pred: Dictionary containing the `y_pred`.

        """
        y_true = data_true[self.y_true_name]
        y_pred = data_pred[self.y_pred_name]
        return self.fn(y_true.to_cpu().data, y_pred.to_cpu().data)
