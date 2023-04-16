from abc import ABC, abstractmethod
from bosk.data import BaseData, GPUData
from typing import Dict, Optional, Callable


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
    """
    def __init__(self, func: Callable, y_true_name: str = 'y',
                 y_pred_name: str = 'output', name: Optional[str] = None) -> None:
        super().__init__(name)
        self.fn = func
        self.y_true_name = y_true_name
        self.y_pred_name = y_pred_name

    def get_score(self, data_true: Dict[str, BaseData], data_pred: Dict[str, BaseData]) -> float:
        y_true = data_true[self.y_true_name]
        if isinstance(y_true, GPUData):
            y_true = y_true.to_cpu()
        y_pred = data_pred[self.y_pred_name]
        if isinstance(y_pred, GPUData):
            y_pred = y_pred.to_cpu()
        y_true, y_pred = y_true.data, y_pred.data
        return self.fn(y_true, y_pred)
