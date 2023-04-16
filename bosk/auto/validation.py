from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Mapping
from sklearn.model_selection import BaseCrossValidator

from ..data import CPUData
from .metrics import MetricsEvaluator


class BasePipelineModelValidator(ABC):
    """Pipeline model validator calculates metrics given dataset and fit, transform methods.
    """
    @abstractmethod
    def __init__(self, cv: Optional[BaseCrossValidator],
                 metrics_cls: Callable[[], MetricsEvaluator]):
        ...

    @abstractmethod
    def calc_metrics(self, data: Mapping[str, CPUData],
                     fitter,
                     transformer,
                     output: str = 'proba') -> Optional[Mapping[str, float]]:
        ...


class CVPipelineModelValidator(BasePipelineModelValidator):
    need_refit = True

    def __init__(self, cv: BaseCrossValidator,
                 metrics_cls: Callable[[], MetricsEvaluator]):
        self.cv = cv
        self.metrics_cls = metrics_cls

    def calc_metrics(self, data: Mapping[str, CPUData],
                     fitter,
                     transformer,
                     output: str = 'proba') -> Optional[Mapping[str, float]]:
        metrics = self.metrics_cls()
        for train_idx, val_idx in self.cv.split(data['X'].data, data['y'].data):
            train_data = {
                k: CPUData(v.data[train_idx])
                for k, v in data.items()
            }
            fitter(train_data)
            val_data = {
                k: CPUData(v.data[val_idx])
                for k, v in data.items()
            }
            preds_val = transformer({k: v for k, v in val_data.items() if k != 'y'})[output].data
            metrics.append_eval(val_data['y'].data, preds_val.data)
        return metrics.average()


class DumbPipelineModelValidator(BasePipelineModelValidator):
    need_refit = True

    def __init__(self, cv: BaseCrossValidator,
                 metrics_cls: Callable[[], MetricsEvaluator]):
        self.cv = cv
        self.metrics_cls = metrics_cls

    def calc_metrics(self, data: Mapping[str, CPUData],
                     fitter,
                     transformer,
                     output: str = 'proba') -> Optional[Mapping[str, float]]:
        return None


class TrainSetPipelineModelValidator(BasePipelineModelValidator):
    need_refit = False

    def __init__(self, cv: BaseCrossValidator,
                 metrics_cls: Callable[[], MetricsEvaluator]):
        self.cv = cv
        self.metrics_cls = metrics_cls

    def calc_metrics(self, data: Mapping[str, CPUData],
                     fitter,
                     transformer,
                     output: str = 'proba') -> Optional[Mapping[str, float]]:
        metrics = self.metrics_cls()
        fitter(data)
        preds = transformer({k: v for k, v in data.items() if k != 'y'})[output].data
        metrics.append_eval(data['y'].data, preds)
        return metrics.average()
