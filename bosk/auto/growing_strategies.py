from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Mapping, Sequence, Type, Union

import numpy as np

from ..pipeline.base import BasePipeline
from ..data import BaseData
from ..stages import Stage
from ..executor.base import BaseExecutor
from .metrics import MetricsEvaluator


class GrowingStrategy(ABC):
    """Growing strategy.

    Determines the stopping criterion for a Deep Forest builder and how to trim the pipelines
    after stopping.

    """
    @abstractmethod
    def need_grow(self, pipeline, metrics, executor_cls, growing_state: dict) -> bool:
        """Check if the pipeline needs to be grown further.

        Args:
            pipeline: The most recent pipeline to check.
            metrics: Metrics evaluation results.
            executor_cls: Executor class to use.
            growing_state: Growing state that stores a context of the strategy.

        Returns:
            True if the pipeline needs to be grown further.

        """
        ...

    def trim(self, pipelines: List[BasePipeline], growing_state: dict) -> List[BasePipeline]:
        """Trim the pipelines after stopping.

        Args:
            pipelines: The pipelines list to trim.
            growing_state: Growing state that stores a context of the strategy.

        Returns:
            Trimmed pipelines

        """
        return pipelines


class DefaultGrowingStrategy(GrowingStrategy):
    """Default growing strategy.

    Never stops.

    """
    def need_grow(self, pipeline, metrics, executor_cls, growing_state: dict) -> bool:
        return True


class EarlyStoppingCV(GrowingStrategy):
    """Early stopping based on cross-validation metrics.

    Args:
        mode: Stop when **any** of the metrics is worse than the best metric or
              **all* of the metrics are worse than the best metric.
        patience: Number of epochs to wait before stopping.

    """
    def __init__(self, mode: Union[Literal['any'], Literal['all']] = 'all', patience: int = 1):
        self.mode = mode
        self.patience = patience

    def need_grow(self, pipeline, metrics, executor_cls, growing_state: dict) -> bool:
        assert metrics is not None, 'Please, calculate CV metrics first'
        if 'best_metrics' not in growing_state:
            growing_state['best_metrics'] = metrics
            growing_state['count'] = 0
            return True
        MODES = {
            'any': any,
            'all': all
        }
        best_metrics = growing_state['best_metrics']
        # check if new metrics are lower than previous
        new_metrics_worse = MODES[self.mode](
            metrics[metric_name] <= best_metrics[metric_name]
            for metric_name in metrics.keys()
        )
        # update the best metrics
        growing_state['best_metrics'] = {
            metric_name: max(metrics[metric_name], best_metrics[metric_name])
            for metric_name in metrics.keys()
        }
        if new_metrics_worse:
            growing_state['count'] += 1
            if growing_state['count'] > self.patience:
                return False
        else:
            growing_state['count'] = 0
        return True

    def trim(self, pipelines: List[BasePipeline], growing_state: dict):
        """Trim the last `count` pipelines.
        """
        count = growing_state['count']
        if count > 0:
            growing_state['count'] = 0
            return pipelines[:-count]
        return pipelines


class EarlyStoppingVal(EarlyStoppingCV):
    """Early stopping based on validation metrics.

    Args:
        data: Data for validation.
        make_metrics_eval: Function that returns a new `MetricsEvaluator`.
        mode: Stop when **any** of the metrics is worse than the best metric or
              **all* of the metrics are worse than the best metric.
        patience: Number of epochs to wait before stopping.

    """
    def __init__(self, data: Mapping[str, BaseData], make_metrics_eval: Callable[[], MetricsEvaluator],
                 **early_stopping_params):
        super().__init__(**early_stopping_params)
        self.initial_data = {k: v for k, v in data.items() if k != 'y'}
        self.y = data['y']
        self.make_metrics_eval = make_metrics_eval

    def need_grow(self, pipeline, metrics, executor_cls: Type[BaseExecutor], growing_state: dict) -> bool:
        if 'metrics' not in growing_state:
            growing_state['current_data'] = self.initial_data
        transformer = executor_cls(pipeline, Stage.TRANSFORM)
        current_input = {
            k: v
            for k, v in growing_state['current_data'].items()
            if k != 'proba'
        }
        predictions = transformer(current_input)
        growing_state['current_data'] = predictions
        metrics_eval = self.make_metrics_eval()
        assert isinstance(self.y.data, np.ndarray)
        assert isinstance(predictions['proba'].data, np.ndarray)
        metrics_eval.append_eval(self.y.data, predictions['proba'].data)
        val_metrics = metrics_eval.average()
        return super().need_grow(pipeline, val_metrics, executor_cls, growing_state)

    def trim(self, pipelines: List[BasePipeline], growing_state: dict):
        """Trim the last `count` pipelines.
        """
        count = growing_state['count']
        if count > 0:
            return pipelines[:-count]
        return pipelines
