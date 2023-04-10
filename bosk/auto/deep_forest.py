"""Auto Deep Forest Construction.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterSampler

from ..data import CPUData
from ..pipeline.base import BasePipeline
from ..executor.base import BaseExecutor
from ..executor.recursive import RecursiveExecutor
from ..utility import get_rand_int, get_random_generator
from ..block.zoo.models.classification.classification_models import ETCBlock, RFCBlock
from ..block.base import BaseBlock

from .metrics import MetricsEvaluator
from .growing_strategies import GrowingStrategy, DefaultGrowingStrategy, EarlyStoppingCV
from .builders import SequentialPipelineBuilder
from .layers import Layer, NativeStackingLayer, StackingLayer
from .validation import (
    BasePipelineModelValidator,
    CVPipelineModelValidator,
    DumbPipelineModelValidator,
    TrainSetPipelineModelValidator,
)


DEFAULT_MAKE_METRICS = lambda: MetricsEvaluator(['f1', 'roc_auc'])
DEFAULT_EXECUTOR_CLS = RecursiveExecutor


class BaseAutoDeepForestConstructor(ABC):
    """Base Auto Deep Forest Construction Algorithm.

    Sequentially constructs a pipeline using given data set.
    Construction is splitted into *steps*, where each step is subdivided into
    *iterations*:

        - At different *steps* the algorithm makes principally different blocks;
        - At different *iterations* the algorithm generates layers of the same structure.

    """
    def __init__(self, executor_cls: BaseExecutor,
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 random_state: Optional[int] = None):
        self.executor_cls = executor_cls or DEFAULT_EXECUTOR_CLS
        self.max_iter = max_iter
        self.cv = cv
        self.make_metrics = make_metrics or DEFAULT_MAKE_METRICS
        if growing_strategy is None:
            if cv is None:
                growing_strategy = DefaultGrowingStrategy()
            else:
                growing_strategy = EarlyStoppingCV()
        self.growing_strategy = growing_strategy
        self.random_state = random_state

    @property
    @abstractmethod
    def n_steps(self) -> int:
        """Number of implemented steps.

        The steps with number < `n_steps` will be passed to `make_step_layer`.
        """

    @abstractmethod
    def make_step_layer(self, step: int, iteration: int,
                        X: np.ndarray, y: np.ndarray,
                        validator: BasePipelineModelValidator,
                        rng: np.random.RandomState) -> Optional[Layer]:
        ...

    def construct(self, X: np.ndarray, y: np.ndarray) -> BasePipeline:
        rng = get_random_generator(self.random_state)
        X = CPUData(X)
        y = CPUData(y)
        # prepare the validator
        if self.cv is None:
            validator = DumbPipelineModelValidator(None, self.make_metrics)
        elif self.cv == 1:
            validator = TrainSetPipelineModelValidator(None, self.make_metrics)
        else:
            validator = CVPipelineModelValidator(
                StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=12345),
                self.make_metrics,
            )
        # prepare the make
        maker = SequentialPipelineBuilder(
            self.executor_cls,
            growing_strategy=self.growing_strategy,
            X=X,
            X_original=X,
            y=y
        )
        for step in range(self.n_steps):
            for i in range(self.max_iter):
                next_layer = self.make_step_layer(step, i, X.data, y.data, validator, rng)
                if next_layer is None:
                    break
                need_continue = maker.append(next_layer)
                if not need_continue:
                    break

        pipeline = maker.build(dict())
        self.history = maker.history
        return pipeline


def _make_base_df_blocks(params: dict, layer_width: int) -> List[BaseBlock]:
    """Make base Deep Forest Blocks for one layer.

    Args:
        params: One Forest parameters (n_estimators, max_depth, etc.).
        layer_width: Number of blocks of one type in the layer.

    Returns:
        List of layer blocks.

    """
    BLOCK_CLASSES = (RFCBlock, ETCBlock)
    result = [
        [block_cls(**params) for _ in range(layer_width)]
        for block_cls in BLOCK_CLASSES
    ]
    # flatten
    return [b for list_of_blocks in result for b in list_of_blocks]


class ClassicalDeepForestConstructor(BaseAutoDeepForestConstructor):
    """Classical Deep Forest iteratively builds layers consisting of different
    Random Forests and Extremely Randomized Trees Classifiers.

    Attributes:
        n_steps: Number of steps. The Classical Deep Forest has just one step.
        rf_params: Parameters of tree ensembles (Random Forests, Extra Trees).

    """
    LAYER_CLS = NativeStackingLayer
    n_steps = 1

    def __init__(self, executor_cls: BaseExecutor, rf_params: Optional[dict] = None,
                 layer_width: int = 2,
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 random_state: Optional[int] = None):
        super().__init__(executor_cls, max_iter, cv, make_metrics, growing_strategy, random_state)
        if rf_params is None:
            rf_params = dict()
        self.rf_params = rf_params
        self.layer_width = layer_width

    def make_step_layer(self, step: int, iteration: int,
                        X: np.ndarray, y: np.ndarray,
                        validator: BasePipelineModelValidator,
                        rng: np.random.RandomState):
        return self.LAYER_CLS(
            make_blocks=lambda: _make_base_df_blocks(self.rf_params, self.layer_width),
            layer_name=f'{self.LAYER_CLS.__name__}({step=}, {iteration=})',
            executor_cls=self.executor_cls,
            validator=validator,
            random_state=get_rand_int(rng)
        )


class HyperparamSearchDeepForestConstructor(BaseAutoDeepForestConstructor):
    """Classical Deep Forest that estimates the best parameters at each step.

    Attributes:
        n_steps: Number of steps. At each step the best parameters are estimated.
        rf_param_grid: Parameters grid for the tree ensembles (Random Forests, Extra Trees).

    """
    n_steps = None  # set at initialization
    LAYER_CLS = NativeStackingLayer

    def __init__(self, executor_cls: BaseExecutor, rf_param_grid: Optional[dict] = None,
                 layer_width: int = 2,
                 n_steps: int = 3,
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 random_state: Optional[int] = None):
        super().__init__(executor_cls, max_iter, cv, make_metrics, growing_strategy, random_state)
        if rf_param_grid is None:
            rf_param_grid = dict()
        self.rf_param_grid = rf_param_grid
        self.layer_width = layer_width
        self.n_steps = n_steps

    def make_step_layer(self, step: int, iteration: int,
                        X: np.ndarray, y: np.ndarray,
                        validator: BasePipelineModelValidator,
                        rng: np.random.Generator):
        sampler = ParameterSampler(self.rf_param_grid, n_iter=1, random_state=get_rand_int(rng))
        rf_params = next(iter(sampler))
        return self.LAYER_CLS(
            make_blocks=lambda: _make_base_df_blocks(rf_params, self.layer_width),
            layer_name=f'{self.LAYER_CLS.__name__}({step=}, {iteration=})',
            executor_cls=self.executor_cls,
            validator=validator,
            random_state=get_rand_int(rng)
        )
