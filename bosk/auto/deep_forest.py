"""Auto Deep Forest Construction.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Tuple, Type
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
from .layers import Layer, MGSRFLayer, NativeStackingLayer, StackingLayer
from .validation import (
    BasePipelineModelValidator,
    CVPipelineModelValidator,
    DumbPipelineModelValidator,
    TrainSetPipelineModelValidator,
)


def DEFAULT_MAKE_METRICS():
    return MetricsEvaluator(['f1', 'roc_auc'])


DEFAULT_EXECUTOR_CLS = RecursiveExecutor


class BaseAutoDeepForestConstructor(ABC):
    """Base Auto Deep Forest Construction Algorithm.

    Sequentially constructs a pipeline using given data set.
    Construction is splitted into *steps*, where each step is subdivided into
    *iterations*:

        - At different *steps* the algorithm makes principally different blocks;
        - At different *iterations* the algorithm generates layers of the same structure.

    """

    def __init__(self, executor_cls: Type[BaseExecutor],
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 block_classes: Optional[Tuple[Type[BaseBlock], ...]] = None,
                 random_state: Optional[int] = None):
        self.executor_cls = executor_cls or DEFAULT_EXECUTOR_CLS
        assert issubclass(self.executor_cls, BaseExecutor)
        self.max_iter = max_iter
        self.cv = cv
        self.make_metrics = make_metrics or DEFAULT_MAKE_METRICS
        if growing_strategy is None:
            if cv is None:
                growing_strategy = DefaultGrowingStrategy()
            else:
                growing_strategy = EarlyStoppingCV()
        self.growing_strategy = growing_strategy
        self.block_classes = block_classes
        self.random_state = random_state

    @property
    @abstractmethod
    def n_steps(self) -> Optional[int]:
        """Number of implemented steps.

        The steps with number < `n_steps` will be passed to `make_step_layer`.
        """

    @abstractmethod
    def make_step_layer(self, step: int, iteration: int,
                        X: np.ndarray, y: np.ndarray,
                        validator: BasePipelineModelValidator,
                        rng: np.random.Generator) -> Optional[Layer]:
        ...

    def construct(self, X_: np.ndarray, y_: np.ndarray) -> BasePipeline:
        assert type(self.n_steps) == int
        rng = get_random_generator(self.random_state)
        X = CPUData(X_)
        y = CPUData(y_)
        # prepare the validator
        validator: BasePipelineModelValidator
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
                next_layer = self.make_step_layer(step, i, X_, y_, validator, rng)
                if next_layer is None:
                    break
                need_continue = maker.append(next_layer)
                if not need_continue:
                    break

        pipeline = maker.build(dict())
        self.history = maker.history
        return pipeline


def _make_base_df_blocks(params: dict, layer_width: int,
                         block_classes: Optional[Tuple[Type[BaseBlock], ...]] = None) -> List[BaseBlock]:
    """Make base Deep Forest Blocks for one layer.

    Args:
        params: One Forest parameters (n_estimators, max_depth, etc.).
        layer_width: Number of blocks of one type in the layer.

    Returns:
        List of layer blocks.

    """
    BLOCK_CLASSES = (RFCBlock, ETCBlock)
    if block_classes is None:
        block_classes = BLOCK_CLASSES
    result = [
        [block_cls(**params) for _ in range(layer_width)]
        for block_cls in block_classes
    ]
    # flatten
    return [b for list_of_blocks in result for b in list_of_blocks]  # type: ignore


class ClassicalDeepForestConstructor(BaseAutoDeepForestConstructor):
    """Classical Deep Forest iteratively builds layers consisting of different
    Random Forests and Extremely Randomized Trees Classifiers.

    Attributes:
        n_steps: Number of steps. The Classical Deep Forest has just one step.
        rf_params: Parameters of tree ensembles (Random Forests, Extra Trees).

    """
    LAYER_CLS = NativeStackingLayer
    n_steps = 1

    def __init__(self, executor_cls: Type[BaseExecutor],
                 rf_params: Optional[dict] = None,
                 layer_width: int = 2,
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 block_classes: Optional[Tuple[Type[BaseBlock], ...]] = None,
                 random_state: Optional[int] = None):
        super().__init__(executor_cls, max_iter, cv, make_metrics, growing_strategy, block_classes, random_state)
        if rf_params is None:
            rf_params = dict()
        self.rf_params = rf_params
        self.layer_width = layer_width

    def make_step_layer(self, step: int, iteration: int,
                        X: np.ndarray, y: np.ndarray,
                        validator: BasePipelineModelValidator,
                        rng: np.random.Generator):
        return self.LAYER_CLS(
            make_blocks=lambda: _make_base_df_blocks(self.rf_params, self.layer_width, self.block_classes),
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

    def __init__(self, executor_cls: Type[BaseExecutor],
                 rf_param_grid: Optional[dict] = None,
                 layer_width: int = 2,
                 n_steps: int = 3,
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 block_classes: Optional[Tuple[Type[BaseBlock], ...]] = None,
                 random_state: Optional[int] = None):
        super().__init__(executor_cls, max_iter, cv, make_metrics, growing_strategy, block_classes, random_state)
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
            make_blocks=lambda: _make_base_df_blocks(rf_params, self.layer_width, self.block_classes),
            layer_name=f'{self.LAYER_CLS.__name__}({step=}, {iteration=})',
            executor_cls=self.executor_cls,
            validator=validator,
            random_state=get_rand_int(rng)
        )


class MGSDeepForestConstructor(BaseAutoDeepForestConstructor):
    """Classical Multi-Grained Scanning Deep Forest.

    It consists of convolutional layers which reduce spatial dimensions (step 1),
    and classical stacking-based layers (step 2).

    Attributes:
        n_steps: Number of steps. The Classical Deep Forest has two steps.
        rf_params: Parameters of tree ensembles (Random Forests, Extra Trees).

    """
    STACKING_LAYER_CLS = NativeStackingLayer
    MGS_LAYER_CLS = MGSRFLayer
    n_steps = 2

    def __init__(self, executor_cls: Type[BaseExecutor],
                 input_shape: Tuple[int],
                 rf_params: Optional[dict] = None,
                 conv_params: Optional[dict] = None,
                 layer_width: int = 2,
                 max_iter: int = 10,
                 cv: Optional[int] = 5,
                 make_metrics: Optional[Callable[[], MetricsEvaluator]] = None,
                 growing_strategy: Optional[GrowingStrategy] = None,
                 block_classes: Optional[Tuple[Type[BaseBlock], ...]] = None,
                 random_state: Optional[int] = None):
        super().__init__(executor_cls, max_iter, cv, make_metrics, growing_strategy, block_classes, random_state)
        self.input_shape = input_shape
        if conv_params is None:
            conv_params = dict()
        self.conv_params = conv_params
        if rf_params is None:
            rf_params = dict()
        self.rf_params = rf_params
        self.layer_width = layer_width

    def make_step_layer(self, step: int, iteration: int,
                        X: np.ndarray, y: np.ndarray,
                        validator: BasePipelineModelValidator,
                        rng: np.random.Generator):
        if step == 0:  # MGS layers
            if iteration > 0:
                return  # only one MGS layer allowed
            return self.MGS_LAYER_CLS(
                input_shape=self.input_shape,
                rf_params=self.rf_params,
                layer_name=f'{self.MGS_LAYER_CLS.__name__}({step=}, {iteration=})',
                executor_cls=self.executor_cls,
                validator=validator,
                random_state=get_rand_int(rng),
                **self.conv_params
            )
        else:  # step == 1, stacking layers
            return self.STACKING_LAYER_CLS(
                make_blocks=lambda: _make_base_df_blocks(self.rf_params, self.layer_width, self.block_classes),
                layer_name=f'{self.STACKING_LAYER_CLS.__name__}({step=}, {iteration=})',
                executor_cls=self.executor_cls,
                validator=validator,
                random_state=get_rand_int(rng)
            )
