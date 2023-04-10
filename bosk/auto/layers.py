import numpy as np
from sklearn.model_selection import StratifiedKFold
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple, Mapping, Type

from ..stages import Stage
from ..utility import get_rand_int, get_random_generator
from ..visitor.group import ModifyGroupVisitor
from ..pipeline.builder.functional import FunctionalPipelineBuilder
from ..block.zoo.models.classification.classification_models import RFCBlock, ETCBlock
from ..block.zoo.routing.cv import CVTrainIndicesBlock, SubsetTrainWrapperBlock
from ..block.base import BaseBlock, BlockOutputData
from ..block.slot import BlockGroup
from ..data import BaseData, CPUData
from ..executor.base import BaseExecutor, DefaultBlockExecutor, InputSlotToDataMapping
from ..executor.block import FitBlacklistBlockExecutor
from ..pipeline.base import BasePipeline
from .validation import CVPipelineModelValidator
from .metrics import MetricsResults


class Layer(ABC):
    def __init__(self, executor_cls: Type[BaseExecutor],
                 validator: CVPipelineModelValidator,
                 layer_name: str = 'forests_layer',
                 random_state: Optional[int] = None):
        self.executor_cls = executor_cls
        self.validator = validator
        self.layer_name = layer_name
        self.random_state = random_state

    @property
    @abstractmethod
    def inputs(self):
        ...

    @abstractmethod
    def fit(self, inputs: Mapping[str, BaseData]) -> Tuple[BasePipeline, MetricsResults]:
        ...

    def calc_metrics(self, data: Mapping[str, CPUData], pipeline: BasePipeline,
                       output: str, fit_outputs: List[str]) -> Mapping[str, float]:
        fitter = self.executor_cls(pipeline, Stage.FIT, outputs=[output, *fit_outputs])
        transformer = self.executor_cls(pipeline, Stage.TRANSFORM)
        return self.validator.calc_metrics(data, fitter, transformer, output)


class MGSLayer(Layer):
    inputs = ['X', 'y']

    def __init__(self, input_shape: Tuple[int], **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape

    def fit(self, data: Mapping[str, BaseData]):
        rng = get_random_generator(self.random_state)
        b = FunctionalPipelineBuilder()
        x_ = b.Input('X')()
        y_ = b.Input('y')()
        reshaped_ = b.Reshape((-1, *self.input_shape))(x_)
        ms = b.MGSRandomFerns(
            n_groups=20,
            n_ferns_in_group=5,
            fern_size=15,
            kind='unary',
            bootstrap=False,
            n_jobs=-1,
            random_state=get_rand_int(rng),
            kernel_size=3,
            stride=3,
            dilation=1,
            padding=None
        )(X=reshaped_, y=y_)
        pooled = b.Pooling(
            kernel_size=2,
            stride=None,
            dilation=1,
            aggregation='max',
        )(X=ms)
        pooled = b.Flatten()(X=pooled)
        proba = b.RandomFerns(
            n_groups=5,
            n_ferns_in_group=5,
            fern_size=5,
            kind='unary',
            bootstrap=True,
            n_jobs=-1,
            random_state=get_rand_int(rng),
        )(X=pooled, y=y_)
        embedding_output = b.Output('X')(pooled)
        proba_output = b.Output('proba')(proba)  # used only for validation
        b.Output('X_original')(embedding_output)
        pipeline = b.build()
        pipeline.accept(ModifyGroupVisitor('add', BlockGroup(self.layer_name)))
        # evaluate the pipeline
        metric_values = self.calc_metrics(data, pipeline, 'proba', ['X'])

        # fit on the whole given data set
        if self.validator.need_refit:
            fitter = self.executor_cls(pipeline, Stage.FIT, outputs=['proba'])
            fitter(data)
        return pipeline, metric_values


class MGSRFLayer(Layer):
    inputs = ['X', 'y']

    def __init__(self, input_shape: Tuple[int], **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape

    def fit(self, data: Mapping[str, BaseData]):
        rng = get_random_generator(self.random_state)
        b = FunctionalPipelineBuilder()
        x_ = b.Input('X')()
        y_ = b.Input('y')()
        reshaped_ = b.Reshape((-1, *self.input_shape))(x_)
        ms = b.MultiGrainedScanningND(
            model=ETCBlock(random_state=get_rand_int(rng)),
            kernel_size=4,
            stride=2,
            dilation=1,
            padding=None
        )(X=reshaped_, y=y_)
        pooled = b.Pooling(
            kernel_size=2,
            stride=None,
            dilation=1,
            aggregation='max',
        )(X=ms)
        pooled = b.Flatten()(X=pooled)
        proba = b.ETC(random_state=get_rand_int(rng))(X=pooled, y=y_)
        embedding_output = b.Output('X')(pooled)

        b.Output('embedding')(embedding_output)
        b.Output('proba')(proba)  # used only for validation

        pipeline = b.build()
        pipeline.accept(ModifyGroupVisitor('add', BlockGroup(self.layer_name)))
        # evaluate the pipeline
        metric_values = self.calc_metrics(data, pipeline, 'proba', ['X'])

        # fit on the whole given data set
        if self.validator.need_refit:
            fitter = self.executor_cls(pipeline, Stage.FIT, outputs=['proba'])
            fitter(data)
        return pipeline, metric_values


class ForestsLayer(Layer):
    inputs = ['X', 'embedding', 'y']

    def __init__(self, make_blocks: Callable[[], Sequence[BaseBlock]],
                 executor_cls: Type[BaseExecutor],
                 validator: CVPipelineModelValidator,
                 layer_name: str = 'forests_layer',
                 random_state: Optional[int] = None):
        super().__init__(
            executor_cls=executor_cls,
            validator=validator,
            layer_name=layer_name,
            random_state=random_state,
        )
        self.make_blocks = make_blocks


    def fit(self, data: Mapping[str, BaseData]):
        rng = get_random_generator(self.random_state)
        # forests layer leverages CPU algorithms
        data = {
            k: v.to_cpu()
            for k, v in data.items()
        }

        # build a fixed pipeline
        b = FunctionalPipelineBuilder()
        x_ = b.Input('X')()
        if 'embedding' not in data:
            embedding_ = x_
        else:
            embedding_ = b.Input('embedding')()
        y_ = b.TargetInput('y')()

        blocks = self.make_blocks()
        block_wrappers = [
            b.wrap(block)(X=embedding_, y=y_)
            for block in blocks
        ]
        # set random states
        for bw in block_wrappers:
            bw.block.set_random_state(get_rand_int(rng))
        block_names = [f'block_{i}' for i in range(len(block_wrappers))]
        named_block_wrappers = dict(zip(block_names, block_wrappers))

        # concatenate embeddings with original feature vector
        new_embedding_ = b.Concat(['X', *block_names], axis=1)(X=x_, **named_block_wrappers)
        proba_ = b.Average(axis=-1)(b.Stack(block_names, axis=-1)(**named_block_wrappers))

        b.Output('embedding')(new_embedding_)
        b.Output('proba')(proba_)

        pipeline = b.build()
        pipeline.accept(ModifyGroupVisitor('add', BlockGroup(self.layer_name)))
        # evaluate the pipeline
        metric_values = self.calc_metrics(data, pipeline, 'proba', ['embedding'])

        # fit on the whole given data set
        if self.validator.need_refit:
            fitter = self.executor_cls(pipeline, Stage.FIT, outputs=['proba', 'embedding'])
            fitter(data)
        return pipeline, metric_values


class StackingLayer(Layer):
    inputs = ['X', 'embedding', 'y']

    def __init__(self, make_blocks: Callable[[], Sequence[BaseBlock]],
                 executor_cls: Type[BaseExecutor],
                 validator: CVPipelineModelValidator,
                 layer_name: str = 'stacking_layer',
                 random_state: Optional[int] = None):
        super().__init__(
            executor_cls=executor_cls,
            validator=validator,
            layer_name=layer_name,
            random_state=random_state,
        )
        self.make_blocks = make_blocks

    def __custom_cross_validate(self, data: Mapping[str, BaseData], pipeline: BasePipeline,
                       blocks: List[BaseBlock], rng: np.random.RandomState) -> Mapping[str, float]:
        fitter = self._make_fitter(pipeline, blocks, rng)
        transformer = self.executor_cls(pipeline, Stage.TRANSFORM)
        return self.validator.calc_metrics(data, fitter, transformer, 'proba')

    def _make_fitter(self, pipeline: BasePipeline, blocks: List[BaseBlock], rng: np.random.RandomState):
        new_rng = get_random_generator(get_rand_int(rng))
        basic_fitter = self.executor_cls(pipeline, Stage.FIT, block_executor=FitBlacklistBlockExecutor(blocks))
        def _fit_fn(inputs: Mapping[str, CPUData]):
            kfold = StratifiedKFold(n_splits=len(blocks), shuffle=True, random_state=get_rand_int(new_rng))
            inp_X = inputs['X'].data
            if 'embedding' in inputs:
                inp_X = inputs['embedding'].data
            inp_y = inputs['y'].data
            split_gen = kfold.split(inp_X, inp_y)
            for i, (block_train_idx, _rest_idx) in enumerate(split_gen):
                blocks[i].fit({'X': inp_X[block_train_idx], 'y': inp_y[block_train_idx]})
            basic_fitter(inputs)
            return None
        return _fit_fn

    def fit(self, data: Mapping[str, BaseData]):
        rng = get_random_generator(self.random_state)
        # forests layer leverages CPU algorithms
        data = {
            k: v.to_cpu()
            for k, v in data.items()
        }

        # build a fixed pipeline
        b = FunctionalPipelineBuilder()
        x_ = b.Input('X')()
        if 'embedding' not in data:
            embedding_ = x_
        else:
            embedding_ = b.Input('embedding')()
        y_ = b.TargetInput('y')()

        blocks = self.make_blocks()
        block_wrappers = [
            b.wrap(block)(X=embedding_, y=y_)
            for block in blocks
        ]
        # set random states
        for bw in block_wrappers:
            bw.block.set_random_state(get_rand_int(rng))
        block_names = [f'block_{i}' for i in range(len(block_wrappers))]
        named_block_wrappers = dict(zip(block_names, block_wrappers))
        # concatenate embeddings with original feature vector
        embedding_ = b.Concat(['X', *block_names], axis=1)(X=x_, **named_block_wrappers)
        proba_ = b.Average(axis=-1)(b.Stack(block_names, axis=-1)(**named_block_wrappers))

        # specify outputs
        b.Output('embedding')(embedding_)
        b.Output('proba')(proba_)

        pipeline = b.build()
        pipeline.accept(ModifyGroupVisitor('add', BlockGroup(self.layer_name)))
        # evaluate the pipeline
        blocks = [w.block for w in block_wrappers]
        metric_values = self.__custom_cross_validate(data, pipeline, blocks, rng)

        # fit on the whole given data set
        if self.validator.need_refit:
            fitter = self._make_fitter(pipeline, blocks, rng)
            fitter(data)
        return pipeline, metric_values


class NativeStackingLayer(Layer):
    """Native Stacking Layer implements stacking (training different models on subsets of data)
    using native BOSK blocks, instead of direct fitting.

    The pipeline made by this layer can be retrained.
    It gives reproducible results, i.e. it is guaranteed
    that the results will not be changed after retraining on the same data.

    """

    inputs = ['X', 'embedding', 'y']

    def __init__(self, make_blocks: Callable[[], Sequence[BaseBlock]],
                 executor_cls: Type[BaseExecutor],
                 validator: CVPipelineModelValidator,
                 layer_name: str = 'stacking_layer',
                 random_state: Optional[int] = None):
        super().__init__(
            executor_cls=executor_cls,
            validator=validator,
            layer_name=layer_name,
            random_state=random_state,
        )
        self.make_blocks = make_blocks

    def fit(self, data: Mapping[str, BaseData]):
        rng = get_random_generator(self.random_state)
        # forests layer leverages CPU algorithms
        data = {
            k: v.to_cpu()
            for k, v in data.items()
        }

        # build a fixed pipeline
        b = FunctionalPipelineBuilder()
        x_ = b.Input('X')()  # original X
        if 'embedding' not in data:
            embedding_ = x_
        else:
            embedding_ = b.Input('embedding')()
        y_ = b.TargetInput('y')()

        blocks = [
            SubsetTrainWrapperBlock(b)
            for b in self.make_blocks()
        ]
        train_indices_ = b.new(
            CVTrainIndicesBlock,
            size=len(blocks),
            random_state=get_rand_int(rng)
        )(X=embedding_, y=y_)
        block_wrappers = [
            b.wrap(block)(X=embedding_, y=y_, training_indices=train_indices_[str(i)])
            for i, block in enumerate(blocks)
        ]
        # set random states
        for bw in block_wrappers:
            bw.block.set_random_state(get_rand_int(rng))
        block_names = [f'block_{i}' for i in range(len(block_wrappers))]
        named_block_wrappers = dict(zip(block_names, block_wrappers))
        # concatenate embeddings with original feature vector
        new_embedding_ = b.Concat(['X', *block_names], axis=1)(X=x_, **named_block_wrappers)
        proba_ = b.Average(axis=-1)(b.Stack(block_names, axis=-1)(**named_block_wrappers))

        # specify outputs
        b.Output('embedding')(new_embedding_)
        b.Output('proba')(proba_)

        pipeline = b.build()
        pipeline.accept(ModifyGroupVisitor('add', BlockGroup(self.layer_name)))
        # evaluate the pipeline
        metric_values = self.calc_metrics(data, pipeline, 'proba', ['embedding'])

        # fit on the whole given data set
        if self.validator.need_refit:
            fitter = self.executor_cls(pipeline, Stage.FIT, outputs=['proba', 'embedding'])
            fitter(data)
        return pipeline, metric_values
