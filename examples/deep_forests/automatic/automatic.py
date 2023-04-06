from collections import defaultdict
from functools import singledispatchmethod
from bosk.block.base import BaseBlock, BaseInputBlock, BlockOutputData
from bosk.block.functional import FunctionalBlockWrapper
from bosk.block.slot import BlockGroup, BlockInputSlot, BlockOutputSlot
from bosk.data import BaseData, CPUData
from bosk.executor.base import BaseBlockExecutor, BaseExecutor, DefaultBlockExecutor, InputSlotToDataMapping
from bosk.executor.recursive import RecursiveExecutor
from bosk.pipeline.base import BasePipeline
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.executor.sklearn_interface import BoskPipelineClassifier
from bosk.pipeline.connection import Connection
from bosk.pipeline.dynamic import BaseDynamicPipeline
from bosk.stages import Stage
from bosk.utility import get_rand_int, get_random_generator
from bosk.visitor.base import BaseVisitor
from bosk.block.zoo.models.classification.classification_models import RFCBlock, ETCBlock
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Mapping, Type


ModificationAction = Literal['add'] | Literal['remove']


class ModifyGroupVisitor(BaseVisitor):
    """Visitor for block groups modification, given a group and action.
    """
    def __init__(self, action: ModificationAction, group: BlockGroup):
        self.action = action
        self.group = group

    @singledispatchmethod
    def visit(self, obj):
        pass  # ignore extra entities

    @visit.register
    def _(self, block: BaseBlock):
        if self.action == 'add':
            block.slots.groups.add(self.group)
        elif self.action == 'remove':
            block.slots.groups.remove(self.group)
        else:
            raise NotImplementedError()


MetricsResults = Mapping[str, float]


class MetricsEvaluator:
    def __init__(self, names: List[str]):
        self.names = set(names)
        self.results = defaultdict(list)

    def append_eval(self, y_true: np.ndarray, y_pred: np.ndarray):
        pred_labels = np.argmax(y_pred, axis=1)
        if 'roc_auc' in self.names:
            self.results['roc_auc'].append(roc_auc_score(y_true, y_pred, multi_class='ovr'))
        if 'f1' in self.names:
            self.results['f1'].append(f1_score(y_true, pred_labels, average='macro'))
        if 'accuracy' in self.names:
            self.results['accuracy'].append(accuracy_score(y_true, pred_labels))

    def average(self) -> MetricsResults:
        return {
            k: np.mean(v)
            for k, v in self.results.items()
        }


class PipelineModelValidator:
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
        for train_idx, val_idx in self.cv.split(X.data, y.data):
            train_data = {
                k: CPUData(v.data[train_idx])
                for k, v in data.items()
            }
            fitter(train_data)
            val_data = {
                k: CPUData(v.data[val_idx])
                for k, v in data.items()
            }
            preds_val = transformer({k: v for k, v in val_data if k != 'y'})[output].data
            metrics.append_eval(val_data['y'], preds_val)
        return metrics.average()


class DumbPipelineModelValidator(PipelineModelValidator):
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


class TrainSetPipelineModelValidator(PipelineModelValidator):
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


class Layer(ABC):
    def __init__(self, executor_cls: Type[BaseExecutor],
                 validator: PipelineModelValidator,
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


class MultiLayerPipelineBuilder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def append(self, layer: Layer):
        pass


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
            n_groups=5,
            n_ferns_in_group=5,
            fern_size=5,
            kind='binary',
            bootstrap=True,
            n_jobs=-1,
            random_state=get_rand_int(rng),
            kernel_size=4,
            stride=4,
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
        embedding_output = b.Output('X')(b.Stack(['pooled', 'proba'])(pooled=pooled, proba=proba))
        proba_output = b.Output('proba')(proba)
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
    inputs = ['X', 'X_original', 'y']

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
        x_original_ = b.Input('X_original')()
        y_ = b.TargetInput('y')()

        rf_ = b.RFC(random_state=get_rand_int(rng))(X=x_, y=y_)
        et_ = b.ETC(random_state=get_rand_int(rng))(X=x_, y=y_)
        embedding_ = b.Concat(['X_original', 'rf', 'et'], axis=1)(X_original=x_original_, rf=rf_, et=et_)
        embedding_output = b.Output('X')(embedding_)
        original_output = b.Output('X_original')(x_original_)

        proba_ = b.Average(axis=-1)(b.Stack(['rf', 'et'], axis=-1)(rf=rf_, et=et_))
        proba_output = b.Output('proba')(proba_)

        pipeline = b.build()
        pipeline.accept(ModifyGroupVisitor('add', BlockGroup(self.layer_name)))
        # evaluate the pipeline
        metric_values = self.calc_metrics(data, pipeline, 'proba', ['X'])

        # fit on the whole given data set
        if self.validator.need_refit:
            fitter = self.executor_cls(pipeline, Stage.FIT, outputs=['proba', 'X'])
            fitter(data)
        return pipeline, metric_values


class FitBlacklistBlockExecutor(DefaultBlockExecutor):
    """Block executor that does not call `fit` method for the blocks in the black list.

    It is used to be able to manually fit some blocks of the pipeline.
    """
    def __init__(self, blacklist: Sequence[BaseBlock]) -> None:
        super().__init__()
        self.blacklist = set(blacklist)

    def execute_block(self, stage: Stage, block: BaseBlock,
                        block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if block in self.blacklist:
            # avoid block fitting
            filtered_block_input_mapping = {
                slot.meta.name: values
                for slot, values in block_input_mapping.items()
                if slot.meta.stages.transform or (stage == Stage.FIT and slot.meta.stages.transform_on_fit)
            }
            return block.wrap(block.transform(filtered_block_input_mapping))
        return super().execute_block(stage, block, block_input_mapping)


class StackingLayer(Layer):
    inputs = ['X', 'X_original', 'y']

    def __init__(self, make_blocks: Callable[[], Sequence[BaseBlock]],
                 executor_cls: Type[BaseExecutor],
                 validator: PipelineModelValidator,
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
            inp_X, inp_y = inputs['X'].data, inputs['y'].data
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
        x_original_ = b.Input('X_original')()
        y_ = b.TargetInput('y')()

        blocks = self.make_blocks()
        block_wrappers = [
            b.wrap(block)(X=x_, y=y_)
            for block in blocks
        ]
        # set random states
        for bw in block_wrappers:
            bw.block.set_random_state(get_rand_int(rng))
        block_names = [f'block_{i}' for i in range(len(block_wrappers))]
        named_block_wrappers = dict(zip(block_names, block_wrappers))

        embedding_ = b.Concat(['X_original', *block_names], axis=1)(X_original=x_original_, **named_block_wrappers)
        embedding_output = b.Output('X')(embedding_)
        original_output = b.Output('X_original')(x_original_)

        proba_ = b.Average(axis=-1)(b.Stack(block_names, axis=-1)(**named_block_wrappers))
        proba_output = b.Output('proba')(proba_)

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


class GrowingStrategy(ABC):
    @abstractmethod
    def need_grow(self, pipeline, metrics, executor_cls, growing_state: dict) -> bool:
        ...

    def trim(self, pipelines: Sequence[BasePipeline]):
        return pipelines


class EarlyStopping(GrowingStrategy):
    def __init__(self, mode: Literal['any'] | Literal['all'] = 'all', patience: int = 1):
        self.mode = mode
        self.patience = patience

    def need_grow(self, pipeline, metrics, executor_cls, growing_state: dict) -> bool:
        assert metrics is not None, 'Please, calculate CV metrics first'
        if 'metrics' not in growing_state:
            growing_state['metrics'] = metrics
            growing_state['count'] = 0
            return True
        MODES = {
            'any': any,
            'all': all
        }
        prev_metrics = growing_state['metrics']
        # check if new metrics are lower than previous
        new_metrics_worse = MODES[self.mode](
            metrics[metric_name] <= prev_metrics[metric_name]
            for metric_name in metrics.keys()
        )
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
            return pipelines[:-count]
        return pipelines


class SequencePipelineMaker(MultiLayerPipelineBuilder):
    def __init__(self, executor_cls: Type[BaseExecutor],
                 growing_strategy: Optional[GrowingStrategy] = None,
                 **inputs: Mapping[str, BaseData]):
        super().__init__()
        self.executor_cls = executor_cls
        self.inputs = inputs
        self.prev_step_inputs = None
        self.pipelines = []
        self.growing_strategy = growing_strategy
        self.__growing_state = dict()
        self.finished = False

    def append(self, layer: Layer) -> bool:
        if self.finished:
            return False
        if self.prev_step_inputs is None:
            step_inputs = self.inputs
        else:
            prev_pipeline = self.pipelines[-1]
            available_outputs = [k for k in layer.inputs if k in prev_pipeline.outputs]
            prev_transformer = self.executor_cls(prev_pipeline, Stage.TRANSFORM, outputs=available_outputs)
            step_inputs = prev_transformer(self.prev_step_inputs)
        # append static inputs (e.g. 'y')
        step_inputs = {**step_inputs}
        step_inputs.update({
            k: self.inputs[k]
            for k in layer.inputs
            if k not in step_inputs
        })
        pipeline, metrics = layer.fit(step_inputs)
        self.pipelines.append(pipeline)
        self.prev_step_inputs = step_inputs
        if not self.growing_strategy.need_grow(pipeline, metrics, self.executor_cls, self.__growing_state):
            self.pipelines = self.growing_strategy.trim(self.pipelines, self.__growing_state)
            self.finished = True
            return False
        return True

    def build(self):
        nodes = []
        connections = []
        inputs = self.pipelines[0].inputs
        outputs = self.pipelines[-1].outputs

        for i, pipeline in enumerate(self.pipelines):
            nodes.extend(pipeline.nodes)
            connections.extend(pipeline.connections)
            if i != 0:
                prev_pipeline = self.pipelines[i - 1]
                for out_name, out_slot in prev_pipeline.outputs.items():
                    if out_name in pipeline.inputs:
                        # matching slots by names
                        connections.append(Connection(out_slot, pipeline.inputs[out_name]))

        return BasePipeline(
            nodes=nodes,
            connections=connections,
            inputs=inputs,
            outputs=outputs,
        )


def make_fit_model(X: np.ndarray, y: np.ndarray):
    X = CPUData(X)
    y = CPUData(y)

    executor_cls = RecursiveExecutor
    # validator = PipelineModelValidator(
    # validator = DumbPipelineModelValidator(
    validator = TrainSetPipelineModelValidator(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=12345),
        lambda: MetricsEvaluator(['f1', 'roc_auc']),
    )
    # pipeline, metrics = MGSLayer(
    #     input_shape=(1, 8, 8),
    #     executor_cls=executor_cls,
    #     validator=validator
    # ).fit(X, y)
    # print(metrics)

    # pipeline, metrics = ForestsLayer(
    #     executor_cls=executor_cls,
    #     validator=validator
    # ).fit(X, y)
    # print(metrics)

    # pipeline, metrics = StackingLayer(
    #     executor_cls=executor_cls,
    #     validator=validator,
    #     random_state=12345
    # ).fit(X, y)
    # print(metrics)



    rng = get_random_generator(12345)
    maker = SequencePipelineMaker(
        executor_cls,
        growing_strategy=EarlyStopping(),
        X=X,
        X_original=X,
        y=y
    )
    maker.append(ForestsLayer(
        executor_cls=executor_cls,
        validator=validator,
        random_state=get_rand_int(rng)
    ))
    maker.append(StackingLayer(
        make_blocks=lambda: [RFCBlock(), ETCBlock(), RFCBlock(), ETCBlock()],
        executor_cls=executor_cls,
        validator=validator,
        random_state=get_rand_int(rng)
    ))
    maker.append(StackingLayer(
        make_blocks=lambda: [RFCBlock(), ETCBlock(), RFCBlock(), ETCBlock()],
        executor_cls=executor_cls,
        validator=validator,
        random_state=get_rand_int(rng)
    ))
    maker.append(StackingLayer(
        make_blocks=lambda: [RFCBlock(), ETCBlock(), RFCBlock(), ETCBlock()],
        executor_cls=executor_cls,
        validator=validator,
        random_state=get_rand_int(rng)
    ))

    pipeline = maker.build()

    # make a scikit-learn model
    model = BoskPipelineClassifier(pipeline, executor_cls=RecursiveExecutor)
    model._classifier_init(y.data)
    return model


def main():
    all_X, all_y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, random_state=12345)
    # fit the model
    model = make_fit_model(X_train, y_train)
    # predict with the model
    test_preds = model.predict(X_test)
    print('Test f1 score:', f1_score(y_test, test_preds, average='macro'))


if __name__ == '__main__':
    main()
