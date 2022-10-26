"""Example of simple Adaptive Weighted Deep Forest definition.

"""
import numpy as np
from typing import Callable, List, Optional
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from bosk.block import auto_block, BaseBlock, BlockInputData, TransformOutputData
from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.naive import NaiveExecutor, LessNaiveExecutor
from bosk.stages import Stage, Stages
from bosk.block import BlockMeta, BlockInputSlot, BlockOutputSlot
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



@auto_block
class RFCBlock(RandomForestClassifier):
    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class ETCBlock(ExtraTreesClassifier):
    def transform(self, X):
        return self.predict_proba(X)


# def make_simple_meta(input_names: List[str], output_names: List[str]):
#     return BlockMeta(
#         inputs=[
#             BlockInputSlot(name=name)
#             for name in input_names
#         ],
#         outputs=[
#             BlockOutputSlot(name=name)
#             for name in output_names
#         ]
#     )


class ConcatBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.axis = axis
        self.ordered_input_names = None
        self.meta = self.make_simple_meta(input_names, ['output'])

    def fit(self, inputs: BlockInputData) -> 'ConcatBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = tuple(
            inputs[name]
            for name in self.ordered_input_names
        )
        concatenated = np.concatenate(ordered_inputs, axis=self.axis)
        return {'output': concatenated}


class StackBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.axis = axis
        self.ordered_input_names = None
        self.meta = self.make_simple_meta(input_names, ['output'])

    def fit(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = tuple(
            inputs[name]
            for name in self.ordered_input_names
        )
        stacked = np.stack(ordered_inputs, axis=self.axis)
        return {'output': stacked}


class AverageBlock(BaseBlock):
    meta = None

    def __init__(self, axis: int = -1):
        self.axis = axis
        self.meta = self.make_simple_meta(['X'], ['output'])

    def fit(self, _inputs: BlockInputData) -> 'AverageBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        averaged = inputs['X'].mean(axis=self.axis)
        return {'output': averaged}


class ArgmaxBlock(BaseBlock):
    meta = None

    def __init__(self, axis: int = -1):
        self.axis = axis
        self.meta = self.make_simple_meta(['X'], ['output'])

    def fit(self, _inputs: BlockInputData) -> 'ArgmaxBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        ids = inputs['X'].argmax(axis=self.axis)
        return {'output': ids}


class InputBlock(BaseBlock):
    meta = None

    def __init__(self):
        self.meta = self.make_simple_meta(['X'], ['X'])

    def fit(self, _inputs: BlockInputData) -> 'InputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs


class TargetInputBlock(BaseBlock):
    meta = None

    def __init__(self):
        TARGET_NAME = 'y'
        self.meta = BlockMeta(
            inputs=[
                BlockInputSlot(
                    name=TARGET_NAME,
                    stages=Stages(transform=False, transform_on_fit=True),
                    parent_block=self
                )
            ],
            outputs=[
                BlockOutputSlot(
                    name=TARGET_NAME,
                    parent_block=self
                )
            ]
        )

    def fit(self, _inputs: BlockInputData) -> 'TargetInputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs


class RocAucBlock(BaseBlock):
    meta = None

    def __init__(self):
        self.meta = BlockMeta(
            inputs=[
                BlockInputSlot(
                    name='pred_probas',
                    stages=Stages(transform=False, transform_on_fit=True),
                    parent_block=self
                ),
                BlockInputSlot(
                    name='gt_y',
                    stages=Stages(transform=False, transform_on_fit=True),
                    parent_block=self
                )
            ],
            outputs=[
                BlockOutputSlot(
                    name='roc-auc',
                    parent_block=self
                )
            ]
        )

    def fit(self, _inputs: BlockInputData) -> 'InputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'roc-auc': roc_auc_score(inputs['gt_y'], inputs['pred_probas'][:, 1])
        }


def make_deep_forest():
    input_x = InputBlock()
    input_y = TargetInputBlock()
    rf_1 = RFCBlock()
    et_1 = ETCBlock()
    concat_1 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_2 = RFCBlock()
    et_2 = ETCBlock()
    concat_2 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_3 = RFCBlock()
    et_3 = ETCBlock()
    stack_3 = StackBlock(['X_0', 'X_1'], axis=1)
    average_3 = AverageBlock(axis=1)
    argmax_3 = ArgmaxBlock(axis=1)
    pipeline = BasePipeline(
        nodes=[
            input_x,
            input_y,
            rf_1,
            et_1,
            concat_1,
            rf_2,
            et_2,
            concat_2,
            rf_3,
            et_3,
            stack_3,
            average_3,
            argmax_3,
        ],
        connections=[
            # input X
            Connection(input_x.meta.outputs['X'], rf_1.meta.inputs['X']),
            Connection(input_x.meta.outputs['X'], et_1.meta.inputs['X']),
            # input y
            Connection(input_y.meta.outputs['y'], rf_1.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], et_1.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], rf_2.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], et_2.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], rf_3.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], et_3.meta.inputs['y']),
            # layers connection
            Connection(rf_1.meta.outputs['output'], concat_1.meta.inputs['X_0']),
            Connection(et_1.meta.outputs['output'], concat_1.meta.inputs['X_1']),
            Connection(concat_1.meta.outputs['output'], rf_2.meta.inputs['X']),
            Connection(concat_1.meta.outputs['output'], et_2.meta.inputs['X']),
            Connection(rf_2.meta.outputs['output'], concat_2.meta.inputs['X_0']),
            Connection(et_2.meta.outputs['output'], concat_2.meta.inputs['X_1']),
            Connection(concat_2.meta.outputs['output'], rf_3.meta.inputs['X']),
            Connection(concat_2.meta.outputs['output'], et_3.meta.inputs['X']),
            Connection(rf_3.meta.outputs['output'], stack_3.meta.inputs['X_0']),
            Connection(et_3.meta.outputs['output'], stack_3.meta.inputs['X_1']),
            Connection(stack_3.meta.outputs['output'], average_3.meta.inputs['X']),
            Connection(average_3.meta.outputs['output'], argmax_3.meta.inputs['X']),
        ]
    )

    fit_executor = NaiveExecutor(
        pipeline,
        stage=Stage.FIT,
        inputs={
            'X': input_x.meta.inputs['X'],
            'y': input_y.meta.inputs['y'],
        },
        outputs={'probas': average_3.meta.outputs['output']},
    )
    transform_executor = NaiveExecutor(
        pipeline,
        stage=Stage.TRANSFORM,
        inputs={'X': input_x.meta.inputs['X']},
        outputs={
            'probas': average_3.meta.outputs['output'],
            'labels': argmax_3.meta.outputs['output'],
        },
    )
    return pipeline, fit_executor, transform_executor


class FunctionalBlockWrapper:
    def __init__(self, block: BaseBlock, output_name: Optional[str] = None):
        self.block = block
        self.output_name = output_name

    def get_input_slot(self, slot_name: Optional[str] = None):
        if slot_name is None:
            if len(self.block.meta.inputs) == 1:
                return list(self.block.meta.inputs.values())[0]
            else:
                raise RuntimeError('Block has more than one input (please, specify it)')
        return self.block.meta.inputs[slot_name]

    def get_output_slot(self) -> BlockOutputSlot:
        if self.output_name is None:
            if len(self.block.meta.outputs) == 1:
                return list(self.block.meta.outputs.values())[0]
            else:
                raise RuntimeError('Block has more than one output')
        return self.block.meta.outputs[self.output_name]

    def __getitem__(self, output_name: str):
        return FunctionalBlockWrapper(self.block, output_name=output_name)


class FunctionalBuilder:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def __getattr__(self, name: str) -> Callable:
        block_name = name + 'Block'
        block_cls = globals().get(block_name, None)
        if block_cls is None:
            raise ValueError(f'Wrong block class: {name} ({block_name} not found)')
        return self._get_block_init(block_cls)

    def _get_block_init(self, block_cls: Callable) -> Callable:
        def block_init(*args, **kwargs):
            block = block_cls(*args, **kwargs)
            self.nodes.append(block)

            def placeholder_fn(*pfn_args, **pfn_kwargs):
                assert len(pfn_args) == 0, "Only kwargs are supported"
                for input_name, input_block_wrapper in pfn_kwargs.items():
                    self.connections.append(
                        Connection(
                            src=input_block_wrapper.get_output_slot(),
                            dst=block.meta.inputs[input_name],
                        )
                    )
                return FunctionalBlockWrapper(block)

            return placeholder_fn

        return block_init

    def new(self, block_cls: Callable, *args, **kwargs) -> Callable:
        return self._get_block_init(block_cls)(*args, **kwargs)

    @property
    def pipeline(self) -> BasePipeline:
        return BasePipeline(self.nodes, self.connections)


def make_deep_forest_functional():
    b = FunctionalBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC()(X=X, y=y)
    et_1 = b.ETC()(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC()(X=concat_1, y=y)
    et_2 = b.ETC()(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC()(X=concat_2, y=y)
    et_3 = b.ETC()(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=stack_3)
    argmax_3 = b.Argmax(axis=1)(X=average_3)

    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = NaiveExecutor(
        b.pipeline,
        stage=Stage.FIT,
        inputs={
            'X': X.get_input_slot(),
            'y': y.get_input_slot(),
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'rf_1_roc-auc': rf_1_roc_auc.get_output_slot(),
            'roc-auc': roc_auc.get_output_slot(),
        },
    )
    transform_executor = NaiveExecutor(
        b.pipeline,
        stage=Stage.TRANSFORM,
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
    )
    return b.pipeline, fit_executor, transform_executor


class CSBlock(BaseBlock):
    meta = None

    def __init__(self, eps: float = 1.0):
        self.eps = eps
        self.meta = self.make_simple_meta(['X'], ['mask', 'best'])

    def fit(self, inputs: BlockInputData) -> 'CSBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        X = inputs['X']
        best_mask = X.max(axis=1) > self.eps
        best = X[best_mask]
        return {
            'mask': ~best_mask,
            'best': best,
        }


class CSFilterBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str]):
        output_names = input_names
        self.input_names = input_names
        self.meta = self.make_simple_meta(input_names + ['mask'], output_names)

    def fit(self, inputs: BlockInputData) -> 'CSFilterBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        mask = inputs['mask']
        return {
            name: inputs[name][mask]
            for name in self.input_names
        }


class CSJoinBlock(BaseBlock):
    meta = None

    def __init__(self):
        self.meta = self.make_simple_meta(['best', 'refined', 'mask'], ['output'])

    def fit(self, inputs: BlockInputData) -> 'CSJoinBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        best = inputs['best']
        refined = inputs['refined']
        mask = inputs['mask']
        n_samples = mask.shape[0]
        rest_dims = best.shape[1:]
        result = np.empty((n_samples, *rest_dims), dtype=best.dtype)
        result[~mask] = best
        result[mask] = refined
        return {'output': result}


def make_deep_forest_layer(b, **inputs):
    rf = b.RFC()(**inputs)
    et = b.ETC()(**inputs)
    stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
    average = b.Average(axis=1)(X=stack)
    return average


def make_deep_forest_functional_confidence_screening():
    b = FunctionalBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC()(X=X, y=y)
    et_1 = b.ETC()(X=X, y=y)
    concat_1 = b.Concat(['rf_1', 'et_1'])(rf_1=rf_1, et_1=et_1)
    stack_1 = b.Stack(['rf_1', 'et_1'], axis=1)(rf_1=rf_1, et_1=et_1)
    average_1 = b.Average(axis=1)(X=stack_1)

    # get confidence screening mask
    cs_1 = b.CS(eps=0.95)(X=average_1)
    # filter X and concatenated predictions samples by CS
    filtered_1 = b.CSFilter(['concat_1', 'X'])(
        concat_1=concat_1,
        X=X,
        mask=cs_1['mask']
    )
    # y should be filtered separately since it is not used at the Transform stage
    filtered_1_y = b.CSFilter(['y'])(y=y, mask=cs_1['mask'])
    concat_all_1 = b.Concat(['filtered_1_X', 'filtered_concat_1'])(
        filtered_1_X=filtered_1['X'],
        filtered_concat_1=filtered_1['concat_1']
    )

    average_2 = make_deep_forest_layer(b, X=concat_all_1, y=filtered_1_y)
    concat_2 = b.Concat(['X', 'average_2'])(X=filtered_1['X'], average_2=average_2)

    # apply sample weighting (see An Adaptive Weighted Deep Forest)
    @auto_block
    class WeightsBlock:
        def __init__(self, ord: int = 1):
            self.ord = ord

        def fit(self, X, y) -> 'WeightsBlock':
            weights = 1 - (np.take_along_axis(X, y[:, np.newaxis], axis=1)) ** self.ord
            self.weights_ = weights.reshape((-1,))
            return self

        def transform(self, X) -> 'np.ndarray':
            return self.weights_

    sample_weight_2 = b.new(WeightsBlock, ord=2)(X=average_2, y=filtered_1_y)

    average_3 = make_deep_forest_layer(b, X=concat_2, y=filtered_1_y, sample_weight=sample_weight_2)

    # join confident samples with screened out ones
    joined_3 = b.CSJoin()(
        best=cs_1['best'],
        refined=average_3,
        mask=cs_1['mask']
    )

    argmax_3 = b.Argmax(axis=1)(X=joined_3)

    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=joined_3)

    fit_executor = LessNaiveExecutor(
        b.pipeline,
        stage=Stage.FIT,
        inputs={
            'X': X.get_input_slot(),
            'y': y.get_input_slot(),
        },
        outputs={
            'probas': joined_3.get_output_slot(),
            'rf_1_roc-auc': rf_1_roc_auc.get_output_slot(),
            'roc-auc': roc_auc.get_output_slot(),
        },
    )
    transform_executor = LessNaiveExecutor(
        b.pipeline,
        stage=Stage.TRANSFORM,
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': joined_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
    )
    return b.pipeline, fit_executor, transform_executor


def my_graph():
    b = FunctionalBuilder()
    x, y = b.Input()(), b.TargetInput()()
    z = b.Input()()
    fake_rf = b.RFC()(X=z, y=z)
    rf_1 = b.RFC()(X=x, y=y)
    concat_1 = b.Concat(['x', 'rf_1'])(x=x, rf_1=rf_1)
    rf_2 = b.RFC()(X=concat_1, y=y)
    fake_rf_2 = b.RFC()(X=concat_1, y=y)
    rf_3 = b.RFC()(X=x, y=y)
    stack = b.Stack(['rf_2', 'rf_3'], axis=1)(rf_2=rf_2, rf_3=rf_3)
    avg = b.Average()(X=stack)
    fit_executor = LessNaiveExecutor(
        b.pipeline,
        stage=Stage.FIT,
        inputs={
            'x': x.get_input_slot(),
            'y': y.get_input_slot(),
        },
        outputs={
            'result': avg.get_output_slot(),
        },
    )
    transform_executor = LessNaiveExecutor(
        b.pipeline,
        stage=Stage.TRANSFORM,
        inputs={
            'x': x.get_input_slot(),
        },
        outputs={
            'result': avg.get_output_slot(),
        },
    )
    return fit_executor, transform_executor

def check_my_graph():
    fit_executor, transform_executor = my_graph()
    train_X, train_y = make_moons(noise=0.5)
    print('fit topological order:')
    fit_executor({'x': train_X, 'y': train_y})
    print("Fit successful")
    print('transform topological order:')
    transform_executor({'x': train_X})

def paint_my_graph():
    fit_executor, _ = my_graph()
    fit_executor.draw('Mygraph')

def paint_cs_graph():
    pipeline, fit_executor, transform_executor = make_deep_forest_functional_confidence_screening()
    fit_executor.draw('Csgraph.pdf')


def main():
    # _pipeline, fit_executor, transform_executor = make_deep_forest()
    # _pipeline, fit_executor, transform_executor = make_deep_forest_functional()
    _pipeline, fit_executor, transform_executor = make_deep_forest_functional_confidence_screening()

    all_X, all_y = make_moons(noise=0.5)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2)
    fit_result = fit_executor({'X': train_X, 'y': train_y})
    print("Fit successful")
    train_result = transform_executor({'X': train_X})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'], train_result['probas']))
    test_result = transform_executor({'X': test_X})
    print(train_result.keys())
    print("Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'][:, 1]))
    print(
        "Train ROC-AUC calculated by fit_executor:",
        fit_result['roc-auc']
    )
    print(
        "Train ROC-AUC for RF_1:",
        fit_result['rf_1_roc-auc']
    )
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'][:, 1]))


if __name__ == "__main__":
    # from time import time
    # np.random.seed(15)
    # time_start = time()
    # check_my_graph()
    # print('time elapded:', round(time() - time_start, 3), ' seconds')

    # paint_my_graph()
    paint_cs_graph()

