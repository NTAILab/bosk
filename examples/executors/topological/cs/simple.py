"""Example of simple Confidence Screening Deep Forest definition.

"""
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.block import BaseBlock
from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.naive import NaiveExecutor
from bosk.executor.topological import TopologicalExecutor
from bosk.stages import Stage
from bosk.slot import BlockOutputSlot
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock, AccuracyBlock, F1ScoreBlock
from bosk.block.zoo.routing import CSBlock, CSJoinBlock, CSFilterBlock


def make_deep_forest(executor, **kwargs):
    input_x = InputBlock()
    input_y = TargetInputBlock()
    rf_1 = RFCBlock(random_state=42)
    et_1 = ETCBlock(random_state=42)
    concat_1 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_2 = RFCBlock(random_state=42)
    et_2 = ETCBlock(random_state=42)
    concat_2 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_3 = RFCBlock(random_state=42)
    et_3 = ETCBlock(random_state=42)
    stack_3 = StackBlock(['X_0', 'X_1'], axis=1)
    average_3 = AverageBlock(axis=1)
    argmax_3 = ArgmaxBlock(axis=1)
    roc_auc = RocAucBlock()
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
            roc_auc
        ],
        connections=[
            # input X
            Connection(input_x.slots.outputs['X'], rf_1.slots.inputs['X']),
            Connection(input_x.slots.outputs['X'], et_1.slots.inputs['X']),
            # input y
            Connection(input_y.slots.outputs['y'], rf_1.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], et_1.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], rf_2.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], et_2.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], rf_3.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], et_3.slots.inputs['y']),
            # layers connection
            Connection(rf_1.slots.outputs['output'], concat_1.slots.inputs['X_0']),
            Connection(et_1.slots.outputs['output'], concat_1.slots.inputs['X_1']),
            Connection(concat_1.slots.outputs['output'], rf_2.slots.inputs['X']),
            Connection(concat_1.slots.outputs['output'], et_2.slots.inputs['X']),
            Connection(rf_2.slots.outputs['output'], concat_2.slots.inputs['X_0']),
            Connection(et_2.slots.outputs['output'], concat_2.slots.inputs['X_1']),
            Connection(concat_2.slots.outputs['output'], rf_3.slots.inputs['X']),
            Connection(concat_2.slots.outputs['output'], et_3.slots.inputs['X']),
            Connection(rf_3.slots.outputs['output'], stack_3.slots.inputs['X_0']),
            Connection(et_3.slots.outputs['output'], stack_3.slots.inputs['X_1']),
            Connection(stack_3.slots.outputs['output'], average_3.slots.inputs['X']),
            Connection(average_3.slots.outputs['output'], argmax_3.slots.inputs['X']),
            Connection(average_3.slots.outputs['output'], roc_auc.slots.inputs['pred_probas']),
            Connection(input_y.slots.outputs['y'], roc_auc.slots.inputs['gt_y']),
        ]
    )

    fit_executor = executor(
        pipeline,
        stage=Stage.FIT,
        inputs={
            'X': input_x.slots.inputs['X'],
            'y': input_y.slots.inputs['y'],
        },
        outputs={
            'probas': average_3.slots.outputs['output'],
            'roc-auc': roc_auc.slots.outputs['roc-auc'],
        },
        **kwargs
    )
    transform_executor = executor(
        pipeline,
        stage=Stage.TRANSFORM,
        inputs={'X': input_x.slots.inputs['X']},
        outputs={
            'probas': average_3.slots.outputs['output'],
            'labels': argmax_3.slots.outputs['output'],
        },
        **kwargs
    )
    return pipeline, fit_executor, transform_executor

class FunctionalBlockWrapper:
    def __init__(self, block: BaseBlock, output_name: Optional[str] = None):
        self.block = block
        self.output_name = output_name

    def get_input_slot(self, slot_name: Optional[str] = None):
        if slot_name is None:
            if len(self.block.slots.inputs) == 1:
                return list(self.block.slots.inputs.values())[0]
            else:
                raise RuntimeError('Block has more than one input (please, specify it)')
        return self.block.meta.inputs[slot_name]

    def get_output_slot(self) -> BlockOutputSlot:
        if self.output_name is None:
            if len(self.block.slots.outputs) == 1:
                return list(self.block.slots.outputs.values())[0]
            else:
                raise RuntimeError('Block has more than one output')
        return self.block.slots.outputs[self.output_name]

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
                            dst=block.slots.inputs[input_name],
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


def make_deep_forest_functional(executor, **kwargs):
    b = FunctionalBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC(random_state=42)(X=X, y=y)
    et_1 = b.ETC(random_state=42)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(random_state=42)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=42)(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC(random_state=42)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=42)(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=stack_3)
    argmax_3 = b.Argmax(axis=1)(X=average_3)

    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = executor(
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
        **kwargs
    )
    transform_executor = executor(
        b.pipeline,
        stage=Stage.TRANSFORM,
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
        **kwargs
    )
    return b.pipeline, fit_executor, transform_executor


def make_deep_forest_layer(b, **inputs):
    rf = b.RFC(random_state=42)(**inputs)
    et = b.ETC(random_state=42)(**inputs)
    stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
    average = b.Average(axis=1)(X=stack)
    return average


def make_deep_forest_functional_confidence_screening(executor, **kwargs):
    b = FunctionalBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC(random_state=42)(X=X, y=y)
    et_1 = b.ETC(random_state=42)(X=X, y=y)
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
    # also slots stages must be different for this block
    # for in_slot_meta in filtered_1_y.block.meta.inputs.values():
    #     in_slot_meta.stages.transform = False

    concat_all_1 = b.Concat(['filtered_1_X', 'filtered_concat_1'])(
        filtered_1_X=filtered_1['X'],
        filtered_concat_1=filtered_1['concat_1']
    )

    average_2 = make_deep_forest_layer(b, X=concat_all_1, y=filtered_1_y)
    concat_2 = b.Concat(['X', 'average_2'])(X=filtered_1['X'], average_2=average_2)

    average_3 = make_deep_forest_layer(b, X=concat_2, y=filtered_1_y)

    # join confident samples with screened out ones
    joined_3 = b.CSJoin()(
        best=cs_1['best'],
        refined=average_3,
        mask=cs_1['mask']
    )

    argmax_3 = b.Argmax(axis=1)(X=joined_3)

    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=joined_3)

    fit_executor = executor(
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
        **kwargs
    )
    transform_executor = executor(
        b.pipeline,
        stage=Stage.TRANSFORM,
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': joined_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
        **kwargs
    )
    return b.pipeline, fit_executor, transform_executor


def main():
    # _, fit_executor, transform_executor = make_deep_forest()
    # _, fit_executor, transform_executor = make_deep_forest_functional()
    test_forest_factory = make_deep_forest

    score_dict = defaultdict(list)

    for name, executor, kw in [('naive', NaiveExecutor, {}), ('topological', TopologicalExecutor, {'figure_dpi': 300, 'figure_rankdir': 'LR'})]:
        print(f'--- Using of the {name} executor ---')
        _, fit_executor, transform_executor = test_forest_factory(executor, **kw)

        all_X, all_y = make_moons(noise=0.5, random_state=42)
        train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
        fit_result = fit_executor({'X': train_X, 'y': train_y})
        print("Fit successful")
        train_result = transform_executor({'X': train_X})
        score_dict["Fit probas == probas on train"].append(np.allclose(fit_result['probas'], train_result['probas']))
        print("Fit probas == probas on train:", score_dict["Fit probas == probas on train"][-1])
        test_result = transform_executor({'X': test_X})
        print(train_result.keys())
        score_dict["Train ROC-AUC"].append(roc_auc_score(train_y, train_result['probas'][:, 1]))
        print("Train ROC-AUC:", score_dict["Train ROC-AUC"][-1])
        score_dict["Train ROC-AUC calculated by fit_executor"].append(fit_result['roc-auc'])
        print(
            "Train ROC-AUC calculated by fit_executor:",
            score_dict["Train ROC-AUC calculated by fit_executor"][-1]
        )
        score_dict["Train ROC-AUC for RF_1"].append(fit_result['roc-auc'])
        print(
            "Train ROC-AUC for RF_1:",
            score_dict["Train ROC-AUC for RF_1"][-1]
        )
        score_dict["Test ROC-AUC"].append(roc_auc_score(test_y, test_result['probas'][:, 1]))
        print("Test ROC-AUC:", score_dict["Test ROC-AUC"][-1])

        if executor is TopologicalExecutor:
            print('Drawing the graphs for fit and transform executors')
            fit_executor.draw('CSGraph_fit.png')
            transform_executor.draw('CSGraph_transform.png')
    
    print('Check the scores diff for the executors:')
    tol = 10 ** -6
    passed = True
    for key, val in score_dict.items():
        res = all([abs(score - val[0]) < tol for score in val])
        passed *= res
        print(key, 'score:', 'pass' if res else 'fail')
    print('Test is', 'passed' if passed else 'failed')


if __name__ == "__main__":
    main()
