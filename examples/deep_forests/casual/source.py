"""Example of the basic deep forest creating using either manual graph definition and the functional API.
"""

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.base import BaseExecutor
from bosk.stages import Stage
from bosk.executor.descriptor import HandlingDescriptor
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder


def make_deep_forest(executor: BaseExecutor, **ex_kw):
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
    roc_auc_rf_1 = RocAucBlock()
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
            roc_auc,
            roc_auc_rf_1
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
            Connection(rf_1.slots.outputs['output'], roc_auc_rf_1.slots.inputs['pred_probas']),
            Connection(input_y.slots.outputs['y'], roc_auc_rf_1.slots.inputs['gt_y']),
        ],
        inputs={
            'X': input_x.slots.inputs['X'],
            'y': input_y.slots.inputs['y'],
        },
        outputs={
            'probas': average_3.slots.outputs['output'],
            'rf_1_roc-auc': roc_auc_rf_1.slots.outputs['roc-auc'],
            'roc-auc': roc_auc.slots.outputs['roc-auc'],
            'labels': argmax_3.slots.outputs['output']
        }
    )

    fit_executor = executor(
        pipeline,
        HandlingDescriptor.from_classes(Stage.FIT),
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        **ex_kw
    )
    transform_executor = executor(
        pipeline,
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs=['X'],
        outputs=['probas', 'labels'],
        **ex_kw
    )
    return fit_executor, transform_executor


def make_deep_forest_functional(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
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
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        HandlingDescriptor.from_classes(Stage.FIT),
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        **ex_kw
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs=['X'],
        outputs=['probas', 'labels'],
        **ex_kw
    )
    return fit_executor, transform_executor


def main():
    executor_class = RecursiveExecutor
    # fit_executor, transform_executor = make_deep_forest(executor_class)
    fit_executor, transform_executor = make_deep_forest_functional(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
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
    main()
