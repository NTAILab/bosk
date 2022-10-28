"""Example of simple Confidence Screening Deep Forest definition.

"""
from typing import Callable, Optional

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.block import BaseBlock
from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.naive import NaiveExecutor
from bosk.stages import Stage
from bosk.slot import BlockOutputSlot
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock, AccuracyBlock, F1ScoreBlock
from bosk.block.zoo.routing import CSBlock, CSJoinBlock, CSFilterBlock
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder


def make_deep_forest():
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


def make_deep_forest_functional():
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


def make_deep_forest_layer(b, **inputs):
    rf = b.RFC(random_state=42)(**inputs)
    et = b.ETC(random_state=42)(**inputs)
    stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
    average = b.Average(axis=1)(X=stack)
    return average


def make_deep_forest_functional_confidence_screening():
    b = FunctionalPipelineBuilder()
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

    fit_executor = NaiveExecutor(
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
    transform_executor = NaiveExecutor(
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


def main():
    # _, fit_executor, transform_executor = make_deep_forest()
    # _, fit_executor, transform_executor = make_deep_forest_functional()
    _, fit_executor, transform_executor = make_deep_forest_functional_confidence_screening()

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
