"""Example of deep forest pipeline construction with "better" functional interface.

"""
import json
from bosk.data import CPUData
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bosk.executor.recursive import RecursiveExecutor

from sklearn.metrics import roc_auc_score
from bosk.block import BaseBlock
from functools import wraps
from typing import Mapping

from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.pipeline.builder.eager import EagerPipelineBuilder
from bosk.data import BaseData
from bosk.executor.block import DefaultBlockExecutor

from bosk.block.zoo.input_plugs import Input, TargetInput
from bosk.block.zoo.models.classification import RFC, ETC
from bosk.block.zoo.data_conversion import Average, Concat, Argmax, Stack
from bosk.block.zoo.metrics import RocAuc


def nested_function_with_pipeline(train_X: np.ndarray, train_y: np.ndarray,
                                  block_executor) -> np.ndarray:
    with EagerPipelineBuilder(block_executor) as b:
        X, y = Input()(CPUData(train_X)), TargetInput()(CPUData(train_y))
        rf = RFC(n_estimators=5, random_state=42)(X=X, y=y)
        train_score = RocAuc()(gt_y=y, pred_probas=rf)
        return train_score.get_output_data().data


def make_deep_forest_functional(executor, forest_params=None, **ex_kw):
    if forest_params is None:
        forest_params = dict()

    with FunctionalPipelineBuilder() as b:
        X, y = Input()(), TargetInput()()
        rf_1 = RFC(random_state=42, **forest_params)(X=X, y=y)
        et_1 = ETC(random_state=42, **forest_params)(X=X, y=y)
        concat_1 = Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
        rf_2 = RFC(random_state=42, **forest_params)(X=concat_1, y=y)
        et_2 = ETC(random_state=42, **forest_params)(X=concat_1, y=y)
        concat_2 = Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
        rf_3 = RFC(random_state=42, **forest_params)(X=concat_2, y=y)
        et_3 = ETC(random_state=42, **forest_params)(X=concat_2, y=y)
        stack_3 = Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)

        average_3 = Average(axis=1)(X=stack_3)
        argmax_3 = Argmax(axis=1)(X=average_3)

        rf_1_roc_auc = RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        stage=Stage.FIT,
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        **ex_kw
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        stage=Stage.TRANSFORM,
        inputs=['X'],
        outputs=['probas', 'labels'],
        **ex_kw
    )
    return fit_executor, transform_executor


def main_functional():
    fit_executor, transform_executor = make_deep_forest_functional(RecursiveExecutor)

    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    fit_result = fit_executor({'X': CPUData(train_X), 'y': CPUData(train_y)})
    print("  Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    print(
        "  Fit probas == probas on train:",
        np.allclose(fit_result['probas'].data, train_result['probas'].data)
    )
    test_result = transform_executor({'X': CPUData(test_X)})
    print("  Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    print("  Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


def main_eager():
    # prepare data first
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    executor_cls = RecursiveExecutor
    block_executor = DefaultBlockExecutor()
    forest_params = dict()

    with EagerPipelineBuilder(block_executor) as b:
        X, y = Input()(CPUData(train_X)), TargetInput()(CPUData(train_y))
        # alternative ways to define input blocks:
        # X, y = Input()(X=CPUData(train_X)), TargetInput()(y=CPUData(train_y))
        # or without input data:
        # X, y = Input()(), TargetInput()()
        # X.execute({X.block.get_single_input(): CPUData(train_X)})
        # y.execute({y.block.get_single_input(): CPUData(train_y)})

        rf_1 = RFC(random_state=42, **forest_params)(X=X, y=y)
        et_1 = ETC(random_state=42, **forest_params)(X=X, y=y)
        concat_1 = Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
        rf_2 = RFC(random_state=42, **forest_params)(X=concat_1, y=y)
        et_2 = ETC(random_state=42, **forest_params)(X=concat_1, y=y)
        concat_2 = Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
        rf_3 = RFC(random_state=42, **forest_params)(X=concat_2, y=y)
        et_3 = ETC(random_state=42, **forest_params)(X=concat_2, y=y)
        stack_3 = Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
        average_3 = Average(axis=1)(X=stack_3)
        argmax_3 = Argmax(axis=1)(X=average_3)

        print(
            'Here we can calculate roc-auc using another pipeline:',
            nested_function_with_pipeline(train_X, train_y, block_executor)
        )

        rf_1_roc_auc = RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = executor_cls(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        stage=Stage.FIT,
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
    )
    transform_executor = executor_cls(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        stage=Stage.TRANSFORM,
        inputs=['X'],
        outputs=['probas', 'labels'],
    )

    train_result = transform_executor({'X': CPUData(train_X)})
    print(
        "  Fit probas == probas on train:",
        np.allclose(average_3.get_output_data().data, train_result['probas'].data)
    )
    test_result = transform_executor({'X': CPUData(test_X)})
    print("  Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    print("  Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


if __name__ == "__main__":
    main_functional()
    main_eager()

