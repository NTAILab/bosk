"""Example of Eager evaluation: applying fit and transform stages during pipeline construnction.

"""
from bosk.data import CPUData
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.executor.block import DefaultBlockExecutor
from bosk.executor.recursive import RecursiveExecutor
from bosk.pipeline.builder.eager import EagerPipelineBuilder


def main():
    # prepare data first
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    executor_cls = RecursiveExecutor
    block_executor = DefaultBlockExecutor()
    forest_params = dict()

    b = EagerPipelineBuilder(block_executor)
    X, y = b.Input()(CPUData(train_X)), b.TargetInput()(CPUData(train_y))
    # alternative ways to define input blocks:
    # X, y = b.Input()(X=CPUData(train_X)), b.TargetInput()(y=CPUData(train_y))
    # or without input data:
    # X, y = b.Input()(), b.TargetInput()()
    # X.execute({X.block.get_single_input(): CPUData(train_X)})
    # y.execute({y.block.get_single_input(): CPUData(train_y)})

    rf_1 = b.RFC(random_state=42, **forest_params)(X=X, y=y)
    et_1 = b.ETC(random_state=42, **forest_params)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(random_state=42, **forest_params)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=42, **forest_params)(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC(random_state=42, **forest_params)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=42, **forest_params)(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=stack_3)
    argmax_3 = b.Argmax(axis=1)(X=average_3)

    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)

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
    main()
