"""Example of deep forest pipeline serialization using skops.

"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bosk.executor.recursive import RecursiveExecutor

from sklearn.metrics import roc_auc_score
from bosk.executor.descriptor import HandlingDescriptor
from bosk.pipeline.serializer.skops import SkopsBlockSerializer
from bosk.pipeline.serializer.zip import ZipPipelineSerializer
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.stages import Stage
from bosk.pipeline.connection import Connection
from bosk.block.base import BaseBlock


def make_deep_forest_functional(executor, forest_params=None, **ex_kw):
    if forest_params is None:
        forest_params = dict()
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
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

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        stage=Stage.FIT,
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        # outputs=['probas'],
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


def main():
    fit_executor, transform_executor = make_deep_forest_functional(RecursiveExecutor)

    # all_X, all_y = make_moons(noise=0.5, random_state=42)
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': train_X, 'y': train_y})
    print("  Fit successful")

    block_serializer = SkopsBlockSerializer()
    serializer = ZipPipelineSerializer(block_serializer)
    transform_pipeline_file = '_test.zip'
    serializer.dump(transform_executor.pipeline, transform_pipeline_file)
    transform_pipeline = serializer.load(transform_pipeline_file)
    transform_executor = RecursiveExecutor(
        transform_pipeline,
        stage=Stage.TRANSFORM,
        inputs=['X'],
        outputs=['probas', 'labels'],
    )

    train_result = transform_executor({'X': train_X})
    print("  Fit probas == probas on train:", np.allclose(fit_result['probas'], train_result['probas']))
    test_result = transform_executor({'X': test_X})
    print("  Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'][:, 1]))
    print(
        "  Train ROC-AUC calculated by fit_executor:",
        fit_result['roc-auc']
    )
    print(
        "  Train ROC-AUC for RF_1:",
        fit_result['rf_1_roc-auc']
    )
    print("  Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'][:, 1]))


if __name__ == "__main__":
    main()
