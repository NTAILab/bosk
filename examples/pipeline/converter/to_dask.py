"""Example of deep forest pipeline conversion to a Dask graph.

"""
import json
from bosk.data import CPUData
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bosk.executor.recursive import RecursiveExecutor
from bosk.pipeline.converter.dask import DaskConverter, FitDaskOperatorSet

from sklearn.metrics import roc_auc_score
from bosk.block import BaseBlock
from dask.threaded import get as dask_get
from dask.optimization import cull as dask_cull


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


class FnEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj.__name__)
        if isinstance(obj, BaseBlock):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def main():
    fit_executor, transform_executor = make_deep_forest_functional(RecursiveExecutor)

    # all_X, all_y = make_moons(noise=0.5, random_state=42)
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    train_converter = DaskConverter(stage=Stage.FIT, operator_set=FitDaskOperatorSet())
    dsk_train = train_converter(fit_executor.pipeline)
    dsk_train['X'] = CPUData(train_X)
    dsk_train['y'] = CPUData(train_y)
    outputs = ['probas']
    dsk_train_culled, dependencies = dask_cull(dsk_train, outputs)
    train_outputs = dask_get(dsk_train_culled, outputs)
    fit_result = dict(zip(outputs, train_outputs))

    # fit_result = fit_executor({'X': train_X, 'y': train_y})
    print("  Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    print(
        "  Fit probas == probas on train:",
        np.allclose(fit_result['probas'].data, train_result['probas'].data)
    )
    test_result = transform_executor({'X': CPUData(test_X)})
    print("  Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    # print(
    #     "  Train ROC-AUC calculated by fit_executor:",
    #     fit_result['roc-auc']
    # )
    # print(
    #     "  Train ROC-AUC for RF_1:",
    #     fit_result['rf_1_roc-auc']
    # )
    print("  Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))

    # execute with Dask

    converter = DaskConverter(Stage.TRANSFORM)
    dsk = converter(fit_executor.pipeline)
    # print("Dask Graph:")
    # print(json.dumps(dsk, cls=FnEncoder, indent=4))
    dsk['X'] = CPUData(test_X)
    outputs = ['probas']
    dsk_culled, dependencies = dask_cull(dsk, outputs)
    test_outputs = dask_get(dsk_culled, outputs)
    print("  Test ROC-AUC (Dask):", roc_auc_score(test_y, test_outputs[0].data[:, 1]))


if __name__ == "__main__":
    main()
