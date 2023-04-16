"""Example of simple xgboost Deep Forest definition.

"""
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.executor.recursive import RecursiveExecutor
from bosk.stages import Stage
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.data import CPUData


def make_deep_forest_functional_gpu(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.XGBClassifier(tree_method="gpu_hist", random_state=42)(X=X, y=y)
    et_1 = b.ETC(random_state=42)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.XGBClassifier(tree_method="gpu_hist", random_state=42)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=42)(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.XGBClassifier(tree_method="gpu_hist", random_state=42)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=42)(X=concat_2, y=y)
    rf_3 = b.MoveTo("GPU")(X=rf_3)
    et_3 = b.MoveTo("GPU")(X=et_3)
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


def make_deep_forest_functional(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.XGBClassifier(random_state=42)(X=X, y=y)
    et_1 = b.ETC(random_state=42)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.XGBClassifier(random_state=42)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=42)(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.XGBClassifier(random_state=42)(X=concat_2, y=y)
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


def run_cpu():
    executor_class = RecursiveExecutor
    fit_executor, transform_executor = make_deep_forest_functional(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': CPUData(train_X), 'y': CPUData(train_y)})
    print("Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'].data, train_result['probas'].data))
    test_result = transform_executor({'X': CPUData(test_X)})
    print(train_result.keys())
    print("Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    print(
        "Train ROC-AUC calculated by fit_executor:",
        fit_result['roc-auc'].data
    )
    print(
        "Train ROC-AUC for RF_1:",
        fit_result['rf_1_roc-auc'].data
    )
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


def run_gpu():
    executor_class = RecursiveExecutor
    fit_executor, transform_executor = make_deep_forest_functional_gpu(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': CPUData(train_X), 'y': CPUData(train_y)})
    print("Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'].data, train_result['probas'].data))
    test_result = transform_executor({'X': CPUData(test_X)})
    print(train_result.keys())
    print("Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    print(
        "Train ROC-AUC calculated by fit_executor:",
        fit_result['roc-auc'].data
    )
    print(
        "Train ROC-AUC for RF_1:",
        fit_result['rf_1_roc-auc'].data
    )
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_cpu()
    end_time = time.time()
    print(f"Time CPU execution: {end_time - start_time} sec")

    start_time = time.time()
    run_gpu()
    end_time = time.time()
    print(f"Time GPU execution: {end_time - start_time} sec")

