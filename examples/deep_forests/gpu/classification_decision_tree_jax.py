from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer,
)

from bosk.data import GPUData


loaders = {
    "iris": load_iris,
    "digits": load_digits,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

from bosk.block.zoo.models.classification.classification_models_jax import DecisionTreeClassifierBlock


def simple_example():
    dataset = load_iris()
    X, y = dataset["data"], dataset["target"]
    n_classes = np.size(np.bincount(y))
    model = DecisionTreeClassifierBlock(
        n_classes=n_classes,
        max_depth=3,
        min_samples=1,
    )
    fitted_model = model.fit(X, y)
    print(fitted_model.score(X, y))

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.executor.recursive import RecursiveExecutor
from bosk.stages import Stage
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder


def make_deep_forest_functional_gpu(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.DecisionTreeClassifier(n_classes=2)(X=X, y=y)
    et_1 = b.ETC(random_state=42)(X=X, y=y)
    et_1 = b.MoveTo("GPU")(X=et_1)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    concat_1 = b.MoveTo("GPU")(X=concat_1)

    rf_2 = b.DecisionTreeClassifier(n_classes=2)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=42)(X=concat_1, y=y)
    et_2 = b.MoveTo("GPU")(X=et_2)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    concat_2 = b.MoveTo("GPU")(X=concat_2)

    rf_3 = b.DecisionTreeClassifier(n_classes=2)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=42)(X=concat_2, y=y)
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

import jax.numpy as jnp

def run_gpu():
    executor_class = RecursiveExecutor
    fit_executor, transform_executor = make_deep_forest_functional_gpu(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': GPUData(train_X), 'y': GPUData(train_y)})
    print("Fit successful")
    train_result = transform_executor({'X': GPUData(train_X)})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'].data, train_result['probas'].data))
    test_result = transform_executor({'X': GPUData(test_X)})
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
    run_gpu()
    end_time = time.time()
    print(f"Time GPU execution: {end_time - start_time} sec")
