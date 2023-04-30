"""Example of simple Confidence Screening Deep Forest definition.

"""
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.executor.recursive import RecursiveExecutor
from bosk.stages import Stage
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.data import CPUData


def make_deep_forest_layer(b, **inputs):
    rf = b.RFC(random_state=42)(**inputs)
    et = b.ETC(random_state=42)(**inputs)
    stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
    average = b.Average(axis=1)(X=stack)
    return average


def make_deep_forest_functional_confidence_screening(executor, **ex_kw):
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

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': joined_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
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
        **ex_kw,
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': joined_3, 'labels': argmax_3}
        ),
        stage=Stage.TRANSFORM,
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': joined_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
        **ex_kw,
    )
    return fit_executor, transform_executor


def main():
    executor_class = RecursiveExecutor
    fit_executor, transform_executor = make_deep_forest_functional_confidence_screening(executor_class)

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
        fit_result['roc-auc']
    )
    print(
        "Train ROC-AUC for RF_1:",
        fit_result['rf_1_roc-auc']
    )
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


if __name__ == "__main__":
    main()
