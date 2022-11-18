"""Example of simple Adaptive Weighted Deep Forest definition in functional style.

"""
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.stages import Stage
from bosk.executor.handlers import SimpleExecutionStrategy, InputSlotStrategy
from bosk.block.zoo.data_weighting import WeightsBlock
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.executor.naive import NaiveExecutor

from examples.deep_forests.cs.simple import make_deep_forest_layer


def make_deep_forest_weighted_confidence_screening(executor, **ex_kw):
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

    sample_weight_2 = b.new(WeightsBlock, ord=2)(X=average_2, y=filtered_1_y)

    average_3 = make_deep_forest_layer(b, X=concat_2, y=filtered_1_y, sample_weight=sample_weight_2)

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
        InputSlotStrategy(Stage.FIT),
        SimpleExecutionStrategy(Stage.FIT),
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
        InputSlotStrategy(Stage.TRANSFORM),
        SimpleExecutionStrategy(Stage.TRANSFORM),
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
    fit_executor, transform_executor = make_deep_forest_weighted_confidence_screening(NaiveExecutor)

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
