"""Example of simple Multi-grained-scanning."""
import numpy as np

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from bosk.executor.recursive import RecursiveExecutor
from bosk.stages import Stage
from bosk.executor.descriptor import HandlingDescriptor
from bosk.block.zoo.multi_grained_scanning import \
    (MultiGrainedScanning1DBlock, MultiGrainedScanning2DBlock)
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder


def make_deep_forest_functional_multi_grained_scanning_1d(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    random_state_value = 42
    X, y = b.Input()(), b.TargetInput()()
    ms = b.new(MultiGrainedScanning1DBlock, models=(RandomForestClassifier(random_state=random_state_value),
                                                    ExtraTreesClassifier(random_state=random_state_value)),
               window_size=2, stride=1)(X=X, y=y)
    rf_1 = b.RFC(random_state=random_state_value)(X=ms, y=y)
    et_1 = b.ETC(random_state=random_state_value)(X=ms, y=y)
    concat_1 = b.Concat(['ms', 'rf_1', 'et_1'])(ms=ms, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(random_state=random_state_value)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=random_state_value)(X=concat_1, y=y)
    concat_2 = b.Concat(['ms', 'rf_2', 'et_2'])(ms=ms, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC(random_state=random_state_value)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=random_state_value)(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=stack_3)
    argmax_3 = b.Argmax(axis=1)(X=average_3)
    #
    rf_1_roc_auc = b.RocAucMultiLabel()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAucMultiLabel()(gt_y=y, pred_probas=average_3)

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        HandlingDescriptor.from_classes(Stage.FIT),
        inputs={
            'X': X.get_input_slot(),
            'y': y.get_input_slot(),
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'rf_1_roc-auc': rf_1_roc_auc.get_output_slot(),
            'roc-auc': roc_auc.get_output_slot(),
        },
        **ex_kw,
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
        **ex_kw,
    )
    return fit_executor, transform_executor


def make_deep_forest_functional_multi_grained_scanning_2d(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    random_state_value = 42
    X, y = b.Input()(), b.TargetInput()()
    ms = b.new(MultiGrainedScanning2DBlock, models=(RandomForestClassifier(random_state=random_state_value),
                                                    ExtraTreesClassifier(random_state=random_state_value)),
               window_size=4, stride=1, shape_sample=[8, 8])(X=X, y=y)
    rf_1 = b.RFC(random_state=random_state_value)(X=ms, y=y)
    et_1 = b.ETC(random_state=random_state_value)(X=ms, y=y)
    concat_1 = b.Concat(['ms', 'rf_1', 'et_1'])(ms=ms, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(random_state=random_state_value)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=random_state_value)(X=concat_1, y=y)
    concat_2 = b.Concat(['ms', 'rf_2', 'et_2'])(ms=ms, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC(random_state=random_state_value)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=random_state_value)(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=stack_3)
    argmax_3 = b.Argmax(axis=1)(X=average_3)
    #
    rf_1_roc_auc = b.RocAucMultiLabel()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAucMultiLabel()(gt_y=y, pred_probas=average_3)

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        HandlingDescriptor.from_classes(Stage.FIT),
        inputs={
            'X': X.get_input_slot(),
            'y': y.get_input_slot(),
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'rf_1_roc-auc': rf_1_roc_auc.get_output_slot(),
            'roc-auc': roc_auc.get_output_slot(),
        },
        **ex_kw,
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
        **ex_kw,
    )
    return fit_executor, transform_executor


def example_iris_dataset():
    print("1D:")
    fit_executor, transform_executor = make_deep_forest_functional_multi_grained_scanning_1d(RecursiveExecutor)
    iris = load_iris()
    all_X = iris.data
    all_y = iris.target
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': train_X, 'y': train_y})
    print("Fit successful")
    train_result = transform_executor({'X': train_X})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'], train_result['probas']))
    test_result = transform_executor({'X': test_X})
    print(train_result.keys())
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'], multi_class="ovr"))


def example_digits_dataset():
    print("2D:")
    fit_executor, transform_executor = make_deep_forest_functional_multi_grained_scanning_2d(RecursiveExecutor)
    digits = load_digits()
    all_X = digits.data
    all_y = digits.target
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': train_X, 'y': train_y})
    print("Fit successful")
    train_result = transform_executor({'X': train_X})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'], train_result['probas']))
    test_result = transform_executor({'X': test_X})
    print(train_result.keys())
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'], multi_class="ovr"))


def main():
    example_iris_dataset()
    example_digits_dataset()


if __name__ == "__main__":
    main()
