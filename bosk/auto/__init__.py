"""Automatic Deep Forest Construction module.

A Deep Forest model can be constructed for different types of data:

1. Tabular data: :py:class:`ClassicalDeepForestConstructor`.
2. Images or sequences: :py:class:`MGSDeepForestConstructor`.

Growing strategies are used to represent stopping criteria.
For cross-validation use :py:class:`EarlyStoppingCV`, for validation
on separately given data set use :py:class:`EarlyStoppingVal`.

Metrics used in cross-validation are defined in :py:class:`MetricsEvaluator`.

Example:

.. code-block:: python

    from bosk.deep_forest import ClassicalDeepForestConstructor, MetricsEvaluator
    from bosk.executor import TopologicalExecutor
    from bosk.executor.sklearn_interface import BoskPipelineClassifier

    constructor = ClassicalDeepForestConstructor(
        TopologicalExecutor,
        rf_params=dict(
            n_estimators=100,
            max_depth=5,
        ),
        layer_width=2,
        max_iter=5,
        cv=3,
        make_metrics=lambda: MetricsEvaluator(['f1']),
        block_classes=(RFCBlock, ETCBlock),
        random_state=12345,
    )
    pipeline = constructor.construct(X_train, y_train)

    model = BoskPipelineClassifier(pipeline, executor_cls=TopologicalExecutor)
    model._classifier_init(y_train)  # apply label encoding to be able to predict labels
    predicted_labels = model.predict(X_test)


"""

from .deep_forest import (
    BaseAutoDeepForestConstructor,
    ClassicalDeepForestConstructor,
    HyperparamSearchDeepForestConstructor,
    MGSDeepForestConstructor,
)
from .growing_strategies import EarlyStoppingCV, EarlyStoppingVal
from .metrics import MetricsEvaluator


__all__ = [
    "BaseAutoDeepForestConstructor",
    "ClassicalDeepForestConstructor",
    "HyperparamSearchDeepForestConstructor",
    "MGSDeepForestConstructor",
    "EarlyStoppingCV",
    "EarlyStoppingVal",
    "MetricsEvaluator",
]
