"""Example of deep forest execution using Greedy parallel engine.

"""
from collections import defaultdict
from time import time
from bosk.block.meta import BlockExecutionProperties
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.topological import TopologicalExecutor
from bosk.executor.parallel.greedy import GreedyParallelExecutor, JoblibParallelEngine, ThreadingParallelEngine
from bosk.painter.topological import TopologicalPainter

# from examples.deep_forests.casual.source import make_deep_forest, make_deep_forest_functional
# from examples.deep_forests.cs.simple import make_deep_forest_functional_confidence_screening
# from examples.deep_forests.mg_scanning.mg_scanning import (make_deep_forest_functional_multi_grained_scanning_1d,
#                                                            make_deep_forest_functional_multi_grained_scanning_2d)
# from examples.deep_forests.weighted_cs.simple import make_deep_forest_weighted_confidence_screening
from sklearn.metrics import roc_auc_score
from bosk.executor.descriptor import HandlingDescriptor
from bosk.block.auto import auto_block
from joblib import parallel_backend


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ParallelRFC(RandomForestClassifier):
    def transform(self, X):
        return self.predict_proba(X)


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ParallelETC(ExtraTreesClassifier):
    def transform(self, X):
        return self.predict_proba(X)


def make_deep_forest_functional(executor, forest_params=None, **ex_kw):
    if forest_params is None:
        forest_params = dict()
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.new(ParallelRFC, random_state=42, **forest_params)(X=X, y=y)
    et_1 = b.new(ParallelETC, random_state=42, **forest_params)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.new(ParallelRFC, random_state=42, **forest_params)(X=concat_1, y=y)
    et_2 = b.new(ParallelETC, random_state=42, **forest_params)(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.new(ParallelRFC, random_state=42, **forest_params)(X=concat_2, y=y)
    et_3 = b.new(ParallelETC, random_state=42, **forest_params)(X=concat_2, y=y)
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
        HandlingDescriptor.from_classes(Stage.FIT),
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
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs=['X'],
        outputs=['probas', 'labels'],
        **ex_kw
    )
    return fit_executor, transform_executor


class SimpleTimer:
    """Context manager for time measurement
    """
    class TimerResult:
        def __init__(self, start: float):
            self.start = start
            self.end = None

        @property
        def duration(self):
            return self.end - self.start

    def __enter__(self):
        self.result = self.TimerResult(time())
        return self.result

    def __exit__(self, _type, _value, _traceback):
        self.result.end = time()


def test_executor(executor_class, ex_kw=None, n_time_iter: int = 1, forest_params=None):
    print("Executor:", executor_class)
    if ex_kw is None:
        ex_kw = dict()
    fit_executor, transform_executor = make_deep_forest_functional(
        executor_class,
        forest_params=forest_params,
        **ex_kw
    )

    # all_X, all_y = make_moons(noise=0.5, random_state=42)
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    all_times = defaultdict(list)
    for _ in range(n_time_iter):
        with SimpleTimer() as t:
            fit_result = fit_executor({'X': train_X, 'y': train_y})
        all_times['fit'].append(t.duration)
    print("  Fit successful")
    for _ in range(n_time_iter):
        with SimpleTimer() as t:
            train_result = transform_executor({'X': train_X})
        all_times['transform'].append(t.duration)
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
    print("  Times:")
    for stage, times in all_times.items():
        times = np.array(times)
        print(f"    {stage} = {times.mean()} ± {times.std()}")


def main():
    params = dict(
        n_time_iter=10,
        forest_params=dict(
            n_estimators=500,
            n_jobs=-1
        ),
    )
    threading_kw = dict(
        parallel_engine=ThreadingParallelEngine(),
    )
    joblib_kw = dict(
        parallel_engine=JoblibParallelEngine(),
    )
    test_executor(RecursiveExecutor, **params)
    print("Threading:")
    test_executor(GreedyParallelExecutor, ex_kw=threading_kw, **params)
    print("Joblib:")
    with parallel_backend('threading', n_jobs=-1):
        test_executor(GreedyParallelExecutor, ex_kw=joblib_kw, **params)
    test_executor(TopologicalExecutor, **params)


if __name__ == "__main__":
    main()
