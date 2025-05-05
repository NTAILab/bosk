from time import sleep
from bosk.block.meta import BlockExecutionProperties
from bosk.data import CPUData
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from bosk.executor.parallel.greedy import (
    GreedyParallelExecutor,
    JoblibParallelEngine,
    MultiprocessingParallelEngine,
)
from bosk.block.auto import auto_block
from joblib import parallel_backend
from ..utility import log_test_name
import logging


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True), fit_argnames={'X', 'y'})
class ParallelRFC(RandomForestClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True), fit_argnames={'X', 'y'})
class ParallelETC(ExtraTreesClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ParallelSleepStub:
    def __init__(self, fit_time: float = 0.1,
                 transform_time: float = 0.05):
        self.fit_time = fit_time
        self.transform_time = transform_time

    def fit(self, X):
        sleep(self.fit_time)
        return self

    def transform(self, X):
        sleep(self.transform_time)
        return CPUData(X)


def make_deep_forest_functional(executor, forest_params=None, **ex_kw):
    if forest_params is None:
        forest_params = dict()
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.new(ParallelRFC, random_state=42, **forest_params)(X=X, y=y)
    et_1 = b.new(ParallelETC, random_state=42, **forest_params)(X=X, y=y)
    rf_1 = b.new(ParallelSleepStub)(X=rf_1)
    et_1 = b.new(ParallelSleepStub)(X=et_1)
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


def test_proc(executor_class, ex_kw=None, forest_params=None):
    if ex_kw is None:
        ex_kw = dict()
    fit_executor, transform_executor = make_deep_forest_functional(
        executor_class,
        forest_params=forest_params,
        **ex_kw
    )
    train_X, train_y = load_breast_cancer(return_X_y=True)
    fit_result = fit_executor({'X': CPUData(train_X), 'y': CPUData(train_y)})
    logging.info("Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    logging.info("Transform successful")
    assert np.allclose(fit_result['probas'].data, train_result['probas'].data),\
        "Fit probas are different from the transform ones"


def greedy_test():
    log_test_name()
    params = dict(
        forest_params=dict(
            n_estimators=11,
            n_jobs=1
        ),
    )
    threading_kw = dict(
        parallel_engine=MultiprocessingParallelEngine(),
    )
    joblib_kw = dict(
        parallel_engine=JoblibParallelEngine(),
    )
    logging.info("Starting test of the threading backend")
    test_proc(GreedyParallelExecutor, ex_kw=threading_kw, **params)
    logging.info("Starting test of the joblib backend")
    with parallel_backend('threading', n_jobs=-1):
        test_proc(GreedyParallelExecutor, ex_kw=joblib_kw, **params)
