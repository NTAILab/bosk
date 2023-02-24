"""Script that contains common tests for all executors."""

from bosk.executor import *
from bosk.executor.base import BaseExecutor
from bosk.executor.parallel.greedy import GreedyParallelExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage
from collections import defaultdict
import numpy as np
from ..pipelines import *
from ..pipelines.base import BasePipelineTest as BPT
from ..utility import fit_pipeline, get_all_subclasses, log_test_name
import logging
from typing import Type, Set

# BasePipelineTest rename is needed to exclude
# this class from pytest test discovery mechanism

EXCLUDED_EXECS_SET = {
    GreedyParallelExecutor,
}


def get_executors() -> Set[Type[BaseExecutor]]:
    executors = get_all_subclasses(BaseExecutor)
    excluded_execs = executors.intersection(EXCLUDED_EXECS_SET)
    executors = executors.difference(excluded_execs)
    logging.info('Following executors will be tested: %r', [e.__name__ for e in executors])
    logging.info('Following executors were excluded: %r', [e.__name__ for e in excluded_execs])
    return executors


def get_pipeline_wrapper() -> BPT:
    """Pipeline that will be used in the fit_transform test."""
    return CasualManualForest()


def fit_transform_test():
    """Test that gets every executor from the `bosk` package and tries to
    fit determined in the `get_pipeline_wrapper` pipeline and than transforms some data
    with the fitted one. To make the executor available to be discovered by this test,
    you should add it to the `__all__` variable of the `bosk.executor` package.
    """
    log_test_name()
    executors = get_executors()
    logging.info('The test is performed with the pipeline from the %s',
                 get_pipeline_wrapper().__class__.__name__)
    status_good = True
    for e_cls in executors:
        logging.info('Starting %s test', e_cls.__name__)
        pipeline_wrapper = get_pipeline_wrapper()
        pipeline = pipeline_wrapper.get_pipeline()
        try:
            logging.info('Starting fit process')
            fitted_pipeline, _ = fit_pipeline(pipeline,
                                              pipeline_wrapper.get_fit_data(), e_cls, *pipeline_wrapper.get_fit_in_out())
            logging.info('Fit process succeeded')
            logging.info('Starting transform process')
            data = pipeline_wrapper.get_transform_data()
            executor = e_cls(fitted_pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                             *pipeline_wrapper.get_transform_in_out())
            executor(data)
            logging.info('Transform process succeeded')
        except Exception as exp:
            logging.error('Executor %s has failed the test with the following exception: %s',
                          e_cls.__name__, exp)
            status_good = False
    assert status_good, "Some executor(s) failed the test"


def cross_test():
    """Test that gets all (determined in the `__all__` variable of the `bosk.tests.pipelines` package)
    test pipelines and all (determined in the `__all__` variable of the `bosk.executor` package)
    executors and than fits every pipeline using every executor, comparing results, retrieved
    by the different executors. The results for every pipeline must be the same regardless of
    the used executor. The same is performed with the transform stage.
    """
    log_test_name()
    tol = 1e-8
    executors = get_executors()
    pip_wrappers = get_all_subclasses(BPT)
    logging.info('Following pipeline wrappers were found: %r', [p.__name__ for p in pip_wrappers])

    def process_scores(exec_dict, postfix):
        for key, val in exec_dict.items():
            if isinstance(val, float) or isinstance(val, int):
                logging.info('\t\t%s: %s', key, val)
            else:
                logging.info('\t\t%s: ... (mean %.4f)', key, np.mean(val))
            score_dict[key + f' ({postfix})'].append(val)
    for p_w_cls in pip_wrappers:
        logging.info('Processing the %s pipeline wrapper', p_w_cls.__name__)
        score_dict = defaultdict(list)
        for e_cls in executors:
            p_w = p_w_cls()
            logging.info('Starting fit with the %s executor', e_cls.__name__)
            fitted_pipeline, fit_dict = fit_pipeline(p_w.get_pipeline(),
                                                     p_w.get_fit_data(), RecursiveExecutor, *p_w.get_fit_in_out())
            logging.info('Fit results:')
            process_scores(fit_dict, 'fit')
            logging.info('Starting transform with the %s executor', e_cls.__name__)
            tf_data = p_w.get_transform_data()
            executor = e_cls(fitted_pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                             *p_w.get_transform_in_out())
            tf_dict = executor(tf_data)
            logging.info('Transform results:')
            process_scores(tf_dict, 'transform')
        for key, val in score_dict.items():
            assert all([np.sum(np.abs(score - val[0])) < tol for score in val]), \
                f'Not all executors have given same results for the {p_w_cls.__name__} case.' +\
                f'The difference at least in the {key} output'
