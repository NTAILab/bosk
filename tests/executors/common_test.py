from bosk.executor import *
from ..pipelines import *
from bosk.executor.base import BaseExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage
from ..pipelines.base import PipelineTestBase
from ..utility import fit_pipeline
from collections import defaultdict
import numpy as np
import logging


def get_pipeline_wrapper() -> PipelineTestBase:
    return CasualManualForestTest()


def fit_transform_test():
    executors = BaseExecutor.__subclasses__()
    logging.info('Following executors were found: %r', [e.__name__ for e in executors])
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
    tol = 1e-8
    executors = BaseExecutor.__subclasses__()
    pip_wrappers = PipelineTestBase.__subclasses__()
    logging.info('Following executors were found: %r', [e.__name__ for e in executors])
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
