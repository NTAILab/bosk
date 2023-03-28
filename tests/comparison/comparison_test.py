from bosk.comparison.cross_val import CVComparator
from bosk.comparison.base import BaseForeignModel
from bosk.comparison.metric import MetricWrapper
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.data import CPUData, BaseData
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np
from time import time
from typing import Dict
from ..utility import log_test_name
import logging


class RFCModel(BaseForeignModel):
    def __init__(self) -> None:
        super().__init__()
        self.forest = RandomForestClassifier(30)

    def fit(self, data: Dict[str, BaseData]) -> None:
        self.forest.fit(data['X'].to_cpu().data,
                        data['y'].to_cpu().data)

    def predict(self, data: Dict[str, BaseData]) -> Dict[str, BaseData]:
        return {'output': CPUData(self.forest.predict_proba(data['X'].to_cpu().data))}

    def set_random_state(self, random_state: int) -> None:
        self.forest.random_state = random_state


class CatBoostModel(BaseForeignModel):
    def __init__(self) -> None:
        super().__init__()
        self.grad_boost = CatBoostClassifier(30, verbose=0)

    def fit(self, data: Dict[str, BaseData]) -> None:
        self.grad_boost.fit(data['X'].to_cpu().data,
                            data['y'].to_cpu().data)

    def predict(self, data: Dict[str, BaseData]) -> Dict[str, BaseData]:
        return {'output': CPUData(self.grad_boost.predict_proba(data['X'].to_cpu().data))}

    def set_random_state(self, random_state: int) -> None:
        self.grad_boost.random_state = random_state


def get_pipeline_1(n_trees):
    # simple pipeline, same as the common part
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    return b.build({'X': X, 'y': y}, {'output': rf_2})


def get_pipeline_2(n_trees):
    # adding just two blocks in the end, results must be the same
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    stack = b.Stack(['rf_2_1', 'rf_2_2'], axis=2)(rf_2_1=rf_2, rf_2_2=rf_2)
    av = b.Average(axis=2)(X=stack)
    return b.build({'X': X, 'y': y}, {'output': av})


def get_pipeline_3(n_trees):
    # little bit wrong definition of the common part,
    # but it must work
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    et_2 = b.ETC(n_estimators=n_trees)(X=concat_1, y=y)
    concat_2 = b.Concat(['rf_2', 'et_2', 'X'])(X=X, rf_2=rf_2, et_2=et_2)
    et_3 = b.ETC(n_estimators=n_trees)(X=concat_2, y=y)
    return b.build({'X': X, 'y': y}, {'output': et_3})


def get_pipeline_4(n_trees):
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    rf_1 = b.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    concat_2 = b.Concat(['rf_2', 'X'])(X=X, rf_2=rf_2, )
    et_3 = b.ETC(n_estimators=n_trees)(X=concat_2, y=y)
    return b.build({'X': X, 'y': y}, {'output': et_3})


def get_pipelines(n_trees=10):
    pip_1 = get_pipeline_1(n_trees)
    common_part = get_pipeline_1(n_trees)
    pip_2 = get_pipeline_2(n_trees)
    pip_3 = get_pipeline_3(n_trees)
    return common_part, [pip_1, pip_2, pip_3]


def my_acc(y_true, y_pred):
    return accuracy_score(y_true, np.int_(y_pred[:, 1]))


def my_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, np.int_(y_pred[:, 1]))


def comparison_cv_basic_test():
    log_test_name()
    random_state = 42
    common_part, pipelines = get_pipelines()
    models = [RFCModel(), CatBoostModel()]
    cv_strat = KFold(shuffle=True, n_splits=3)
    comparator = CVComparator(pipelines, common_part, models, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    models_keys = [f'pipeline_{i}' for i in range(len(pipelines))]
    models_keys += [f'model_{i}' for i in range(len(models))]
    assert all([key in cv_res for key in models_keys]), \
        "Not all the models are in the comparison result"
    for model_name, nested_dict in cv_res.items():
        for key, val in nested_dict.items():
            assert len(val) == cv_strat.get_n_splits(), \
                f"Not all the folds were proceeded for the {model_name} model"
    logging.info("CV comparator done the work, following results were retrieved")
    for model_name, nested_dict in cv_res.items():
        logging.info(model_name)
        for key, val in nested_dict.items():
            logging.info(f'\t{key}: {val}')


def no_pipelines_test():
    log_test_name()
    random_state = 42
    models = [RFCModel(), CatBoostModel()]
    cv_strat = KFold(shuffle=True, n_splits=3)
    comparator = CVComparator(None, None, models, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    models_keys = [f'model_{i}' for i in range(len(models))]
    assert all([key in cv_res for key in models_keys]), \
        "Not all the models are in the comparison result"
    for model_name, nested_dict in cv_res.items():
        for key, val in nested_dict.items():
            assert len(val) == cv_strat.get_n_splits(), \
                f"Not all the folds were proceeded for the {model_name} model"
    logging.info("CV comparator done the work, following results were retrieved")
    for model_name, nested_dict in cv_res.items():
        logging.info(model_name)
        for key, val in nested_dict.items():
            logging.info(f'\t{key}: {val}')


def no_foreign_models_test():
    log_test_name()
    random_state = 42
    common_part, pipelines = get_pipelines()
    cv_strat = KFold(shuffle=True, n_splits=3)
    comparator = CVComparator(pipelines, common_part, None, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    models_keys = [f'pipeline_{i}' for i in range(len(pipelines))]
    assert all([key in cv_res for key in models_keys]), \
        "Not all the models are in the comparison result"
    for model_name, nested_dict in cv_res.items():
        for key, val in nested_dict.items():
            assert len(val) == cv_strat.get_n_splits(), \
                f"Not all the folds were proceeded for the {model_name} model"
    logging.info("CV comparator done the work, following results were retrieved")
    for model_name, nested_dict in cv_res.items():
        logging.info(model_name)
        for key, val in nested_dict.items():
            logging.info(f'\t{key}: {val}')


def get_optim_test_pipelines(n_trees=13):
    pip_1 = get_pipeline_1(n_trees)
    pip_2 = get_pipeline_2(n_trees)
    pip_3 = get_pipeline_3(n_trees)
    common_part = get_pipeline_1(n_trees)
    return common_part, [pip_1, pip_2, pip_3]


def get_unoptim_res(random_state):
    _, pipelines = get_pipelines()
    cv_strat = KFold(shuffle=True, n_splits=5)
    comparator = CVComparator(pipelines, None, None, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    return cv_res


def get_optim_res(random_state):
    common_part, pipelines = get_pipelines()
    cv_strat = KFold(shuffle=True, n_splits=5)
    comparator = CVComparator(pipelines, common_part, None, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    return cv_res


def optimization_test():
    log_test_name()
    random_state = 42
    tol = 1e-6
    time_stamp = time()
    unoptim_res = get_unoptim_res(random_state)
    unoptim_time = time() - time_stamp
    time_stamp = time()
    optim_res = get_optim_res(random_state)
    optim_time = time() - time_stamp

    for model_name, nested_dict in unoptim_res.items():
        assert model_name in optim_res, "Different results were retrieved"
        for key, val in nested_dict.items():
            assert key in optim_res[model_name], "Different results were retrieved"
            assert len(val) == len(optim_res[model_name][key]), "Different results were retrieved"
            assert all(abs(val[i] - optim_res[model_name][key][i]) < tol for i in range(len(val))), \
                "Different results were retrieved"

    assert optim_time < unoptim_time, "Optimization was useless"
    logging.info('Time of unoptimized run: %f s.', unoptim_time)
    logging.info('Time of optimized run: %f s.', optim_time)
