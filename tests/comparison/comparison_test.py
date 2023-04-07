from bosk.comparison.cross_val import CVComparator
from bosk.comparison.base import BaseForeignModel
from bosk.comparison.metric import MetricWrapper
from bosk.pipeline import BasePipeline
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.executor.topological import TopologicalExecutor
from bosk.executor.timer import TimerBlockHandler
from bosk.stages import Stage
from bosk.data import CPUData, BaseData
from bosk.utility import timer_wrap
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np
import random
from typing import Dict
from ..utility import log_test_name
import logging
import warnings


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
    models_num = len(pipelines) + len(models)
    assert len(set(cv_res.loc[:, 'model name'])) == models_num, \
        "Not all the models are in the comparison result"
    assert cv_res.loc[:, 'fold #'].max() + 1 == cv_strat.get_n_splits(), \
        "Not all the folds were proceeded"


def shuffle_test():
    # same as the basic test, but with shuffled
    # blocks and connections
    log_test_name()
    random_state = 42
    random.seed(random_state)
    common_part, pipelines = get_pipelines()
    cp_nodes = common_part.nodes
    cp_conns = common_part.connections
    random.shuffle(cp_nodes)
    random.shuffle(cp_conns)
    common_part = BasePipeline(cp_nodes, cp_conns, common_part.inputs, common_part.outputs)
    for i in range(len(pipelines)):
        pip = pipelines[i]
        pip_nodes = pip.nodes
        pip_conns = pip.connections
        random.shuffle(pip_nodes)
        random.shuffle(pip_conns)
        pipelines[i] = BasePipeline(pip_nodes, pip_conns, pip.inputs, pip.outputs)
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
    models_num = len(pipelines) + len(models)
    assert len(set(cv_res.loc[:, 'model name'])) == models_num, \
        "Not all the models are in the comparison result"
    assert cv_res.loc[:, 'fold #'].max() + 1 == cv_strat.get_n_splits(), \
        "Not all the folds were proceeded"


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
    models_num = len(models)
    assert len(set(cv_res.loc[:, 'model name'])) == models_num, \
        "Not all the models are in the comparison result"
    assert cv_res.loc[:, 'fold #'].max() + 1 == cv_strat.get_n_splits(), \
        "Not all the folds were proceeded"


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
    models_num = len(pipelines)
    assert len(set(cv_res.loc[:, 'model name'])) == models_num, \
        "Not all the models are in the comparison result"
    assert cv_res.loc[:, 'fold #'].max() + 1 == cv_strat.get_n_splits(), \
        "Not all the folds were proceeded"


def get_optim_test_pipelines(n_trees=13):
    pip_1 = get_pipeline_1(n_trees)
    pip_2 = get_pipeline_2(n_trees)
    pip_3 = get_pipeline_4(n_trees)
    common_part = get_pipeline_1(n_trees)
    return common_part, [pip_1, pip_2, pip_3]


@timer_wrap
def get_unoptim_res(random_state):
    _, pipelines = get_optim_test_pipelines()
    cv_strat = KFold(shuffle=True, n_splits=5)
    comparator = CVComparator(pipelines, None, None, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    return cv_res, comparator


@timer_wrap
def get_optim_res(random_state):
    common_part, pipelines = get_optim_test_pipelines()
    cv_strat = KFold(shuffle=True, n_splits=5)
    comparator = CVComparator(pipelines, common_part, None, cv_strat, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    return cv_res, comparator


def optimization_test():
    log_test_name()
    random_state = 42
    tol = 1e-6
    (unoptim_res, unopt_cmp), unoptim_time = get_unoptim_res(random_state)
    (optim_res, opt_cmp), optim_time = get_optim_res(random_state)
    diff = unoptim_res.compare(optim_res, 0)
    if len(diff.columns) > 1 or 'time' not in diff.columns:
        numerical_diff = diff.select_dtypes(include=np.number)
        assert numerical_diff.columns == diff.columns,\
            "Different results for non-numerical data were retrieved"
        reason_diff = diff.loc[::2, diff.columns != 'time'].to_numpy()\
            - diff.loc[1::2, diff.columns != 'time'].to_numpy()
        assert np.sum(reason_diff) < tol, "Different results were retrieved"
    _, pipelines = get_optim_test_pipelines()
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        common_res = TopologicalExecutor(opt_cmp._common_pipeline, Stage.FIT)(data)
    for true_pipeline, unopt_pipeline, opt_pipeline, extra_blocks in zip(
        pipelines,
        unopt_cmp._optim_pipelines,
        opt_cmp._optim_pipelines,
        opt_cmp._new_blocks_list):
        true_bl_ex = TimerBlockHandler()
        unopt_bl_ex = TimerBlockHandler()
        opt_bl_ex = TimerBlockHandler()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            TopologicalExecutor(true_pipeline, Stage.FIT, block_executor=true_bl_ex)(data)
            TopologicalExecutor(unopt_pipeline, Stage.FIT, block_executor=unopt_bl_ex)(data)
            opt_input = opt_cmp._get_pers_inp_dict(opt_pipeline, common_res, data)
            TopologicalExecutor(opt_pipeline, Stage.FIT, block_executor=opt_bl_ex)(opt_input)
            true_proc_num = len(true_bl_ex.blocks_time)
            unopt_proc_num = len(unopt_bl_ex.blocks_time)
            opt_proc_num = len(opt_bl_ex.blocks_time)
            assert true_proc_num == unopt_proc_num, \
                "Number of proceeded blocks in the original pipeline must " + \
                "be equal to the ones in the unoptimized pipeline"
            assert opt_proc_num - len(extra_blocks) < true_proc_num, \
                "Number of proceeded blocks in the optimized pipeline must" + \
                "be less than the ones in the original pipeline"

    logging.info('Time of unoptimized run: %f s.', unoptim_time)
    logging.info('Time of optimized run: %f s.', optim_time)


def blocks_times_test():
    log_test_name()
    random_state = 42
    common_part, pipelines = get_pipelines()
    models = [RFCModel(), CatBoostModel()]
    cv_strat = KFold(shuffle=True, n_splits=3)
    comparator = CVComparator(pipelines, common_part, models, cv_strat,
                              get_blocks_times=True, random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    for i, pipeline in enumerate(pipelines):
        fit_bl_ex = TimerBlockHandler()
        tf_bl_ex = TimerBlockHandler()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            TopologicalExecutor(pipeline, Stage.FIT, block_executor=fit_bl_ex)(data)
            TopologicalExecutor(pipeline, Stage.TRANSFORM, block_executor=tf_bl_ex)(data)
        true_fit_blocks_time = fit_bl_ex.blocks_time
        true_tf_blocks_time = tf_bl_ex.blocks_time
        sub_df = cv_res.loc[cv_res.loc[:, 'model name'] ==
                            f'deep forest {i}', ['train/test', 'time', 'blocks time']]
        blocks_time_dict_list = sub_df.loc[:, 'blocks time']
        for idx, time_dict in blocks_time_dict_list.items():
            if sub_df.loc[idx, 'train/test'] == 'train':
                pip_blocks = set(true_fit_blocks_time.keys())
            else:
                pip_blocks = set(true_tf_blocks_time.keys())
            assert set(time_dict.keys()) == pip_blocks, \
                "Blocks, presented in the profiling result, do not match with the ones in the pipeline"
            assert sum(time_dict.values()) < sub_df.loc[idx, 'time'], \
                "Sum of blocks' times must be smaller than total execution time"
    for i in range(len(models)):
        model_blocks_time = cv_res.loc[cv_res.loc[:, 'model name'] == f'model {i}', 'blocks time']
        assert all(model_blocks_time.apply(lambda x: x == None)), \
            "For foreign models blocks times must be None"
