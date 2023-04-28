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
from typing import Dict
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


def get_pipelines(n_trees=10):
    # simple pipeline, same as the common part
    b_1 = FunctionalPipelineBuilder()
    X, y = b_1.Input()(), b_1.TargetInput()()
    rf_1 = b_1.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_1.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_1.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_1.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    pip_1 = b_1.build({'X': X, 'y': y}, {'output': rf_2})

    # for making different objects
    b_c = FunctionalPipelineBuilder()
    X, y = b_c.Input()(), b_c.TargetInput()()
    rf_1 = b_c.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_c.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_c.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_c.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    common_part = b_c.build({'X': X, 'y': y}, {'output': rf_2})

    # adding just two blocks in the end, results must be the same
    b_2 = FunctionalPipelineBuilder()
    X, y = b_2.Input()(), b_2.TargetInput()()
    rf_1 = b_2.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_2.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_2.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_2.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    stack = b_2.Stack(['rf_2_1', 'rf_2_2'], axis=2)(rf_2_1=rf_2, rf_2_2=rf_2)
    av = b_2.Average(axis=2)(X=stack)
    pip_2 = b_2.build({'X': X, 'y': y}, {'output': av})

    # adding more forests
    b_3 = FunctionalPipelineBuilder()
    X, y = b_3.Input()(), b_3.TargetInput()()
    rf_1 = b_3.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_3.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_3.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_3.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    et_2 = b_3.ETC(n_estimators=n_trees)(X=concat_1, y=y)
    concat_2 = b_3.Concat(['rf_2', 'et_2', 'X'])(X=X, rf_2=rf_2, et_2=et_2)
    et_3 = b_3.ETC(n_estimators=n_trees)(X=concat_2, y=y)
    pip_3 = b_3.build({'X': X, 'y': y}, {'output': et_3})

    return common_part, [pip_1, pip_2, pip_3]


def my_acc(y_true, y_pred):
    return accuracy_score(y_true, np.int_(y_pred[:, 1]))


def my_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, np.int_(y_pred[:, 1]))


def main():
    logging.basicConfig(level=logging.INFO)
    random_state = 42
    common_part, pipelines = get_pipelines()
    models = [RFCModel(), CatBoostModel()]
    comparator = CVComparator(pipelines, models,
                              KFold(shuffle=True, n_splits=3), random_state=random_state)
    x, y = make_moons(noise=0.5, random_state=random_state)
    data = {
        'X': CPUData(x),
        'y': CPUData(y),
    }
    metrics = [MetricWrapper(my_acc, name='accuracy'), MetricWrapper(my_roc_auc, name='roc_auc')]
    cv_res = comparator.get_score(data, metrics)
    print(cv_res.to_string())


if __name__ == '__main__':
    main()
