from bosk.auto.deep_forest import ClassicalDeepForestConstructor
from bosk.block.zoo.models.classification.classification_models import CatBoostClassifierBlock, XGBClassifierBlock
from bosk.comparison.cross_val import CVComparator
from bosk.comparison.base import BaseForeignModel
from bosk.comparison.metric import MetricWrapper
from bosk.executor.parallel.greedy import GreedyParallelExecutor
from bosk.painter.topological import TopologicalPainter
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.data import CPUData, BaseData
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from typing import Dict
import logging
from functools import partial
from pathlib import Path


def make_foreign_model(make_base_estimator: callable, name: str) -> BaseForeignModel:
    """Make a foreign model wrapper for the given estimator.

    Args:
        make_base_estimator: A function that returns the estimator.
        name: Desired class name.

    Returns:
        A foreign model wrapper.
    """
    class ForeignModelCls(BaseForeignModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.forest = make_base_estimator(*args, **kwargs)

        def fit(self, data: Dict[str, BaseData]) -> None:
            self.forest.fit(data['X'].to_cpu().data,
                            data['y'].to_cpu().data)

        def predict(self, data: Dict[str, BaseData]) -> Dict[str, BaseData]:
            return {'proba': CPUData(self.forest.predict_proba(data['X'].to_cpu().data))}

        def set_random_state(self, random_state: int) -> None:
            self.forest.random_state = random_state

    ForeignModelCls.__name__ = name
    return ForeignModelCls


RFCModel = make_foreign_model(RandomForestClassifier, 'RFCModel')
CatBoostModel = make_foreign_model(partial(CatBoostClassifier, verbose=0), 'CatBoostModel')
XGBoostModel = make_foreign_model(partial(XGBClassifier), 'XGBoostModel')
MLPModel = make_foreign_model(MLPClassifier, 'MLPModel')


def get_pipelines(n_trees=10):
    # simple pipeline, same as the common part
    b_1 = FunctionalPipelineBuilder()
    X, y = b_1.Input()(), b_1.TargetInput()()
    rf_1 = b_1.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_1.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_1.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_1.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    pip_1 = b_1.build({'X': X, 'y': y}, {'proba': rf_2})

    # for making different objects
    b_c = FunctionalPipelineBuilder()
    X, y = b_c.Input()(), b_c.TargetInput()()
    rf_1 = b_c.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_c.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_c.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_c.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    common_part = b_c.build({'X': X, 'y': y}, {'proba': rf_2})

    # adding just two blocks in the end, results must be the same
    b_2 = FunctionalPipelineBuilder()
    X, y = b_2.Input()(), b_2.TargetInput()()
    rf_1 = b_2.RFC(n_estimators=n_trees)(X=X, y=y)
    et_1 = b_2.ETC(n_estimators=n_trees)(X=X, y=y)
    concat_1 = b_2.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b_2.RFC(n_estimators=n_trees)(X=concat_1, y=y)
    stack = b_2.Stack(['rf_2_1', 'rf_2_2'], axis=2)(rf_2_1=rf_2, rf_2_2=rf_2)
    av = b_2.Average(axis=2)(X=stack)
    pip_2 = b_2.build({'X': X, 'y': y}, {'proba': av})

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
    pip_3 = b_3.build({'X': X, 'y': y}, {'proba': et_3})

    return common_part, [pip_1, pip_2, pip_3]


def accuracy_metric(y_true, y_pred):
    return accuracy_score(y_true, np.argmax(y_pred, axis=1))


def roc_auc_metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred[:, 1])


def f1_metric(y_true, y_pred):
    return f1_score(y_true, np.argmax(y_pred, axis=1))


def main():
    logging.basicConfig(level=logging.INFO)
    results_path = Path(__file__).parent.absolute() / 'results'
    results_path.mkdir(parents=True, exist_ok=True)

    random_state = 42
    common_part, pipelines_list = get_pipelines()
    pipelines = {
        f'deep forest (tiny #{i})': pipeline
        for i, pipeline in enumerate(pipelines_list)
    }

    moons_x, moons_y = make_moons(n_samples=100, noise=0.1, random_state=12345)
    rng = np.random.RandomState(12345)
    moons_x = np.concatenate((moons_x, rng.normal(size=(moons_x.shape[0], 8))), axis=1)
    bc_x, bc_y = load_breast_cancer(return_X_y=True)
    datasets = {
        'breast_cancer': {
            'X': CPUData(bc_x),
            'y': CPUData(bc_y),
        },
        'moons': {
            'X': CPUData(moons_x),
            'y': CPUData(moons_y),
        }
    }
    print('Automatically construct the pipeline ')
    constructor = ClassicalDeepForestConstructor(
        GreedyParallelExecutor,
        rf_params=dict(),
        max_iter=1,
        layer_width=8,
        cv=None,
        random_state=12345,
    )
    pipeline = constructor.construct(bc_x, bc_y)

    constructor = ClassicalDeepForestConstructor(
        GreedyParallelExecutor,
        rf_params=dict(),
        max_iter=1,
        layer_width=16,
        cv=None,
        random_state=12345,
    )
    pipeline_2 = constructor.construct(bc_x, bc_y)

    constructor = ClassicalDeepForestConstructor(
        GreedyParallelExecutor,
        rf_params=dict(),
        max_iter=2,
        layer_width=16,
        cv=None,
        random_state=12345,
    )
    pipeline_3 = constructor.construct(bc_x, bc_y)

    constructor = ClassicalDeepForestConstructor(
        GreedyParallelExecutor,
        rf_params=dict(
            verbose=0,
            random_state=0,  # initialize random state field
        ),
        max_iter=1,
        layer_width=4,
        cv=None,
        block_classes=(CatBoostClassifierBlock, XGBClassifierBlock),
        random_state=12345,
    )
    pipeline_4 = constructor.construct(bc_x, bc_y)

    print('Pipelines constructed')

    painter = TopologicalPainter()
    painter.from_pipeline(pipeline_3)
    painter.render('auto_pipeline', 'png')

    pipelines['deep forest (small)'] = pipeline
    pipelines['deep forest (medium)'] = pipeline_2
    pipelines['deep forest (large)'] = pipeline_3
    pipelines['deep forest (catboost)'] = pipeline_4

    models = {
        'random_forest': RFCModel(),
        'catboost': CatBoostModel(),
        'xgboost': XGBoostModel(),
        'mlp#1': MLPModel([8] * 3),
        'mlp#2': MLPModel([32] * 3),
        'mlp#3': MLPModel([64] * 3),
    }
    foreign_model_names = {
        f'model {i}': name
        for i, name in enumerate(models.keys())
    }
    comparator = CVComparator(list(pipelines.values()), list(models.values()),
                              KFold(shuffle=True, n_splits=5),
                              f_optimize_pipelines=True,
                              exec_cls=GreedyParallelExecutor,
                              random_state=random_state)

    metrics = [
        MetricWrapper(accuracy_metric, name='accuracy', y_pred_name='proba'),
        MetricWrapper(roc_auc_metric, name='roc_auc', y_pred_name='proba'),
        MetricWrapper(f1_metric, name='f1', y_pred_name='proba'),
    ]
    cv_res = dict()
    for name, data in datasets.items():
        cv_res[name] = comparator.get_score(data, metrics)
        cv_res[name].loc[:, 'dataset'] = name
    cv_res = pd.concat(cv_res.values(), ignore_index=True).reset_index()

    model_names = {
        f'deep forest {i}': name
        for i, name in enumerate(pipelines.keys())
    }
    model_names.update(foreign_model_names)
    cv_res.loc[:, 'model name'] = cv_res.loc[:, 'model name'].apply(lambda n: model_names.get(n, n))

    measurement_names = ['time'] + [m.name for m in metrics]
    melted = pd.melt(
        cv_res,
        id_vars=set(cv_res.columns).difference(measurement_names),
        value_vars=measurement_names,
        var_name='metric'
    ).drop('index', axis=1)
    by = set(melted.columns).difference(['value', 'fold #'])
    agg_over_folds = melted.groupby(by=list(by)).agg({'value': ['mean', 'std']}).reset_index()
    agg_over_folds.columns = [
        '_'.join(col) if len(col[1]) > 0 else col[0] for col in agg_over_folds.columns
    ]  # droplevel with concat
    agg_over_folds.loc[:, 'value'] = agg_over_folds[['value_mean', 'value_std']].apply(
        lambda ms: f'{ms[0]:.3f} ± {ms[1]:.3f}',
        axis=1
    )
    agg_over_folds.to_csv(results_path / 'cv_all_results.csv')

    print('CV test metrics:')
    test_set_pivot = pd.pivot(
        agg_over_folds.query('`train/test` == "test" and metric != "time"'),
        index=['dataset', 'model name'],
        columns=['metric'],
        values=['value']
    )
    print(test_set_pivot)
    test_set_pivot.to_html(results_path / 'cv_test_metrics.html')

    print('Training performance:')
    perf_pivot = pd.pivot(
        agg_over_folds.query('`train/test` == "train" and metric == "time"'),
        index=['dataset', 'model name'],
        columns='metric',
        values=['value']
    )
    perf_pivot.to_html(results_path / 'cv_train_performance.html')
    print(perf_pivot)


if __name__ == '__main__':
    main()
