from bosk.block.zoo.models.classification.ferns import RandomFernsBlock
from bosk.comparison.cross_val import CVComparator
from bosk.comparison.metric import MetricWrapper
from bosk.executor.parallel.greedy import GreedyParallelExecutor
from bosk.executor.topological import TopologicalExecutor
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.data import CPUData
from bosk.stages import Stage
from sklearn.calibration import LabelEncoder
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import logging
from pathlib import Path


def make_mgs_forest_pipeline(image_shape: tuple):
    b = FunctionalPipelineBuilder()
    X = b.Input(name='X')()
    y = b.TargetInput(name='y')()
    X = b.Reshape((-1, 1, *image_shape))(X)
    X = b.FitLambda(
        function=(lambda inp: print("X shape:", inp['X'].data.shape)),
        inputs=['X']
    )(X=X)
    mgs = b.MultiGrainedScanningND(
        model=RandomForestClassifier(),
        kernel_size=3,
        stride=3,
        dilation=1,
        padding=None
    )(X=X, y=y)
    mgs = b.FitLambda(
        function=(lambda inp: print("mgs shape:", inp['mgs'].data.shape)),
        inputs=['mgs']
    )(mgs=mgs)
    global_pooling = b.GlobalAveragePooling()(mgs)
    global_pooling = b.FitLambda(
        function=(lambda inp: print("global pooling shape:", inp['global_pooling'].data.shape)),
        inputs=['global_pooling']
    )(global_pooling=global_pooling)
    b.Output(name='proba')(global_pooling)
    return b.build()


def make_mgs_pipeline(image_shape: tuple):
    b = FunctionalPipelineBuilder()
    X = b.Input(name='X')()
    y = b.TargetInput(name='y')()
    X = b.Reshape((-1, 1, *image_shape))(X)
    X = b.FitLambda(
        function=(lambda inp: print("X shape:", inp['X'].data.shape)),
        inputs=['X']
    )(X=X)
    mgs = b.MGSRandomFerns(
        n_groups=20,
        n_ferns_in_group=5,
        fern_size=15,
        kind='unary',
        bootstrap=True,
        n_jobs=-1,
        kernel_size=3,
        stride=3,
        dilation=1,
        padding=None
    )(X=X, y=y)
    mgs = b.FitLambda(
        function=(lambda inp: print("mgs shape:", inp['mgs'].data.shape)),
        inputs=['mgs']
    )(mgs=mgs)
    global_pooling = b.GlobalAveragePooling()(mgs)
    global_pooling = b.FitLambda(
        function=(lambda inp: print("global pooling shape:", inp['global_pooling'].data.shape)),
        inputs=['global_pooling']
    )(global_pooling=global_pooling)
    b.Output(name='proba')(global_pooling)
    return b.build()


def make_mgs_ferns_pipeline(image_shape: tuple):
    b = FunctionalPipelineBuilder()
    X = b.Input(name='X')()
    y = b.TargetInput(name='y')()
    X = b.Reshape((-1, 1, *image_shape))(X)
    X = b.FitLambda(
        function=(lambda inp: print("X shape:", inp['X'].data.shape)),
        inputs=['X']
    )(X=X)
    mgs = b.MultiGrainedScanningND(
        model=RandomFernsBlock(
            n_groups=20,
            n_ferns_in_group=5,
            fern_size=15,
            kind='unary',
            bootstrap=True,
            n_jobs=-1,
            random_state=12345
        ),
        kernel_size=3,
        stride=3,
        dilation=1,
        padding=None
    )(X=X, y=y)
    mgs = b.FitLambda(
        function=(lambda inp: print("mgs shape:", inp['mgs'].data.shape)),
        inputs=['mgs']
    )(mgs=mgs)
    global_pooling = b.GlobalAveragePooling()(mgs)
    global_pooling = b.FitLambda(
        function=(lambda inp: print("global pooling shape:", inp['global_pooling'].data.shape)),
        inputs=['global_pooling']
    )(global_pooling=global_pooling)
    b.Output(name='proba')(global_pooling)
    return b.build()


def roc_auc_metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, multi_class='ovr')


def main():
    logging.basicConfig(level=logging.INFO)
    results_path = Path(__file__).parent.absolute() / 'results'
    results_path.mkdir(parents=True, exist_ok=True)

    random_state = 42
    pipelines = dict()

    logging.info('Loding MNIST dataset')
    mnist_X, mnist_y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
    mnist_y = LabelEncoder().fit_transform(mnist_y)
    logging.info('MNIST dataset loaded')
    sample_slice = slice(0, 5000)
    mnist_X = mnist_X[sample_slice]
    mnist_y = mnist_y[sample_slice]
    image_shape = (28, 28)

    datasets = {
        'mnist': {
            'X': CPUData(mnist_X),
            'y': CPUData(mnist_y),
        },
    }
    print('Automatically construct the pipeline ')

    mgs_ferns_pipeline = make_mgs_pipeline(image_shape)
    print('Warm up')
    fit_executor = TopologicalExecutor(mgs_ferns_pipeline, stage=Stage.FIT)
    fit_executor(datasets['mnist'])

    mgs_forest_pipeline = make_mgs_forest_pipeline(image_shape)
    mgs_plus_ferns_pipeline = make_mgs_ferns_pipeline(image_shape)

    pipelines['deep forest (mgs ferns)'] = mgs_ferns_pipeline
    pipelines['deep forest (mgs forest)'] = mgs_forest_pipeline
    pipelines['deep forest (mgs + ferns)'] = mgs_plus_ferns_pipeline

    models = dict()
    foreign_model_names = {
        f'model {i}': name
        for i, name in enumerate(models.keys())
    }
    comparator = CVComparator(list(pipelines.values()), list(models.values()),
                              KFold(shuffle=True, n_splits=2),
                              f_optimize_pipelines=False,
                              exec_cls=GreedyParallelExecutor,
                              random_state=random_state)

    metrics = [
        MetricWrapper(roc_auc_metric, name='roc_auc', y_pred_name='proba'),
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
    agg_over_folds.columns = ['_'.join(col) if len(col[1]) > 0 else col[0] for col in agg_over_folds.columns]  # droplevel with concat
    agg_over_folds.loc[:, 'value'] = agg_over_folds[['value_mean', 'value_std']].apply(
        lambda ms: f'{ms[0]:.3f} ± {ms[1]:.3f}',
        axis=1
    )

    print('Training performance:')
    perf_pivot = pd.pivot(
        agg_over_folds.query('`train/test` == "train" and metric == "time"'),
        index=['dataset', 'model name'],
        columns='metric',
        values=['value']
    )
    perf_pivot.to_html(results_path / 'image_performance.html')
    print(perf_pivot)


if __name__ == '__main__':
    main()
