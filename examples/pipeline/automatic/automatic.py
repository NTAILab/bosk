import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from bosk.auto.deep_forest import ClassicalDeepForestConstructor
from bosk.data import CPUData
from bosk.executor.topological import TopologicalExecutor
from bosk.painter.topological import TopologicalPainter
from bosk.executor.sklearn_interface import BoskPipelineClassifier


def make_fit_model(X: np.ndarray, y: np.ndarray):
    X = CPUData(X)
    y = CPUData(y)

    constructor = ClassicalDeepForestConstructor(
        TopologicalExecutor,
        rf_params=dict(),
        max_iter=10,
        layer_width=2,
        cv=2,
        random_state=12345,
    )
    pipeline = constructor.construct(X.data, y.data)

    print(pd.DataFrame(constructor.history))

    # make a scikit-learn model
    model = BoskPipelineClassifier(pipeline, executor_cls=TopologicalExecutor)
    model._classifier_init(y.data)
    return model


def main():
    all_X, all_y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, random_state=12345)
    # fit the model
    model = make_fit_model(X_train, y_train)
    # predict with the model
    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)
    print('Test f1 score:', f1_score(y_test, test_preds, average='macro'))
    print('Test roc_auc score:', roc_auc_score(y_test, test_proba, multi_class='ovr'))
    # draw
    painter = TopologicalPainter()
    painter.from_pipeline(model.pipeline)
    filename = 'deep_forest'
    painter.render(filename, 'png')
    # retraining
    print('Retrain')
    model.fit(X_train, y_train)
    print('Predict')
    test_preds = model.predict(X_test)
    print('Predict proba')
    test_proba = model.predict_proba(X_test)
    print('Test f1 score:', f1_score(y_test, test_preds, average='macro'))
    print('Test roc_auc score:', roc_auc_score(y_test, test_proba, multi_class='ovr'))


if __name__ == '__main__':
    main()
