import numpy as np
from typing import Tuple
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from inspect import getsourcelines


def without_bosk_framework(data: Tuple[np.ndarray, ...]):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.model_selection import StratifiedKFold


    class TwoLayerDeepForest(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.first_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=12345)
            self.first_layer = [
                ExtraTreesClassifier(random_state=1),
                RandomForestClassifier(random_state=2),
            ]
            self.second_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            self.second_layer = [
                ExtraTreesClassifier(random_state=3),
                RandomForestClassifier(random_state=4),
            ]
            self.final_et = ExtraTreesClassifier(random_state=5)
            first_predictions = []
            for model, (train_ids, _test_ids) in zip(self.first_layer, self.first_kfold.split(X, y)):
                model.fit(X[train_ids], y[train_ids])
                first_predictions.append(model.predict_proba(X))
            first_feature_vector = np.concatenate([X] + first_predictions, axis=1)
            second_predictions = []
            for model, (train_ids, _test_ids) in zip(self.second_layer, self.second_kfold.split(X, y)):
                model.fit(first_feature_vector[train_ids], y[train_ids])
                second_predictions.append(model.predict_proba(first_feature_vector))
            second_feature_vector = np.concatenate([X] + second_predictions, axis=1)
            self.final_et.fit(second_feature_vector, y)
            return self

        def predict_proba(self, X):
            first_predictions = []
            for model in self.first_layer:
                first_predictions.append(model.predict_proba(X))
            first_feature_vector = np.concatenate([X] + first_predictions, axis=1)
            second_predictions = []
            for model in self.second_layer:
                second_predictions.append(model.predict_proba(first_feature_vector))
            second_feature_vector = np.concatenate([X] + second_predictions, axis=1)
            return self.final_et.predict_proba(second_feature_vector)


    classifier = TwoLayerDeepForest()
    X_train, X_test, y_train, y_test = data
    classifier.fit(X_train, y_train)
    return classifier.predict_proba(X_test)


def with_bosk_framework(data: Tuple[np.ndarray, ...]):
    from bosk.pipeline.builder import FunctionalPipelineBuilder
    from bosk.executor.sklearn_interface import BoskPipelineClassifier
    from bosk.block.zoo.models.classification import RFCBlock, ETCBlock


    b = FunctionalPipelineBuilder()
    X = b.Input('X')()
    y = b.TargetInput('y')()
    cv_1 = b.CVTrainIndices(2, 12345)(X=X, y=y)
    et_1 = b.SubsetTrainWrapper(ETCBlock(random_state=1))(X=X, y=y, training_indices=cv_1['0'])
    rf_1 = b.SubsetTrainWrapper(RFCBlock(random_state=2))(X=X, y=y, training_indices=cv_1['1'])
    features = b.Concat(['X', 'et', 'rf'])(rf=rf_1, et=et_1, X=X)
    cv_2 = b.CVTrainIndices(2, 42)(X=X, y=y)
    et_2 = b.SubsetTrainWrapper(ETCBlock(random_state=3))(X=features, y=y, training_indices=cv_2['0'])
    rf_2 = b.SubsetTrainWrapper(RFCBlock(random_state=4))(X=features, y=y, training_indices=cv_2['1'])
    final_et = b.ETC(random_state=5)(X=b.Concat(['X', 'et', 'rf'])(rf=rf_2, et=et_2, X=X), y=y)
    b.Output('proba')(final_et)
    pipeline = b.build()
    classifier = BoskPipelineClassifier(pipeline)
    X_train, X_test, y_train, y_test = data
    classifier.fit(X_train, y_train)
    return classifier.predict_proba(X_test)


def main():
    data = train_test_split(*make_moons(random_state=12345), random_state=42)
    without_bosk_result = without_bosk_framework(data)
    with_bosk_result = with_bosk_framework(data)
    assert np.allclose(without_bosk_result, with_bosk_result)


def get_code_size(fn) -> Tuple[int, int]:
    lines = getsourcelines(fn)[0]
    n_lines = sum(1 if len(line.strip()) > 0 else 0 for line in lines)
    n_chars = sum(len(line.strip()) for line in lines)
    return dict(lines=n_lines, chars=n_chars)


def code_size():
    sizes = {
        'without': get_code_size(without_bosk_framework),
        'with Bosk': get_code_size(with_bosk_framework),
    }
    for name, size in sizes.items():
        print(name, size)
    for k, v in sizes['without'].items():
        percent = sizes['with Bosk'][k] / v * 100.0
        print(
            f"{k}:",
            "bosk / (without bosk) = {:.0f}%;".format(percent),
            "code shortage = {:.0f}%".format(100.0 - percent)
        )


if __name__ == "__main__":
    main()
    code_size()
