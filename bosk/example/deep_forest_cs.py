from abc import ABC, abstractmethod
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.naive import NaiveExecutor
from bosk.stages import Stage
from bosk.block import auto_block
from bosk.data import Data


# Currently, all parameters set by user
# Not cross-validation approach


@dataclass(frozen=True)
class TransferCascadeForestData:
    samples: Data
    labels: Optional[Data] = None
    prediction_matrix: Optional[np.ndarray] = None
    original_feature_data: Optional[Data] = None



class BaseModel(ABC):
    """Base class for all deep forests modules."""
    @abstractmethod
    def fit(self, input_data: TransferCascadeForestData):
        pass

    @abstractmethod
    def transform(self, input_data: TransferCascadeForestData):
        pass


class ConfidenceScreening:
    def __init__(self, eps: float):
        self._eps = eps

    def __call__(self, proba: np.ndarray, labels: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
        """Returns matrix with final predictions"""
        x = np.argmax(proba, axis=1)
        for i in range(proba.shape[0]):
            if prediction_matrix[i] is None:
                if proba[i][x[i]] > self._eps:
                    prediction_matrix[i] = x[i]
        return prediction_matrix


# @auto_block
# class CascadeRandomForestsBlock:  #Without Confidence Screening
#     pass


@auto_block
class CascadeRandomForestsCFBlock:
    def __init__(self, eps: float = 0.5, count_estimators: int = 5, count_forest: int = 5):
        self._count_estimators: int = count_estimators
        self._count_forest: int = count_forest
        self._forests: defaultdict = defaultdict(Dict[str, RandomForestClassifier])
        self._confidence_screening: ConfidenceScreening= ConfidenceScreening(eps)
        self._prediction_matrix: Optional[np.ndarray] = None

    def fit(self, input_data: TransferCascadeForestData):
        assert input_data.labels is not None
        if input_data.prediction_matrix is None:
            self._prediction_matrix = np.full(input_data.samples.shape[0], None)
        else:
            self._prediction_matrix = input_data.prediction_matrix
        if input_data.original_feature_data is not None:
            samples = np.column_stack((input_data.samples,
                                       input_data.original_feature_data))[input_data.prediction_matrix == None]
        else:
            samples = input_data.samples[self._prediction_matrix == None]  # it should be checked that samples exist
        labels = input_data.labels[self._prediction_matrix == None]
        for i in range(self._count_forest):
            assert samples.shape[0] == labels.shape[0]
            nrf = RandomForestClassifier(n_estimators=self._count_estimators,
                                         max_features='sqrt', oob_score=True)
            nrf.fit(samples, labels)
            rf = RandomForestClassifier(n_estimators=self._count_estimators,
                                        max_features=1, oob_score=True)
            rf.fit(samples, labels)
            self._forests[i] = {"not random": nrf,
                                "random": rf}
        return self

    def transform(self, input_data: TransferCascadeForestData):
        print("transform")
        samples = input_data.samples
        if input_data.original_feature_data is not None:
            samples = np.column_stack((input_data.samples, input_data.original_feature_data))
        res = []
        for key, forest in self._forests.items():
            res.append(forest["not random"].predict_proba(samples))
            res.append(forest["random"].predict_proba(samples))
        output_samples = np.column_stack(np.array(res))
        prediction_matrix = self._confidence_screening(np.mean(res, axis=0), input_data.labels,
                                                       self._prediction_matrix)
        if input_data.original_feature_data is None:
            original_data = input_data.samples
        else:
            original_data = input_data.original_feature_data
        output_data = TransferCascadeForestData(samples=output_samples,
                                                labels=input_data.labels,
                                                prediction_matrix=prediction_matrix,
                                                original_feature_data=original_data)
        return output_data


@auto_block
class AverageBlock:
    def __init__(self):
        self._count_labels = 0

    def fit(self, input_data: TransferCascadeForestData):
        return self

    def transform(self, input_data: TransferCascadeForestData):
        result = input_data.prediction_matrix
        if input_data.labels is not None:
            self._count_labels = len(np.unique(input_data.labels))
        for i in range(input_data.samples.shape[0]):
            avg_prediction = []
            for score_prediction in range(0, self._count_labels):
                label_prediction = input_data.samples[i][score_prediction:
                                                         len(input_data.samples[i]):
                                                         self._count_labels]
                avg_prediction.append(np.mean(label_prediction, axis=0))
            x = np.argmax(avg_prediction)
            if input_data.prediction_matrix[i] is None:
                if avg_prediction[x] > self._eps:
                    result[i] = avg_prediction[x]
        return result


class DeepForestConfidenceScreeningExample(BaseModel):
    """Confidence screening deep forest"""
    def __init__(self, count_blocks: Optional[int] = None):
        self._count_blocks = count_blocks
        self._node_1 = CascadeRandomForestsCFBlock(0.5, 2, 2) #ConfidenceScreeningBlock
        self._node_2 = CascadeRandomForestsCFBlock(0.5, 2, 2) #ConfidenceScreeningBlock
        self._node_3 = AverageBlock()
        self._nodes = []
        pipeline = BasePipeline(
            nodes=[self._node_1, self._node_2, self._node_3],
            connections=[
                Connection(self._node_1.meta.outputs['output'], self._node_2.meta.inputs['input_data']),
                Connection(self._node_2.meta.outputs['output'], self._node_3.meta.inputs['input_data'])
            ]
        )
        self._fit_executor = NaiveExecutor(
            pipeline,
            stage=Stage.FIT,
            inputs={
                'input_data': self._node_1.meta.inputs['input_data'],
            },
            outputs={'output': self._node_3.meta.outputs['output']},
        )
        self._transform_executor = NaiveExecutor(
            pipeline,
            stage=Stage.TRANSFORM,
            inputs={'input_data': self._node_1.meta.inputs['input_data']},
            outputs={
                'output_res': self._node_3.meta.outputs['output'],
            },
        )

    def fit(self, input_data: TransferCascadeForestData):
        self._fit_executor({'input_data': input_data})
        print("Fit successful")

    def transform(self, input_data: TransferCascadeForestData) -> Data:
        transform_input_data = TransferCascadeForestData(samples=np.random.normal(scale=10.0, size=(100, 5)))
        result = self._transform_executor({'input_data': transform_input_data})
        output_res = result['output_res']
        return output_res


def main():
    deep_forest = DeepForestConfidenceScreeningExample()
    test_x = np.random.normal(scale=10.0, size=(100, 5))
    test_y = np.random.binomial(n=1, p=0.5, size=test_x.shape[0])
    train_input_data = TransferCascadeForestData(samples=np.random.normal(scale=10.0, size=(100, 5)),
                                                 labels=test_y)
    transform_input_data = TransferCascadeForestData(samples=np.random.normal(scale=10.0, size=(100, 5)))
    deep_forest.fit(train_input_data)
    res = deep_forest.transform(transform_input_data)
    print("Res = ", res)


if __name__ == "__main__":
    main()
