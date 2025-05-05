"""Eager evaluations tests."""

from typing import Dict, Optional, Sequence, Tuple
from bosk.pipeline.base import BasePipeline
from bosk.data import BaseData, CPUData
from bosk.executor.topological import TopologicalExecutor
from bosk.pipeline.builder.eager import EagerPipelineBuilder
from bosk.stages import Stage
from ..utility import fit_pipeline, log_test_name
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import logging

from bosk.pipeline.builder import FunctionalPipelineBuilder
from bosk.block.zoo.input_plugs import Input, TargetInput
from bosk.block.zoo.data_conversion import Argmax, Average, Concat, Stack
from bosk.block.zoo.models.classification import RFC, ETC
from bosk.block.zoo.models.regression import RFR, ETR
from bosk.block.zoo.output_plugs import Output
from bosk.block.zoo.metrics import RocAuc


class FunctionalBlockReuseTest:
    """Test case of the basic deep forest constructed with functional pipeline."""

    random_seed = 42
    executor_cls = TopologicalExecutor

    def reuse_twice_test(self):
        all_X, all_y = make_moons(noise=0.5, random_state=self.random_seed)
        train_X, test_X, train_y, test_y = train_test_split(
            all_X,
            all_y,
            test_size=0.2,
            random_state=self.random_seed
        )

        with FunctionalPipelineBuilder() as b:
            X, y = Input()(), TargetInput()()
            X2 = Input()()
            X3 = Input()()

            rfr = RFR()
            out1 = rfr(X=X, y=y)
            out2 = rfr(X=X2)  # reuse
            out3 = rfr(X=X3)  # reuse

            pipeline = b.build(
                {'X': X, 'y': y, 'X2': X2, 'X3': X3},
                {'out1': out1 , 'out2': out2, 'out3': out3},
            )

        fit_executor = self.executor_cls(
            pipeline,
            stage=Stage.FIT,
        )
        transform_executor = self.executor_cls(
            pipeline,
            stage=Stage.TRANSFORM,
        )

        fit_executor({'X': train_X, 'y': train_y, 'X2': train_X[:10], 'X3': train_X[:7]})
        train_result = transform_executor({'X': train_X, 'X2': test_X, 'X3': test_X[:5]}).numpy()
        assert train_result['out1'].shape[0] == train_X.shape[0]
        assert train_result['out2'].shape[0] == test_X.shape[0]
        assert train_result['out3'].shape[0] == 5

    def reuse_separately_test(self):
        all_X, all_y = make_moons(noise=0.5, random_state=self.random_seed)
        train_X, test_X, train_y, test_y = train_test_split(
            all_X,
            all_y,
            test_size=0.2,
            random_state=self.random_seed
        )

        with FunctionalPipelineBuilder() as b:
            X, y = Input()(), TargetInput()()
            X2 = Input()()
            X3 = Input()()

            rfr = RFR()
            out1 = rfr(X=X, y=y)
            out2 = rfr(X=X2)  # reuse
            out3 = rfr(X=X3)  # reuse

            pipeline = b.build(
                {'X': X, 'y': y, 'X2': X2, 'X3': X3},
                {'out1': out1 , 'out2': out2, 'out3': out3},
            )

        fit_executor = self.executor_cls(
            pipeline,
            stage=Stage.FIT,
            inputs=['X', 'y'],
            outputs=['out1'],

        )
        transform_executor = self.executor_cls(
            pipeline,
            stage=Stage.TRANSFORM,
            inputs=['X', 'X2', 'X3'],
            outputs=['out2', 'out3'],
        )

        fit_executor({'X': train_X, 'y': train_y})
        train_result = transform_executor({'X': test_X, 'X2': test_X, 'X3': test_X[:5]}).numpy()
        assert train_result['out2'].shape[0] == test_X.shape[0]
        assert train_result['out3'].shape[0] == 5


class EagerBlockReuseTest:
    """Test case of the basic deep forest constructed with functional pipeline."""

    random_seed = 42
    executor_cls = TopologicalExecutor

    def reuse_twice_test(self):
        all_X, all_y = make_moons(noise=0.5, random_state=self.random_seed)
        train_X, test_X, train_y, test_y = train_test_split(
            all_X,
            all_y,
            test_size=0.2,
            random_state=self.random_seed
        )

        with EagerPipelineBuilder() as b:
            X, y = Input()(train_X), TargetInput()(train_y)
            X2 = Input()(test_X)
            X3 = Input()(test_X[:5])

            rfr = RFR()
            out1 = rfr(X=X, y=y)
            out2 = rfr(X=X2)  # reuse
            out3 = rfr(X=X3)  # reuse

            pipeline = b.build(
                {'X': X, 'y': y, 'X2': X2, 'X3': X3},
                {'out1': out1 , 'out2': out2, 'out3': out3},
            )

        transform_executor = self.executor_cls(
            pipeline,
            stage=Stage.TRANSFORM,
        )

        train_result = transform_executor({'X': train_X, 'X2': test_X, 'X3': test_X[:5]}).numpy()
        assert np.allclose(out1.data, train_result['out1'])
        assert np.allclose(out2.data, train_result['out2'])
        assert np.allclose(out3.data, train_result['out3'])
