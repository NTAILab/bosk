"""Eager evaluations tests."""

from typing import Dict, Optional, Sequence, Tuple
from bosk.pipeline.base import BasePipeline
from bosk.data import BaseData, CPUData
from bosk.executor.topological import TopologicalExecutor
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


class FunctionalPipelineTest:
    """Test case of the basic deep forest constructed with functional pipeline."""

    random_seed = 42
    executor_cls = TopologicalExecutor

    def functional_fit_transform_test(self):
        all_X, all_y = make_moons(noise=0.5, random_state=self.random_seed)
        train_X, test_X, train_y, test_y = train_test_split(
            all_X,
            all_y,
            test_size=0.2,
            random_state=self.random_seed
        )

        with FunctionalPipelineBuilder() as b:
            X, y = Input()(), TargetInput()()
            forest_params = dict(n_estimators=3, max_depth=2)

            rf_1 = RFC(random_state=self.random_seed, **forest_params)(X=X, y=y)
            et_1 = ETC(random_state=self.random_seed, **forest_params)(X=X, y=y)
            concat_1 = Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
            rf_2 = RFC(random_state=self.random_seed, **forest_params)(X=concat_1, y=y)
            et_2 = ETC(random_state=self.random_seed, **forest_params)(X=concat_1, y=y)
            concat_2 = Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
            rf_3 = RFC(random_state=self.random_seed, **forest_params)(X=concat_2, y=y)
            et_3 = ETC(random_state=self.random_seed, **forest_params)(X=concat_2, y=y)
            stack_3 = Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
            average_3 = Average(axis=1)(X=stack_3)
            argmax_3 = Argmax(axis=1)(X=average_3)

            rf_1_roc_auc = RocAuc()(gt_y=y, pred_probas=rf_1)
            roc_auc = RocAuc()(gt_y=y, pred_probas=average_3)

            pipeline = b.build(
                {'X': X, 'y': y},
                {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc, 'labels': argmax_3}
            )

        fit_executor = self.executor_cls(
            pipeline,
            stage=Stage.FIT,
            inputs=['X', 'y'],
            outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        )
        transform_executor = self.executor_cls(
            pipeline,
            stage=Stage.TRANSFORM,
            inputs=['X'],
            outputs=['probas', 'labels'],
        )

        fit_executor({'X': train_X, 'y': train_y})
        train_result = transform_executor({'X': CPUData(train_X)})
