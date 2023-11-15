"""Eager evaluations tests."""

from typing import Dict, Optional, Sequence, Tuple
from bosk.pipeline.base import BasePipeline
from bosk.data import BaseData, CPUData
from bosk.executor.block import DefaultBlockExecutor
from bosk.executor.recursive import RecursiveExecutor
from bosk.stages import Stage
from bosk.pipeline.builder.eager import EagerPipelineBuilder
from ..utility import fit_pipeline, log_test_name
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import logging


class EagerEvaluationTest:
    """Test case of the basic deep forest constructed with eager evaluations."""

    random_seed = 42
    executor_cls = RecursiveExecutor

    def eager_fit_transform_test(self):
        all_X, all_y = make_moons(noise=0.5, random_state=self.random_seed)
        train_X, test_X, train_y, test_y = train_test_split(
            all_X,
            all_y,
            test_size=0.2,
            random_state=self.random_seed
        )

        block_executor = DefaultBlockExecutor()
        b = EagerPipelineBuilder(block_executor)
        X, y = b.Input()(CPUData(train_X)), b.TargetInput()(CPUData(train_y))
        forest_params = dict(n_estimators=3, max_depth=2)

        rf_1 = b.RFC(random_state=self.random_seed, **forest_params)(X=X, y=y)
        et_1 = b.ETC(random_state=self.random_seed, **forest_params)(X=X, y=y)
        concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
        rf_2 = b.RFC(random_state=self.random_seed, **forest_params)(X=concat_1, y=y)
        et_2 = b.ETC(random_state=self.random_seed, **forest_params)(X=concat_1, y=y)
        concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
        rf_3 = b.RFC(random_state=self.random_seed, **forest_params)(X=concat_2, y=y)
        et_3 = b.ETC(random_state=self.random_seed, **forest_params)(X=concat_2, y=y)
        stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
        average_3 = b.Average(axis=1)(X=stack_3)
        argmax_3 = b.Argmax(axis=1)(X=average_3)

        rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)

        eager_probas = average_3.get_output_data().data

        fit_executor = self.executor_cls(
            b.build(
                {'X': X, 'y': y},
                {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
            ),
            stage=Stage.FIT,
            inputs=['X', 'y'],
            outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        )
        transform_executor = self.executor_cls(
            b.build(
                {'X': X, 'y': y},
                {'probas': average_3, 'labels': argmax_3}
            ),
            stage=Stage.TRANSFORM,
            inputs=['X'],
            outputs=['probas', 'labels'],
        )

        train_result = transform_executor({'X': CPUData(train_X)})
        assert np.allclose(eager_probas, train_result['probas'].data), \
                "Probabilities obtained at pipeline construction should match the predicted"

