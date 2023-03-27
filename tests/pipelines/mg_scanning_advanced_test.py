from sklearn.datasets import load_iris, load_digits
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from bosk.data import Data, CPUData
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.block.zoo.multi_grained_scanning import \
    (MultiGrainedScanning1DBlock, MultiGrainedScanning2DBlock)
from .base import BasePipelineTest as BPT
from typing import Dict, Optional, Sequence, Tuple
import logging


def log_shape(inp):
    logging.info('Shape: %s', inp['X'].data.shape)


class AdvancedMGScanning2DTest(BPT):

    random_state: int = 42
    n_trees: int = 13

    def _get_pipeline(self):
        b = FunctionalPipelineBuilder()
        X, y = b.Input()(), b.TargetInput()()
        ms = b.new(MultiGrainedScanning2DBlock, models=(
            RandomForestClassifier(random_state=self.random_state, n_estimators=self.n_trees),
            ExtraTreesClassifier(random_state=self.random_state, n_estimators=self.n_trees)),
            window_size=3, stride=1, shape_sample=[8, 8])(X=X, y=y)
        ms = b.Reshape((-1, 20, 6, 6))(X=ms)
        printer = b.FitLambda(
            function=log_shape,
            inputs=['X']
        )(X=ms)
        # downsample
        pooled = b.Pooling(
            kernel_size=2,
            stride=None,
            dilation=1,
            aggregation='max',
        )(X=printer)
        pooled = b.FitLambda(
            function=log_shape,
            inputs=['X']
        )(X=pooled)
        pooled = b.Flatten()(X=pooled)
        rf_1 = b.RFC(random_state=self.random_state)(X=pooled, y=y)
        et_1 = b.ETC(random_state=self.random_state)(X=pooled, y=y)
        concat_1 = b.Concat(['ms', 'rf_1', 'et_1'])(ms=pooled, rf_1=rf_1, et_1=et_1)
        rf_2 = b.RFC(random_state=self.random_state)(X=concat_1, y=y)
        et_2 = b.ETC(random_state=self.random_state)(X=concat_1, y=y)
        concat_2 = b.Concat(['ms', 'rf_2', 'et_2'])(ms=pooled, rf_2=rf_2, et_2=et_2)
        rf_3 = b.RFC(random_state=self.random_state)(X=concat_2, y=y)
        et_3 = b.ETC(random_state=self.random_state)(X=concat_2, y=y)
        stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
        average_3 = b.Average(axis=1)(X=stack_3)
        argmax_3 = b.Argmax(axis=1)(X=average_3)
        #
        rf_1_roc_auc = b.RocAucMultiLabel()(gt_y=y, pred_probas=rf_1)
        roc_auc = b.RocAucMultiLabel()(gt_y=y, pred_probas=average_3)
        return b.build({'X': X, 'y': y},
                    {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc,
                        'roc-auc': roc_auc, 'labels': argmax_3})

    def make_dataset(self):
        digits = load_digits()
        self.x, self.y = CPUData(digits.data), CPUData(digits.target)

    def get_fit_data(self) -> Dict[str, Data]:
        if not hasattr(self, 'x'):
            self.make_dataset()
        return {'X': self.x, 'y': self.y}

    def get_fit_in_out(self) -> Tuple[Optional[Sequence[str]], Optional[Sequence[str]]]:
        return ['X', 'y'], ['probas', 'rf_1_roc-auc', 'roc-auc']

    def get_transform_data(self) -> Dict[str, Data]:
        if not hasattr(self, 'x'):
            self.make_dataset()
        return {'X': self.x}

    def get_transform_in_out(self) -> Tuple[Optional[Sequence[str]], Optional[Sequence[str]]]:
        return ['X'], ['probas', 'labels']
