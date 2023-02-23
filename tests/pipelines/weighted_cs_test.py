from sklearn.datasets import make_moons

from bosk.data import Data
from bosk.block.zoo.data_weighting import WeightsBlock
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from .base import BasePipelineTest as BPT
from typing import Dict, Optional, Sequence, Tuple


class WeightedCSForestTest(BPT):

    random_state: int = 42
    n_trees: int = 23

    def make_deep_forest_layer(self, b, **inputs):
        rf = b.RFC(random_state=self.random_state, n_estimators=self.n_trees)(**inputs)
        et = b.ETC(random_state=self.random_state, n_estimators=self.n_trees)(**inputs)
        stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
        average = b.Average(axis=1)(X=stack)
        return average

    def get_pipeline(self):
        b = FunctionalPipelineBuilder()
        X, y = b.Input()(), b.TargetInput()()
        rf_1 = b.RFC(random_state=42)(X=X, y=y)
        et_1 = b.ETC(random_state=42)(X=X, y=y)
        concat_1 = b.Concat(['rf_1', 'et_1'])(rf_1=rf_1, et_1=et_1)
        stack_1 = b.Stack(['rf_1', 'et_1'], axis=1)(rf_1=rf_1, et_1=et_1)
        average_1 = b.Average(axis=1)(X=stack_1)

        # get confidence screening mask
        cs_1 = b.CS(eps=0.95)(X=average_1)
        # filter X and concatenated predictions samples by CS
        filtered_1 = b.CSFilter(['concat_1', 'X'])(
            concat_1=concat_1,
            X=X,
            mask=cs_1['mask']
        )
        # y should be filtered separately since it is not used at the Transform stage
        filtered_1_y = b.CSFilter(['y'])(y=y, mask=cs_1['mask'])
        concat_all_1 = b.Concat(['filtered_1_X', 'filtered_concat_1'])(
            filtered_1_X=filtered_1['X'],
            filtered_concat_1=filtered_1['concat_1']
        )

        average_2 = self.make_deep_forest_layer(b, X=concat_all_1, y=filtered_1_y)
        concat_2 = b.Concat(['X', 'average_2'])(X=filtered_1['X'], average_2=average_2)

        sample_weight_2 = b.new(WeightsBlock, ord=2)(X=average_2, y=filtered_1_y)

        average_3 = self.make_deep_forest_layer(
            b, X=concat_2, y=filtered_1_y, sample_weight=sample_weight_2)

        # join confident samples with screened out ones
        joined_3 = b.CSJoin()(
            best=cs_1['best'],
            refined=average_3,
            mask=cs_1['mask']
        )

        argmax_3 = b.Argmax(axis=1)(X=joined_3)

        rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = b.RocAuc()(gt_y=y, pred_probas=joined_3)
        return b.build({'X': X, 'y': y},
                       {'probas': joined_3, 'rf_1_roc-auc': rf_1_roc_auc,
                        'roc-auc': roc_auc, 'labels': argmax_3})

    def make_dataset(self):
        self.x, self.y = make_moons(noise=0.5, random_state=self.random_state)

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
