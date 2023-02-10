from bosk.data import Data
from bosk.pipeline.base import BasePipeline
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from .base import PipelineTestBase
from sklearn.datasets import make_moons
from typing import Dict, Optional, Sequence, Tuple

class CasualFuncForestTest(PipelineTestBase):
    random_state: int = 42
    
    def get_pipeline(self) -> BasePipeline:
        b = FunctionalPipelineBuilder()
        X, y = b.Input()(), b.TargetInput()()
        rf_1 = b.RFC(random_state=self.random_state)(X=X, y=y)
        et_1 = b.ETC(random_state=self.random_state)(X=X, y=y)
        concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
        rf_2 = b.RFC(random_state=self.random_state)(X=concat_1, y=y)
        et_2 = b.ETC(random_state=self.random_state)(X=concat_1, y=y)
        concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
        rf_3 = b.RFC(random_state=self.random_state)(X=concat_2, y=y)
        et_3 = b.ETC(random_state=self.random_state)(X=concat_2, y=y)
        stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
        average_3 = b.Average(axis=1)(X=stack_3)
        argmax_3 = b.Argmax(axis=1)(X=average_3)

        rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)
        return b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 
            'roc-auc': roc_auc, 'labels': argmax_3}
            )
        
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