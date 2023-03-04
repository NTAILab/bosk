from bosk.pipeline.base import BasePipeline, Connection
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock
from .base import BasePipelineTest as BPT
from sklearn.datasets import make_moons
from typing import Dict, Optional, Sequence, Tuple
from bosk.data import Data, CPUData


class CasualManualForestTest(BPT):
    """Test case of the basic deep forest made manually with the connections."""

    n_trees: int = 37

    def _get_pipeline(self) -> BasePipeline:
        input_x = InputBlock()
        input_y = TargetInputBlock()
        rf_1 = RFCBlock(n_estimators=self.n_trees)
        et_1 = ETCBlock(n_estimators=self.n_trees)
        concat_1 = ConcatBlock(['X_0', 'X_1'], axis=1)
        rf_2 = RFCBlock(n_estimators=self.n_trees)
        et_2 = ETCBlock(n_estimators=self.n_trees)
        concat_2 = ConcatBlock(['X_0', 'X_1'], axis=1)
        rf_3 = RFCBlock(n_estimators=self.n_trees)
        et_3 = ETCBlock(n_estimators=self.n_trees)
        stack_3 = StackBlock(['X_0', 'X_1'], axis=1)
        average_3 = AverageBlock(axis=1)
        argmax_3 = ArgmaxBlock(axis=1)
        roc_auc = RocAucBlock()
        roc_auc_rf_1 = RocAucBlock()
        return BasePipeline(
            nodes=[
                input_x,
                input_y,
                rf_1,
                et_1,
                concat_1,
                rf_2,
                et_2,
                concat_2,
                rf_3,
                et_3,
                stack_3,
                average_3,
                argmax_3,
                roc_auc,
                roc_auc_rf_1
            ],
            connections=[
                # input X
                Connection(input_x.slots.outputs['X'], rf_1.slots.inputs['X']),
                Connection(input_x.slots.outputs['X'], et_1.slots.inputs['X']),
                # input y
                Connection(input_y.slots.outputs['y'], rf_1.slots.inputs['y']),
                Connection(input_y.slots.outputs['y'], et_1.slots.inputs['y']),
                Connection(input_y.slots.outputs['y'], rf_2.slots.inputs['y']),
                Connection(input_y.slots.outputs['y'], et_2.slots.inputs['y']),
                Connection(input_y.slots.outputs['y'], rf_3.slots.inputs['y']),
                Connection(input_y.slots.outputs['y'], et_3.slots.inputs['y']),
                # layers connection
                Connection(rf_1.slots.outputs['output'], concat_1.slots.inputs['X_0']),
                Connection(et_1.slots.outputs['output'], concat_1.slots.inputs['X_1']),
                Connection(concat_1.slots.outputs['output'], rf_2.slots.inputs['X']),
                Connection(concat_1.slots.outputs['output'], et_2.slots.inputs['X']),
                Connection(rf_2.slots.outputs['output'], concat_2.slots.inputs['X_0']),
                Connection(et_2.slots.outputs['output'], concat_2.slots.inputs['X_1']),
                Connection(concat_2.slots.outputs['output'], rf_3.slots.inputs['X']),
                Connection(concat_2.slots.outputs['output'], et_3.slots.inputs['X']),
                Connection(rf_3.slots.outputs['output'], stack_3.slots.inputs['X_0']),
                Connection(et_3.slots.outputs['output'], stack_3.slots.inputs['X_1']),
                Connection(stack_3.slots.outputs['output'], average_3.slots.inputs['X']),
                Connection(average_3.slots.outputs['output'], argmax_3.slots.inputs['X']),
                Connection(average_3.slots.outputs['output'], roc_auc.slots.inputs['pred_probas']),
                Connection(input_y.slots.outputs['y'], roc_auc.slots.inputs['gt_y']),
                Connection(rf_1.slots.outputs['output'], roc_auc_rf_1.slots.inputs['pred_probas']),
                Connection(input_y.slots.outputs['y'], roc_auc_rf_1.slots.inputs['gt_y']),
            ],
            inputs={
                'X': input_x.slots.inputs['X'],
                'y': input_y.slots.inputs['y'],
            },
            outputs={
                'probas': average_3.slots.outputs['output'],
                'rf_1_roc-auc': roc_auc_rf_1.slots.outputs['roc-auc'],
                'roc-auc': roc_auc.slots.outputs['roc-auc'],
                'labels': argmax_3.slots.outputs['output']
            }
        )

    def make_dataset(self):
        x, y = make_moons(noise=0.5, random_state=self.random_seed)
        self.x = CPUData(x)
        self.y = CPUData(y)

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
