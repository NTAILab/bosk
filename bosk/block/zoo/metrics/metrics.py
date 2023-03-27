from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score

from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...meta import BlockMeta, BlockExecutionProperties
from ....data import CPUData
from ....stages import Stages
from ...slot import InputSlotMeta, OutputSlotMeta


class RocAucBlock(BaseBlock):
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred_probas',
                stages=Stages(transform=True, transform_on_fit=True),
            ),
            InputSlotMeta(
                name='gt_y',
                stages=Stages(transform=True, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='roc-auc',
            )
        ],
        execution_props=BlockExecutionProperties()
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'RocAucBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'roc-auc': CPUData(roc_auc_score(inputs['gt_y'].data, inputs['pred_probas'].data[:, 1]))
        }


class RocAucMultiLabelBlock(BaseBlock):
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred_probas',
                stages=Stages(transform=False, transform_on_fit=True),
            ),
            InputSlotMeta(
                name='gt_y',
                stages=Stages(transform=False, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='roc-auc',
            )
        ],
        execution_props=BlockExecutionProperties()
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'RocAucMultiLabelBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'roc-auc': CPUData(roc_auc_score(inputs['gt_y'].data, inputs['pred_probas'].data, multi_class='ovr'))
        }


class AccuracyBlock(BaseBlock):
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred_probas',
                stages=Stages(transform=False, transform_on_fit=True),
            ),
            InputSlotMeta(
                name='gt_y',
                stages=Stages(transform=False, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='accuracy',
            )
        ],
        execution_props=BlockExecutionProperties()
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'AccuracyBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'accuracy': CPUData(accuracy_score(inputs['gt_y'].data, inputs['pred_probas'].data[:, 1]))
        }


class F1ScoreBlock(BaseBlock):
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred_probas',
                stages=Stages(transform=False, transform_on_fit=True),
            ),
            InputSlotMeta(
                name='gt_y',
                stages=Stages(transform=False, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='f1-score',
            )
        ],
        execution_props=BlockExecutionProperties()
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'F1ScoreBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'f1-score': CPUData(f1_score(inputs['gt_y'].data, inputs['pred_probas'].data[:, 1]))
        }


class R2ScoreBlock(BaseBlock):
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred_probas',
                stages=Stages(transform=False, transform_on_fit=True),
            ),
            InputSlotMeta(
                name='gt_y',
                stages=Stages(transform=False, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='r2-score',
            )
        ],
        execution_props=BlockExecutionProperties()
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'R2ScoreBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'r2-score': CPUData(r2_score(inputs['gt_y'].data, inputs['pred_probas'].data[:, 1]))
        }
