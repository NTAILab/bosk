from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from bosk.block import BaseBlock, BlockInputData, TransformOutputData
from bosk.stages import Stages
from bosk.block import BlockMeta
from bosk.slot import InputSlotMeta, OutputSlotMeta


class RocAucBlock(BaseBlock):
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
        ]
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'RocAucBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'roc-auc': roc_auc_score(inputs['gt_y'], inputs['pred_probas'][:, 1])
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
                name='roc-auc',
            )
        ]
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'AccuracyBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'accuracy': accuracy_score(inputs['gt_y'], inputs['pred_probas'][:, 1])
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
        ]
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'F1ScoreBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'f1-score': f1_score(inputs['gt_y'], inputs['pred_probas'][:, 1])
        }
