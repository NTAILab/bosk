"""Metric blocks.
"""

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score
from functools import wraps

from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...placeholder import PlaceholderMixin
from ...meta import BlockMeta, BlockExecutionProperties, InputSlotMeta, OutputSlotMeta
from ....data import CPUData
from ....stages import Stages


__all__ = [
    "Accuracy",
    "RocAuc",
    "RocAucMultiLabel",
    "F1Score",
    "R2Score",
    # for backward compatibility:
    "AccuracyBlock",
    "RocAucBlock",
    "RocAucMultiLabelBlock",
    "F1ScoreBlock",
    "R2ScoreBlock",
]


class RocAuc(PlaceholderMixin, BaseBlock):
    """ROC-AUC block.

    Args:
        **kwargs: Arguments for `sklearn.metrics.roc_auc_score`.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - pred_probas: Predicted probabilities.
        - gt_y: Ground truth labels.

    Output slots
    ------------

        - roc-auc: ROC-AUC score.

    """

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

    @wraps(roc_auc_score)  # type: ignore
    def __init__(self, **kwargs):
        super().__init__()
        self.roc_auc_score_kwargs = kwargs

    def fit(self, _inputs: BlockInputData) -> 'RocAucBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        pred_probas = inputs['pred_probas'].data
        if pred_probas.shape[1] == 2:
            # the binary case is special for the `roc_auc_score`
            pred_probas = pred_probas[:, 1]
        return {
            'roc-auc': CPUData(
                roc_auc_score(
                    inputs['gt_y'].data,
                    pred_probas,
                    **self.roc_auc_score_kwargs
                )
            )
        }


class RocAucMultiLabel(PlaceholderMixin, BaseBlock):
    """ROC-AUC Multilabel block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - pred_probas: Predicted probabilities.
        - gt_y: Ground truth labels.

    Output slots
    ------------

        - roc-auc: ROC-AUC score.

    """

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
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'roc-auc': CPUData(roc_auc_score(inputs['gt_y'].data, inputs['pred_probas'].data, multi_class='ovr'))
        }


class Accuracy(PlaceholderMixin, BaseBlock):
    """Accuracy block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - pred: Predicted labels.
        - gt_y: Ground truth labels.

    Output slots
    ------------

        - accuracy: Accuracy score.

    """

    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred',
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
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'accuracy': CPUData(accuracy_score(inputs['gt_y'].data, inputs['pred'].data))
        }


class F1Score(PlaceholderMixin, BaseBlock):
    """F1 metric block.

    Args:
        **kwargs: Arguments for `sklearn.metrics.f1_score`.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.


    Transform inputs
    ~~~~~~~~~~~~~~~~

        - pred: Predicted labels.
        - gt_y: Ground truth labels.

    Output slots
    ------------

        - f1-score: F1 score.

    """

    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred',
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

    @wraps(f1_score)  # type: ignore
    def __init__(self, **kwargs):
        super().__init__()
        self.f1_score_kwargs = kwargs

    def fit(self, _inputs: BlockInputData) -> 'F1ScoreBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'f1-score': CPUData(
                f1_score(
                    inputs['gt_y'].data,
                    inputs['pred'].data,
                    **self.f1_score_kwargs
                )
            )
        }


class R2Score(PlaceholderMixin, BaseBlock):
    """R2 metric block.

    Args:
        **kwargs: Arguments for `sklearn.metrics.r2_score`.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - pred: Predicted target estimation.
        - gt_y: Ground truth target variable value.

    Output slots
    ------------

        - r2-score: R2 score.

    """

    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='pred',
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

    @wraps(r2_score)  # type: ignore
    def __init__(self, **kwargs):
        super().__init__()
        self.r2_score_kwargs = kwargs

    def fit(self, _inputs: BlockInputData) -> 'R2ScoreBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {
            'r2-score': CPUData(
                r2_score(
                    inputs['gt_y'].data,
                    inputs['pred'].data,
                    **self.r2_score_kwargs
                )
            )
        }


AccuracyBlock = Accuracy
RocAucBlock = RocAuc
RocAucMultiLabelBlock = RocAucMultiLabel
F1ScoreBlock = F1Score
R2ScoreBlock = R2Score

