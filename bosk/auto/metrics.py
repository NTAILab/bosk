import numpy as np
from typing import List, Mapping
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


MetricsResults = Mapping[str, float]


class MetricsEvaluator:
    """Calculates, collects and aggregates the specified metrics.

    Any metric follows the rule "the higher is better", i.e.
    if originally metric decreases when quality increases, it is multiplied by (-1).

    Args:
        names: A list of metric names to be evaluated.

    """

    names: set[str]
    """A list of metric names to be evaluated.
    """
    results: defaultdict[str, List[float]]
    """A mapping (dictionary) of metric names to lists of metric values.
    """

    def __init__(self, names: List[str]):
        self.names = set(names)
        self.results = defaultdict(list)

    def append_eval(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate and append the evaluation results.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        """
        pred_labels = np.argmax(y_pred, axis=1)
        if 'roc_auc' in self.names:
            if y_pred.shape[1] == 2:
                ras_y = y_pred[:, 1]
            else:
                ras_y = y_pred
            self.results['roc_auc'].append(roc_auc_score(y_true, ras_y, multi_class='ovr'))
        if 'f1' in self.names:
            self.results['f1'].append(f1_score(y_true, pred_labels, average='macro'))
        if 'accuracy' in self.names:
            self.results['accuracy'].append(accuracy_score(y_true, pred_labels))

    def average(self) -> MetricsResults:
        """Averages the evaluation results.

        Returns:
            A mapping (dictionary) of metric names to average metric values.
        """
        return {
            k: np.mean(v)
            for k, v in self.results.items()
        }
