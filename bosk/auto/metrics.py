import numpy as np
from typing import List, Mapping
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


MetricsResults = Mapping[str, float]


class MetricsEvaluator:
    """Calculates, collects and aggregates the specified metrics.

    Any metric follows the rule "the higher is better", i.e.
    if originally metric decreases when quality increases, it is multiplied by (-1).
    """
    def __init__(self, names: List[str]):
        self.names = set(names)
        self.results = defaultdict(list)

    def append_eval(self, y_true: np.ndarray, y_pred: np.ndarray):
        pred_labels = np.argmax(y_pred, axis=1)
        if 'roc_auc' in self.names:
            self.results['roc_auc'].append(roc_auc_score(y_true, y_pred, multi_class='ovr'))
        if 'f1' in self.names:
            self.results['f1'].append(f1_score(y_true, pred_labels, average='macro'))
        if 'accuracy' in self.names:
            self.results['accuracy'].append(accuracy_score(y_true, pred_labels))

    def average(self) -> MetricsResults:
        return {
            k: np.mean(v)
            for k, v in self.results.items()
        }
