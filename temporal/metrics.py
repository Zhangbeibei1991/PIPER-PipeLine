import numpy as np

from typing import Dict, Any
from overrides import overrides
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from allennlp.training.metrics import Metric

from .ILP import ILPSolver


class ILPMetric(Metric):

    def __init__(self, label2idx, labels, flip):
        self._fwd_probs = []
        self._bwd_probs = []
        self._fwd_pairs = []
        self._bwd_pairs = []
        self._labels = []

        self.label2idx = label2idx
        self.labels = labels
        self.flip = flip

    @overrides
    def __call__(
            self, fwd_probs, fwd_pairs, bwd_probs, bwd_pairs, labels
    ):
        self._fwd_probs.append(fwd_probs)
        self._fwd_pairs.extend(fwd_pairs)
        self._labels.append(labels)

        if bwd_probs is not None:
            self._bwd_probs.append(bwd_probs)
            self._bwd_pairs.extend(bwd_pairs)

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        if len(self._bwd_probs) > 0 and self.flip:
            probs = np.concatenate(self._fwd_probs + self._bwd_probs)
            pairs = self._fwd_pairs + self._bwd_pairs
            flip = True
        else:
            probs = np.concatenate(self._fwd_probs)
            pairs = self._fwd_pairs
            flip = False

        gold = np.concatenate(self._labels)
        pred = ILPSolver(pairs, probs, self.label2idx, flip=flip).inference()[:gold.shape[0]]

        f1 = f1_score(y_true=gold, y_pred=pred, average='micro', labels=self.labels)
        r = recall_score(y_true=gold, y_pred=pred, average='micro', labels=self.labels)
        p = precision_score(y_true=gold, y_pred=pred, average='micro', labels=self.labels)

        if reset:
            self.reset()

        return {
            "ilp_fscore": f1,
            "ilp_recall": r,
            "ilp_precision": p
        }

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self._fwd_probs.clear()
        self._bwd_probs.clear()
        self._fwd_pairs.clear()
        self._bwd_pairs.clear()
        self._labels.clear()

