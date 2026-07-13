from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | list[list[int]]]:
    pred = probabilities.argmax(axis=1)
    onehot = np.eye(probabilities.shape[1])[y_true]
    output: dict[str, float | list[list[int]]] = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "stemi_recall": float(recall_score(y_true, pred, labels=[1], average=None, zero_division=0)[0]),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
    }
    try:
        output["macro_auroc"] = float(roc_auc_score(onehot, probabilities, average="macro", multi_class="ovr"))
        output["macro_auprc"] = float(average_precision_score(onehot, probabilities, average="macro"))
    except ValueError:
        output["macro_auroc"] = float("nan")
        output["macro_auprc"] = float("nan")
    return output

