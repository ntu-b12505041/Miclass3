from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def compose_soft_cascade(mi_probability: np.ndarray, stemi_given_mi_probability: np.ndarray) -> np.ndarray:
    """Compose P(MI) and P(STEMI|MI) into three mutually exclusive probabilities.

    Class order follows Miclass3: non_mi, stemi_proxy, nstemi_proxy.
    """
    p_mi = np.asarray(mi_probability, dtype=float).reshape(-1)
    p_stemi_mi = np.asarray(stemi_given_mi_probability, dtype=float).reshape(-1)
    if p_mi.shape != p_stemi_mi.shape:
        raise ValueError("mi_probability and stemi_given_mi_probability must have the same shape")
    p_mi = np.clip(p_mi, 0.0, 1.0)
    p_stemi_mi = np.clip(p_stemi_mi, 0.0, 1.0)
    return np.column_stack(
        [
            1.0 - p_mi,
            p_mi * p_stemi_mi,
            p_mi * (1.0 - p_stemi_mi),
        ]
    )


def compose_hard_cascade(
    mi_probability: np.ndarray,
    stemi_given_mi_probability: np.ndarray,
    mi_threshold: float = 0.5,
    stemi_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a literal two-stage gate and return three-class scores and predictions.

    Records below the MI threshold stop at stage 1 and become non_mi. Records
    above it are routed to stage 2 and classified as STEMI-proxy or
    NSTEMI-proxy. The returned score rows sum to one so existing Miclass3
    reporting utilities can still compute AUROC/AUPRC.
    """
    p_mi = np.asarray(mi_probability, dtype=float).reshape(-1)
    p_stemi_mi = np.asarray(stemi_given_mi_probability, dtype=float).reshape(-1)
    if p_mi.shape != p_stemi_mi.shape:
        raise ValueError("mi_probability and stemi_given_mi_probability must have the same shape")
    if not 0.0 <= float(mi_threshold) <= 1.0 or not 0.0 <= float(stemi_threshold) <= 1.0:
        raise ValueError("cascade thresholds must be in [0, 1]")

    scores = compose_soft_cascade(p_mi, p_stemi_mi)
    predictions = np.zeros(len(p_mi), dtype=int)
    routed = p_mi >= float(mi_threshold)
    predictions[routed & (p_stemi_mi >= float(stemi_threshold))] = 1
    predictions[routed & (p_stemi_mi < float(stemi_threshold))] = 2

    # Make argmax reproduce the explicit routing decision while preserving the
    # soft probabilities as much as possible within the selected branch.
    hard_scores = scores.copy()
    stop = ~routed
    hard_scores[stop, 0] = np.maximum(hard_scores[stop, 0], 0.500001)
    hard_scores[stop, 1:] *= (1.0 - hard_scores[stop, 0]) / np.maximum(hard_scores[stop, 1:].sum(axis=1, keepdims=True), 1e-12)
    for index in np.where(routed)[0]:
        hard_scores[index, 0] = 0.0
        if predictions[index] == 1:
            hard_scores[index, 1] = max(hard_scores[index, 1], 0.500001)
            hard_scores[index, 2] = 1.0 - hard_scores[index, 1]
        else:
            hard_scores[index, 2] = max(hard_scores[index, 2], 0.500001)
            hard_scores[index, 1] = 1.0 - hard_scores[index, 2]
    hard_scores /= np.maximum(hard_scores.sum(axis=1, keepdims=True), 1e-12)
    return hard_scores, predictions


def calibrate_binary_threshold(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    minimum_recall: float | None = None,
) -> tuple[float, dict[str, float | bool | int | None], pd.DataFrame]:
    """Choose a validation-only binary threshold by F1, then balanced accuracy."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.asarray(positive_probability, dtype=float).reshape(-1)
    if y_true.shape != probability.shape:
        raise ValueError("y_true and positive_probability must have the same shape")
    if minimum_recall is not None and not 0.0 <= float(minimum_recall) <= 1.0:
        raise ValueError("minimum_recall must be in [0, 1]")

    thresholds = np.unique(np.concatenate(([0.0], probability, [1.0])))
    rows: list[dict[str, float | bool]] = []
    for threshold in thresholds:
        pred = (probability >= threshold).astype(int)
        recall = float(recall_score(y_true, pred, zero_division=0))
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                "recall": recall,
                "meets_minimum_recall": minimum_recall is None or recall >= float(minimum_recall),
            }
        )
    table = pd.DataFrame(rows)
    feasible = table[table["meets_minimum_recall"]]
    pool = feasible if not feasible.empty else table
    chosen = pool.sort_values(
        ["f1", "balanced_accuracy", "recall", "threshold"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    summary = {
        "objective": "f1",
        "minimum_recall": None if minimum_recall is None else float(minimum_recall),
        "constraint_satisfied": bool(not feasible.empty),
        "candidate_count": int(len(table)),
        "f1": float(chosen["f1"]),
        "balanced_accuracy": float(chosen["balanced_accuracy"]),
        "recall": float(chosen["recall"]),
    }
    return float(chosen["threshold"]), summary, table
