from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def _clip_probability(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def apply_binary_logit_bias(probability: np.ndarray, bias: float) -> np.ndarray:
    """Shift binary probability in log-odds space without changing ranking."""
    p = _clip_probability(probability)
    logit = np.log(p) - np.log1p(-p)
    shifted = logit + float(bias)
    return 1.0 / (1.0 + np.exp(-shifted))


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


def compose_calibrated_soft_cascade(
    mi_probability: np.ndarray,
    stemi_given_mi_probability: np.ndarray,
    mi_logit_bias: float = 0.0,
    stemi_logit_bias: float = 0.0,
) -> np.ndarray:
    """Compose a soft cascade after validation-fitted logit-bias calibration.

    The two biases adjust decision calibration only; they do not change either
    binary model's AUROC ranking.  Values must be chosen on validation data and
    frozen before evaluating the held-out test fold.
    """
    p_mi = apply_binary_logit_bias(mi_probability, mi_logit_bias)
    p_stemi = apply_binary_logit_bias(stemi_given_mi_probability, stemi_logit_bias)
    return compose_soft_cascade(p_mi, p_stemi)


def calibrate_soft_cascade(
    y_true: np.ndarray,
    mi_probability: np.ndarray,
    stemi_given_mi_probability: np.ndarray,
    minimum_stemi_recall: float | None = None,
    bias_min: float = -2.0,
    bias_max: float = 2.0,
    bias_step: float = 0.1,
) -> tuple[dict[str, float | bool | int | None], pd.DataFrame]:
    """Fit two hierarchy-preserving logit biases on validation data only.

    The search maximizes macro-F1, then balanced accuracy, then STEMI recall.
    An optional STEMI-recall floor can be imposed.  This calibrates the final
    three-class decision while preserving the probabilistic hierarchy:

      P(non-MI) = 1 - P(MI)
      P(STEMI)  = P(MI) * P(STEMI | MI)
      P(NSTEMI) = P(MI) * (1 - P(STEMI | MI))
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p_mi = np.asarray(mi_probability, dtype=float).reshape(-1)
    p_stemi = np.asarray(stemi_given_mi_probability, dtype=float).reshape(-1)
    if y.shape != p_mi.shape or y.shape != p_stemi.shape:
        raise ValueError("y_true and cascade probabilities must have the same shape")
    if minimum_stemi_recall is not None and not 0.0 <= float(minimum_stemi_recall) <= 1.0:
        raise ValueError("minimum_stemi_recall must be in [0, 1]")
    if bias_step <= 0 or bias_max < bias_min:
        raise ValueError("invalid bias search range")

    bias_values = np.arange(float(bias_min), float(bias_max) + 0.5 * float(bias_step), float(bias_step))
    rows: list[dict[str, float | bool]] = []
    for mi_bias in bias_values:
        calibrated_mi = apply_binary_logit_bias(p_mi, float(mi_bias))
        for stemi_bias in bias_values:
            calibrated_stemi = apply_binary_logit_bias(p_stemi, float(stemi_bias))
            probabilities = compose_soft_cascade(calibrated_mi, calibrated_stemi)
            pred = probabilities.argmax(axis=1)
            macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
            balanced = float(balanced_accuracy_score(y, pred))
            stemi_recall = float(recall_score(y, pred, labels=[1], average=None, zero_division=0)[0])
            rows.append(
                {
                    "mi_logit_bias": float(mi_bias),
                    "stemi_logit_bias": float(stemi_bias),
                    "macro_f1": macro_f1,
                    "balanced_accuracy": balanced,
                    "stemi_recall": stemi_recall,
                    "meets_minimum_stemi_recall": minimum_stemi_recall is None
                    or stemi_recall >= float(minimum_stemi_recall),
                }
            )

    table = pd.DataFrame(rows)
    feasible = table[table["meets_minimum_stemi_recall"]]
    constraint_satisfied = not feasible.empty
    pool = feasible if constraint_satisfied else table
    chosen = pool.sort_values(
        ["macro_f1", "balanced_accuracy", "stemi_recall", "mi_logit_bias", "stemi_logit_bias"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).iloc[0]
    summary: dict[str, float | bool | int | None] = {
        "objective": "macro_f1",
        "minimum_stemi_recall": None if minimum_stemi_recall is None else float(minimum_stemi_recall),
        "constraint_satisfied": bool(constraint_satisfied),
        "candidate_count": int(len(table)),
        "mi_logit_bias": float(chosen["mi_logit_bias"]),
        "stemi_logit_bias": float(chosen["stemi_logit_bias"]),
        "macro_f1": float(chosen["macro_f1"]),
        "balanced_accuracy": float(chosen["balanced_accuracy"]),
        "stemi_recall": float(chosen["stemi_recall"]),
    }
    return summary, table


def compose_hard_cascade(
    mi_probability: np.ndarray,
    stemi_given_mi_probability: np.ndarray,
    mi_threshold: float = 0.5,
    stemi_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a literal two-stage gate and return three-class scores and predictions."""
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

    hard_scores = scores.copy()
    stop = ~routed
    if np.any(stop):
        hard_scores[stop, 0] = np.maximum(hard_scores[stop, 0], 0.500001)
        remaining = (1.0 - hard_scores[stop, 0])[:, None]
        branch_sum = hard_scores[stop, 1:].sum(axis=1, keepdims=True)
        fallback = np.full((int(stop.sum()), 2), 0.5, dtype=float)
        branch_mix = np.divide(
            hard_scores[stop, 1:],
            branch_sum,
            out=fallback,
            where=branch_sum > 1e-12,
        )
        hard_scores[stop, 1:] = branch_mix * remaining

    for index in np.where(routed)[0]:
        hard_scores[index, 0] = 0.0
        if predictions[index] == 1:
            hard_scores[index, 1] = max(float(hard_scores[index, 1]), 0.500001)
            hard_scores[index, 2] = 1.0 - hard_scores[index, 1]
        else:
            hard_scores[index, 2] = max(float(hard_scores[index, 2]), 0.500001)
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
