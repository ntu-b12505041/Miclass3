from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strict80 import SearchResult, search_strict80_thresholds


@dataclass(frozen=True)
class ThreeWayParentBlendResult:
    weight_stage1: float
    weight_direct_aux: float
    weight_direct_class: float
    result: SearchResult
    rows: list[dict[str, object]]


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    out = np.empty_like(value, dtype=float)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def blend_three_parent_log_odds(
    p_stage1_mi: np.ndarray,
    p_direct_aux_mi: np.ndarray,
    p_direct_class_mi: np.ndarray,
    weight_stage1: float,
    weight_direct_aux: float,
) -> np.ndarray:
    """Blend three existing MI-vs-nonMI signals in log-odds space.

    The third weight is implicit: 1 - weight_stage1 - weight_direct_aux.
    All three weights must be non-negative.  This keeps the experiment a
    calibration/fusion change only; no ECG model is retrained.
    """
    p1 = np.asarray(p_stage1_mi, dtype=float)
    p2 = np.asarray(p_direct_aux_mi, dtype=float)
    p3 = np.asarray(p_direct_class_mi, dtype=float)
    if not (p1.shape == p2.shape == p3.shape):
        raise ValueError("all parent probability arrays must have identical shapes")
    w1 = float(weight_stage1)
    w2 = float(weight_direct_aux)
    w3 = 1.0 - w1 - w2
    if w1 < -1e-12 or w2 < -1e-12 or w3 < -1e-12:
        raise ValueError("parent blend weights must be non-negative and sum to <= 1")
    w1 = max(w1, 0.0)
    w2 = max(w2, 0.0)
    w3 = max(w3, 0.0)
    score = w1 * _logit(p1) + w2 * _logit(p2) + w3 * _logit(p3)
    return _sigmoid(score)


def _simplex_grid(step: float) -> list[tuple[float, float]]:
    if step <= 0:
        raise ValueError("weight step must be positive")
    values = np.arange(0.0, 1.0 + step * 0.5, step)
    pairs: list[tuple[float, float]] = []
    for w1 in values:
        for w2 in values:
            if w1 + w2 <= 1.0 + 1e-9:
                pairs.append((float(min(w1, 1.0)), float(min(w2, 1.0))))
    return pairs


def _fine_simplex_grid(
    center_stage1: float,
    center_aux: float,
    radius: float,
    step: float,
) -> list[tuple[float, float]]:
    values1 = np.arange(max(0.0, center_stage1 - radius), min(1.0, center_stage1 + radius) + step * 0.5, step)
    values2 = np.arange(max(0.0, center_aux - radius), min(1.0, center_aux + radius) + step * 0.5, step)
    pairs = []
    for w1 in values1:
        for w2 in values2:
            if w1 + w2 <= 1.0 + 1e-9:
                pairs.append((float(w1), float(w2)))
    return pairs


def search_threeway_parent_blend(
    y_true: np.ndarray,
    p_stage1_mi: np.ndarray,
    p_direct_aux_mi: np.ndarray,
    p_direct_class_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    *,
    target: float = 0.80,
    weight_coarse_step: float = 0.10,
    weight_fine_radius: float = 0.10,
    weight_fine_step: float = 0.02,
    threshold_coarse_step: float = 0.02,
    threshold_fine_radius: float = 0.03,
    threshold_fine_step: float = 0.002,
) -> ThreeWayParentBlendResult:
    """Fold-9-only search over three parent MI signals and two cascade thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    all_rows: list[dict[str, object]] = []
    best_key = None
    best_weights: tuple[float, float] | None = None
    best_result: SearchResult | None = None

    def evaluate(weight_pairs: list[tuple[float, float]], phase: str) -> None:
        nonlocal best_key, best_weights, best_result
        for w_stage1, w_aux in weight_pairs:
            w_class = 1.0 - w_stage1 - w_aux
            parent = blend_three_parent_log_odds(
                p_stage1_mi,
                p_direct_aux_mi,
                p_direct_class_mi,
                w_stage1,
                w_aux,
            )
            result, _ = search_strict80_thresholds(
                y_true,
                parent,
                p_stemi_given_mi,
                target=target,
                coarse_step=threshold_coarse_step,
                fine_radius=threshold_fine_radius,
                fine_step=threshold_fine_step,
            )
            summary = result.summary
            key = (
                float(summary["minimum_of_six"]),
                int(summary["n_metrics_strictly_above_target"]),
                float(summary["mean_of_six"]),
                float(summary["accuracy"]),
                -max(w_stage1, w_aux, w_class),
            )
            all_rows.append(
                {
                    "phase": phase,
                    "weight_stage1": w_stage1,
                    "weight_direct_aux": w_aux,
                    "weight_direct_class": w_class,
                    "mi_threshold": result.mi_threshold,
                    "stemi_threshold": result.stemi_threshold,
                    "minimum_of_six": summary["minimum_of_six"],
                    "mean_of_six": summary["mean_of_six"],
                    "accuracy": summary["accuracy"],
                    "all_six_strictly_above_target": summary["all_six_strictly_above_target"],
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_weights = (w_stage1, w_aux)
                best_result = result

    evaluate(_simplex_grid(weight_coarse_step), "coarse")
    if best_weights is None or best_result is None:
        raise RuntimeError("three-way parent coarse search produced no candidates")
    coarse_best = best_weights
    evaluate(
        _fine_simplex_grid(
            coarse_best[0],
            coarse_best[1],
            weight_fine_radius,
            weight_fine_step,
        ),
        "fine",
    )
    if best_weights is None or best_result is None:
        raise RuntimeError("three-way parent fine search produced no candidates")
    w1, w2 = best_weights
    return ThreeWayParentBlendResult(
        weight_stage1=w1,
        weight_direct_aux=w2,
        weight_direct_class=1.0 - w1 - w2,
        result=best_result,
        rows=all_rows,
    )
