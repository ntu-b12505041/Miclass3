from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .strict80 import six_metric_summary


@dataclass(frozen=True)
class DirectAuxHierarchyResult:
    mi_threshold: float
    stemi_threshold: float
    summary: dict[str, object]
    confusion_matrix: np.ndarray
    predictions: np.ndarray


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def direct_aux_hierarchy_predict(
    p_mi_aux: np.ndarray,
    p_stemi_aux: np.ndarray,
    mi_threshold: float,
    stemi_threshold: float,
) -> np.ndarray:
    """Decode the Direct model's existing binary auxiliary heads hierarchically.

    Stage 1 uses the Direct model's MI-vs-non-MI auxiliary head. Records routed
    to MI are then split by its STEMI-vs-rest auxiliary head. No ECG weights are
    retrained and no target-side features are introduced.
    """
    p_mi = np.asarray(p_mi_aux, dtype=float)
    p_stemi = np.asarray(p_stemi_aux, dtype=float)
    if p_mi.shape != p_stemi.shape:
        raise ValueError("p_mi_aux and p_stemi_aux must have identical shapes")

    pred = np.zeros(len(p_mi), dtype=int)
    routed = p_mi >= float(mi_threshold)
    pred[routed & (p_stemi >= float(stemi_threshold))] = 1
    pred[routed & (p_stemi < float(stemi_threshold))] = 2
    return pred


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    return np.arange(float(start), float(stop) + 0.5 * float(step), float(step))


def _search(
    y_true: np.ndarray,
    p_mi: np.ndarray,
    p_stemi: np.ndarray,
    mi_values: Iterable[float],
    stemi_values: Iterable[float],
    target: float,
) -> tuple[DirectAuxHierarchyResult, list[dict[str, object]]]:
    best = None
    best_key = None
    rows: list[dict[str, object]] = []
    for t_mi in mi_values:
        for t_st in stemi_values:
            pred = direct_aux_hierarchy_predict(p_mi, p_stemi, float(t_mi), float(t_st))
            summary, cm = six_metric_summary(y_true, pred, target=target)
            key = (
                float(summary["minimum_of_six"]),
                int(summary["n_metrics_strictly_above_target"]),
                float(summary["mean_of_six"]),
                float(summary["accuracy"]),
            )
            rows.append(
                {
                    "mi_threshold": float(t_mi),
                    "stemi_threshold": float(t_st),
                    "minimum_of_six": float(summary["minimum_of_six"]),
                    "mean_of_six": float(summary["mean_of_six"]),
                    "accuracy": float(summary["accuracy"]),
                    "all_six_strictly_above_target": bool(summary["all_six_strictly_above_target"]),
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best = DirectAuxHierarchyResult(
                    mi_threshold=float(t_mi),
                    stemi_threshold=float(t_st),
                    summary=summary,
                    confusion_matrix=cm,
                    predictions=pred,
                )
    if best is None:
        raise RuntimeError("Direct auxiliary hierarchy search produced no candidates")
    return best, rows


def search_direct_aux_hierarchy(
    y_true: np.ndarray,
    p_mi_aux: np.ndarray,
    p_stemi_aux: np.ndarray,
    *,
    target: float = 0.80,
    coarse_step: float = 0.01,
    fine_radius: float = 0.03,
    fine_step: float = 0.001,
) -> tuple[DirectAuxHierarchyResult, list[dict[str, object]]]:
    """Fold-9-only threshold search maximizing the weakest of six metrics."""
    y = np.asarray(y_true, dtype=int)
    p_mi = np.asarray(p_mi_aux, dtype=float)
    p_stemi = np.asarray(p_stemi_aux, dtype=float)
    coarse_values = _grid(0.0, 1.0, coarse_step)
    coarse_best, coarse_rows = _search(
        y, p_mi, p_stemi, coarse_values, coarse_values, target
    )
    mi_fine = _grid(
        max(0.0, coarse_best.mi_threshold - fine_radius),
        min(1.0, coarse_best.mi_threshold + fine_radius),
        fine_step,
    )
    st_fine = _grid(
        max(0.0, coarse_best.stemi_threshold - fine_radius),
        min(1.0, coarse_best.stemi_threshold + fine_radius),
        fine_step,
    )
    fine_best, fine_rows = _search(y, p_mi, p_stemi, mi_fine, st_fine, target)
    for row in coarse_rows:
        row["phase"] = "coarse"
    for row in fine_rows:
        row["phase"] = "fine"
    return fine_best, coarse_rows + fine_rows
