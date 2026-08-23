from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


@dataclass(frozen=True)
class SearchResult:
    mi_threshold: float
    stemi_threshold: float
    summary: dict[str, object]
    confusion_matrix: np.ndarray
    predictions: np.ndarray


def hierarchical_predict(
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    mi_threshold: float,
    stemi_threshold: float,
) -> np.ndarray:
    """Hard hierarchical decision: non-MI first, then STEMI/NSTEMI within MI."""
    p_mi = np.asarray(p_mi, dtype=float)
    p_stemi_given_mi = np.asarray(p_stemi_given_mi, dtype=float)
    if p_mi.shape != p_stemi_given_mi.shape:
        raise ValueError("p_mi and p_stemi_given_mi must have identical shapes")

    prediction = np.zeros(p_mi.shape[0], dtype=int)
    routed_to_mi = p_mi >= float(mi_threshold)
    prediction[routed_to_mi & (p_stemi_given_mi >= float(stemi_threshold))] = 1
    prediction[routed_to_mi & (p_stemi_given_mi < float(stemi_threshold))] = 2
    return prediction


def six_metric_summary(
    y_true: np.ndarray,
    prediction: np.ndarray,
    target: float = 0.80,
) -> tuple[dict[str, object], np.ndarray]:
    """Return one-vs-rest sensitivity/specificity for all three proxy classes."""
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1, 2])
    total = int(cm.sum())

    per_class: dict[str, dict[str, float | int]] = {}
    six_values: list[float] = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(cm[class_id, class_id])
        fn = int(cm[class_id, :].sum() - tp)
        fp = int(cm[:, class_id].sum() - tp)
        tn = int(total - tp - fn - fp)
        sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
        specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
        per_class[name] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "support": int(tp + fn),
        }
        six_values.extend([sensitivity, specificity])

    finite = np.asarray(six_values, dtype=float)
    minimum = float(np.nanmin(finite))
    mean = float(np.nanmean(finite))
    strictly_above = int(np.sum(finite > float(target)))
    at_or_above = int(np.sum(finite >= float(target)))

    summary: dict[str, object] = {
        "target": float(target),
        "per_class": per_class,
        "minimum_of_six": minimum,
        "mean_of_six": mean,
        "n_metrics_strictly_above_target": strictly_above,
        "n_metrics_at_or_above_target": at_or_above,
        "all_six_strictly_above_target": bool(np.all(finite > float(target))),
        "all_six_at_or_above_target": bool(np.all(finite >= float(target))),
        "accuracy": float(np.trace(cm) / total) if total else float("nan"),
    }
    return summary, cm


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    values = np.arange(float(start), float(stop) + step * 0.5, float(step))
    return np.clip(values, 0.0, 1.0)


def _search_grid(
    y_true: np.ndarray,
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    mi_thresholds: Iterable[float],
    stemi_thresholds: Iterable[float],
    target: float,
) -> tuple[SearchResult, list[dict[str, float | int | bool | str]]]:
    best: SearchResult | None = None
    best_key: tuple[float, int, float, float] | None = None
    rows: list[dict[str, float | int | bool | str]] = []

    for mi_threshold in mi_thresholds:
        for stemi_threshold in stemi_thresholds:
            prediction = hierarchical_predict(
                p_mi,
                p_stemi_given_mi,
                float(mi_threshold),
                float(stemi_threshold),
            )
            summary, cm = six_metric_summary(y_true, prediction, target=target)
            key = (
                float(summary["minimum_of_six"]),
                int(summary["n_metrics_strictly_above_target"]),
                float(summary["mean_of_six"]),
                float(summary["accuracy"]),
            )
            rows.append(
                {
                    "mi_threshold": float(mi_threshold),
                    "stemi_threshold": float(stemi_threshold),
                    "minimum_of_six": float(summary["minimum_of_six"]),
                    "mean_of_six": float(summary["mean_of_six"]),
                    "n_metrics_strictly_above_target": int(summary["n_metrics_strictly_above_target"]),
                    "all_six_strictly_above_target": bool(summary["all_six_strictly_above_target"]),
                    "accuracy": float(summary["accuracy"]),
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best = SearchResult(
                    mi_threshold=float(mi_threshold),
                    stemi_threshold=float(stemi_threshold),
                    summary=summary,
                    confusion_matrix=cm,
                    predictions=prediction,
                )

    if best is None:
        raise RuntimeError("threshold search produced no candidates")
    return best, rows


def search_strict80_thresholds(
    y_true: np.ndarray,
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    target: float = 0.80,
    coarse_step: float = 0.01,
    fine_radius: float = 0.03,
    fine_step: float = 0.001,
) -> tuple[SearchResult, list[dict[str, float | int | bool | str]]]:
    """Validation-only two-stage threshold search maximizing the weakest of six metrics.

    The primary objective is the minimum sensitivity/specificity over the three
    one-vs-rest classes. Ties prefer more metrics strictly above the requested
    target, then a higher mean of the six metrics, then accuracy.
    """
    coarse = _grid(0.0, 1.0, coarse_step)
    coarse_best, coarse_rows = _search_grid(
        y_true,
        p_mi,
        p_stemi_given_mi,
        coarse,
        coarse,
        target,
    )

    mi_fine = _grid(
        max(0.0, coarse_best.mi_threshold - fine_radius),
        min(1.0, coarse_best.mi_threshold + fine_radius),
        fine_step,
    )
    stemi_fine = _grid(
        max(0.0, coarse_best.stemi_threshold - fine_radius),
        min(1.0, coarse_best.stemi_threshold + fine_radius),
        fine_step,
    )
    fine_best, fine_rows = _search_grid(
        y_true,
        p_mi,
        p_stemi_given_mi,
        mi_fine,
        stemi_fine,
        target,
    )
    for row in coarse_rows:
        row["search_phase"] = "coarse"
    for row in fine_rows:
        row["search_phase"] = "fine"
    return fine_best, coarse_rows + fine_rows
