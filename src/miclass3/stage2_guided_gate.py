from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strict80 import six_metric_summary


@dataclass(frozen=True)
class Stage2GuidedResult:
    mi_threshold: float
    stemi_threshold: float
    rescue_floor: float
    rescue_stemi_max: float
    summary: dict[str, object]
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    rescue_mask: np.ndarray


def stage2_guided_predict(
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    *,
    mi_threshold: float,
    stemi_threshold: float,
    rescue_floor: float,
    rescue_stemi_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard cascade with a narrow Stage2-guided rescue for parent-gate misses.

    The normal Stage1 MI threshold remains the default gate. A record that
    falls below that gate can still enter the MI branch only when:

    1. P(MI) is close to the parent boundary (>= rescue_floor), and
    2. Stage2 is strongly NSTEMI-like (P(STEMI|MI) <= rescue_stemi_max).

    The rescue threshold is constrained by the search to be no larger than the
    ordinary STEMI threshold, so rescued records are a targeted NSTEMI path.
    No labels, SCP codes, or rule outputs are used at inference time.
    """
    p_mi = np.asarray(p_mi, dtype=float)
    p_stemi = np.asarray(p_stemi_given_mi, dtype=float)
    if p_mi.shape != p_stemi.shape:
        raise ValueError("p_mi and p_stemi_given_mi must have identical shapes")

    primary_route = p_mi >= float(mi_threshold)
    rescue = (
        (~primary_route)
        & (p_mi >= float(rescue_floor))
        & (p_stemi <= float(rescue_stemi_max))
    )
    routed = primary_route | rescue

    prediction = np.zeros(p_mi.shape[0], dtype=int)
    prediction[routed & (p_stemi >= float(stemi_threshold))] = 1
    prediction[routed & (p_stemi < float(stemi_threshold))] = 2
    return prediction, rescue


def _values(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    return np.arange(float(start), float(stop) + 0.5 * float(step), float(step))


def search_stage2_guided_gate(
    y_true: np.ndarray,
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    *,
    target: float = 0.80,
    mi_center: float = 0.324,
    stemi_center: float = 0.276,
    mi_radius: float = 0.05,
    stemi_radius: float = 0.04,
    threshold_step: float = 0.004,
    rescue_band_min: float = 0.02,
    rescue_band_max: float = 0.18,
    rescue_band_step: float = 0.02,
    rescue_stemi_max_min: float = 0.04,
    rescue_stemi_max_max: float = 0.28,
    rescue_stemi_max_step: float = 0.02,
) -> tuple[Stage2GuidedResult, list[dict[str, object]]]:
    """Validation-only search for a conservative Stage2-guided parent rescue."""
    y = np.asarray(y_true, dtype=int)
    p_mi = np.asarray(p_mi, dtype=float)
    p_stemi = np.asarray(p_stemi_given_mi, dtype=float)

    mi_values = np.clip(
        _values(mi_center - mi_radius, mi_center + mi_radius, threshold_step), 0.0, 1.0
    )
    stemi_values = np.clip(
        _values(stemi_center - stemi_radius, stemi_center + stemi_radius, threshold_step),
        0.0,
        1.0,
    )
    band_values = _values(rescue_band_min, rescue_band_max, rescue_band_step)
    rescue_stemi_values = np.clip(
        _values(rescue_stemi_max_min, rescue_stemi_max_max, rescue_stemi_max_step),
        0.0,
        1.0,
    )

    best_key = None
    best = None
    rows: list[dict[str, object]] = []

    for t_mi in mi_values:
        for t_stemi in stemi_values:
            # A rescue is intended only for samples that Stage2 itself would
            # regard as NSTEMI-like, never for STEMI-like samples.
            allowed_rescue_stemi = rescue_stemi_values[rescue_stemi_values <= t_stemi]
            for band in band_values:
                floor = max(0.0, float(t_mi - band))
                for rescue_stemi_max in allowed_rescue_stemi:
                    pred, rescue = stage2_guided_predict(
                        p_mi,
                        p_stemi,
                        mi_threshold=float(t_mi),
                        stemi_threshold=float(t_stemi),
                        rescue_floor=floor,
                        rescue_stemi_max=float(rescue_stemi_max),
                    )
                    summary, cm = six_metric_summary(y, pred, target=target)
                    rescued_total = int(rescue.sum())
                    rescued_nstemi = int(np.sum(rescue & (y == 2)))
                    rescued_nonmi = int(np.sum(rescue & (y == 0)))
                    rescued_stemi = int(np.sum(rescue & (y == 1)))
                    key = (
                        float(summary["minimum_of_six"]),
                        int(summary["n_metrics_strictly_above_target"]),
                        float(summary["mean_of_six"]),
                        float(summary["accuracy"]),
                        -rescued_nonmi,
                        rescued_nstemi,
                        -float(band),
                        -float(rescue_stemi_max),
                    )
                    rows.append(
                        {
                            "mi_threshold": float(t_mi),
                            "stemi_threshold": float(t_stemi),
                            "rescue_floor": floor,
                            "rescue_band": float(band),
                            "rescue_stemi_max": float(rescue_stemi_max),
                            "rescued_total": rescued_total,
                            "rescued_nonmi": rescued_nonmi,
                            "rescued_stemi": rescued_stemi,
                            "rescued_nstemi": rescued_nstemi,
                            "minimum_of_six": float(summary["minimum_of_six"]),
                            "mean_of_six": float(summary["mean_of_six"]),
                            "accuracy": float(summary["accuracy"]),
                            "all_six_strictly_above_target": bool(
                                summary["all_six_strictly_above_target"]
                            ),
                        }
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best = Stage2GuidedResult(
                            mi_threshold=float(t_mi),
                            stemi_threshold=float(t_stemi),
                            rescue_floor=floor,
                            rescue_stemi_max=float(rescue_stemi_max),
                            summary=summary,
                            confusion_matrix=cm,
                            predictions=pred,
                            rescue_mask=rescue,
                        )

    if best is None:
        raise RuntimeError("Stage2-guided rescue search produced no candidates")
    return best, rows
