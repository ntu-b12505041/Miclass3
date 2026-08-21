from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strict80 import hierarchical_predict, six_metric_summary


@dataclass(frozen=True)
class RescueResult:
    mi_threshold: float
    stemi_threshold: float
    rescue_margin: float
    rescue_floor: float
    summary: dict[str, object]
    confusion_matrix: np.ndarray
    predictions: np.ndarray


def nstemi_rescue_predict(
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    p_aux_non_mi: np.ndarray,
    p_aux_nstemi: np.ndarray,
    *,
    mi_threshold: float,
    stemi_threshold: float,
    rescue_margin: float,
    rescue_floor: float,
) -> np.ndarray:
    """Hard cascade with a narrow auxiliary NSTEMI rescue path.

    The standard Stage1 MI gate remains the primary decision. Records that
    fail the gate can be rescued only when both conditions hold:

    1. primary P(MI) is still near the MI boundary (>= rescue_floor), and
    2. the auxiliary subtype head prefers NSTEMI over non-MI by at least
       rescue_margin.

    Rescued records are sent through the existing Stage2 classifier rather
    than being forced directly to NSTEMI, so Stage2 remains the subtype judge.
    """
    p_mi = np.asarray(p_mi, dtype=float)
    p_stemi = np.asarray(p_stemi_given_mi, dtype=float)
    aux_non = np.asarray(p_aux_non_mi, dtype=float)
    aux_nstemi = np.asarray(p_aux_nstemi, dtype=float)
    if not (p_mi.shape == p_stemi.shape == aux_non.shape == aux_nstemi.shape):
        raise ValueError("all probability arrays must have identical shapes")

    prediction = hierarchical_predict(p_mi, p_stemi, mi_threshold, stemi_threshold)
    failed_parent_gate = p_mi < float(mi_threshold)
    aux_margin = aux_nstemi - aux_non
    rescue = (
        failed_parent_gate
        & (p_mi >= float(rescue_floor))
        & (aux_margin >= float(rescue_margin))
    )
    prediction[rescue & (p_stemi >= float(stemi_threshold))] = 1
    prediction[rescue & (p_stemi < float(stemi_threshold))] = 2
    return prediction


def _values(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    return np.arange(float(start), float(stop) + step * 0.5, float(step))


def search_nstemi_rescue(
    y_true: np.ndarray,
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    p_aux_non_mi: np.ndarray,
    p_aux_nstemi: np.ndarray,
    *,
    target: float = 0.80,
    mi_center: float = 0.324,
    stemi_center: float = 0.276,
    mi_radius: float = 0.06,
    stemi_radius: float = 0.04,
    threshold_step: float = 0.004,
    rescue_margin_min: float = -0.10,
    rescue_margin_max: float = 0.30,
    rescue_margin_step: float = 0.04,
    rescue_band_min: float = 0.02,
    rescue_band_max: float = 0.16,
    rescue_band_step: float = 0.02,
) -> tuple[RescueResult, list[dict[str, object]]]:
    """Validation-only local search around the accepted strict-80 cascade.

    rescue_floor is expressed as mi_threshold - rescue_band, ensuring rescue
    applies only to records near the original parent decision boundary.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_mi = np.asarray(p_mi, dtype=float)
    p_stemi = np.asarray(p_stemi_given_mi, dtype=float)
    aux_non = np.asarray(p_aux_non_mi, dtype=float)
    aux_nstemi = np.asarray(p_aux_nstemi, dtype=float)

    mi_values = np.clip(_values(mi_center - mi_radius, mi_center + mi_radius, threshold_step), 0, 1)
    st_values = np.clip(_values(stemi_center - stemi_radius, stemi_center + stemi_radius, threshold_step), 0, 1)
    margin_values = _values(rescue_margin_min, rescue_margin_max, rescue_margin_step)
    band_values = _values(rescue_band_min, rescue_band_max, rescue_band_step)

    best_key = None
    best = None
    rows: list[dict[str, object]] = []

    for t_mi in mi_values:
        for t_st in st_values:
            for band in band_values:
                floor = max(0.0, float(t_mi - band))
                for margin in margin_values:
                    pred = nstemi_rescue_predict(
                        p_mi,
                        p_stemi,
                        aux_non,
                        aux_nstemi,
                        mi_threshold=float(t_mi),
                        stemi_threshold=float(t_st),
                        rescue_margin=float(margin),
                        rescue_floor=floor,
                    )
                    summary, cm = six_metric_summary(y_true, pred, target=target)
                    key = (
                        float(summary["minimum_of_six"]),
                        int(summary["n_metrics_strictly_above_target"]),
                        float(summary["mean_of_six"]),
                        float(summary["accuracy"]),
                        -float(band),
                        float(margin),
                    )
                    rows.append(
                        {
                            "mi_threshold": float(t_mi),
                            "stemi_threshold": float(t_st),
                            "rescue_band": float(band),
                            "rescue_floor": floor,
                            "rescue_margin": float(margin),
                            "minimum_of_six": float(summary["minimum_of_six"]),
                            "mean_of_six": float(summary["mean_of_six"]),
                            "accuracy": float(summary["accuracy"]),
                            "all_six_strictly_above_target": bool(summary["all_six_strictly_above_target"]),
                        }
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best = RescueResult(
                            mi_threshold=float(t_mi),
                            stemi_threshold=float(t_st),
                            rescue_margin=float(margin),
                            rescue_floor=floor,
                            summary=summary,
                            confusion_matrix=cm,
                            predictions=pred,
                        )
    if best is None:
        raise RuntimeError("NSTEMI rescue search produced no candidates")
    return best, rows
