from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strict80 import SearchResult, search_strict80_thresholds


@dataclass(frozen=True)
class DirectMIParentBlendResult:
    alpha_stage1: float
    result: SearchResult
    rows: list[dict[str, object]]


def _logit(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def blend_parent_log_odds(
    p_stage1_mi: np.ndarray,
    p_direct_mi_aux: np.ndarray,
    alpha_stage1: float,
) -> np.ndarray:
    """Blend independent parent-level MI evidence in log-odds space.

    alpha_stage1=1 reproduces Stage1-v3 parent probabilities, while
    alpha_stage1=0 uses only the Direct MorphologyFusion MI auxiliary head.
    """
    stage1 = np.asarray(p_stage1_mi, dtype=float)
    direct = np.asarray(p_direct_mi_aux, dtype=float)
    if stage1.shape != direct.shape:
        raise ValueError("Stage1 and Direct MI probabilities must have identical shapes")
    alpha = float(alpha_stage1)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_stage1 must be in [0, 1]")
    blended_logit = alpha * _logit(stage1) + (1.0 - alpha) * _logit(direct)
    return 1.0 / (1.0 + np.exp(-np.clip(blended_logit, -40.0, 40.0)))


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    return np.clip(
        np.arange(float(start), float(stop) + 0.5 * float(step), float(step)),
        0.0,
        1.0,
    )


def search_direct_mi_parent_blend(
    y_true: np.ndarray,
    p_stage1_mi: np.ndarray,
    p_direct_mi_aux: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    *,
    target: float = 0.80,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    alpha_coarse_step: float = 0.05,
    alpha_fine_radius: float = 0.05,
    alpha_fine_step: float = 0.01,
    threshold_coarse_step: float = 0.01,
    threshold_fine_radius: float = 0.03,
    threshold_fine_step: float = 0.001,
) -> DirectMIParentBlendResult:
    """Validation-only search over one parent blend weight and two cascade thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    p_stage1_mi = np.asarray(p_stage1_mi, dtype=float)
    p_direct_mi_aux = np.asarray(p_direct_mi_aux, dtype=float)
    p_stemi = np.asarray(p_stemi_given_mi, dtype=float)
    if not (
        y_true.shape == p_stage1_mi.shape == p_direct_mi_aux.shape == p_stemi.shape
    ):
        raise ValueError("labels and probability arrays must have identical shapes")

    best_key = None
    best_alpha = None
    best_result = None
    all_rows: list[dict[str, object]] = []

    def evaluate(alphas: np.ndarray, phase: str) -> None:
        nonlocal best_key, best_alpha, best_result
        for alpha in np.unique(alphas):
            parent = blend_parent_log_odds(p_stage1_mi, p_direct_mi_aux, float(alpha))
            result, _ = search_strict80_thresholds(
                y_true,
                parent,
                p_stemi,
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
                -abs(float(alpha) - 0.5),
            )
            all_rows.append(
                {
                    "phase": phase,
                    "alpha_stage1": float(alpha),
                    "alpha_direct_mi_aux": 1.0 - float(alpha),
                    "mi_threshold": float(result.mi_threshold),
                    "stemi_threshold": float(result.stemi_threshold),
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
                best_alpha = float(alpha)
                best_result = result

    coarse = _grid(alpha_min, alpha_max, alpha_coarse_step)
    evaluate(coarse, "coarse_alpha")
    if best_alpha is None or best_result is None:
        raise RuntimeError("parent blend coarse search produced no candidates")

    fine = _grid(
        max(alpha_min, best_alpha - alpha_fine_radius),
        min(alpha_max, best_alpha + alpha_fine_radius),
        alpha_fine_step,
    )
    evaluate(fine, "fine_alpha")
    if best_alpha is None or best_result is None:
        raise RuntimeError("parent blend fine search produced no candidates")

    return DirectMIParentBlendResult(
        alpha_stage1=best_alpha,
        result=best_result,
        rows=all_rows,
    )
