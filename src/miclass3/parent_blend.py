from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strict80 import SearchResult, search_strict80_thresholds


@dataclass(frozen=True)
class BlendSearchResult:
    alpha_primary: float
    result: SearchResult
    rows: list[dict[str, float | int | bool | str]]


def blend_parent_probability(
    p_mi_primary: np.ndarray,
    p_aux_non_mi: np.ndarray,
    alpha_primary: float,
) -> np.ndarray:
    """Blend the primary MI head with MI probability implied by the auxiliary head.

    The auxiliary three-class head yields P(non-MI), P(STEMI-proxy), and
    P(NSTEMI-proxy).  Its parent-level MI evidence is therefore
    1 - P_aux(non-MI).  alpha_primary is intentionally constrained near 1.0 in
    the experiment so the auxiliary signal only re-ranks borderline records.
    """
    primary = np.asarray(p_mi_primary, dtype=float)
    aux_non_mi = np.asarray(p_aux_non_mi, dtype=float)
    if primary.shape != aux_non_mi.shape:
        raise ValueError("primary and auxiliary probabilities must have the same shape")
    alpha = float(alpha_primary)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_primary must be in [0, 1]")
    aux_mi = 1.0 - aux_non_mi
    return np.clip(alpha * primary + (1.0 - alpha) * aux_mi, 0.0, 1.0)


def _alpha_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("alpha step must be positive")
    values = np.arange(float(start), float(stop) + step * 0.5, float(step))
    return np.clip(values, 0.0, 1.0)


def search_parent_blend(
    y_true: np.ndarray,
    p_mi_primary: np.ndarray,
    p_aux_non_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    *,
    target: float = 0.80,
    alpha_min: float = 0.70,
    alpha_max: float = 1.00,
    alpha_coarse_step: float = 0.05,
    alpha_fine_radius: float = 0.05,
    alpha_fine_step: float = 0.01,
    threshold_coarse_step: float = 0.02,
    threshold_fine_radius: float = 0.03,
    threshold_fine_step: float = 0.002,
) -> BlendSearchResult:
    """Validation-only search over a small auxiliary correction to Stage1-v3.

    The primary Stage1-v3 probability remains dominant.  For each alpha, the
    existing strict-80 search selects MI and STEMI|MI thresholds by maximizing
    the weakest of the six class-wise sensitivity/specificity metrics.
    """
    all_rows: list[dict[str, float | int | bool | str]] = []
    best: tuple[tuple[float, int, float, float, float], float, SearchResult] | None = None

    def evaluate_alphas(alphas: np.ndarray, phase: str) -> None:
        nonlocal best
        for alpha in alphas:
            blended = blend_parent_probability(p_mi_primary, p_aux_non_mi, float(alpha))
            result, rows = search_strict80_thresholds(
                y_true,
                blended,
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
                float(alpha),
            )
            all_rows.append(
                {
                    "search_phase": phase,
                    "alpha_primary": float(alpha),
                    "mi_threshold": float(result.mi_threshold),
                    "stemi_threshold": float(result.stemi_threshold),
                    "minimum_of_six": float(summary["minimum_of_six"]),
                    "mean_of_six": float(summary["mean_of_six"]),
                    "n_metrics_strictly_above_target": int(summary["n_metrics_strictly_above_target"]),
                    "all_six_strictly_above_target": bool(summary["all_six_strictly_above_target"]),
                    "accuracy": float(summary["accuracy"]),
                }
            )
            if best is None or key > best[0]:
                best = (key, float(alpha), result)

    coarse_alphas = _alpha_grid(alpha_min, alpha_max, alpha_coarse_step)
    evaluate_alphas(coarse_alphas, "coarse_alpha")
    if best is None:
        raise RuntimeError("alpha search produced no candidates")

    coarse_best_alpha = best[1]
    fine_alphas = _alpha_grid(
        max(alpha_min, coarse_best_alpha - alpha_fine_radius),
        min(alpha_max, coarse_best_alpha + alpha_fine_radius),
        alpha_fine_step,
    )
    evaluate_alphas(fine_alphas, "fine_alpha")
    if best is None:
        raise RuntimeError("alpha search produced no candidates")

    return BlendSearchResult(alpha_primary=best[1], result=best[2], rows=all_rows)
