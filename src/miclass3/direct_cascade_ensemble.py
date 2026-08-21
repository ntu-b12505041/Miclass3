from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strict80 import six_metric_summary


@dataclass(frozen=True)
class EnsembleResult:
    alpha_direct: float
    stemi_bias: float
    nstemi_bias: float
    summary: dict[str, object]
    confusion_matrix: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray


def cascade_soft_probabilities(p_mi: np.ndarray, p_stemi_given_mi: np.ndarray) -> np.ndarray:
    p_mi = np.clip(np.asarray(p_mi, dtype=float), 0.0, 1.0)
    p_stemi = np.clip(np.asarray(p_stemi_given_mi, dtype=float), 0.0, 1.0)
    if p_mi.shape != p_stemi.shape:
        raise ValueError("p_mi and p_stemi_given_mi must have identical shapes")
    return np.column_stack(
        [
            1.0 - p_mi,
            p_mi * p_stemi,
            p_mi * (1.0 - p_stemi),
        ]
    )


def blend_probabilities(
    direct_probabilities: np.ndarray,
    cascade_probabilities: np.ndarray,
    alpha_direct: float,
) -> np.ndarray:
    direct = np.asarray(direct_probabilities, dtype=float)
    cascade = np.asarray(cascade_probabilities, dtype=float)
    if direct.shape != cascade.shape or direct.ndim != 2 or direct.shape[1] != 3:
        raise ValueError("direct and cascade probabilities must both have shape (n, 3)")
    alpha = float(alpha_direct)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_direct must be in [0, 1]")
    mixed = alpha * direct + (1.0 - alpha) * cascade
    mixed = np.clip(mixed, 1e-8, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def apply_class_biases(
    probabilities: np.ndarray,
    stemi_bias: float,
    nstemi_bias: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    logits[:, 1] += float(stemi_bias)
    logits[:, 2] += float(nstemi_bias)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("grid step must be positive")
    return np.arange(float(start), float(stop) + 0.5 * float(step), float(step))


def search_direct_cascade_ensemble(
    y_true: np.ndarray,
    direct_probabilities: np.ndarray,
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    *,
    target: float = 0.80,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    alpha_coarse_step: float = 0.05,
    bias_min: float = -1.2,
    bias_max: float = 1.2,
    bias_coarse_step: float = 0.10,
    alpha_fine_radius: float = 0.05,
    alpha_fine_step: float = 0.01,
    bias_fine_radius: float = 0.10,
    bias_fine_step: float = 0.02,
) -> tuple[EnsembleResult, list[dict[str, object]]]:
    y_true = np.asarray(y_true, dtype=int)
    direct = np.asarray(direct_probabilities, dtype=float)
    cascade = cascade_soft_probabilities(p_mi, p_stemi_given_mi)
    if len(y_true) != len(direct) or len(y_true) != len(cascade):
        raise ValueError("labels and probability arrays must have matching lengths")

    rows: list[dict[str, object]] = []
    best_key = None
    best: EnsembleResult | None = None

    def evaluate(alphas: np.ndarray, st_biases: np.ndarray, ns_biases: np.ndarray, phase: str) -> None:
        nonlocal best_key, best
        for alpha in alphas:
            mixed = blend_probabilities(direct, cascade, float(alpha))
            for st_bias in st_biases:
                for ns_bias in ns_biases:
                    calibrated = apply_class_biases(mixed, float(st_bias), float(ns_bias))
                    pred = calibrated.argmax(axis=1)
                    summary, cm = six_metric_summary(y_true, pred, target=target)
                    key = (
                        float(summary["minimum_of_six"]),
                        int(summary["n_metrics_strictly_above_target"]),
                        float(summary["mean_of_six"]),
                        float(summary["accuracy"]),
                        -abs(float(alpha) - 0.5),
                        -abs(float(st_bias)) - abs(float(ns_bias)),
                    )
                    rows.append(
                        {
                            "phase": phase,
                            "alpha_direct": float(alpha),
                            "stemi_bias": float(st_bias),
                            "nstemi_bias": float(ns_bias),
                            "minimum_of_six": float(summary["minimum_of_six"]),
                            "mean_of_six": float(summary["mean_of_six"]),
                            "accuracy": float(summary["accuracy"]),
                            "all_six_strictly_above_target": bool(summary["all_six_strictly_above_target"]),
                        }
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best = EnsembleResult(
                            alpha_direct=float(alpha),
                            stemi_bias=float(st_bias),
                            nstemi_bias=float(ns_bias),
                            summary=summary,
                            confusion_matrix=cm,
                            probabilities=calibrated,
                            predictions=pred,
                        )

    evaluate(
        _grid(alpha_min, alpha_max, alpha_coarse_step),
        _grid(bias_min, bias_max, bias_coarse_step),
        _grid(bias_min, bias_max, bias_coarse_step),
        "coarse",
    )
    if best is None:
        raise RuntimeError("ensemble coarse search produced no candidates")

    coarse = best
    evaluate(
        np.clip(
            _grid(coarse.alpha_direct - alpha_fine_radius, coarse.alpha_direct + alpha_fine_radius, alpha_fine_step),
            alpha_min,
            alpha_max,
        ),
        _grid(coarse.stemi_bias - bias_fine_radius, coarse.stemi_bias + bias_fine_radius, bias_fine_step),
        _grid(coarse.nstemi_bias - bias_fine_radius, coarse.nstemi_bias + bias_fine_radius, bias_fine_step),
        "fine",
    )
    if best is None:
        raise RuntimeError("ensemble fine search produced no candidates")
    return best, rows
