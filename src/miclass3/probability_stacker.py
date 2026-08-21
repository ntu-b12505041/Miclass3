from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .strict80 import six_metric_summary

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")
FEATURE_NAMES = (
    "mi_logit",
    "stemi_given_mi_logit",
    "mi_x_stemi_interaction",
    "stemi_confidence",
    "aux_mi_logit",
    "aux_stemi_vs_nonmi",
    "aux_nstemi_vs_nonmi",
)


def _clip_probability(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def _logit(values: np.ndarray) -> np.ndarray:
    p = _clip_probability(values)
    return np.log(p) - np.log1p(-p)


def build_stacker_features(
    p_mi: np.ndarray,
    p_stemi_given_mi: np.ndarray,
    p_aux_non_mi: np.ndarray,
    p_aux_stemi: np.ndarray,
    p_aux_nstemi: np.ndarray,
) -> np.ndarray:
    """Build a tiny set of probability-derived features for final calibration.

    No ECG backbone is retrained.  The stacker only sees outputs already
    produced by Stage1-v3 and the existing Stage2 classifier.
    """
    arrays = [
        np.asarray(p_mi, dtype=float),
        np.asarray(p_stemi_given_mi, dtype=float),
        np.asarray(p_aux_non_mi, dtype=float),
        np.asarray(p_aux_stemi, dtype=float),
        np.asarray(p_aux_nstemi, dtype=float),
    ]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays[1:]):
        raise ValueError("all probability arrays must have identical shapes")

    p_mi, p_stemi, aux_non, aux_stemi, aux_nstemi = arrays
    mi_logit = _logit(p_mi)
    stemi_logit = _logit(p_stemi)
    aux_mi_logit = _logit(1.0 - aux_non)
    eps = 1e-5
    aux_stemi_vs_non = np.log(np.clip(aux_stemi, eps, 1.0)) - np.log(
        np.clip(aux_non, eps, 1.0)
    )
    aux_nstemi_vs_non = np.log(np.clip(aux_nstemi, eps, 1.0)) - np.log(
        np.clip(aux_non, eps, 1.0)
    )
    return np.column_stack(
        [
            mi_logit,
            stemi_logit,
            mi_logit * stemi_logit,
            np.abs(stemi_logit),
            aux_mi_logit,
            aux_stemi_vs_non,
            aux_nstemi_vs_non,
        ]
    ).astype(np.float64)


def class_weight_dict(y: np.ndarray, power: float) -> dict[int, float]:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts ** (-float(power))
    weights /= weights.mean()
    return {i: float(weights[i]) for i in range(3)}


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)) / np.asarray(
        scale, dtype=float
    )


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def model_probabilities(
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    z = standardize(x, mean, scale)
    return softmax(z @ np.asarray(coef, dtype=float).T + np.asarray(intercept, dtype=float))


def apply_class_biases(
    probabilities: np.ndarray,
    stemi_bias: float,
    nstemi_bias: float,
) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    scores = np.log(p)
    scores[:, 1] += float(stemi_bias)
    scores[:, 2] += float(nstemi_bias)
    return softmax(scores)


@dataclass(frozen=True)
class BiasResult:
    stemi_bias: float
    nstemi_bias: float
    probabilities: np.ndarray
    predictions: np.ndarray
    summary: dict[str, object]
    confusion_matrix: np.ndarray


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    return np.arange(float(start), float(stop) + step * 0.5, float(step))


def search_class_biases(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    target: float = 0.80,
    bias_min: float = -0.8,
    bias_max: float = 0.8,
    coarse_step: float = 0.1,
    fine_radius: float = 0.15,
    fine_step: float = 0.02,
) -> BiasResult:
    y_true = np.asarray(y_true, dtype=int)
    best_key = None
    best = None

    def evaluate(stemi_values: Iterable[float], nstemi_values: Iterable[float]) -> None:
        nonlocal best_key, best
        for b_st in stemi_values:
            for b_ns in nstemi_values:
                calibrated = apply_class_biases(probabilities, float(b_st), float(b_ns))
                pred = calibrated.argmax(axis=1)
                summary, cm = six_metric_summary(y_true, pred, target=target)
                key = (
                    float(summary["minimum_of_six"]),
                    int(summary["n_metrics_strictly_above_target"]),
                    float(summary["mean_of_six"]),
                    float(summary["accuracy"]),
                    -abs(float(b_st)) - abs(float(b_ns)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = BiasResult(
                        stemi_bias=float(b_st),
                        nstemi_bias=float(b_ns),
                        probabilities=calibrated,
                        predictions=pred,
                        summary=summary,
                        confusion_matrix=cm,
                    )

    coarse = _grid(bias_min, bias_max, coarse_step)
    evaluate(coarse, coarse)
    if best is None:
        raise RuntimeError("class-bias search produced no candidates")
    stemi_fine = _grid(
        max(bias_min, best.stemi_bias - fine_radius),
        min(bias_max, best.stemi_bias + fine_radius),
        fine_step,
    )
    nstemi_fine = _grid(
        max(bias_min, best.nstemi_bias - fine_radius),
        min(bias_max, best.nstemi_bias + fine_radius),
        fine_step,
    )
    evaluate(stemi_fine, nstemi_fine)
    if best is None:
        raise RuntimeError("class-bias search produced no candidates")
    return best


@dataclass(frozen=True)
class StackerSearchResult:
    c_value: float
    class_weight_power: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    bias_result: BiasResult


def search_logistic_stacker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_values: Iterable[float],
    class_weight_powers: Iterable[float],
    target: float = 0.80,
    bias_min: float = -0.8,
    bias_max: float = 0.8,
    bias_coarse_step: float = 0.1,
    bias_fine_radius: float = 0.15,
    bias_fine_step: float = 0.02,
) -> tuple[StackerSearchResult, list[dict[str, object]]]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    x_val = np.asarray(x_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)
    mean, scale = fit_standardizer(x_train)
    z_train = standardize(x_train, mean, scale)
    z_val = standardize(x_val, mean, scale)

    best_key = None
    best = None
    rows: list[dict[str, object]] = []
    for c_value in c_values:
        for power in class_weight_powers:
            model = LogisticRegression(
                C=float(c_value),
                class_weight=class_weight_dict(y_train, float(power)),
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            )
            model.fit(z_train, y_train)
            if not np.array_equal(model.classes_, np.array([0, 1, 2])):
                raise ValueError(f"unexpected logistic classes: {model.classes_}")
            val_prob = model.predict_proba(z_val)
            bias = search_class_biases(
                y_val,
                val_prob,
                target=target,
                bias_min=bias_min,
                bias_max=bias_max,
                coarse_step=bias_coarse_step,
                fine_radius=bias_fine_radius,
                fine_step=bias_fine_step,
            )
            summary = bias.summary
            key = (
                float(summary["minimum_of_six"]),
                int(summary["n_metrics_strictly_above_target"]),
                float(summary["mean_of_six"]),
                float(summary["accuracy"]),
                -abs(np.log10(float(c_value))),
                -float(power),
            )
            rows.append(
                {
                    "C": float(c_value),
                    "class_weight_power": float(power),
                    "stemi_bias": bias.stemi_bias,
                    "nstemi_bias": bias.nstemi_bias,
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
                best = StackerSearchResult(
                    c_value=float(c_value),
                    class_weight_power=float(power),
                    mean=mean.copy(),
                    scale=scale.copy(),
                    coef=model.coef_.copy(),
                    intercept=model.intercept_.copy(),
                    bias_result=bias,
                )
    if best is None:
        raise RuntimeError("logistic stacker search produced no candidate")
    return best, rows
