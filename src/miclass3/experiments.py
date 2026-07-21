from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


# These are the overall metrics that the training pipeline can report for all
# three official folds. Per-class support varies by fold, so the experiment
# selector uses the comparable overall metrics rather than an unstable single
# cell of a confusion matrix.
DEFAULT_REQUIRED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_auroc",
    "macro_auprc",
    "macro_f1",
    "stemi_recall",
)


def metric_floor(metrics: Mapping[str, object], required_metrics: Iterable[str] = DEFAULT_REQUIRED_METRICS) -> float:
    """Return the weakest required metric, or -inf if any metric is absent."""
    values: list[float] = []
    for name in required_metrics:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return float("-inf")
        if not np.isfinite(value):
            return float("-inf")
        values.append(value)
    return min(values) if values else float("-inf")


def build_validation_leaderboard(
    trials: Iterable[Mapping[str, object]],
    required_metrics: Iterable[str] = DEFAULT_REQUIRED_METRICS,
    target: float = 0.85,
) -> pd.DataFrame:
    """Rank validation-only trial results against an explicit quality floor.

    The primary sort key is the minimum required metric. This prevents a model
    with high AUPRC but weak STEMI recall or balanced accuracy from winning the
    hyperparameter search. Macro-AUPRC and macro-F1 break ties because they are
    the existing project selection objectives.
    """
    required = tuple(required_metrics)
    if not required:
        raise ValueError("required_metrics must not be empty")
    if not 0.0 <= float(target) <= 1.0:
        raise ValueError("target must be in [0, 1]")

    rows: list[dict[str, object]] = []
    for trial in trials:
        name = str(trial.get("trial", trial.get("name", "unnamed")))
        raw_metrics = trial.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raw_metrics = {}
        row: dict[str, object] = {"trial": name}
        for metric in required:
            try:
                row[metric] = float(raw_metrics[metric])
            except (KeyError, TypeError, ValueError):
                row[metric] = float("nan")
        floor = metric_floor(raw_metrics, required)
        row["validation_metric_floor"] = floor
        row["passes_target"] = bool(floor >= float(target))
        row["error"] = trial.get("error")
        rows.append(row)

    columns = ["trial", *required, "validation_metric_floor", "passes_target", "error"]
    leaderboard = pd.DataFrame(rows, columns=columns)
    if leaderboard.empty:
        return leaderboard
    return leaderboard.sort_values(
        ["passes_target", "validation_metric_floor", "macro_auprc", "macro_f1", "balanced_accuracy", "stemi_recall"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
