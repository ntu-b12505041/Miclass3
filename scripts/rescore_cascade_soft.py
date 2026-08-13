from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.cascade import calibrate_soft_cascade, compose_calibrated_soft_cascade, compose_soft_cascade
from miclass3.metrics import write_split_artifacts


def load_details(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"actual_id", "p_mi", "p_stemi_given_mi"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def fit_operating_point(
    val: pd.DataFrame,
    *,
    minimum_stemi_recall: float | None,
    bias_min: float,
    bias_max: float,
    bias_step: float,
):
    return calibrate_soft_cascade(
        val["actual_id"].to_numpy(dtype=int),
        val["p_mi"].to_numpy(dtype=float),
        val["p_stemi_given_mi"].to_numpy(dtype=float),
        minimum_stemi_recall=minimum_stemi_recall,
        bias_min=bias_min,
        bias_max=bias_max,
        bias_step=bias_step,
    )


def score_mode(
    frame: pd.DataFrame,
    *,
    mode: str,
    mi_bias: float = 0.0,
    stemi_bias: float = 0.0,
) -> np.ndarray:
    p_mi = frame["p_mi"].to_numpy(dtype=float)
    p_stemi = frame["p_stemi_given_mi"].to_numpy(dtype=float)
    if mode == "default":
        return compose_soft_cascade(p_mi, p_stemi)
    return compose_calibrated_soft_cascade(
        p_mi,
        p_stemi,
        mi_logit_bias=mi_bias,
        stemi_logit_bias=stemi_bias,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score an existing cascade run into three validation-safe operating points "
            "without retraining: default, balanced, and high-sensitivity."
        )
    )
    parser.add_argument("--source-dir", default="artifacts/cascade_v3")
    parser.add_argument("--out-dir", default="artifacts/cascade_v3_operating_points")
    parser.add_argument("--high-sensitivity-recall", type=float, default=0.75)
    parser.add_argument("--bias-min", type=float, default=-2.0)
    parser.add_argument("--bias-max", type=float, default=2.0)
    parser.add_argument("--bias-step", type=float, default=0.1)
    args = parser.parse_args()

    source = ROOT / args.source_dir
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    val = load_details(source / "val_cascade_details.csv")

    # Balanced: validation-only search with no STEMI-recall floor. The existing
    # calibration routine maximizes macro-F1, then balanced accuracy, then
    # STEMI recall. This is the general-purpose operating point.
    balanced_summary, balanced_candidates = fit_operating_point(
        val,
        minimum_stemi_recall=None,
        bias_min=args.bias_min,
        bias_max=args.bias_max,
        bias_step=args.bias_step,
    )
    balanced_candidates.to_csv(out / "val_balanced_calibration_candidates.csv", index=False)

    # High sensitivity: same validation-only search, but require a STEMI recall
    # floor. This is intentionally a separate operating point rather than the
    # primary general-purpose classifier.
    high_summary, high_candidates = fit_operating_point(
        val,
        minimum_stemi_recall=args.high_sensitivity_recall,
        bias_min=args.bias_min,
        bias_max=args.bias_max,
        bias_step=args.bias_step,
    )
    high_candidates.to_csv(out / "val_high_sensitivity_calibration_candidates.csv", index=False)

    operating_points = {
        "default": {
            "description": "Uncalibrated probabilistic hierarchy; no fitted bias.",
            "mi_logit_bias": 0.0,
            "stemi_logit_bias": 0.0,
            "validation_selection": None,
        },
        "balanced": {
            "description": "Validation-selected general-purpose point maximizing macro-F1, then balanced accuracy.",
            "mi_logit_bias": float(balanced_summary["mi_logit_bias"]),
            "stemi_logit_bias": float(balanced_summary["stemi_logit_bias"]),
            "validation_selection": balanced_summary,
        },
        "high_sensitivity": {
            "description": (
                "Validation-selected point maximizing macro-F1 subject to the configured STEMI-recall floor."
            ),
            "mi_logit_bias": float(high_summary["mi_logit_bias"]),
            "stemi_logit_bias": float(high_summary["stemi_logit_bias"]),
            "validation_selection": high_summary,
        },
    }

    results: dict[str, object] = {
        "source_dir": str(args.source_dir),
        "high_sensitivity_recall_floor": float(args.high_sensitivity_recall),
        "operating_points": operating_points,
        "splits": {},
    }

    comparison_rows: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        path = source / f"{split}_cascade_details.csv"
        if not path.is_file():
            continue
        frame = load_details(path)
        y = frame["actual_id"].to_numpy(dtype=int)
        results["splits"][split] = {}

        export = frame.copy()
        for mode, spec in operating_points.items():
            probabilities = score_mode(
                frame,
                mode=mode,
                mi_bias=float(spec["mi_logit_bias"]),
                stemi_bias=float(spec["stemi_logit_bias"]),
            )
            mode_dir = out / mode
            metrics = write_split_artifacts(
                mode_dir,
                split,
                y,
                probabilities,
                records=None,
                stemi_threshold=None,
            )
            results["splits"][split][mode] = metrics

            export[f"{mode}_prob_non_mi"] = probabilities[:, 0]
            export[f"{mode}_prob_stemi_proxy"] = probabilities[:, 1]
            export[f"{mode}_prob_nstemi_proxy"] = probabilities[:, 2]
            export[f"{mode}_prediction"] = probabilities.argmax(axis=1)

            comparison_rows.append(
                {
                    "split": split,
                    "mode": mode,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "stemi_recall": metrics.get("stemi_recall"),
                    "macro_auroc": metrics.get("macro_auroc"),
                    "macro_auprc": metrics.get("macro_auprc"),
                    "mi_logit_bias": float(spec["mi_logit_bias"]),
                    "stemi_logit_bias": float(spec["stemi_logit_bias"]),
                }
            )

        export.to_csv(out / f"{split}_operating_point_details.csv", index=False)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "operating_point_comparison.csv", index=False)
    (out / "operating_point_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    test_results = results.get("splits", {}).get("test", {})
    print(json.dumps({"test": test_results, "operating_points": operating_points}, indent=2))


if __name__ == "__main__":
    main()
