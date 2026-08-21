from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.strict80 import hierarchical_predict, search_strict80_thresholds, six_metric_summary


CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_joined_predictions(stage1_path: Path, stage2_path: Path) -> pd.DataFrame:
    stage1 = pd.read_csv(stage1_path)
    stage2 = pd.read_csv(stage2_path)
    _require_columns(stage1, ["ecg_id", "actual_id", "p_mi"], stage1_path)
    _require_columns(stage2, ["ecg_id", "actual_id", "p_stemi_given_mi"], stage2_path)

    left = stage1[["ecg_id", "actual_id", "p_mi"]].copy()
    right = stage2[["ecg_id", "actual_id", "p_stemi_given_mi"]].copy()
    joined = left.merge(
        right,
        on="ecg_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_stage1", "_stage2"),
    )
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError(
            "Stage1 and Stage2 prediction files do not contain exactly the same ECG IDs "
            f"({len(left)} vs {len(right)} vs joined {len(joined)})"
        )
    mismatch = joined["actual_id_stage1"] != joined["actual_id_stage2"]
    if mismatch.any():
        examples = joined.loc[mismatch, ["ecg_id", "actual_id_stage1", "actual_id_stage2"]].head()
        raise ValueError(f"label mismatch between Stage1 and Stage2 files:\n{examples}")

    joined = joined.rename(columns={"actual_id_stage1": "actual_id"}).drop(
        columns=["actual_id_stage2"]
    )
    if not np.isfinite(joined[["p_mi", "p_stemi_given_mi"]].to_numpy(dtype=float)).all():
        raise ValueError("prediction probabilities contain NaN or infinity")
    return joined


def flatten_metrics(summary: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "target": summary["target"],
        "minimum_of_six": summary["minimum_of_six"],
        "mean_of_six": summary["mean_of_six"],
        "accuracy": summary["accuracy"],
        "n_metrics_strictly_above_target": summary["n_metrics_strictly_above_target"],
        "all_six_strictly_above_target": summary["all_six_strictly_above_target"],
        "all_six_at_or_above_target": summary["all_six_at_or_above_target"],
    }
    per_class = summary["per_class"]
    for name in CLASS_NAMES:
        metrics = per_class[name]
        output[f"{name}_sensitivity"] = metrics["sensitivity"]
        output[f"{name}_specificity"] = metrics["specificity"]
    return output


def print_result(title: str, summary: dict[str, object], cm: np.ndarray) -> None:
    print(f"\n{'#' * 46}")
    print(title)
    print(f"{'#' * 46}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print("\nPer-class sensitivity / specificity:")
    for name in CLASS_NAMES:
        metrics = summary["per_class"][name]
        print(
            f"{name:13s} sensitivity={metrics['sensitivity']:.4f} "
            f"specificity={metrics['specificity']:.4f}"
        )
    print(f"\nMinimum of six metrics = {summary['minimum_of_six']:.4f}")
    print(
        f"All six strictly > {float(summary['target']):.2f}: "
        f"{summary['all_six_strictly_above_target']}"
    )


def save_prediction_table(
    joined: pd.DataFrame,
    prediction: np.ndarray,
    mi_threshold: float,
    stemi_threshold: float,
    path: Path,
) -> None:
    output = joined.copy()
    output["predicted_id"] = prediction.astype(int)
    output["predicted_label"] = [CLASS_NAMES[int(value)] for value in prediction]
    output["mi_threshold"] = float(mi_threshold)
    output["stemi_threshold"] = float(stemi_threshold)
    output.to_csv(path, index=False)


def run_validation(cfg: dict, config_path: str) -> None:
    inputs = cfg["inputs"]
    output_cfg = cfg["output"]
    target = float(cfg.get("target", 0.80))
    search_cfg = cfg.get("search", {}) or {}

    stage1_path = ROOT / inputs["stage1_validation_predictions"]
    stage2_path = ROOT / inputs["stage2_validation_details"]
    if not stage1_path.is_file():
        raise FileNotFoundError(stage1_path)
    if not stage2_path.is_file():
        raise FileNotFoundError(stage2_path)

    joined = load_joined_predictions(stage1_path, stage2_path)
    y = joined["actual_id"].to_numpy(dtype=int)
    p_mi = joined["p_mi"].to_numpy(dtype=float)
    p_stemi = joined["p_stemi_given_mi"].to_numpy(dtype=float)

    best, rows = search_strict80_thresholds(
        y,
        p_mi,
        p_stemi,
        target=target,
        coarse_step=float(search_cfg.get("coarse_step", 0.01)),
        fine_radius=float(search_cfg.get("fine_radius", 0.03)),
        fine_step=float(search_cfg.get("fine_step", 0.001)),
    )

    out_dir = ROOT / output_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "validation_threshold_candidates.csv", index=False)
    save_prediction_table(
        joined,
        best.predictions,
        best.mi_threshold,
        best.stemi_threshold,
        out_dir / "validation_predictions.csv",
    )
    pd.DataFrame(
        best.confusion_matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(out_dir / "validation_confusion_matrix.csv")

    validation = flatten_metrics(best.summary)
    validation["mi_threshold"] = best.mi_threshold
    validation["stemi_threshold"] = best.stemi_threshold
    validation["n_records"] = int(len(joined))

    summary = {
        "experiment": "stage1_v3_plus_existing_stage2_strict80",
        "config": config_path,
        "decision_rule": "hard_hierarchical_thresholds",
        "selection_objective": "maximize_minimum_of_six_classwise_sensitivity_specificity",
        "target": target,
        "inputs": {
            "stage1_validation_predictions": str(inputs["stage1_validation_predictions"]),
            "stage2_validation_details": str(inputs["stage2_validation_details"]),
        },
        "validation": validation,
        "validation_per_class": best.summary["per_class"],
        "validation_confusion_matrix": best.confusion_matrix.tolist(),
        "test_evaluated": False,
        "test": None,
        "note": (
            "Thresholds were selected on PTB-XL Fold 9 only. Fold 10 is not read by this mode. "
            "Run finalize-only only if the validation result is accepted."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined validation ECGs: {len(joined)}")
    print(f"Selected MI threshold: {best.mi_threshold:.6f}")
    print(f"Selected STEMI|MI threshold: {best.stemi_threshold:.6f}")
    print_result("VALIDATION STRICT-80 CASCADE", best.summary, best.confusion_matrix)
    print(f"\nSaved: {out_dir / 'metrics.json'}")
    if not bool(best.summary["all_six_strictly_above_target"]):
        print("Fold 10 remains locked because the strict validation target was not met.")
    else:
        print("Validation target PASSED. Fold 10 remains untouched until explicit --finalize-only.")


def run_finalize(cfg: dict) -> None:
    inputs = cfg["inputs"]
    out_dir = ROOT / cfg["output"]["out_dir"]
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Run validation calibration before --finalize-only")

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = summary["validation"]
    if not bool(validation["all_six_strictly_above_target"]):
        raise RuntimeError(
            "Validation did not meet the strict >80% target for all six metrics; "
            "refusing to evaluate Fold 10."
        )

    stage1_path = ROOT / inputs["stage1_test_predictions"]
    stage2_path = ROOT / inputs["stage2_test_details"]
    if not stage1_path.is_file():
        raise FileNotFoundError(
            f"Missing {stage1_path}. First generate frozen Stage1-v3 Fold-10 probabilities with:\n"
            "python scripts/train_stage1_v3.py --config configs/stage1_v3_morphology_fusion.yaml "
            "--manifest data/label_manifest_custom65_common.csv --device cuda --finalize-only"
        )
    if not stage2_path.is_file():
        raise FileNotFoundError(stage2_path)

    joined = load_joined_predictions(stage1_path, stage2_path)
    mi_threshold = float(validation["mi_threshold"])
    stemi_threshold = float(validation["stemi_threshold"])
    prediction = hierarchical_predict(
        joined["p_mi"].to_numpy(dtype=float),
        joined["p_stemi_given_mi"].to_numpy(dtype=float),
        mi_threshold,
        stemi_threshold,
    )
    test_summary, cm = six_metric_summary(
        joined["actual_id"].to_numpy(dtype=int),
        prediction,
        target=float(summary["target"]),
    )
    flattened = flatten_metrics(test_summary)
    flattened["mi_threshold"] = mi_threshold
    flattened["stemi_threshold"] = stemi_threshold
    flattened["n_records"] = int(len(joined))

    save_prediction_table(
        joined,
        prediction,
        mi_threshold,
        stemi_threshold,
        out_dir / "test_predictions.csv",
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv"
    )
    summary["test"] = flattened
    summary["test_per_class"] = test_summary["per_class"]
    summary["test_confusion_matrix"] = cm.tolist()
    summary["test_evaluated"] = True
    summary["note"] = (
        "Fold-10 thresholds were frozen from Fold 9; no Fold-10 threshold or model selection was performed."
    )
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_result("FROZEN THRESHOLDS -> FOLD 10 TEST", test_summary, cm)
    print(f"\nUpdated: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Stage1-v3 MI probabilities with the existing common-cohort Stage2 and "
            "select Fold-9 thresholds by the weakest of six class-wise sensitivity/specificity metrics."
        )
    )
    parser.add_argument("--config", default="configs/stage1_v3_stage2_strict80.yaml")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.finalize_only:
        run_finalize(cfg)
    else:
        run_validation(cfg, args.config)


if __name__ == "__main__":
    main()
