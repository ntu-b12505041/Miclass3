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

from miclass3.direct_cascade_ensemble import (
    apply_class_biases,
    blend_probabilities,
    cascade_soft_probabilities,
    search_direct_cascade_ensemble,
)
from miclass3.strict80 import six_metric_summary

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def _require(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def load_joined(direct_path: Path, stage1_path: Path, stage2_path: Path) -> pd.DataFrame:
    direct = pd.read_csv(direct_path)
    stage1 = pd.read_csv(stage1_path)
    stage2 = pd.read_csv(stage2_path)

    _require(
        direct,
        ["ecg_id", "actual_id", "prob_non_mi", "prob_stemi_proxy", "prob_nstemi_proxy"],
        direct_path,
    )
    _require(stage1, ["ecg_id", "actual_id", "p_mi"], stage1_path)
    _require(stage2, ["ecg_id", "actual_id", "p_stemi_given_mi"], stage2_path)

    d = direct[
        ["ecg_id", "actual_id", "prob_non_mi", "prob_stemi_proxy", "prob_nstemi_proxy"]
    ].rename(columns={"actual_id": "actual_id_direct"})
    s1 = stage1[["ecg_id", "actual_id", "p_mi"]].rename(
        columns={"actual_id": "actual_id_stage1"}
    )
    s2 = stage2[["ecg_id", "actual_id", "p_stemi_given_mi"]].rename(
        columns={"actual_id": "actual_id_stage2"}
    )

    joined = d.merge(s1, on="ecg_id", how="inner", validate="one_to_one").merge(
        s2, on="ecg_id", how="inner", validate="one_to_one"
    )
    if not (len(joined) == len(d) == len(s1) == len(s2)):
        raise ValueError(
            "prediction ECG sets differ: "
            f"direct={len(d)} stage1={len(s1)} stage2={len(s2)} joined={len(joined)}"
        )
    labels = joined[["actual_id_direct", "actual_id_stage1", "actual_id_stage2"]]
    if not (labels.nunique(axis=1) == 1).all():
        raise ValueError("Direct/Stage1/Stage2 actual labels do not match")
    joined["actual_id"] = joined["actual_id_direct"].astype(int)
    return joined


def print_summary(title: str, summary: dict[str, object], cm: np.ndarray) -> None:
    print("\n" + "#" * 58)
    print(title)
    print("#" * 58)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print("Per-class sensitivity / specificity:")
    for name in CLASS_NAMES:
        metrics = summary["per_class"][name]
        print(
            f"{name:13s} sensitivity={metrics['sensitivity']:.4f} "
            f"specificity={metrics['specificity']:.4f}"
        )
    print(f"Minimum of six metrics = {summary['minimum_of_six']:.4f}")
    print(
        f"All six strictly > {float(summary['target']):.2f}: "
        f"{summary['all_six_strictly_above_target']}"
    )


def probability_inputs(joined: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    direct_prob = joined[["prob_non_mi", "prob_stemi_proxy", "prob_nstemi_proxy"]].to_numpy(float)
    cascade_prob = cascade_soft_probabilities(
        joined["p_mi"].to_numpy(float),
        joined["p_stemi_given_mi"].to_numpy(float),
    )
    return direct_prob, cascade_prob


def save_ensemble_predictions(
    joined: pd.DataFrame,
    cascade_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> None:
    table = joined[
        ["ecg_id", "actual_id", "prob_non_mi", "prob_stemi_proxy", "prob_nstemi_proxy", "p_mi", "p_stemi_given_mi"]
    ].copy()
    table["cascade_prob_non_mi"] = cascade_prob[:, 0]
    table["cascade_prob_stemi_proxy"] = cascade_prob[:, 1]
    table["cascade_prob_nstemi_proxy"] = cascade_prob[:, 2]
    table["ensemble_prob_non_mi"] = ensemble_prob[:, 0]
    table["ensemble_prob_stemi_proxy"] = ensemble_prob[:, 1]
    table["ensemble_prob_nstemi_proxy"] = ensemble_prob[:, 2]
    table["predicted_id"] = predictions.astype(int)
    table["predicted_label"] = [CLASS_NAMES[int(x)] for x in predictions]
    table.to_csv(path, index=False)


def run_validation(cfg: dict) -> None:
    inputs = cfg["inputs"]
    target = float(cfg.get("target", 0.80))
    promotion = float(cfg.get("promotion_minimum_of_six", target))

    direct_path = ROOT / inputs["direct_validation_predictions"]
    stage1_path = ROOT / inputs["stage1_validation_predictions"]
    stage2_path = ROOT / inputs["stage2_validation_details"]
    for path in (direct_path, stage1_path, stage2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    joined = load_joined(direct_path, stage1_path, stage2_path)
    y = joined["actual_id"].to_numpy(dtype=int)
    direct_prob, cascade_prob = probability_inputs(joined)

    direct_summary, direct_cm = six_metric_summary(y, direct_prob.argmax(axis=1), target=target)
    cascade_summary, cascade_cm = six_metric_summary(y, cascade_prob.argmax(axis=1), target=target)

    best, rows = search_direct_cascade_ensemble(
        y,
        direct_prob,
        joined["p_mi"].to_numpy(float),
        joined["p_stemi_given_mi"].to_numpy(float),
        target=target,
        **(cfg.get("search", {}) or {}),
    )

    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "validation_candidates.csv", index=False)
    pd.DataFrame(best.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "validation_confusion_matrix.csv"
    )
    save_ensemble_predictions(
        joined,
        cascade_prob,
        best.probabilities,
        best.predictions,
        out_dir / "validation_predictions.csv",
    )

    validation = {
        "alpha_direct": best.alpha_direct,
        "alpha_cascade": 1.0 - best.alpha_direct,
        "stemi_bias": best.stemi_bias,
        "nstemi_bias": best.nstemi_bias,
        "minimum_of_six": best.summary["minimum_of_six"],
        "mean_of_six": best.summary["mean_of_six"],
        "accuracy": best.summary["accuracy"],
        "all_six_strictly_above_target": best.summary["all_six_strictly_above_target"],
        "promotion_minimum_of_six": promotion,
        "promotion_passed": bool(best.summary["minimum_of_six"] >= promotion),
        "n_records": int(len(joined)),
    }
    summary = {
        "experiment": "direct_common65_matched_plus_stage1_v3_cascade_ensemble",
        "selection_objective": "Fold9 maximize minimum of six class-wise sensitivity/specificity metrics",
        "target": target,
        "promotion_minimum_of_six": promotion,
        "validation": validation,
        "validation_per_class": best.summary["per_class"],
        "validation_confusion_matrix": best.confusion_matrix.tolist(),
        "direct_argmax_baseline": {
            "minimum_of_six": direct_summary["minimum_of_six"],
            "per_class": direct_summary["per_class"],
            "confusion_matrix": direct_cm.tolist(),
        },
        "cascade_soft_argmax_baseline": {
            "minimum_of_six": cascade_summary["minimum_of_six"],
            "per_class": cascade_summary["per_class"],
            "confusion_matrix": cascade_cm.tolist(),
        },
        "test_evaluated": False,
        "test": None,
        "note": "All ensemble weights and class biases are selected on Fold 9 only; Fold 10 is not read.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined validation ECGs: {len(joined)}")
    print(f"Direct-only minimum of six: {direct_summary['minimum_of_six']:.4f}")
    print(f"Cascade soft-argmax minimum of six: {cascade_summary['minimum_of_six']:.4f}")
    print(f"Selected Direct weight alpha: {best.alpha_direct:.3f}")
    print(f"Selected Cascade weight: {1.0 - best.alpha_direct:.3f}")
    print(f"Selected STEMI logit bias: {best.stemi_bias:.3f}")
    print(f"Selected NSTEMI logit bias: {best.nstemi_bias:.3f}")
    print_summary("VALIDATION DIRECT + CASCADE ENSEMBLE", best.summary, best.confusion_matrix)
    print(f"Promotion minimum required = {promotion:.4f}")
    print(f"Promotion passed: {validation['promotion_passed']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


def run_finalize(cfg: dict) -> None:
    inputs = cfg["inputs"]
    target = float(cfg.get("target", 0.80))
    out_dir = ROOT / cfg["output"]["out_dir"]
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Run Fold-9 ensemble selection first")

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = summary["validation"]
    if not bool(validation["all_six_strictly_above_target"]):
        raise RuntimeError(
            "Fold-9 ensemble did not meet the user's strict >0.80 target for all six metrics; refusing finalize"
        )

    direct_path = ROOT / inputs["direct_test_predictions"]
    stage1_path = ROOT / inputs["stage1_test_predictions"]
    stage2_path = ROOT / inputs["stage2_test_details"]
    for path in (direct_path, stage1_path, stage2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    joined = load_joined(direct_path, stage1_path, stage2_path)
    y = joined["actual_id"].to_numpy(dtype=int)
    direct_prob, cascade_prob = probability_inputs(joined)

    alpha_direct = float(validation["alpha_direct"])
    stemi_bias = float(validation["stemi_bias"])
    nstemi_bias = float(validation["nstemi_bias"])
    mixed = blend_probabilities(direct_prob, cascade_prob, alpha_direct)
    ensemble_prob = apply_class_biases(mixed, stemi_bias, nstemi_bias)
    predictions = ensemble_prob.argmax(axis=1)
    test_summary, cm = six_metric_summary(y, predictions, target=target)

    save_ensemble_predictions(
        joined,
        cascade_prob,
        ensemble_prob,
        predictions,
        out_dir / "test_predictions.csv",
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv"
    )

    summary["test"] = {
        "alpha_direct": alpha_direct,
        "alpha_cascade": 1.0 - alpha_direct,
        "stemi_bias": stemi_bias,
        "nstemi_bias": nstemi_bias,
        "minimum_of_six": test_summary["minimum_of_six"],
        "mean_of_six": test_summary["mean_of_six"],
        "accuracy": test_summary["accuracy"],
        "all_six_strictly_above_target": test_summary["all_six_strictly_above_target"],
        "n_records": int(len(joined)),
    }
    summary["test_per_class"] = test_summary["per_class"]
    summary["test_confusion_matrix"] = cm.tolist()
    summary["test_evaluated"] = True
    summary["note"] = (
        "Fold-10 ensemble uses alpha and class biases frozen from Fold 9. "
        "No Fold-10 search, calibration, threshold tuning, or retraining is performed."
    )
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Frozen Direct weight alpha: {alpha_direct:.3f}")
    print(f"Frozen Cascade weight: {1.0 - alpha_direct:.3f}")
    print(f"Frozen STEMI logit bias: {stemi_bias:.3f}")
    print(f"Frozen NSTEMI logit bias: {nstemi_bias:.3f}")
    print_summary("FROZEN DIRECT + CASCADE ENSEMBLE -> FOLD 10", test_summary, cm)
    print(f"Updated: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct + Stage1-v3 Cascade ensemble validation calibration and frozen Fold-10 evaluation."
    )
    parser.add_argument("--config", default="configs/direct_cascade_ensemble80.yaml")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.finalize_only:
        run_finalize(cfg)
    else:
        run_validation(cfg)


if __name__ == "__main__":
    main()
