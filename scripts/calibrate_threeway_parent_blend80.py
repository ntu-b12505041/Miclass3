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

from miclass3.strict80 import six_metric_summary
from miclass3.threeway_parent_blend import search_threeway_parent_blend

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame[columns].copy()


def load_joined(cfg: dict) -> pd.DataFrame:
    inputs = cfg["inputs"]
    s1 = _read(ROOT / inputs["stage1_validation_predictions"], ["ecg_id", "actual_id", "p_mi"]).rename(columns={"actual_id": "actual_s1"})
    direct = _read(
        ROOT / inputs["direct_validation_predictions"],
        ["ecg_id", "actual_id", "prob_non_mi"],
    ).rename(columns={"actual_id": "actual_direct"})
    aux = _read(
        ROOT / inputs["direct_aux_validation_predictions"],
        ["ecg_id", "actual_id", "p_mi_aux"],
    ).rename(columns={"actual_id": "actual_aux"})
    s2 = _read(
        ROOT / inputs["stage2_validation_details"],
        ["ecg_id", "actual_id", "p_stemi_given_mi"],
    ).rename(columns={"actual_id": "actual_s2"})
    joined = s1.merge(direct, on="ecg_id", validate="one_to_one").merge(aux, on="ecg_id", validate="one_to_one").merge(s2, on="ecg_id", validate="one_to_one")
    expected = len(s1)
    if not (len(joined) == expected == len(direct) == len(aux) == len(s2)):
        raise ValueError("prediction ECG sets do not match exactly")
    labels = joined[["actual_s1", "actual_direct", "actual_aux", "actual_s2"]]
    if not (labels.nunique(axis=1) == 1).all():
        raise ValueError("actual labels disagree across prediction sources")
    joined["actual_id"] = joined["actual_s1"].astype(int)
    joined["p_direct_class_mi"] = 1.0 - joined["prob_non_mi"].astype(float)
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold9-only three-way MI parent blend calibration.")
    parser.add_argument("--config", default="configs/threeway_parent_blend80.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    joined = load_joined(cfg)
    y = joined["actual_id"].to_numpy(dtype=int)
    target = float(cfg.get("target", 0.80))
    promotion = float(cfg.get("promotion_minimum_of_six", 0.82))

    best = search_threeway_parent_blend(
        y,
        joined["p_mi"].to_numpy(float),
        joined["p_mi_aux"].to_numpy(float),
        joined["p_direct_class_mi"].to_numpy(float),
        joined["p_stemi_given_mi"].to_numpy(float),
        target=target,
        **(cfg.get("search", {}) or {}),
    )
    result = best.result
    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best.rows).to_csv(out_dir / "validation_candidates.csv", index=False)
    pd.DataFrame(result.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "validation_confusion_matrix.csv")

    summary = {
        "experiment": "threeway_parent_blend_stage1_direct_aux_direct_class",
        "selection_objective": "Fold9 maximize minimum of six sensitivity/specificity metrics",
        "target": target,
        "validation": {
            "weight_stage1": best.weight_stage1,
            "weight_direct_aux": best.weight_direct_aux,
            "weight_direct_class": best.weight_direct_class,
            "mi_threshold": result.mi_threshold,
            "stemi_threshold": result.stemi_threshold,
            "minimum_of_six": result.summary["minimum_of_six"],
            "mean_of_six": result.summary["mean_of_six"],
            "accuracy": result.summary["accuracy"],
            "all_six_strictly_above_target": result.summary["all_six_strictly_above_target"],
            "promotion_minimum_of_six": promotion,
            "promotion_passed": bool(result.summary["minimum_of_six"] >= promotion),
            "n_records": int(len(joined)),
        },
        "validation_per_class": result.summary["per_class"],
        "validation_confusion_matrix": result.confusion_matrix.tolist(),
        "test_evaluated": False,
        "note": "All weights and thresholds are selected on Fold9 only; Fold10 is not read.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined validation ECGs: {len(joined)}")
    print(f"Selected Stage1-v3 MI weight: {best.weight_stage1:.3f}")
    print(f"Selected Direct MI-aux weight: {best.weight_direct_aux:.3f}")
    print(f"Selected Direct class-derived MI weight: {best.weight_direct_class:.3f}")
    print(f"Selected MI threshold: {result.mi_threshold:.6f}")
    print(f"Selected STEMI|MI threshold: {result.stemi_threshold:.6f}")
    print("\n##########################################################")
    print("VALIDATION THREE-WAY PARENT BLEND")
    print("##########################################################")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(result.confusion_matrix)
    print("Per-class sensitivity / specificity:")
    for name in CLASS_NAMES:
        metrics = result.summary["per_class"][name]
        print(f"{name:13s} sensitivity={metrics['sensitivity']:.4f} specificity={metrics['specificity']:.4f}")
    print(f"Minimum of six metrics = {result.summary['minimum_of_six']:.4f}")
    print(f"All six strictly > {target:.2f}: {result.summary['all_six_strictly_above_target']}")
    print(f"Promotion minimum required = {promotion:.4f}")
    print(f"Promotion passed: {summary['validation']['promotion_passed']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
