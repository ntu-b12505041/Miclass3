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

from miclass3.direct_mi_parent_blend import (
    blend_parent_log_odds,
    search_direct_mi_parent_blend,
)

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def _require(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def load_joined(stage1_path: Path, direct_aux_path: Path, stage2_path: Path) -> pd.DataFrame:
    stage1 = pd.read_csv(stage1_path)
    direct = pd.read_csv(direct_aux_path)
    stage2 = pd.read_csv(stage2_path)

    _require(stage1, ["ecg_id", "actual_id", "p_mi"], stage1_path)
    _require(direct, ["ecg_id", "actual_id", "p_mi_aux"], direct_aux_path)
    _require(stage2, ["ecg_id", "actual_id", "p_stemi_given_mi"], stage2_path)

    s1 = stage1[["ecg_id", "actual_id", "p_mi"]].rename(
        columns={"actual_id": "actual_id_stage1"}
    )
    direct = direct[["ecg_id", "actual_id", "p_mi_aux"]].rename(
        columns={"actual_id": "actual_id_direct"}
    )
    s2 = stage2[["ecg_id", "actual_id", "p_stemi_given_mi"]].rename(
        columns={"actual_id": "actual_id_stage2"}
    )

    joined = s1.merge(direct, on="ecg_id", how="inner", validate="one_to_one").merge(
        s2, on="ecg_id", how="inner", validate="one_to_one"
    )
    if not (len(joined) == len(s1) == len(direct) == len(s2)):
        raise ValueError(
            "prediction ECG sets differ: "
            f"stage1={len(s1)} direct_aux={len(direct)} stage2={len(s2)} joined={len(joined)}"
        )
    labels = joined[["actual_id_stage1", "actual_id_direct", "actual_id_stage2"]]
    if not (labels.nunique(axis=1) == 1).all():
        raise ValueError("Stage1/Direct-aux/Stage2 actual labels do not match")
    joined["actual_id"] = joined["actual_id_stage1"].astype(int)
    return joined


def print_summary(summary: dict[str, object], cm: np.ndarray) -> None:
    print("\n" + "#" * 58)
    print("VALIDATION DIRECT-MI-AUX + STAGE1 PARENT BLEND")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only blend of Stage1-v3 and Direct MI auxiliary parent evidence."
    )
    parser.add_argument("--config", default="configs/direct_mi_parent_blend80.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    target = float(cfg.get("target", 0.80))
    promotion = float(cfg.get("promotion_minimum_of_six", target))
    inputs = cfg["inputs"]

    joined = load_joined(
        ROOT / inputs["stage1_validation_predictions"],
        ROOT / inputs["direct_aux_validation_predictions"],
        ROOT / inputs["stage2_validation_details"],
    )
    y = joined["actual_id"].to_numpy(dtype=int)

    best = search_direct_mi_parent_blend(
        y,
        joined["p_mi"].to_numpy(float),
        joined["p_mi_aux"].to_numpy(float),
        joined["p_stemi_given_mi"].to_numpy(float),
        target=target,
        **(cfg.get("search", {}) or {}),
    )
    result = best.result
    parent = blend_parent_log_odds(
        joined["p_mi"].to_numpy(float),
        joined["p_mi_aux"].to_numpy(float),
        best.alpha_stage1,
    )

    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best.rows).to_csv(out_dir / "validation_alpha_candidates.csv", index=False)
    pd.DataFrame(result.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "validation_confusion_matrix.csv"
    )

    prediction_table = joined[
        ["ecg_id", "actual_id", "p_mi", "p_mi_aux", "p_stemi_given_mi"]
    ].copy()
    prediction_table["p_mi_parent_blend"] = parent
    prediction_table["predicted_id"] = result.predictions
    prediction_table.to_csv(out_dir / "validation_predictions.csv", index=False)

    validation = {
        "alpha_stage1": best.alpha_stage1,
        "alpha_direct_mi_aux": 1.0 - best.alpha_stage1,
        "mi_threshold": result.mi_threshold,
        "stemi_threshold": result.stemi_threshold,
        "minimum_of_six": result.summary["minimum_of_six"],
        "mean_of_six": result.summary["mean_of_six"],
        "accuracy": result.summary["accuracy"],
        "all_six_strictly_above_target": result.summary[
            "all_six_strictly_above_target"
        ],
        "promotion_minimum_of_six": promotion,
        "promotion_passed": bool(result.summary["minimum_of_six"] >= promotion),
        "n_records": int(len(joined)),
    }
    summary = {
        "experiment": "direct_mi_aux_plus_stage1_v3_parent_logodds_blend",
        "selection_objective": "Fold9 maximize minimum of six class-wise sensitivity/specificity metrics",
        "target": target,
        "promotion_minimum_of_six": promotion,
        "validation": validation,
        "validation_per_class": result.summary["per_class"],
        "validation_confusion_matrix": result.confusion_matrix.tolist(),
        "test_evaluated": False,
        "test": None,
        "note": "All blend weights and thresholds are selected on Fold 9 only; Fold 10 is not read.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined validation ECGs: {len(joined)}")
    print(f"Selected Stage1-v3 parent weight alpha: {best.alpha_stage1:.3f}")
    print(f"Selected Direct MI-aux weight: {1.0 - best.alpha_stage1:.3f}")
    print(f"Selected MI threshold: {result.mi_threshold:.6f}")
    print(f"Selected STEMI|MI threshold: {result.stemi_threshold:.6f}")
    print_summary(result.summary, result.confusion_matrix)
    print(f"Promotion minimum required = {promotion:.4f}")
    print(f"Promotion passed: {validation['promotion_passed']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
