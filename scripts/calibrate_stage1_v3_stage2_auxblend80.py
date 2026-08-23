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

from miclass3.parent_blend import blend_parent_probability, search_parent_blend
from miclass3.strict80 import hierarchical_predict, six_metric_summary


CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_joined(stage1_path: Path, stage2_path: Path) -> pd.DataFrame:
    stage1 = pd.read_csv(stage1_path)
    stage2 = pd.read_csv(stage2_path)
    require_columns(
        stage1,
        ["ecg_id", "actual_id", "p_mi", "p_aux_non_mi"],
        stage1_path,
    )
    require_columns(stage2, ["ecg_id", "actual_id", "p_stemi_given_mi"], stage2_path)

    left = stage1[["ecg_id", "actual_id", "p_mi", "p_aux_non_mi"]].copy()
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
            "Stage1 and Stage2 files do not contain exactly the same ECG IDs "
            f"({len(left)} vs {len(right)} vs joined {len(joined)})"
        )
    mismatch = joined["actual_id_stage1"] != joined["actual_id_stage2"]
    if mismatch.any():
        raise ValueError("Stage1 and Stage2 labels disagree for one or more ECG IDs")
    joined = joined.rename(columns={"actual_id_stage1": "actual_id"}).drop(
        columns=["actual_id_stage2"]
    )
    values = joined[["p_mi", "p_aux_non_mi", "p_stemi_given_mi"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction probabilities contain NaN or infinity")
    return joined


def flatten(summary: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "target": summary["target"],
        "minimum_of_six": summary["minimum_of_six"],
        "mean_of_six": summary["mean_of_six"],
        "accuracy": summary["accuracy"],
        "all_six_strictly_above_target": summary["all_six_strictly_above_target"],
    }
    for name in CLASS_NAMES:
        metrics = summary["per_class"][name]
        output[f"{name}_sensitivity"] = metrics["sensitivity"]
        output[f"{name}_specificity"] = metrics["specificity"]
    return output


def print_summary(title: str, summary: dict[str, object], cm: np.ndarray) -> None:
    print("\n" + "#" * 54)
    print(title)
    print("#" * 54)
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


def save_predictions(
    joined: pd.DataFrame,
    blended_mi: np.ndarray,
    prediction: np.ndarray,
    alpha_primary: float,
    mi_threshold: float,
    stemi_threshold: float,
    path: Path,
) -> None:
    out = joined.copy()
    out["p_aux_mi"] = 1.0 - out["p_aux_non_mi"]
    out["p_mi_blended"] = blended_mi
    out["predicted_id"] = prediction.astype(int)
    out["predicted_label"] = [CLASS_NAMES[int(value)] for value in prediction]
    out["alpha_primary"] = float(alpha_primary)
    out["mi_threshold"] = float(mi_threshold)
    out["stemi_threshold"] = float(stemi_threshold)
    out.to_csv(path, index=False)


def validation_mode(cfg: dict, config_path: str) -> None:
    inputs = cfg["inputs"]
    target = float(cfg.get("target", 0.80))
    promotion_minimum = float(cfg.get("promotion_minimum_of_six", 0.82))
    search_cfg = cfg.get("search", {}) or {}

    stage1_path = ROOT / inputs["stage1_validation_predictions"]
    stage2_path = ROOT / inputs["stage2_validation_details"]
    joined = load_joined(stage1_path, stage2_path)

    y = joined["actual_id"].to_numpy(dtype=int)
    result = search_parent_blend(
        y,
        joined["p_mi"].to_numpy(dtype=float),
        joined["p_aux_non_mi"].to_numpy(dtype=float),
        joined["p_stemi_given_mi"].to_numpy(dtype=float),
        target=target,
        alpha_min=float(search_cfg.get("alpha_min", 0.70)),
        alpha_max=float(search_cfg.get("alpha_max", 1.00)),
        alpha_coarse_step=float(search_cfg.get("alpha_coarse_step", 0.05)),
        alpha_fine_radius=float(search_cfg.get("alpha_fine_radius", 0.05)),
        alpha_fine_step=float(search_cfg.get("alpha_fine_step", 0.01)),
        threshold_coarse_step=float(search_cfg.get("threshold_coarse_step", 0.02)),
        threshold_fine_radius=float(search_cfg.get("threshold_fine_radius", 0.03)),
        threshold_fine_step=float(search_cfg.get("threshold_fine_step", 0.002)),
    )

    best = result.result
    blended = blend_parent_probability(
        joined["p_mi"].to_numpy(dtype=float),
        joined["p_aux_non_mi"].to_numpy(dtype=float),
        result.alpha_primary,
    )
    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.rows).to_csv(out_dir / "validation_alpha_candidates.csv", index=False)
    save_predictions(
        joined,
        blended,
        best.predictions,
        result.alpha_primary,
        best.mi_threshold,
        best.stemi_threshold,
        out_dir / "validation_predictions.csv",
    )
    pd.DataFrame(best.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "validation_confusion_matrix.csv"
    )

    validation = flatten(best.summary)
    validation.update(
        {
            "alpha_primary": float(result.alpha_primary),
            "alpha_auxiliary": float(1.0 - result.alpha_primary),
            "mi_threshold": float(best.mi_threshold),
            "stemi_threshold": float(best.stemi_threshold),
            "n_records": int(len(joined)),
            "promotion_minimum_of_six": promotion_minimum,
            "promotion_passed": bool(float(best.summary["minimum_of_six"]) >= promotion_minimum),
        }
    )
    summary = {
        "experiment": "stage1_v3_aux_parent_blend_plus_existing_stage2",
        "config": config_path,
        "decision_rule": "small_auxiliary_parent_probability_blend_then_hard_hierarchy",
        "selection_objective": "maximize_minimum_of_six_classwise_sensitivity_specificity",
        "target": target,
        "promotion_minimum_of_six": promotion_minimum,
        "validation": validation,
        "validation_per_class": best.summary["per_class"],
        "validation_confusion_matrix": best.confusion_matrix.tolist(),
        "test_evaluated": False,
        "test": None,
        "note": (
            "Only Fold 9 is read in validation mode. The auxiliary head is used as a small "
            "parent-level MI correction; no model weights are changed. Because Fold 10 has "
            "already been inspected in an earlier experiment, this experiment requires a "
            "stronger Fold-9 promotion margin before any additional benchmark check."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined validation ECGs: {len(joined)}")
    print(f"Selected primary MI weight alpha: {result.alpha_primary:.3f}")
    print(f"Auxiliary MI correction weight: {1.0 - result.alpha_primary:.3f}")
    print(f"Selected MI threshold: {best.mi_threshold:.6f}")
    print(f"Selected STEMI|MI threshold: {best.stemi_threshold:.6f}")
    print_summary("VALIDATION AUX-BLEND STRICT-80 CASCADE", best.summary, best.confusion_matrix)
    print(f"Promotion minimum required = {promotion_minimum:.4f}")
    print(f"Promotion passed: {validation['promotion_passed']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


def finalize_mode(cfg: dict) -> None:
    out_dir = ROOT / cfg["output"]["out_dir"]
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Run validation mode first")
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = summary["validation"]
    if not bool(validation["promotion_passed"]):
        raise RuntimeError(
            "Validation did not reach the stronger promotion margin; refusing another Fold-10 check."
        )

    inputs = cfg["inputs"]
    joined = load_joined(
        ROOT / inputs["stage1_test_predictions"],
        ROOT / inputs["stage2_test_details"],
    )
    alpha = float(validation["alpha_primary"])
    blended = blend_parent_probability(
        joined["p_mi"].to_numpy(dtype=float),
        joined["p_aux_non_mi"].to_numpy(dtype=float),
        alpha,
    )
    prediction = hierarchical_predict(
        blended,
        joined["p_stemi_given_mi"].to_numpy(dtype=float),
        float(validation["mi_threshold"]),
        float(validation["stemi_threshold"]),
    )
    test_summary, cm = six_metric_summary(
        joined["actual_id"].to_numpy(dtype=int),
        prediction,
        target=float(summary["target"]),
    )
    save_predictions(
        joined,
        blended,
        prediction,
        alpha,
        float(validation["mi_threshold"]),
        float(validation["stemi_threshold"]),
        out_dir / "test_predictions.csv",
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv"
    )
    summary["test"] = flatten(test_summary)
    summary["test_per_class"] = test_summary["per_class"]
    summary["test_confusion_matrix"] = cm.tolist()
    summary["test_evaluated"] = True
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary("FROZEN AUX-BLEND -> FOLD 10 BENCHMARK", test_summary, cm)
    print(f"Updated: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use the existing Stage1-v3 auxiliary head as a small MI-ranking correction before the existing Stage2."
    )
    parser.add_argument("--config", default="configs/stage1_v3_stage2_auxblend80.yaml")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.finalize_only:
        finalize_mode(cfg)
    else:
        validation_mode(cfg, args.config)


if __name__ == "__main__":
    main()
