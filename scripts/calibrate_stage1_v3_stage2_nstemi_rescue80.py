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

from miclass3.nstemi_rescue import nstemi_rescue_predict, search_nstemi_rescue
from miclass3.strict80 import six_metric_summary

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def _require(frame: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def load_joined(stage1_path: Path, stage2_path: Path) -> pd.DataFrame:
    s1 = pd.read_csv(stage1_path)
    s2 = pd.read_csv(stage2_path)
    _require(
        s1,
        ["ecg_id", "actual_id", "p_mi", "p_aux_non_mi", "p_aux_nstemi_proxy"],
        stage1_path,
    )
    _require(s2, ["ecg_id", "actual_id", "p_stemi_given_mi"], stage2_path)

    left = s1[
        ["ecg_id", "actual_id", "p_mi", "p_aux_non_mi", "p_aux_nstemi_proxy"]
    ].copy()
    right = s2[["ecg_id", "actual_id", "p_stemi_given_mi"]].copy()
    joined = left.merge(
        right,
        on="ecg_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_s1", "_s2"),
    )
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError(
            f"prediction ECG sets differ: stage1={len(left)}, stage2={len(right)}, joined={len(joined)}"
        )
    mismatch = joined["actual_id_s1"] != joined["actual_id_s2"]
    if mismatch.any():
        raise ValueError("Stage1/Stage2 actual labels do not match")
    joined = joined.rename(columns={"actual_id_s1": "actual_id"}).drop(
        columns=["actual_id_s2"]
    )
    return joined


def flatten(summary: dict[str, object]) -> dict[str, object]:
    out = {
        "target": summary["target"],
        "minimum_of_six": summary["minimum_of_six"],
        "mean_of_six": summary["mean_of_six"],
        "accuracy": summary["accuracy"],
        "all_six_strictly_above_target": summary["all_six_strictly_above_target"],
    }
    for name in CLASS_NAMES:
        out[f"{name}_sensitivity"] = summary["per_class"][name]["sensitivity"]
        out[f"{name}_specificity"] = summary["per_class"][name]["specificity"]
    return out


def print_result(title: str, summary: dict[str, object], cm: np.ndarray) -> None:
    print("\n" + "#" * 54)
    print(title)
    print("#" * 54)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print("Per-class sensitivity / specificity:")
    for name in CLASS_NAMES:
        m = summary["per_class"][name]
        print(
            f"{name:13s} sensitivity={m['sensitivity']:.4f} specificity={m['specificity']:.4f}"
        )
    print(f"Minimum of six metrics = {summary['minimum_of_six']:.4f}")
    print(
        f"All six strictly > {float(summary['target']):.2f}: "
        f"{summary['all_six_strictly_above_target']}"
    )


def save_predictions(
    joined: pd.DataFrame,
    pred: np.ndarray,
    *,
    mi_threshold: float,
    stemi_threshold: float,
    rescue_margin: float,
    rescue_floor: float,
    path: Path,
) -> None:
    out = joined.copy()
    out["predicted_id"] = pred.astype(int)
    out["predicted_label"] = [CLASS_NAMES[int(x)] for x in pred]
    out["mi_threshold"] = mi_threshold
    out["stemi_threshold"] = stemi_threshold
    out["rescue_margin"] = rescue_margin
    out["rescue_floor"] = rescue_floor
    out["aux_nstemi_minus_nonmi"] = out["p_aux_nstemi_proxy"] - out["p_aux_non_mi"]
    out.to_csv(path, index=False)


def run_validation(cfg: dict, config_path: str) -> None:
    inputs = cfg["inputs"]
    target = float(cfg.get("target", 0.80))
    promotion = float(cfg.get("promotion_minimum_of_six", target))
    search = cfg.get("search", {}) or {}

    joined = load_joined(
        ROOT / inputs["stage1_validation_predictions"],
        ROOT / inputs["stage2_validation_details"],
    )
    y = joined["actual_id"].to_numpy(dtype=int)

    best, rows = search_nstemi_rescue(
        y,
        joined["p_mi"].to_numpy(float),
        joined["p_stemi_given_mi"].to_numpy(float),
        joined["p_aux_non_mi"].to_numpy(float),
        joined["p_aux_nstemi_proxy"].to_numpy(float),
        target=target,
        **search,
    )

    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "validation_candidates.csv", index=False)
    pd.DataFrame(best.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "validation_confusion_matrix.csv"
    )
    save_predictions(
        joined,
        best.predictions,
        mi_threshold=best.mi_threshold,
        stemi_threshold=best.stemi_threshold,
        rescue_margin=best.rescue_margin,
        rescue_floor=best.rescue_floor,
        path=out_dir / "validation_predictions.csv",
    )

    validation = flatten(best.summary)
    validation.update(
        {
            "mi_threshold": best.mi_threshold,
            "stemi_threshold": best.stemi_threshold,
            "rescue_margin": best.rescue_margin,
            "rescue_floor": best.rescue_floor,
            "rescue_band": best.mi_threshold - best.rescue_floor,
            "promotion_minimum_of_six": promotion,
            "promotion_passed": bool(best.summary["minimum_of_six"] >= promotion),
            "n_records": int(len(joined)),
        }
    )
    summary = {
        "experiment": "stage1_v3_stage2_borderline_nstemi_rescue",
        "config": config_path,
        "selection_objective": "validation_only_maximize_minimum_of_six",
        "target": target,
        "promotion_minimum_of_six": promotion,
        "validation": validation,
        "validation_per_class": best.summary["per_class"],
        "validation_confusion_matrix": best.confusion_matrix.tolist(),
        "test_evaluated": False,
        "test": None,
        "note": "All rescue and threshold parameters are selected on Fold 9 only.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined validation ECGs: {len(joined)}")
    print(f"Selected MI threshold: {best.mi_threshold:.6f}")
    print(f"Selected STEMI|MI threshold: {best.stemi_threshold:.6f}")
    print(f"Selected rescue floor: {best.rescue_floor:.6f}")
    print(f"Selected rescue band: {best.mi_threshold - best.rescue_floor:.6f}")
    print(f"Selected aux NSTEMI-nonMI margin: {best.rescue_margin:.6f}")
    print_result("VALIDATION BORDERLINE NSTEMI RESCUE", best.summary, best.confusion_matrix)
    print(f"Promotion minimum required = {promotion:.4f}")
    print(f"Promotion passed: {validation['promotion_passed']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


def run_finalize(cfg: dict) -> None:
    inputs = cfg["inputs"]
    out_dir = ROOT / cfg["output"]["out_dir"]
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("run validation selection first")
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    val = summary["validation"]
    if not bool(val["promotion_passed"]):
        raise RuntimeError("validation promotion margin was not met; refusing finalize")

    joined = load_joined(
        ROOT / inputs["stage1_test_predictions"],
        ROOT / inputs["stage2_test_details"],
    )
    pred = nstemi_rescue_predict(
        joined["p_mi"].to_numpy(float),
        joined["p_stemi_given_mi"].to_numpy(float),
        joined["p_aux_non_mi"].to_numpy(float),
        joined["p_aux_nstemi_proxy"].to_numpy(float),
        mi_threshold=float(val["mi_threshold"]),
        stemi_threshold=float(val["stemi_threshold"]),
        rescue_margin=float(val["rescue_margin"]),
        rescue_floor=float(val["rescue_floor"]),
    )
    test_summary, cm = six_metric_summary(
        joined["actual_id"].to_numpy(dtype=int), pred, target=float(summary["target"])
    )
    test = flatten(test_summary)
    test["n_records"] = int(len(joined))
    summary["test"] = test
    summary["test_per_class"] = test_summary["per_class"]
    summary["test_confusion_matrix"] = cm.tolist()
    summary["test_evaluated"] = True
    summary["note"] = "Finalize uses parameters frozen from Fold 9; no Fold-10 search is performed."
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv"
    )
    save_predictions(
        joined,
        pred,
        mi_threshold=float(val["mi_threshold"]),
        stemi_threshold=float(val["stemi_threshold"]),
        rescue_margin=float(val["rescue_margin"]),
        rescue_floor=float(val["rescue_floor"]),
        path=out_dir / "test_predictions.csv",
    )
    print_result("FROZEN RESCUE RULE -> FOLD 10", test_summary, cm)
    print(f"Updated: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/stage1_v3_stage2_nstemi_rescue80.yaml"
    )
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.finalize_only:
        run_finalize(cfg)
    else:
        run_validation(cfg, args.config)


if __name__ == "__main__":
    main()
