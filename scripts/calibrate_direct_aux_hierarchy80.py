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

from miclass3.direct_aux_hierarchy import (
    direct_aux_hierarchy_predict,
    search_direct_aux_hierarchy,
)
from miclass3.strict80 import six_metric_summary

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def print_result(title: str, summary: dict[str, object], cm: np.ndarray) -> None:
    print("\n" + "#" * 56)
    print(title)
    print("#" * 56)
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
        description="Calibrate the frozen Direct model's MI/STEMI auxiliary hierarchy on Fold 9."
    )
    parser.add_argument("--config", default="configs/direct_aux_hierarchy80.yaml")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    target = float(cfg.get("target", 0.80))
    promotion = float(cfg.get("promotion_minimum_of_six", target))
    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    if not args.finalize_only:
        prediction_path = ROOT / cfg["inputs"]["validation_aux_predictions"]
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing {prediction_path}. First run:\n"
                "python scripts/export_direct_aux_predictions.py --split val --device cpu"
            )
        frame = pd.read_csv(prediction_path)
        required = ["ecg_id", "actual_id", "p_mi_aux", "p_stemi_aux"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{prediction_path} missing columns: {missing}")

        y = frame["actual_id"].to_numpy(dtype=int)
        best, rows = search_direct_aux_hierarchy(
            y,
            frame["p_mi_aux"].to_numpy(float),
            frame["p_stemi_aux"].to_numpy(float),
            target=target,
            **(cfg.get("search", {}) or {}),
        )
        pd.DataFrame(rows).to_csv(out_dir / "validation_candidates.csv", index=False)
        pd.DataFrame(best.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            out_dir / "validation_confusion_matrix.csv"
        )
        prediction_table = frame.copy()
        prediction_table["predicted_id"] = best.predictions.astype(int)
        prediction_table.to_csv(out_dir / "validation_predictions.csv", index=False)

        validation = {
            "mi_threshold": best.mi_threshold,
            "stemi_threshold": best.stemi_threshold,
            "minimum_of_six": best.summary["minimum_of_six"],
            "mean_of_six": best.summary["mean_of_six"],
            "accuracy": best.summary["accuracy"],
            "all_six_strictly_above_target": best.summary["all_six_strictly_above_target"],
            "promotion_minimum_of_six": promotion,
            "promotion_passed": bool(best.summary["minimum_of_six"] >= promotion),
            "n_records": int(len(frame)),
        }
        summary = {
            "experiment": "direct_existing_binary_auxiliary_heads_hierarchy",
            "selection_objective": "Fold9 maximize minimum of six class-wise sensitivity/specificity metrics",
            "target": target,
            "promotion_minimum_of_six": promotion,
            "validation": validation,
            "validation_per_class": best.summary["per_class"],
            "validation_confusion_matrix": best.confusion_matrix.tolist(),
            "test_evaluated": False,
            "test": None,
            "note": "No ECG model weights were retrained; only Fold-9 thresholds for existing auxiliary heads were selected.",
        }
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Validation ECGs: {len(frame)}")
        print(f"Selected MI-aux threshold: {best.mi_threshold:.6f}")
        print(f"Selected STEMI-aux threshold: {best.stemi_threshold:.6f}")
        print_result("VALIDATION DIRECT AUXILIARY HIERARCHY", best.summary, best.confusion_matrix)
        print(f"Promotion minimum required = {promotion:.4f}")
        print(f"Promotion passed: {validation['promotion_passed']}")
        print(f"Saved: {metrics_path}")
        return

    if not metrics_path.is_file():
        raise FileNotFoundError("Run Fold-9 calibration first")
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    val = summary["validation"]
    if not bool(val["all_six_strictly_above_target"]):
        raise RuntimeError("Fold-9 auxiliary hierarchy did not meet the strict >80% target; refusing finalize")

    prediction_path = ROOT / cfg["inputs"]["test_aux_predictions"]
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Missing {prediction_path}. Generate it only after accepting validation with:\n"
            "python scripts/export_direct_aux_predictions.py --split test --device cpu"
        )
    frame = pd.read_csv(prediction_path)
    pred = direct_aux_hierarchy_predict(
        frame["p_mi_aux"].to_numpy(float),
        frame["p_stemi_aux"].to_numpy(float),
        float(val["mi_threshold"]),
        float(val["stemi_threshold"]),
    )
    test_summary, cm = six_metric_summary(
        frame["actual_id"].to_numpy(dtype=int), pred, target=float(summary["target"])
    )
    summary["test"] = {
        "minimum_of_six": test_summary["minimum_of_six"],
        "mean_of_six": test_summary["mean_of_six"],
        "accuracy": test_summary["accuracy"],
        "all_six_strictly_above_target": test_summary["all_six_strictly_above_target"],
        "n_records": int(len(frame)),
    }
    summary["test_per_class"] = test_summary["per_class"]
    summary["test_confusion_matrix"] = cm.tolist()
    summary["test_evaluated"] = True
    summary["note"] = "Fold-10 used thresholds frozen from Fold 9; no Fold-10 selection was performed."
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "test_confusion_matrix.csv")
    output = frame.copy()
    output["predicted_id"] = pred.astype(int)
    output.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"Frozen MI-aux threshold: {float(val['mi_threshold']):.6f}")
    print(f"Frozen STEMI-aux threshold: {float(val['stemi_threshold']):.6f}")
    print_result("FROZEN DIRECT AUXILIARY HIERARCHY -> FOLD 10", test_summary, cm)
    print(f"Updated: {metrics_path}")


if __name__ == "__main__":
    main()
