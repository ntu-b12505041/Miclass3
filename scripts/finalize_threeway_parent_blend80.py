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

from miclass3.strict80 import hierarchical_predict, six_metric_summary
from miclass3.threeway_parent_blend import blend_three_parent_log_odds

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame[columns].copy()


def load_test(cfg: dict) -> pd.DataFrame:
    inputs = cfg["inputs"]
    s1 = _read(ROOT / inputs["stage1_test_predictions"], ["ecg_id", "actual_id", "p_mi"]).rename(columns={"actual_id": "actual_s1"})
    direct = _read(
        ROOT / inputs["direct_test_predictions"],
        ["ecg_id", "actual_id", "prob_non_mi"],
    ).rename(columns={"actual_id": "actual_direct"})
    aux = _read(
        ROOT / inputs["direct_aux_test_predictions"],
        ["ecg_id", "actual_id", "p_mi_aux"],
    ).rename(columns={"actual_id": "actual_aux"})
    s2 = _read(
        ROOT / inputs["stage2_test_details"],
        ["ecg_id", "actual_id", "p_stemi_given_mi"],
    ).rename(columns={"actual_id": "actual_s2"})

    joined = s1.merge(direct, on="ecg_id", validate="one_to_one").merge(
        aux, on="ecg_id", validate="one_to_one"
    ).merge(s2, on="ecg_id", validate="one_to_one")
    expected = len(s1)
    if not (len(joined) == expected == len(direct) == len(aux) == len(s2)):
        raise ValueError("Fold10 prediction ECG sets do not match exactly")
    labels = joined[["actual_s1", "actual_direct", "actual_aux", "actual_s2"]]
    if not (labels.nunique(axis=1) == 1).all():
        raise ValueError("Fold10 actual labels disagree across prediction sources")
    joined["actual_id"] = joined["actual_s1"].astype(int)
    joined["p_direct_class_mi"] = 1.0 - joined["prob_non_mi"].astype(float)
    return joined


def print_result(summary: dict[str, object], cm: np.ndarray) -> None:
    print("\n##########################################################")
    print("FROZEN THREE-WAY PARENT BLEND -> FOLD 10")
    print("##########################################################")
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
        description="Evaluate the Fold9-frozen three-way parent blend on Fold10 without any test-side search."
    )
    parser.add_argument("--config", default="configs/threeway_parent_blend80.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["out_dir"]
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Run Fold9 calibration first so frozen parameters exist")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    val = stored["validation"]
    if not bool(val.get("all_six_strictly_above_target", False)):
        raise RuntimeError("Fold9 did not meet the >0.80 acceptance target; refusing finalize")

    w1 = float(val["weight_stage1"])
    w2 = float(val["weight_direct_aux"])
    w3 = float(val["weight_direct_class"])
    t_mi = float(val["mi_threshold"])
    t_stemi = float(val["stemi_threshold"])

    joined = load_test(cfg)
    parent = blend_three_parent_log_odds(
        joined["p_mi"].to_numpy(float),
        joined["p_mi_aux"].to_numpy(float),
        joined["p_direct_class_mi"].to_numpy(float),
        w1,
        w2,
    )
    pred = hierarchical_predict(
        parent,
        joined["p_stemi_given_mi"].to_numpy(float),
        t_mi,
        t_stemi,
    )
    target = float(stored.get("target", cfg.get("target", 0.80)))
    summary, cm = six_metric_summary(joined["actual_id"].to_numpy(int), pred, target=target)

    test = {
        "n_records": int(len(joined)),
        "weight_stage1": w1,
        "weight_direct_aux": w2,
        "weight_direct_class": w3,
        "mi_threshold": t_mi,
        "stemi_threshold": t_stemi,
        "minimum_of_six": summary["minimum_of_six"],
        "mean_of_six": summary["mean_of_six"],
        "accuracy": summary["accuracy"],
        "all_six_strictly_above_target": summary["all_six_strictly_above_target"],
    }
    stored["test_evaluated"] = True
    stored["test"] = test
    stored["test_per_class"] = summary["per_class"]
    stored["test_confusion_matrix"] = cm.tolist()
    stored["note"] = (
        "Fold10 used the exact weights and thresholds frozen from Fold9. "
        "No Fold10 search, calibration, or model selection was performed."
    )
    metrics_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv"
    )
    prediction_table = pd.DataFrame(
        {
            "ecg_id": joined["ecg_id"].to_numpy(),
            "actual_id": joined["actual_id"].to_numpy(int),
            "parent_probability": parent,
            "p_stemi_given_mi": joined["p_stemi_given_mi"].to_numpy(float),
            "predicted_id": pred.astype(int),
        }
    )
    prediction_table.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"Frozen Stage1-v3 MI weight: {w1:.3f}")
    print(f"Frozen Direct MI-aux weight: {w2:.3f}")
    print(f"Frozen Direct class-derived MI weight: {w3:.3f}")
    print(f"Frozen MI threshold: {t_mi:.6f}")
    print(f"Frozen STEMI|MI threshold: {t_stemi:.6f}")
    print_result(summary, cm)
    print(f"Updated: {metrics_path}")


if __name__ == "__main__":
    main()
