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

from miclass3.probability_stacker import (
    CLASS_NAMES,
    FEATURE_NAMES,
    apply_class_biases,
    build_stacker_features,
    model_probabilities,
    search_logistic_stacker,
)
from miclass3.strict80 import six_metric_summary


def _require(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def load_joined(stage1_path: Path, stage2_path: Path) -> pd.DataFrame:
    s1 = pd.read_csv(stage1_path)
    s2 = pd.read_csv(stage2_path)
    _require(
        s1,
        [
            "ecg_id",
            "actual_id",
            "p_mi",
            "p_aux_non_mi",
            "p_aux_stemi_proxy",
            "p_aux_nstemi_proxy",
        ],
        stage1_path,
    )
    _require(s2, ["ecg_id", "actual_id", "p_stemi_given_mi"], stage2_path)

    left = s1[
        [
            "ecg_id",
            "actual_id",
            "p_mi",
            "p_aux_non_mi",
            "p_aux_stemi_proxy",
            "p_aux_nstemi_proxy",
        ]
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


def make_features(frame: pd.DataFrame) -> np.ndarray:
    return build_stacker_features(
        frame["p_mi"].to_numpy(float),
        frame["p_stemi_given_mi"].to_numpy(float),
        frame["p_aux_non_mi"].to_numpy(float),
        frame["p_aux_stemi_proxy"].to_numpy(float),
        frame["p_aux_nstemi_proxy"].to_numpy(float),
    )


def flatten(summary: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {
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
    print("\n" + "#" * 56)
    print(title)
    print("#" * 56)
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
    probabilities: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> None:
    out = joined.copy()
    out["stacker_non_mi"] = probabilities[:, 0]
    out["stacker_stemi_proxy"] = probabilities[:, 1]
    out["stacker_nstemi_proxy"] = probabilities[:, 2]
    out["predicted_id"] = predictions.astype(int)
    out["predicted_label"] = [CLASS_NAMES[int(x)] for x in predictions]
    out.to_csv(path, index=False)


def run_validation(cfg: dict, config_path: str) -> None:
    inputs = cfg["inputs"]
    target = float(cfg.get("target", 0.80))
    promotion = float(cfg.get("promotion_minimum_of_six", target))
    search = cfg.get("search", {}) or {}

    train = load_joined(
        ROOT / inputs["stage1_train_predictions"],
        ROOT / inputs["stage2_train_details"],
    )
    val = load_joined(
        ROOT / inputs["stage1_validation_predictions"],
        ROOT / inputs["stage2_validation_details"],
    )
    x_train = make_features(train)
    x_val = make_features(val)
    y_train = train["actual_id"].to_numpy(dtype=int)
    y_val = val["actual_id"].to_numpy(dtype=int)

    best, rows = search_logistic_stacker(
        x_train,
        y_train,
        x_val,
        y_val,
        c_values=search.get("c_values", [0.03, 0.1, 0.3, 1.0, 3.0]),
        class_weight_powers=search.get(
            "class_weight_powers", [0.0, 0.25, 0.5, 0.75]
        ),
        target=target,
        bias_min=float(search.get("bias_min", -0.8)),
        bias_max=float(search.get("bias_max", 0.8)),
        bias_coarse_step=float(search.get("bias_coarse_step", 0.1)),
        bias_fine_radius=float(search.get("bias_fine_radius", 0.15)),
        bias_fine_step=float(search.get("bias_fine_step", 0.02)),
    )

    out_dir = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "validation_model_candidates.csv", index=False)

    bias = best.bias_result
    pd.DataFrame(
        bias.confusion_matrix, index=CLASS_NAMES, columns=CLASS_NAMES
    ).to_csv(out_dir / "validation_confusion_matrix.csv")
    save_predictions(
        val,
        bias.probabilities,
        bias.predictions,
        out_dir / "validation_predictions.csv",
    )

    np.savez(
        out_dir / "stacker_model.npz",
        mean=best.mean,
        scale=best.scale,
        coef=best.coef,
        intercept=best.intercept,
        C=np.asarray([best.c_value], dtype=float),
        class_weight_power=np.asarray([best.class_weight_power], dtype=float),
        stemi_bias=np.asarray([bias.stemi_bias], dtype=float),
        nstemi_bias=np.asarray([bias.nstemi_bias], dtype=float),
    )

    validation = flatten(bias.summary)
    validation.update(
        {
            "C": best.c_value,
            "class_weight_power": best.class_weight_power,
            "stemi_bias": bias.stemi_bias,
            "nstemi_bias": bias.nstemi_bias,
            "promotion_minimum_of_six": promotion,
            "promotion_passed": bool(bias.summary["minimum_of_six"] >= promotion),
            "n_records": int(len(val)),
        }
    )
    summary = {
        "experiment": "stage1_v3_stage2_regularized_logistic_stacker",
        "config": config_path,
        "feature_names": list(FEATURE_NAMES),
        "selection_objective": "train_fitted_stacker_plus_fold9_maximin_six_metric_selection",
        "target": target,
        "promotion_minimum_of_six": promotion,
        "train_records": int(len(train)),
        "validation": validation,
        "validation_per_class": bias.summary["per_class"],
        "validation_confusion_matrix": bias.confusion_matrix.tolist(),
        "test_evaluated": False,
        "test": None,
        "note": (
            "The multinomial stacker is fitted on train-fold predictions only. "
            "C/class weighting and small class biases are selected on Fold 9."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Joined train ECGs: {len(train)}")
    print(f"Joined validation ECGs: {len(val)}")
    print("Stacker features:", ", ".join(FEATURE_NAMES))
    print(f"Selected C: {best.c_value:.6g}")
    print(f"Selected class-weight power: {best.class_weight_power:.3f}")
    print(f"Selected STEMI logit bias: {bias.stemi_bias:.3f}")
    print(f"Selected NSTEMI logit bias: {bias.nstemi_bias:.3f}")
    print_result("VALIDATION REGULARIZED LOGISTIC STACKER", bias.summary, bias.confusion_matrix)
    print(f"Promotion minimum required = {promotion:.4f}")
    print(f"Promotion passed: {validation['promotion_passed']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


def run_finalize(cfg: dict) -> None:
    inputs = cfg["inputs"]
    out_dir = ROOT / cfg["output"]["out_dir"]
    metrics_path = out_dir / "metrics.json"
    model_path = out_dir / "stacker_model.npz"
    if not metrics_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("run validation stacker selection first")

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not bool(summary["validation"]["promotion_passed"]):
        raise RuntimeError("validation promotion margin was not met; refusing finalize")

    test = load_joined(
        ROOT / inputs["stage1_test_predictions"],
        ROOT / inputs["stage2_test_details"],
    )
    x_test = make_features(test)
    saved = np.load(model_path)
    raw = model_probabilities(
        x_test,
        saved["mean"],
        saved["scale"],
        saved["coef"],
        saved["intercept"],
    )
    calibrated = apply_class_biases(
        raw,
        float(saved["stemi_bias"][0]),
        float(saved["nstemi_bias"][0]),
    )
    pred = calibrated.argmax(axis=1)
    test_summary, cm = six_metric_summary(
        test["actual_id"].to_numpy(dtype=int), pred, target=float(summary["target"])
    )
    summary["test"] = flatten(test_summary)
    summary["test"]["n_records"] = int(len(test))
    summary["test_per_class"] = test_summary["per_class"]
    summary["test_confusion_matrix"] = cm.tolist()
    summary["test_evaluated"] = True
    summary["note"] = (
        "Fold-10 benchmark uses the frozen train-fitted stacker and Fold-9-selected hyperparameters/biases; "
        "no Fold-10 fitting or selection is performed."
    )
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv"
    )
    save_predictions(test, calibrated, pred, out_dir / "test_predictions.csv")
    print_result("FROZEN LOGISTIC STACKER -> FOLD 10", test_summary, cm)
    print(f"Updated: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/stage1_v3_stage2_logistic_stacker80.yaml"
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
