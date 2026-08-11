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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score an existing cascade run with validation-calibrated soft hierarchy; no retraining required."
    )
    parser.add_argument("--source-dir", default="artifacts/cascade_v2")
    parser.add_argument("--out-dir", default="artifacts/cascade_v2_soft_rescore")
    parser.add_argument("--minimum-stemi-recall", type=float, default=0.75)
    parser.add_argument("--bias-min", type=float, default=-2.0)
    parser.add_argument("--bias-max", type=float, default=2.0)
    parser.add_argument("--bias-step", type=float, default=0.1)
    args = parser.parse_args()

    source = ROOT / args.source_dir
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    val = load_details(source / "val_cascade_details.csv")
    summary, candidates = calibrate_soft_cascade(
        val["actual_id"].to_numpy(dtype=int),
        val["p_mi"].to_numpy(dtype=float),
        val["p_stemi_given_mi"].to_numpy(dtype=float),
        minimum_stemi_recall=args.minimum_stemi_recall,
        bias_min=args.bias_min,
        bias_max=args.bias_max,
        bias_step=args.bias_step,
    )
    candidates.to_csv(out / "val_soft_calibration_candidates.csv", index=False)

    mi_bias = float(summary["mi_logit_bias"])
    stemi_bias = float(summary["stemi_logit_bias"])
    results: dict[str, object] = {"calibration": summary, "splits": {}}

    for split in ("train", "val", "test"):
        path = source / f"{split}_cascade_details.csv"
        if not path.is_file():
            continue
        frame = load_details(path)
        y = frame["actual_id"].to_numpy(dtype=int)
        p_mi = frame["p_mi"].to_numpy(dtype=float)
        p_stemi = frame["p_stemi_given_mi"].to_numpy(dtype=float)
        raw = compose_soft_cascade(p_mi, p_stemi)
        calibrated = compose_calibrated_soft_cascade(
            p_mi,
            p_stemi,
            mi_logit_bias=mi_bias,
            stemi_logit_bias=stemi_bias,
        )
        metrics = write_split_artifacts(out, split, y, calibrated, records=None, stemi_threshold=None)
        raw_metrics = write_split_artifacts(out / "uncalibrated", split, y, raw, records=None, stemi_threshold=None)
        results["splits"][split] = {"calibrated_soft": metrics, "uncalibrated_soft": raw_metrics}

        export = frame.copy()
        export["cal_prob_non_mi"] = calibrated[:, 0]
        export["cal_prob_stemi_proxy"] = calibrated[:, 1]
        export["cal_prob_nstemi_proxy"] = calibrated[:, 2]
        export["calibrated_prediction"] = calibrated.argmax(axis=1)
        export.to_csv(out / f"{split}_cascade_details.csv", index=False)

    (out / "soft_rescore_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results.get("splits", {}).get("test", {}), indent=2))


if __name__ == "__main__":
    main()
