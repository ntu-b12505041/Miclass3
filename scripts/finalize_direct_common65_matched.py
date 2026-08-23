from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.data import FeatureTransform, PTBXL500Dataset


def transform_from_dict(payload: dict[str, object]) -> FeatureTransform:
    return FeatureTransform(
        columns=tuple(payload["columns"]),
        impute_values=np.asarray(payload["impute_values"], dtype=np.float32),
        centers=np.asarray(payload["centers"], dtype=np.float32),
        scales=np.asarray(payload["scales"], dtype=np.float32),
        binary_mask=np.asarray(payload["binary_mask"], dtype=bool),
        clip=float(payload.get("clip", 6.0)),
        add_missing_indicators=bool(payload.get("add_missing_indicators", True)),
    )


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from miclass3.models import make_model

    parser = argparse.ArgumentParser(
        description="Run the frozen Direct common-cohort checkpoint on PTB-XL Fold 10 without retraining."
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/direct_common65_matched/best_model.pt",
    )
    parser.add_argument(
        "--manifest",
        default="data/label_manifest_custom65_common_directmatched.csv",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--out",
        default="artifacts/direct_common65_matched/test_predictions.csv",
    )
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    manifest_path = ROOT / args.manifest
    out_path = ROOT / args.out
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    data_cfg = cfg["data"]
    data_dir = ROOT / data_cfg["data_dir"]

    manifest = pd.read_csv(manifest_path).query(
        "label_tier != 'excluded' and label_id.notna()", engine="python"
    ).copy()
    manifest["label_id"] = manifest["label_id"].astype(int)
    test = manifest[manifest.strat_fold.isin(data_cfg["test_folds"])].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("Fold-10 test split is empty")
    if set(test["label_id"].unique()) != {0, 1, 2}:
        raise ValueError("Fold-10 test split must contain all three proxy classes")

    feature_columns = list(checkpoint["feature_columns"])
    transform = transform_from_dict(checkpoint["feature_transform"])
    if tuple(feature_columns) != transform.columns:
        raise ValueError("checkpoint feature columns and transform columns do not match")

    dataset = PTBXL500Dataset(
        test,
        data_dir,
        feature_columns=feature_columns,
        feature_transform=transform,
        waveform_normalization=str(data_cfg.get("waveform_normalization", "per_lead_zscore")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    architecture = cfg["models"].get("architecture", {}) or {}
    model = make_model(
        "morphology_fusion",
        dataset.feature_dim,
        architecture=architecture,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    probabilities = []
    labels = []
    with torch.no_grad():
        for x, y, features in loader:
            logits = model(
                x.to(device, non_blocking=True),
                features.to(device, non_blocking=True),
            )["class_logits"]
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())

    prob = np.concatenate(probabilities)
    y = np.concatenate(labels).astype(int)
    if len(prob) != len(test):
        raise RuntimeError("prediction row count does not match Fold-10 manifest")

    output = pd.DataFrame(
        {
            "ecg_id": test["ecg_id"].to_numpy(),
            "actual_id": y,
            "prob_non_mi": prob[:, 0],
            "prob_stemi_proxy": prob[:, 1],
            "prob_nstemi_proxy": prob[:, 2],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"Frozen Direct Fold-10 inference complete: {len(output)} ECGs")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved: {out_path}")
    print("No training or Fold-10 model/threshold selection was performed.")


if __name__ == "__main__":
    main()
