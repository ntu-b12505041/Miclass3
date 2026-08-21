from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.data import FeatureTransform, PTBXL500Dataset
from miclass3.direct_aux_hierarchy import sigmoid


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
        description="Export frozen Direct class/MI/STEMI probabilities for one PTB-XL split."
    )
    parser.add_argument("--checkpoint", default="artifacts/direct_common65_matched/best_model.pt")
    parser.add_argument("--manifest", default="data/label_manifest_custom65_common_directmatched.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    manifest_path = ROOT / args.manifest
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
    fold_key = {"train": "train_folds", "val": "val_folds", "test": "test_folds"}[args.split]

    manifest = pd.read_csv(manifest_path).query(
        "label_tier != 'excluded' and label_id.notna()", engine="python"
    ).copy()
    manifest["label_id"] = manifest["label_id"].astype(int)
    frame = manifest[manifest.strat_fold.isin(data_cfg[fold_key])].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{args.split} split is empty")

    feature_columns = list(checkpoint["feature_columns"])
    transform = transform_from_dict(checkpoint["feature_transform"])
    if tuple(feature_columns) != transform.columns:
        raise ValueError("checkpoint feature columns and transform columns do not match")

    dataset = PTBXL500Dataset(
        frame,
        ROOT / data_cfg["data_dir"],
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
    model = make_model("morphology_fusion", dataset.feature_dim, architecture=architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    labels = []
    class_probabilities = []
    mi_logits = []
    stemi_logits = []
    with torch.no_grad():
        for x, y, features in loader:
            output = model(
                x.to(device, non_blocking=True),
                features.to(device, non_blocking=True),
            )
            class_probabilities.append(torch.softmax(output["class_logits"], dim=1).cpu().numpy())
            mi_logits.append(output["mi_logits"].cpu().numpy())
            stemi_logits.append(output["stemi_logits"].cpu().numpy())
            labels.append(y.numpy())

    class_prob = np.concatenate(class_probabilities)
    mi_logit = np.concatenate(mi_logits)
    stemi_logit = np.concatenate(stemi_logits)
    y = np.concatenate(labels).astype(int)

    output = pd.DataFrame(
        {
            "ecg_id": frame["ecg_id"].to_numpy(),
            "actual_id": y,
            "prob_non_mi": class_prob[:, 0],
            "prob_stemi_proxy": class_prob[:, 1],
            "prob_nstemi_proxy": class_prob[:, 2],
            "mi_aux_logit": mi_logit,
            "p_mi_aux": sigmoid(mi_logit),
            "stemi_aux_logit": stemi_logit,
            "p_stemi_aux": sigmoid(stemi_logit),
        }
    )
    out_path = ROOT / (args.out or f"artifacts/direct_common65_matched/{args.split}_aux_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"Frozen Direct auxiliary export complete: split={args.split} n={len(output)}")
    print(f"Saved: {out_path}")
    print("No training or split-specific model selection was performed.")


if __name__ == "__main__":
    main()
