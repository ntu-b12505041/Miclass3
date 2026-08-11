from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.cascade import calibrate_binary_threshold, compose_hard_cascade, compose_soft_cascade
from miclass3.data import FeatureTransform, PTBXL500Dataset
from miclass3.metrics import write_split_artifacts
from miclass3.models import MorphologyFusion, SEResNet

# Avoid direct target shortcuts such as modified_sgarbossa_positive. These
# continuous / structural ECG features still expose clinically relevant
# morphology without feeding the final proxy-label rule itself to stage 2.
CASCADE_STAGE2_FEATURES = (
    "max_st_j_mv",
    "max_st_j60_mv",
    "max_st_s_ratio",
    "qrs_duration_ms",
    "lbbb",
)


def normalized_class_weights(counts: np.ndarray, power: float) -> np.ndarray:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    weights = np.power(counts, -float(power))
    return weights / weights.mean()


def binary_metrics(y_true: np.ndarray, positive_probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(positive_probability, dtype=float)
    pred = (probability >= float(threshold)).astype(int)
    output: dict[str, float | int] = {
        "n_records": int(len(y_true)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    try:
        output["auroc"] = float(roc_auc_score(y_true, probability))
        output["auprc"] = float(average_precision_score(y_true, probability))
    except ValueError:
        output["auroc"] = float("nan")
        output["auprc"] = float("nan")
    return output


def target_for_stage(labels, stage: int):
    if stage == 1:
        return (labels != 0).long()
    if stage == 2:
        return (labels == 1).long()
    raise ValueError("stage must be 1 or 2")


def forward_logits(model, x, features, device, stage: int):
    if stage == 1:
        return model(x.to(device, non_blocking=True))["class_logits"]
    return model(
        x.to(device, non_blocking=True),
        features.to(device, non_blocking=True),
    )["class_logits"]


def predict_binary(model, loader, device, stage: int):
    import torch

    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for x, y, features in loader:
            logits = forward_logits(model, x, features, device, stage)
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            targets.append(target_for_stage(y, stage).numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def train_stage(stage: int, train_loader, val_loader, device, cfg: dict):
    import torch
    from torch import nn

    training = cfg["training"]
    arch = cfg["architecture"]["stage1" if stage == 1 else "stage2"]

    if stage == 1:
        model = SEResNet(
            in_channels=12,
            classes=2,
            width=int(arch.get("width", 64)),
            dropout=float(arch.get("block_dropout", 0.1)),
        ).to(device)
        class_weight_power = float(training.get("stage1_class_weight_power", 0.5))
    else:
        feature_dim = train_loader.dataset.feature_dim
        if feature_dim <= 0:
            raise ValueError("stage 2 morphology fusion requires non-empty morphology features")
        model = MorphologyFusion(
            in_channels=12,
            feature_dim=feature_dim,
            classes=2,
            width=int(arch.get("width", 64)),
            block_dropout=float(arch.get("block_dropout", 0.1)),
            fusion_dropout=float(arch.get("fusion_dropout", 0.25)),
        ).to(device)
        class_weight_power = float(training.get("stage2_class_weight_power", 0.8))

    original = train_loader.dataset.records["label_id"].to_numpy(dtype=int)
    binary = (original != 0).astype(int) if stage == 1 else (original == 1).astype(int)
    counts = np.bincount(binary, minlength=2)
    weights = normalized_class_weights(counts, class_weight_power)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    best_score = -np.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        losses = []
        start = time.perf_counter()
        for x, y, features in train_loader:
            logits = forward_logits(model, x, features, device, stage)
            target = target_for_stage(y.to(device, non_blocking=True), stage)
            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip_norm", 1.0)))
            optimizer.step()
            losses.append(float(loss.item()))

        y_val, p_val = predict_binary(model, val_loader, device, stage)
        metrics = binary_metrics(y_val, p_val, threshold=0.5)
        metrics.update(epoch=epoch, loss=float(np.mean(losses)), epoch_seconds=float(time.perf_counter() - start))
        history.append(metrics)
        score = float(metrics["auprc"])
        print(
            f"stage={stage} epoch={epoch} loss={metrics['loss']:.4f} "
            f"val_auprc={score:.4f} val_auroc={metrics['auroc']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break

    if best_state is None:
        raise RuntimeError(f"stage {stage} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, {
        "binary_class_counts": counts.tolist(),
        "class_weights": weights.tolist(),
        "class_weight_power": class_weight_power,
        "model": "seresnet" if stage == 1 else "morphology_fusion",
    }


def ensure_waveform_paths(manifest: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Restore filename_hr from official PTB-XL metadata when an older manifest omitted it."""
    if "filename_hr" in manifest.columns and manifest["filename_hr"].notna().all():
        return manifest
    metadata_path = data_dir / "ptbxl_database.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"manifest is missing filename_hr and metadata was not found at {metadata_path}"
        )
    meta = pd.read_csv(metadata_path, usecols=["ecg_id", "filename_hr"])
    if "filename_hr" in manifest.columns:
        manifest = manifest.drop(columns=["filename_hr"])
    merged = manifest.merge(meta, on="ecg_id", how="left", validate="many_to_one")
    missing = int(merged["filename_hr"].isna().sum())
    if missing:
        raise ValueError(f"{missing} manifest records could not be matched to PTB-XL filename_hr")
    print("Restored filename_hr from ptbxl_database.csv")
    return merged


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train an optimized MI -> STEMI/NSTEMI ECG cascade.")
    parser.add_argument("--config", default="configs/cascade.yaml")
    parser.add_argument("--manifest", default="data/label_manifest.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    device = torch.device(device_name)
    data_cfg = cfg["data"]
    data_dir = ROOT / data_cfg["data_dir"]
    if not (data_dir / "records500").is_dir():
        raise FileNotFoundError(f"Missing PTB-XL records500 under {data_dir / 'records500'}")

    manifest = pd.read_csv(ROOT / args.manifest).query("label_tier != 'excluded' and label_id.notna()", engine="python").copy()
    manifest["label_id"] = manifest["label_id"].astype(int)
    manifest = ensure_waveform_paths(manifest, data_dir)
    if args.max_records:
        manifest = (
            manifest.groupby(["strat_fold", "label_id"], group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, args.max_records // 30)), random_state=seed))
            .reset_index(drop=True)
        )

    full_frames = {
        "train": manifest[manifest.strat_fold.isin(data_cfg["train_folds"])].copy(),
        "val": manifest[manifest.strat_fold.isin(data_cfg["val_folds"])].copy(),
        "test": manifest[manifest.strat_fold.isin(data_cfg["test_folds"])].copy(),
    }
    stage2_frames = {name: frame[frame.label_id.isin([1, 2])].copy() for name, frame in full_frames.items()}
    for split, frame in stage2_frames.items():
        if frame.empty or frame.label_id.nunique() < 2:
            raise ValueError(f"stage 2 requires both STEMI-proxy and NSTEMI-proxy in {split}")

    feature_columns = [name for name in CASCADE_STAGE2_FEATURES if name in manifest.columns]
    if not feature_columns:
        raise ValueError(
            "No safe stage-2 morphology features were found in the manifest. "
            "Expected one or more of: " + ", ".join(CASCADE_STAGE2_FEATURES)
        )
    print("Stage 2 morphology features:", ", ".join(feature_columns))
    feature_cfg = cfg["training"].get("feature_transform", {}) or {}
    stage2_transform = FeatureTransform.fit(
        stage2_frames["train"],
        feature_columns,
        clip=float(feature_cfg.get("clip", 6.0)),
        add_missing_indicators=bool(feature_cfg.get("add_missing_indicators", True)),
    )

    def stage1_dataset(frame):
        return PTBXL500Dataset(
            frame,
            data_dir,
            feature_columns=[],
            waveform_normalization=str(data_cfg.get("waveform_normalization", "per_lead_zscore")),
        )

    def stage2_dataset(frame):
        return PTBXL500Dataset(
            frame,
            data_dir,
            feature_columns=feature_columns,
            feature_transform=stage2_transform,
            waveform_normalization=str(data_cfg.get("waveform_normalization", "per_lead_zscore")),
        )

    batch_size = int(cfg["training"]["batch_size"])
    workers = int(cfg["training"]["num_workers"])
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    stage1_datasets = {name: stage1_dataset(frame) for name, frame in full_frames.items()}
    stage2_datasets = {name: stage2_dataset(frame) for name, frame in stage2_frames.items()}
    stage2_full_datasets = {name: stage2_dataset(frame) for name, frame in full_frames.items()}

    stage1_loaders = {
        name: DataLoader(ds, shuffle=(name == "train"), **loader_kwargs)
        for name, ds in stage1_datasets.items()
    }
    stage2_loaders = {
        name: DataLoader(ds, shuffle=(name == "train"), **loader_kwargs)
        for name, ds in stage2_datasets.items()
    }
    stage2_full_loaders = {
        name: DataLoader(ds, shuffle=False, **loader_kwargs)
        for name, ds in stage2_full_datasets.items()
    }

    out_dir = ROOT / (args.out_dir or cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Training stage 1: MI vs non-MI (high-recall gate)")
    stage1, stage1_history, stage1_train_info = train_stage(1, stage1_loaders["train"], stage1_loaders["val"], device, cfg)
    y1_val, p1_val = predict_binary(stage1, stage1_loaders["val"], device, stage=1)
    mi_threshold, mi_threshold_summary, mi_candidates = calibrate_binary_threshold(
        y1_val,
        p1_val,
        minimum_recall=cfg["cascade"].get("stage1_minimum_recall"),
    )
    mi_candidates.to_csv(out_dir / "stage1_mi_threshold_candidates.csv", index=False)
    print(
        f"Selected stage-1 threshold={mi_threshold:.6f} "
        f"with validation recall={mi_threshold_summary['recall']:.4f}"
    )

    print("Training stage 2: morphology-fusion STEMI-proxy vs NSTEMI-proxy")
    stage2, stage2_history, stage2_train_info = train_stage(2, stage2_loaders["train"], stage2_loaders["val"], device, cfg)
    y2_val, p2_val = predict_binary(stage2, stage2_loaders["val"], device, stage=2)
    stemi_threshold, stemi_threshold_summary, stemi_candidates = calibrate_binary_threshold(
        y2_val,
        p2_val,
        minimum_recall=cfg["cascade"].get("stage2_minimum_recall"),
    )
    stemi_candidates.to_csv(out_dir / "stage2_stemi_threshold_candidates.csv", index=False)

    summary: dict[str, object] = {
        "device": str(device),
        "stage2_feature_columns": feature_columns,
        "stage2_feature_transform": stage2_transform.to_dict(),
        "mi_threshold": mi_threshold,
        "stemi_given_mi_threshold": stemi_threshold,
        "stage1_threshold_selection": mi_threshold_summary,
        "stage2_threshold_selection": stemi_threshold_summary,
        "stage1_train": stage1_train_info,
        "stage2_train": stage2_train_info,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        y1, p_mi = predict_binary(stage1, stage1_loaders[split], device, stage=1)
        stage1_metrics = binary_metrics(y1, p_mi, threshold=mi_threshold)

        y2, p_stemi_mi_subset = predict_binary(stage2, stage2_loaders[split], device, stage=2)
        stage2_metrics = binary_metrics(y2, p_stemi_mi_subset, threshold=stemi_threshold)

        _, p_stemi_given_mi = predict_binary(stage2, stage2_full_loaders[split], device, stage=2)
        soft_probabilities = compose_soft_cascade(p_mi, p_stemi_given_mi)
        hard_probabilities, hard_predictions = compose_hard_cascade(
            p_mi,
            p_stemi_given_mi,
            mi_threshold=mi_threshold,
            stemi_threshold=stemi_threshold,
        )

        y_three = full_frames[split]["label_id"].to_numpy(dtype=int)
        final_metrics = write_split_artifacts(
            out_dir,
            split,
            y_three,
            hard_probabilities,
            full_frames[split],
            stemi_threshold=None,
        )
        pd.DataFrame(
            {
                "ecg_id": full_frames[split]["ecg_id"].to_numpy() if "ecg_id" in full_frames[split] else np.arange(len(y_three)),
                "actual_id": y_three,
                "p_mi": p_mi,
                "p_stemi_given_mi": p_stemi_given_mi,
                "soft_prob_non_mi": soft_probabilities[:, 0],
                "soft_prob_stemi_proxy": soft_probabilities[:, 1],
                "soft_prob_nstemi_proxy": soft_probabilities[:, 2],
                "hard_cascade_prediction": hard_predictions,
            }
        ).to_csv(out_dir / f"{split}_cascade_details.csv", index=False)
        summary["splits"][split] = {
            "stage1_mi_vs_non_mi": stage1_metrics,
            "stage2_stemi_vs_nstemi_on_true_mi": stage2_metrics,
            "final_three_class": final_metrics,
            "routed_to_stage2": int((p_mi >= mi_threshold).sum()),
            "total_records": int(len(p_mi)),
        }

    torch.save(
        {
            "state_dict": stage1.state_dict(),
            "architecture": cfg["architecture"]["stage1"],
            "threshold": mi_threshold,
            "target": "MI_vs_non_MI",
        },
        out_dir / "stage1_mi_vs_non_mi.pt",
    )
    torch.save(
        {
            "state_dict": stage2.state_dict(),
            "architecture": cfg["architecture"]["stage2"],
            "threshold": stemi_threshold,
            "target": "STEMI_proxy_vs_NSTEMI_proxy_given_MI",
            "feature_columns": feature_columns,
            "feature_transform": stage2_transform.to_dict(),
        },
        out_dir / "stage2_stemi_vs_nstemi_morphology_fusion.pt",
    )
    (out_dir / "cascade_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"].get("test", {}), indent=2))


if __name__ == "__main__":
    main()
