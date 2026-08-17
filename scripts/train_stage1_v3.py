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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from miclass3.data import FeatureTransform, PTBXL500Dataset
from miclass3.stage1_v3 import Stage1V3MorphologyFusion
from train_stage1_v2 import (
    choose_route_threshold,
    ensure_waveform_paths,
    make_loader,
    normalized_class_weights,
    prediction_frame,
    probability_metrics,
    route_metrics,
    target_flags,
)

CORE_FEATURES = (
    "max_st_j_mv",
    "max_st_j60_mv",
    "max_st_s_ratio",
    "qrs_duration_ms",
    "lbbb",
)
LEAD_PREFIXES = ("st_j_", "st_j60_", "s_depth_", "qrs_polarity_")
FORBIDDEN_FEATURES = {
    "label",
    "label_id",
    "label_reason",
    "label_tier",
    "standard_stemi",
    "modified_sgarbossa_positive",
    "mi_scp_codes",
    "scp_codes",
}


def select_features(frame: pd.DataFrame, profile: str) -> list[str]:
    profile = str(profile).lower()
    if profile not in {"core", "lead_aware"}:
        raise ValueError("features.profile must be core or lead_aware")
    selected = [name for name in CORE_FEATURES if name in frame.columns]
    if profile == "lead_aware":
        selected.extend(
            sorted(
                name
                for name in frame.columns
                if name not in FORBIDDEN_FEATURES
                and any(name.startswith(prefix) for prefix in LEAD_PREFIXES)
            )
        )
    return list(dict.fromkeys(selected))


def predict(model, loader, device):
    import torch

    model.eval()
    labels, p_mi, subtype_probabilities = [], [], []
    with torch.no_grad():
        for x, y, features in loader:
            output = model(
                x.to(device, non_blocking=True),
                features.to(device, non_blocking=True),
            )
            labels.append(y.numpy())
            p_mi.append(torch.softmax(output["class_logits"], dim=1)[:, 1].cpu().numpy())
            subtype_probabilities.append(
                torch.softmax(output["subtype_logits"], dim=1).cpu().numpy()
            )
    return (
        np.concatenate(labels).astype(int),
        np.concatenate(p_mi),
        np.concatenate(subtype_probabilities),
    )


def evaluate_split(model, loader, frame, device, threshold: float):
    y, p_mi, subtype = predict(model, loader, device)
    metrics = route_metrics(y, p_mi, threshold)
    metrics.update(probability_metrics(y, p_mi))
    return metrics, prediction_frame(frame, y, p_mi, subtype, threshold)


def main() -> None:
    import torch
    from torch import nn

    parser = argparse.ArgumentParser(
        description="Train Stage1-v3: full-resolution SE-ResNet plus safe lead-aware morphology fusion."
    )
    parser.add_argument("--config", default="configs/stage1_v3_morphology_fusion.yaml")
    parser.add_argument("--manifest", default="data/label_manifest_custom65_common.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    device = torch.device(device_name)

    data_cfg = cfg["data"]
    data_dir = ROOT / data_cfg["data_dir"]
    if not (data_dir / "records500").is_dir():
        raise FileNotFoundError(f"Missing PTB-XL records500 under {data_dir / 'records500'}")

    manifest = pd.read_csv(ROOT / args.manifest).query(
        "label_tier != 'excluded' and label_id.notna()", engine="python"
    ).copy()
    manifest["label_id"] = manifest["label_id"].astype(int)
    manifest = ensure_waveform_paths(manifest, data_dir)

    frames = {
        "train": manifest[manifest.strat_fold.isin(data_cfg["train_folds"])].copy(),
        "val": manifest[manifest.strat_fold.isin(data_cfg["val_folds"])].copy(),
        "test": manifest[manifest.strat_fold.isin(data_cfg["test_folds"])].copy(),
    }
    for name, frame in frames.items():
        if set(frame["label_id"].unique()) != {0, 1, 2}:
            raise ValueError(f"{name} split must contain all three proxy classes")

    feature_cfg = cfg.get("features", {}) or {}
    feature_columns = select_features(manifest, feature_cfg.get("profile", "lead_aware"))
    if not feature_columns:
        raise ValueError("Stage1-v3 found no safe morphology features")
    feature_transform = FeatureTransform.fit(
        frames["train"],
        feature_columns,
        clip=float(feature_cfg.get("clip", 6.0)),
        add_missing_indicators=bool(feature_cfg.get("add_missing_indicators", True)),
    )
    print(f"Stage1-v3 safe morphology columns: {len(feature_columns)}")
    print(", ".join(feature_columns))

    normalization = str(data_cfg.get("waveform_normalization", "per_lead_zscore"))
    datasets = {
        name: PTBXL500Dataset(
            frame,
            data_dir,
            feature_columns=feature_columns,
            feature_transform=feature_transform,
            waveform_normalization=normalization,
        )
        for name, frame in frames.items()
    }

    training = cfg["training"]
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(datasets["train"], batch_size, workers, pin_memory, True),
        "train_eval": make_loader(datasets["train"], batch_size, workers, pin_memory, False),
        "val": make_loader(datasets["val"], batch_size, workers, pin_memory, False),
        "test": make_loader(datasets["test"], batch_size, workers, pin_memory, False),
    }

    arch = cfg["architecture"]
    model = Stage1V3MorphologyFusion(
        in_channels=12,
        feature_dim=feature_transform.output_dim,
        width=int(arch.get("width", 64)),
        block_dropout=float(arch.get("block_dropout", 0.15)),
        fusion_dropout=float(arch.get("fusion_dropout", 0.35)),
    ).to(device)

    out_dir = ROOT / (args.out_dir or cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_stage1_v3.pt"
    metrics_path = out_dir / "metrics.json"

    if args.finalize_only:
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError("finalize-only requires a completed validation run")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(summary["validation"]["threshold"])
        test_metrics, test_predictions = evaluate_split(
            model, loaders["test"], frames["test"], device, threshold
        )
        test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)
        summary["test"] = test_metrics
        summary["test"].update(target_flags(test_metrics, cfg))
        summary["test_evaluated"] = True
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["test"], indent=2))
        return

    train_labels = frames["train"]["label_id"].to_numpy(dtype=int)
    binary_counts = np.bincount((train_labels != 0).astype(int), minlength=2)
    primary_weights = normalized_class_weights(
        binary_counts, float(training.get("primary_class_weight_power", 0.5))
    )
    subtype_counts = np.bincount(train_labels, minlength=3)
    auxiliary_weights = normalized_class_weights(
        subtype_counts, float(training.get("auxiliary_class_weight_power", 0.5))
    )
    auxiliary_weights[2] *= float(training.get("nstemi_aux_weight", 1.0))
    auxiliary_weights /= auxiliary_weights.mean()

    primary_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(primary_weights, dtype=torch.float32, device=device),
        label_smoothing=float(training.get("primary_label_smoothing", 0.0)),
    )
    auxiliary_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(auxiliary_weights, dtype=torch.float32, device=device),
        label_smoothing=float(training.get("auxiliary_label_smoothing", 0.0)),
    )
    auxiliary_weight = float(training.get("auxiliary_weight", 0.20))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    best_key = None
    best_state = None
    best_epoch = None
    stale = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        losses, primary_losses, auxiliary_losses = [], [], []
        epoch_start = time.perf_counter()
        for x, y, features in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            output = model(x, features)
            primary_loss = primary_criterion(output["class_logits"], (y != 0).long())
            auxiliary_loss = auxiliary_criterion(output["subtype_logits"], y)
            loss = primary_loss + auxiliary_weight * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip_norm", 1.0))
            )
            optimizer.step()
            losses.append(float(loss.item()))
            primary_losses.append(float(primary_loss.item()))
            auxiliary_losses.append(float(auxiliary_loss.item()))

        y_val, p_val, _ = predict(model, loaders["val"], device)
        threshold, route_summary, _ = choose_route_threshold(y_val, p_val)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "primary_loss": float(np.mean(primary_losses)),
            "auxiliary_loss": float(np.mean(auxiliary_losses)),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
            **route_summary,
            **probability_metrics(y_val, p_val),
        }
        history.append(row)
        key = (
            row["worst_route_recall"],
            row["mean_route_recall"],
            row["nstemi_route_recall"],
            row["auprc"],
        )
        print(
            f"epoch={epoch} loss={row['loss']:.4f} "
            f"val_worst_route={row['worst_route_recall']:.4f} "
            f"nonMI={row['non_mi_recall']:.4f} "
            f"STEMIroute={row['stemi_route_recall']:.4f} "
            f"NSTEMIroute={row['nstemi_route_recall']:.4f} "
            f"threshold={threshold:.4f} val_auprc={row['auprc']:.4f}"
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training.get("patience", 8)):
            break

    if best_state is None:
        raise RuntimeError("Stage1-v3 did not produce a checkpoint")
    model.load_state_dict(best_state)

    y_val, p_val, subtype_val = predict(model, loaders["val"], device)
    threshold, val_summary, threshold_candidates = choose_route_threshold(y_val, p_val)
    val_summary.update(probability_metrics(y_val, p_val))
    val_summary.update(target_flags(val_summary, cfg))
    threshold_candidates.to_csv(out_dir / "val_threshold_candidates.csv", index=False)
    prediction_frame(frames["val"], y_val, p_val, subtype_val, threshold).to_csv(
        out_dir / "val_predictions.csv", index=False
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    train_metrics, train_predictions = evaluate_split(
        model, loaders["train_eval"], frames["train"], device, threshold
    )
    train_predictions.to_csv(out_dir / "train_predictions.csv", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": cfg["architecture"],
            "feature_columns": feature_columns,
            "feature_transform": feature_transform.to_dict(),
            "waveform_normalization": normalization,
            "validation_threshold": threshold,
            "best_epoch": best_epoch,
            "target": "MI_vs_non_MI_with_safe_morphology_and_three_class_proxy_auxiliary",
        },
        checkpoint_path,
    )

    summary = {
        "experiment": "stage1_v3_fullres_morphology_fusion",
        "device": str(device),
        "manifest": args.manifest,
        "waveform_normalization": normalization,
        "architecture": cfg["architecture"],
        "feature_profile": feature_cfg.get("profile", "lead_aware"),
        "feature_columns": feature_columns,
        "feature_transform": feature_transform.to_dict(),
        "selection": cfg.get("selection", {}),
        "training": {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "fit_seconds": float(time.perf_counter() - training_start),
            "binary_class_counts": binary_counts.tolist(),
            "primary_class_weights": primary_weights.tolist(),
            "subtype_class_counts": subtype_counts.tolist(),
            "auxiliary_class_weights": auxiliary_weights.tolist(),
            "auxiliary_weight": auxiliary_weight,
        },
        "train": train_metrics,
        "validation": val_summary,
        "test_evaluated": False,
        "test": None,
        "note": "Fold 10 is intentionally untouched until --finalize-only is run after validation selection.",
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSelected Stage1-v3 validation operating point:")
    print(json.dumps(val_summary, indent=2))
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print("Fold 10 has NOT been evaluated. Only finalize if validation is accepted.")


if __name__ == "__main__":
    main()
