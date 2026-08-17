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
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.data import PTBXL500Dataset
from miclass3.stage1_v2 import Stage1V2MultiScaleSE


CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")


def normalized_class_weights(counts: np.ndarray, power: float) -> np.ndarray:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    weights = np.power(counts, -float(power))
    return weights / weights.mean()


def ensure_waveform_paths(manifest: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if "filename_hr" in manifest.columns and manifest["filename_hr"].notna().all():
        return manifest
    metadata_path = data_dir / "ptbxl_database.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"manifest is missing filename_hr and metadata was not found at {metadata_path}"
        )
    metadata = pd.read_csv(metadata_path, usecols=["ecg_id", "filename_hr"])
    if "filename_hr" in manifest.columns:
        manifest = manifest.drop(columns=["filename_hr"])
    manifest = manifest.merge(metadata, on="ecg_id", how="left", validate="many_to_one")
    missing = int(manifest["filename_hr"].isna().sum())
    if missing:
        raise ValueError(f"{missing} records could not be matched to PTB-XL filename_hr")
    print("Restored filename_hr from ptbxl_database.csv")
    return manifest


def make_model(cfg: dict, device):
    arch = cfg["architecture"]
    return Stage1V2MultiScaleSE(
        in_channels=12,
        width=int(arch.get("width", 64)),
        kernels=tuple(arch.get("kernels", [11, 25, 49])),
        bottleneck=int(arch.get("bottleneck", 32)),
        block_dropout=float(arch.get("block_dropout", 0.10)),
        head_dropout=float(arch.get("head_dropout", 0.20)),
    ).to(device)


def make_loader(dataset, batch_size: int, workers: int, pin_memory: bool, shuffle: bool):
    from torch.utils.data import DataLoader

    kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle,
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    labels, p_mi, subtype_probabilities = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            output = model(x.to(device, non_blocking=True))
            labels.append(y.numpy())
            p_mi.append(torch.softmax(output["class_logits"], dim=1)[:, 1].cpu().numpy())
            subtype_probabilities.append(torch.softmax(output["subtype_logits"], dim=1).cpu().numpy())
    return (
        np.concatenate(labels).astype(int),
        np.concatenate(p_mi),
        np.concatenate(subtype_probabilities),
    )


def route_metrics(y_three: np.ndarray, p_mi: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_three, dtype=int)
    routed = np.asarray(p_mi, dtype=float) >= float(threshold)

    def fraction(mask: np.ndarray, decision: np.ndarray) -> float:
        return float(decision[mask].mean()) if mask.any() else float("nan")

    non_mi_recall = fraction(y == 0, ~routed)
    stemi_route_recall = fraction(y == 1, routed)
    nstemi_route_recall = fraction(y == 2, routed)
    worst = float(np.nanmin([non_mi_recall, stemi_route_recall, nstemi_route_recall]))
    mean = float(np.nanmean([non_mi_recall, stemi_route_recall, nstemi_route_recall]))

    binary_true = (y != 0).astype(int)
    binary_pred = routed.astype(int)
    cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    binary_sensitivity = float(tp / max(tp + fn, 1))
    binary_specificity = float(tn / max(tn + fp, 1))

    return {
        "threshold": float(threshold),
        "non_mi_recall": non_mi_recall,
        "stemi_route_recall": stemi_route_recall,
        "nstemi_route_recall": nstemi_route_recall,
        "worst_route_recall": worst,
        "mean_route_recall": mean,
        "binary_mi_sensitivity": binary_sensitivity,
        "binary_mi_specificity": binary_specificity,
    }


def choose_route_threshold(
    y_three: np.ndarray,
    p_mi: np.ndarray,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    probabilities = np.asarray(p_mi, dtype=float)
    thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    rows = [route_metrics(y_three, probabilities, threshold) for threshold in thresholds]
    table = pd.DataFrame(rows)
    chosen = table.sort_values(
        [
            "worst_route_recall",
            "mean_route_recall",
            "nstemi_route_recall",
            "stemi_route_recall",
            "threshold",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).iloc[0]
    summary = {key: float(chosen[key]) for key in table.columns}
    return float(chosen["threshold"]), summary, table


def probability_metrics(y_three: np.ndarray, p_mi: np.ndarray) -> dict[str, float]:
    y_binary = (np.asarray(y_three, dtype=int) != 0).astype(int)
    output = {}
    try:
        output["auroc"] = float(roc_auc_score(y_binary, p_mi))
        output["auprc"] = float(average_precision_score(y_binary, p_mi))
    except ValueError:
        output["auroc"] = float("nan")
        output["auprc"] = float("nan")
    return output


def target_flags(summary: dict[str, float], cfg: dict) -> dict[str, bool]:
    selection = cfg.get("selection", {}) or {}
    return {
        "passes_target_worst_route_recall": summary["worst_route_recall"]
        >= float(selection.get("target_worst_route_recall", 0.85)),
        "passes_stretch_non_mi_recall": summary["non_mi_recall"]
        >= float(selection.get("stretch_non_mi_recall", 0.88)),
        "passes_stretch_stemi_route_recall": summary["stemi_route_recall"]
        >= float(selection.get("stretch_stemi_route_recall", 0.92)),
        "passes_stretch_nstemi_route_recall": summary["nstemi_route_recall"]
        >= float(selection.get("stretch_nstemi_route_recall", 0.95)),
    }


def prediction_frame(frame: pd.DataFrame, y: np.ndarray, p_mi: np.ndarray, subtype: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": frame["ecg_id"].to_numpy() if "ecg_id" in frame else np.arange(len(frame)),
            "actual_id": y,
            "actual_label": [CLASS_NAMES[int(value)] for value in y],
            "p_mi": p_mi,
            "p_aux_non_mi": subtype[:, 0],
            "p_aux_stemi_proxy": subtype[:, 1],
            "p_aux_nstemi_proxy": subtype[:, 2],
            "mi_threshold": float(threshold),
            "routed_to_mi": (p_mi >= float(threshold)).astype(int),
        }
    )


def evaluate_split(model, loader, frame, device, threshold: float) -> tuple[dict, pd.DataFrame]:
    y, p_mi, subtype = predict(model, loader, device)
    metrics = route_metrics(y, p_mi, threshold)
    metrics.update(probability_metrics(y, p_mi))
    return metrics, prediction_frame(frame, y, p_mi, subtype, threshold)


def main() -> None:
    import torch
    from torch import nn

    parser = argparse.ArgumentParser(
        description="Train Stage1-v2: 500 Hz multi-scale SE-Inception MI router with NSTEMI-aware auxiliary supervision."
    )
    parser.add_argument("--config", default="configs/stage1_v2_multiscale.yaml")
    parser.add_argument("--manifest", default="data/label_manifest_custom65_common.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not train. Load the saved best checkpoint/validation threshold and evaluate fold 10 once.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
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

    frames = {
        "train": manifest[manifest.strat_fold.isin(data_cfg["train_folds"])].copy(),
        "val": manifest[manifest.strat_fold.isin(data_cfg["val_folds"])].copy(),
        "test": manifest[manifest.strat_fold.isin(data_cfg["test_folds"])].copy(),
    }
    for name, frame in frames.items():
        if set(frame["label_id"].unique()) != {0, 1, 2}:
            raise ValueError(f"{name} split must contain all three proxy classes")

    normalization = str(data_cfg.get("waveform_normalization", "per_lead_zscore"))
    datasets = {
        name: PTBXL500Dataset(frame, data_dir, feature_columns=[], waveform_normalization=normalization)
        for name, frame in frames.items()
    }
    training = cfg["training"]
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(datasets["train"], batch_size, workers, pin_memory, shuffle=True),
        "train_eval": make_loader(datasets["train"], batch_size, workers, pin_memory, shuffle=False),
        "val": make_loader(datasets["val"], batch_size, workers, pin_memory, shuffle=False),
        "test": make_loader(datasets["test"], batch_size, workers, pin_memory, shuffle=False),
    }

    out_dir = ROOT / (args.out_dir or cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_stage1_v2.pt"
    metrics_path = out_dir / "metrics.json"

    model = make_model(cfg, device)

    if args.finalize_only:
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError("finalize-only requires best_stage1_v2.pt and metrics.json from a completed validation run")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(summary["validation"]["threshold"])
        test_metrics, test_predictions = evaluate_split(model, loaders["test"], frames["test"], device, threshold)
        test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)
        summary["test"] = test_metrics
        summary["test"].update(target_flags(test_metrics, cfg))
        summary["test_evaluated"] = True
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["test"], indent=2))
        return

    train_labels = frames["train"]["label_id"].to_numpy(dtype=int)
    binary_counts = np.bincount((train_labels != 0).astype(int), minlength=2)
    primary_weights = normalized_class_weights(binary_counts, float(training.get("primary_class_weight_power", 0.5)))

    subtype_counts = np.bincount(train_labels, minlength=3)
    auxiliary_weights = normalized_class_weights(subtype_counts, float(training.get("auxiliary_class_weight_power", 0.5)))
    auxiliary_weights[2] *= float(training.get("nstemi_aux_weight", 1.0))
    auxiliary_weights /= auxiliary_weights.mean()

    primary_criterion = nn.CrossEntropyLoss(weight=torch.tensor(primary_weights, dtype=torch.float32, device=device))
    auxiliary_criterion = nn.CrossEntropyLoss(weight=torch.tensor(auxiliary_weights, dtype=torch.float32, device=device))
    auxiliary_weight = float(training.get("auxiliary_weight", 0.35))

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
        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            output = model(x)
            primary_target = (y != 0).long()
            primary_loss = primary_criterion(output["class_logits"], primary_target)
            auxiliary_loss = auxiliary_criterion(output["subtype_logits"], y)
            loss = primary_loss + auxiliary_weight * auxiliary_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip_norm", 1.0)))
            optimizer.step()

            losses.append(float(loss.item()))
            primary_losses.append(float(primary_loss.item()))
            auxiliary_losses.append(float(auxiliary_loss.item()))

        y_val, p_val, _ = predict(model, loaders["val"], device)
        threshold, route_summary, _ = choose_route_threshold(y_val, p_val)
        probability = probability_metrics(y_val, p_val)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "primary_loss": float(np.mean(primary_losses)),
            "auxiliary_loss": float(np.mean(auxiliary_losses)),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
            **route_summary,
            **probability,
        }
        history.append(row)
        key = (
            row["worst_route_recall"],
            row["mean_route_recall"],
            row["nstemi_route_recall"],
            row["auprc"],
        )
        print(
            f"epoch={epoch} loss={row['loss']:.4f} val_worst_route={row['worst_route_recall']:.4f} "
            f"nonMI={row['non_mi_recall']:.4f} STEMIroute={row['stemi_route_recall']:.4f} "
            f"NSTEMIroute={row['nstemi_route_recall']:.4f} threshold={threshold:.4f} val_auprc={row['auprc']:.4f}"
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training.get("patience", 10)):
            break

    if best_state is None:
        raise RuntimeError("Stage1-v2 did not produce a checkpoint")
    model.load_state_dict(best_state)

    y_val, p_val, subtype_val = predict(model, loaders["val"], device)
    threshold, val_summary, threshold_candidates = choose_route_threshold(y_val, p_val)
    val_summary.update(probability_metrics(y_val, p_val))
    val_summary.update(target_flags(val_summary, cfg))
    threshold_candidates.to_csv(out_dir / "val_threshold_candidates.csv", index=False)
    prediction_frame(frames["val"], y_val, p_val, subtype_val, threshold).to_csv(out_dir / "val_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    train_metrics, train_predictions = evaluate_split(model, loaders["train_eval"], frames["train"], device, threshold)
    train_predictions.to_csv(out_dir / "train_predictions.csv", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": cfg["architecture"],
            "waveform_normalization": normalization,
            "validation_threshold": threshold,
            "best_epoch": best_epoch,
            "target": "MI_vs_non_MI_with_three_class_proxy_auxiliary",
        },
        checkpoint_path,
    )

    summary = {
        "experiment": "stage1_v2_multiscale_nstemi_aware",
        "device": str(device),
        "manifest": args.manifest,
        "waveform_normalization": normalization,
        "architecture": cfg["architecture"],
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

    print("\nSelected Stage1-v2 validation operating point:")
    print(json.dumps(val_summary, indent=2))
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print("Fold 10 has NOT been evaluated. If validation is accepted, finalize exactly once with --finalize-only.")


if __name__ == "__main__":
    main()
