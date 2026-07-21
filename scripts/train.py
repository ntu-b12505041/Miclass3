from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miclass3.data import FeatureTransform, PTBXL500Dataset, select_model_features
from miclass3.metrics import calibrate_stemi_threshold, multiclass_metrics, write_split_artifacts


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict(model, loader, device):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for x, y, f in loader:
            out = model(x.to(device), f.to(device))["class_logits"]
            probs.append(torch.softmax(out, 1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def normalized_class_weights(counts: np.ndarray, power: float) -> np.ndarray:
    """Return mean-one inverse-frequency weights with a tunable exponent."""
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    weights = np.power(counts, -float(power))
    return weights / weights.mean()


def make_train_loader(dataset, labels: np.ndarray, cfg: dict):
    """Build a shuffled or moderately class-aware training loader.

    ``sqrt`` sampling changes the expected class mix to be proportional to
    sqrt(class count), which is substantially less aggressive than fully
    balanced oversampling and is appropriate for the small STEMI-proxy class.
    """
    training_cfg = cfg["training"]
    strategy = str(training_cfg.get("sampling_strategy", "none")).lower()
    batch_size = int(training_cfg["batch_size"])
    if strategy == "none":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_cfg["num_workers"],
        )
    if strategy not in {"sqrt", "balanced"}:
        raise ValueError("training.sampling_strategy must be none, sqrt, or balanced")
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    power = float(training_cfg.get("sampling_power", 0.5 if strategy == "sqrt" else 1.0))
    if strategy == "balanced":
        power = 1.0
    class_weights = np.power(np.maximum(counts, 1.0), -power)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=training_cfg["num_workers"],
    )


def subsample_majority_class(
    frame: pd.DataFrame, training_cfg: dict, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Cap the large non-MI class for the training split only.

    The cap is expressed relative to ``reference_label`` (NSTEMI-proxy by
    default).  Validation and test rows are never touched, and the sample is
    deterministic for reproducible runs.  This reduces repeated easy
    non-MI examples while preserving the existing square-root sampler for the
    much smaller STEMI-proxy class.
    """
    spec = training_cfg.get("majority_subsample", {}) or {}
    enabled = bool(spec.get("enabled", False))
    majority_label = int(spec.get("majority_label", 0))
    reference_label = int(spec.get("reference_label", 2))
    ratio = float(spec.get("max_ratio", 2.0))
    before = frame["label_id"].value_counts().reindex([0, 1, 2], fill_value=0).astype(int)
    info = {
        "enabled": enabled,
        "majority_label": majority_label,
        "reference_label": reference_label,
        "max_ratio": ratio,
        "before_counts": {str(i): int(before[i]) for i in range(3)},
    }
    if not enabled:
        info["after_counts"] = info["before_counts"].copy()
        info["removed_count"] = 0
        return frame, info
    if ratio <= 0:
        raise ValueError("training.majority_subsample.max_ratio must be positive")
    majority = frame[frame["label_id"] == majority_label]
    reference_count = int((frame["label_id"] == reference_label).sum())
    cap = int(np.ceil(reference_count * ratio))
    if reference_count == 0 or len(majority) <= cap:
        info["cap"] = int(cap)
        info["after_counts"] = info["before_counts"].copy()
        info["removed_count"] = 0
        return frame, info
    keep_majority = majority.sample(n=max(1, cap), random_state=seed)
    keep_other = frame[frame["label_id"] != majority_label]
    sampled = pd.concat([keep_other, keep_majority], axis=0).sample(frac=1.0, random_state=seed)
    after = sampled["label_id"].value_counts().reindex([0, 1, 2], fill_value=0).astype(int)
    info["cap"] = int(cap)
    info["after_counts"] = {str(i): int(after[i]) for i in range(3)}
    info["removed_count"] = int(len(frame) - len(sampled))
    return sampled.reset_index(drop=True), info


def main() -> None:
    global torch, nn, F, DataLoader, WeightedRandomSampler, make_model
    run_start = time.perf_counter()
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    p = argparse.ArgumentParser(description="Train one selected records500 Miclass3 model.")
    p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--manifest", default="data/label_manifest.csv")
    p.add_argument("--model", choices=["morphology_fusion", "inceptiontime", "seresnet"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"]); p.add_argument("--max-records", type=int)
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="Write train/validation artifacts only; use during validation-only hyperparameter tuning.",
    )
    args = p.parse_args(); cfg = yaml.safe_load((ROOT / args.config).read_text())
    folds = cfg["data"]
    data_dir = ROOT / folds["data_dir"]
    records500 = data_dir / "records500"
    if not records500.is_dir():
        raise FileNotFoundError(
            f"Missing raw PTB-XL records500 at {records500}. Download the official data before training."
        )
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from miclass3.models import make_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for training. Install the project's requirements first."
        ) from exc
    seed = int(cfg["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manifest = pd.read_csv(ROOT / args.manifest).query("label_tier != 'excluded' and label_id.notna()", engine="python").copy()
    manifest["label_id"] = manifest["label_id"].astype(int)
    if args.max_records: manifest = manifest.groupby(["strat_fold", "label_id"], group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, args.max_records // 30)), random_state=seed))
    feature_profile = str(cfg["training"].get("feature_profile", "core"))
    feature_columns = select_model_features(manifest, feature_profile)
    lbbb_feature_index = feature_columns.index("lbbb") if "lbbb" in feature_columns else None
    subsets = {
        "train": manifest[manifest.strat_fold.isin(folds["train_folds"])].copy(),
        "val": manifest[manifest.strat_fold.isin(folds["val_folds"])].copy(),
        "test": manifest[manifest.strat_fold.isin(folds["test_folds"])].copy(),
    }
    subsets["train"], subsample_info = subsample_majority_class(
        subsets["train"], cfg["training"], seed
    )
    feature_cfg = cfg["training"].get("feature_transform", {}) or {}
    feature_transform = FeatureTransform.fit(
        subsets["train"],
        feature_columns,
        clip=float(feature_cfg.get("clip", 6.0)),
        add_missing_indicators=bool(feature_cfg.get("add_missing_indicators", True)),
    )
    datasets = {
        name: PTBXL500Dataset(
            frame,
            data_dir,
            feature_columns,
            feature_transform,
            waveform_normalization=str(folds.get("waveform_normalization", "per_lead_zscore")),
        )
        for name, frame in subsets.items()
    }
    loaders = {
        "train": make_train_loader(datasets["train"], subsets["train"]["label_id"].to_numpy(), cfg),
        **{
            name: DataLoader(
                datasets[name],
                batch_size=cfg["training"]["batch_size"],
                shuffle=False,
                num_workers=cfg["training"]["num_workers"],
            )
            for name in ("val", "test")
        },
    }
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model_name = args.model or cfg["models"]["primary"]
    architecture = cfg["models"].get("architecture", {}) or {}
    model = make_model(model_name, datasets["train"].feature_dim, architecture=architecture).to(device)
    counts = subsets["train"].label_id.value_counts().reindex([0, 1, 2], fill_value=0).to_numpy()
    if np.any(counts == 0):
        raise ValueError(f"Every training fold must contain all three classes; counts={counts.tolist()}")
    training_cfg = cfg["training"]
    class_weight_power = float(training_cfg.get("class_weight_power", 1.0))
    class_weights = normalized_class_weights(counts, class_weight_power)
    loss_name = str(training_cfg.get("classification_loss", "cross_entropy")).lower()
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=label_smoothing)
    elif loss_name == "focal":
        focal_gamma = float(training_cfg.get("focal_gamma", 1.5))
        if focal_gamma < 0:
            raise ValueError("training.focal_gamma must be non-negative")

        def criterion(logits, target):
            cross_entropy = F.cross_entropy(
                logits,
                target,
                weight=class_weight_tensor,
                label_smoothing=label_smoothing,
                reduction="none",
            )
            target_probability = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
            return ((1.0 - target_probability).pow(focal_gamma) * cross_entropy).mean()
    else:
        raise ValueError("training.classification_loss must be cross_entropy or focal")
    stemi_positive = int(counts[1]); stemi_negative = int(counts.sum() - stemi_positive)
    sampling_strategy = str(training_cfg.get("sampling_strategy", "none")).lower()
    aux_power = float(training_cfg.get("auxiliary_pos_weight_power", 1.0))
    aux_pos_weight = (stemi_negative / max(stemi_positive, 1)) ** aux_power
    aux_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(aux_pos_weight, dtype=torch.float32, device=device)
    )
    mi_positive = int(counts[1] + counts[2])
    mi_negative = int(counts[0])
    mi_pos_weight = (mi_negative / max(mi_positive, 1)) ** aux_power
    mi_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(mi_pos_weight, dtype=torch.float32, device=device)
    )
    lbbb_criterion = None
    lbbb_pos_weight = None
    if lbbb_feature_index is not None and model_name == "morphology_fusion":
        lbbb_values = pd.to_numeric(subsets["train"]["lbbb"], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)
        lbbb_positive = int(lbbb_values.sum())
        lbbb_negative = int(len(lbbb_values) - lbbb_positive)
        if lbbb_positive and lbbb_negative:
            lbbb_pos_weight = (lbbb_negative / lbbb_positive) ** float(training_cfg.get("auxiliary_pos_weight_power", 0.5))
            lbbb_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(lbbb_pos_weight, dtype=torch.float32, device=device)
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler_cfg = training_cfg.get("lr_scheduler", {}) or {}
    scheduler = None
    if bool(scheduler_cfg.get("enabled", False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 3)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
        )
    best, best_state, stale = -np.inf, None, 0
    history=[]
    fit_start = time.perf_counter()
    fit_started_at_utc = datetime.now(timezone.utc).isoformat()
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        sync_device(device)
        epoch_start = time.perf_counter()
        model.train(); losses=[]
        for x, y, f in loaders["train"]:
            outputs = model(x.to(device), f.to(device)); loss = criterion(outputs["class_logits"], y.to(device))
            if "stemi_logits" in outputs:
                target = (y.to(device) == 1).float()
                loss = loss + training_cfg["auxiliary_weight"] * aux_criterion(outputs["stemi_logits"], target)
            if "mi_logits" in outputs:
                mi_target = (y.to(device) != 0).float()
                loss = loss + training_cfg.get("mi_auxiliary_weight", 0.15) * mi_criterion(outputs["mi_logits"], mi_target)
            if lbbb_criterion is not None and "lbbb_logits" in outputs:
                lbbb_target = f[:, lbbb_feature_index].to(device).clamp(0, 1)
                loss = loss + training_cfg.get("lbbb_auxiliary_weight", 0.0) * lbbb_criterion(outputs["lbbb_logits"], lbbb_target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        sync_device(device)
        train_seconds = time.perf_counter() - epoch_start
        val_start = time.perf_counter()
        yv, pv = predict(model, loaders["val"], device); metrics = multiclass_metrics(yv, pv); metrics.update(epoch=epoch, loss=float(np.mean(losses)))
        sync_device(device)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        metrics.update(
            epoch_seconds=epoch_seconds,
            train_seconds=train_seconds,
            val_seconds=val_seconds,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        history.append(metrics)
        score = metrics["macro_auprc"]
        print(f"epoch={epoch} loss={metrics['loss']:.4f} val_macro_auprc={score:.4f} epoch_seconds={epoch_seconds:.1f}")
        if score > best: best, stale, best_state = score, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if scheduler is not None:
            scheduler.step(score)
        if stale >= cfg["training"]["patience"]: break
    sync_device(device)
    fit_wall_clock_seconds = time.perf_counter() - fit_start
    model.load_state_dict(best_state); results={}
    threshold = None
    threshold_calibration_seconds = None
    if bool(cfg.get("metrics", {}).get("calibrate_stemi_threshold", True)):
        threshold_start = time.perf_counter()
        yv, pv = predict(model, loaders["val"], device)
        threshold_cfg = cfg.get("metrics", {}).get("stemi_threshold", {}) or {}
        threshold, threshold_summary, threshold_candidates = calibrate_stemi_threshold(
            yv,
            pv,
            minimum_stemi_recall=threshold_cfg.get("minimum_stemi_recall"),
        )
        threshold_score = threshold_summary["macro_f1"]
        sync_device(device)
        threshold_calibration_seconds = time.perf_counter() - threshold_start
    else:
        threshold_score, threshold_summary, threshold_candidates = None, None, None
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    if threshold_candidates is not None:
        threshold_candidates.to_csv(out / "val_stemi_threshold_candidates.csv", index=False)
    evaluation_seconds_by_split = {}
    splits_to_evaluate = ("train", "val") if args.skip_test else ("train", "val", "test")
    for split in splits_to_evaluate:
        loader = loaders[split]
        if split == "train":
            # Re-evaluate train deterministically; the training sampler is not
            # used for reporting and should not change the reported support.
            loader = DataLoader(datasets[split], batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])
        eval_start = time.perf_counter()
        y, prob = predict(model, loader, device)
        sync_device(device)
        results[split] = write_split_artifacts(out, split, y, prob, subsets[split], threshold)
        evaluation_seconds_by_split[split] = time.perf_counter() - eval_start
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "feature_transform": feature_transform.to_dict(),
            "config": cfg,
        },
        out / "best_model.pt",
    )
    epoch_times = [float(row["epoch_seconds"]) for row in history]
    total_wall_clock_seconds = time.perf_counter() - run_start
    training_summary = {
        "train_class_counts": {str(i): int(counts[i]) for i in range(3)},
        "feature_profile": feature_profile,
        "feature_columns": feature_columns,
        "waveform_normalization": str(folds.get("waveform_normalization", "per_lead_zscore")),
        "architecture": architecture,
        "classification_loss": loss_name,
        "label_smoothing": label_smoothing,
        "focal_gamma": float(training_cfg.get("focal_gamma", 1.5)) if loss_name == "focal" else None,
        "majority_subsample": subsample_info,
        "class_weights": {str(i): float(class_weights[i]) for i in range(3)},
        "sampling_strategy": sampling_strategy,
        "sampling_power": float(training_cfg.get("sampling_power", 0.5)),
        "auxiliary_stemi_pos_weight": float(aux_pos_weight),
        "mi_auxiliary_weight": float(training_cfg.get("mi_auxiliary_weight", 0.15)),
        "auxiliary_mi_pos_weight": float(mi_pos_weight),
        "auxiliary_pos_weight_power": aux_power,
        "lbbb_auxiliary_weight": float(training_cfg.get("lbbb_auxiliary_weight", 0.0)),
        "lbbb_auxiliary_pos_weight": lbbb_pos_weight,
        "feature_transform": feature_transform.to_dict(),
        "lr_scheduler": {
            "enabled": scheduler is not None,
            "factor": scheduler_cfg.get("factor") if scheduler is not None else None,
            "patience": scheduler_cfg.get("patience") if scheduler is not None else None,
            "min_lr": scheduler_cfg.get("min_lr") if scheduler is not None else None,
        },
        "stemi_threshold": threshold,
        "validation_macro_f1_at_threshold": threshold_score,
        "threshold_calibration": threshold_summary,
        "timing": {
            "run_started_at_utc": run_started_at_utc,
            "fit_started_at_utc": fit_started_at_utc,
            "fit_wall_clock_seconds": float(fit_wall_clock_seconds),
            "total_wall_clock_seconds": float(total_wall_clock_seconds),
            "epochs_completed": int(len(history)),
            "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
            "median_epoch_seconds": float(np.median(epoch_times)) if epoch_times else None,
            "threshold_calibration_seconds": None if threshold_calibration_seconds is None else float(threshold_calibration_seconds),
            "evaluation_seconds_by_split": {name: float(seconds) for name, seconds in evaluation_seconds_by_split.items()},
        },
    }
    (out / "metrics.json").write_text(json.dumps({"history": history, "results": results, "training": training_summary}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
