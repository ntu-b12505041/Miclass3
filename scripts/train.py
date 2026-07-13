from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miclass3.data import PTBXL500Dataset
from miclass3.metrics import classification_tables, multiclass_metrics, write_split_artifacts
from miclass3.models import make_model


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


def calibrate_stemi_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select a STEMI threshold on validation data by macro-F1."""
    best_threshold, best_score = 0.5, -np.inf
    for threshold in np.arange(0.05, 0.96, 0.01):
        score = float(classification_tables(y_true, probabilities, float(threshold))[0]["macro_f1"])
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def main() -> None:
    p = argparse.ArgumentParser(description="Train one selected records500 Miclass3 model.")
    p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--manifest", default="data/label_manifest.csv")
    p.add_argument("--model", choices=["morphology_fusion", "inceptiontime", "seresnet"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"]); p.add_argument("--max-records", type=int)
    p.add_argument("--out-dir", default="artifacts")
    args = p.parse_args(); cfg = yaml.safe_load((ROOT / args.config).read_text())
    seed = int(cfg["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manifest = pd.read_csv(ROOT / args.manifest).query("label_tier != 'excluded' and label_id.notna()", engine="python").copy()
    manifest["label_id"] = manifest["label_id"].astype(int)
    if args.max_records: manifest = manifest.groupby(["strat_fold", "label_id"], group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, args.max_records // 30)), random_state=seed))
    feature_columns = [c for c in ("max_st_j_mv", "max_st_j60_mv", "max_st_s_ratio", "qrs_duration_ms", "lbbb", "modified_sgarbossa_positive") if c in manifest]
    lbbb_feature_index = feature_columns.index("lbbb") if "lbbb" in feature_columns else None
    folds = cfg["data"]
    subsets = {"train": manifest[manifest.strat_fold.isin(folds["train_folds"])], "val": manifest[manifest.strat_fold.isin(folds["val_folds"])], "test": manifest[manifest.strat_fold.isin(folds["test_folds"])]}
    data_dir = ROOT / folds["data_dir"]
    datasets = {name: PTBXL500Dataset(frame, data_dir, feature_columns) for name, frame in subsets.items()}
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
    model = make_model(args.model or cfg["models"]["primary"], len(feature_columns)).to(device)
    counts = subsets["train"].label_id.value_counts().reindex([0, 1, 2], fill_value=0).to_numpy()
    if np.any(counts == 0):
        raise ValueError(f"Every training fold must contain all three classes; counts={counts.tolist()}")
    training_cfg = cfg["training"]
    class_weight_power = float(training_cfg.get("class_weight_power", 1.0))
    class_weights = normalized_class_weights(counts, class_weight_power)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    stemi_positive = int(counts[1]); stemi_negative = int(counts.sum() - stemi_positive)
    sampling_strategy = str(training_cfg.get("sampling_strategy", "none")).lower()
    aux_power = float(training_cfg.get("auxiliary_pos_weight_power", 1.0))
    aux_pos_weight = (stemi_negative / max(stemi_positive, 1)) ** aux_power
    aux_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(aux_pos_weight, dtype=torch.float32, device=device)
    )
    lbbb_criterion = None
    lbbb_pos_weight = None
    if lbbb_feature_index is not None and args.model != "seresnet" and (args.model or cfg["models"]["primary"]) == "morphology_fusion":
        lbbb_values = pd.to_numeric(subsets["train"]["lbbb"], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)
        lbbb_positive = int(lbbb_values.sum())
        lbbb_negative = int(len(lbbb_values) - lbbb_positive)
        if lbbb_positive and lbbb_negative:
            lbbb_pos_weight = (lbbb_negative / lbbb_positive) ** float(training_cfg.get("auxiliary_pos_weight_power", 0.5))
            lbbb_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(lbbb_pos_weight, dtype=torch.float32, device=device)
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    best, best_state, stale = -np.inf, None, 0
    history=[]
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train(); losses=[]
        for x, y, f in loaders["train"]:
            outputs = model(x.to(device), f.to(device)); loss = criterion(outputs["class_logits"], y.to(device))
            if "stemi_logits" in outputs:
                target = (y.to(device) == 1).float()
                loss = loss + training_cfg["auxiliary_weight"] * aux_criterion(outputs["stemi_logits"], target)
            if lbbb_criterion is not None and "lbbb_logits" in outputs:
                lbbb_target = f[:, lbbb_feature_index].to(device).clamp(0, 1)
                loss = loss + training_cfg.get("lbbb_auxiliary_weight", 0.0) * lbbb_criterion(outputs["lbbb_logits"], lbbb_target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        yv, pv = predict(model, loaders["val"], device); metrics = multiclass_metrics(yv, pv); metrics.update(epoch=epoch, loss=float(np.mean(losses))); history.append(metrics)
        score = metrics["macro_auprc"]
        print(f"epoch={epoch} loss={metrics['loss']:.4f} val_macro_auprc={score:.4f}")
        if score > best: best, stale, best_state = score, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= cfg["training"]["patience"]: break
    model.load_state_dict(best_state); results={}
    threshold = None
    if bool(cfg.get("metrics", {}).get("calibrate_stemi_threshold", True)):
        yv, pv = predict(model, loaders["val"], device)
        threshold, threshold_score = calibrate_stemi_threshold(yv, pv)
    else:
        threshold_score = None
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    for split, loader in loaders.items():
        if split == "train":
            # Re-evaluate train deterministically; the training sampler is not
            # used for reporting and should not change the reported support.
            loader = DataLoader(datasets[split], batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])
        y, prob = predict(model, loader, device)
        results[split] = write_split_artifacts(out, split, y, prob, subsets[split], threshold)
    torch.save({"state_dict": model.state_dict(), "feature_columns": feature_columns, "config": cfg}, out / "best_model.pt")
    training_summary = {
        "train_class_counts": {str(i): int(counts[i]) for i in range(3)},
        "class_weights": {str(i): float(class_weights[i]) for i in range(3)},
        "sampling_strategy": sampling_strategy,
        "sampling_power": float(training_cfg.get("sampling_power", 0.5)),
        "auxiliary_stemi_pos_weight": float(aux_pos_weight),
        "auxiliary_pos_weight_power": aux_power,
        "lbbb_auxiliary_weight": float(training_cfg.get("lbbb_auxiliary_weight", 0.0)),
        "lbbb_auxiliary_pos_weight": lbbb_pos_weight,
        "stemi_threshold": threshold,
        "validation_macro_f1_at_threshold": threshold_score,
    }
    (out / "metrics.json").write_text(json.dumps({"history": history, "results": results, "training": training_summary}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
