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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miclass3.data import PTBXL500Dataset
from miclass3.metrics import multiclass_metrics
from miclass3.models import make_model


def predict(model, loader, device):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for x, y, f in loader:
            out = model(x.to(device), f.to(device))["class_logits"]
            probs.append(torch.softmax(out, 1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


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
    folds = cfg["data"]
    subsets = {"train": manifest[manifest.strat_fold.isin(folds["train_folds"])], "val": manifest[manifest.strat_fold.isin(folds["val_folds"])], "test": manifest[manifest.strat_fold.isin(folds["test_folds"])]}
    data_dir = ROOT / folds["data_dir"]
    loaders = {name: DataLoader(PTBXL500Dataset(frame, data_dir, feature_columns), batch_size=cfg["training"]["batch_size"], shuffle=name == "train", num_workers=cfg["training"]["num_workers"]) for name, frame in subsets.items()}
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = make_model(args.model or cfg["models"]["primary"], len(feature_columns)).to(device)
    counts = subsets["train"].label_id.value_counts().reindex([0, 1, 2], fill_value=1).to_numpy()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    best, best_state, stale = -np.inf, None, 0
    history=[]
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train(); losses=[]
        for x, y, f in loaders["train"]:
            outputs = model(x.to(device), f.to(device)); loss = criterion(outputs["class_logits"], y.to(device))
            if "stemi_logits" in outputs:
                target = (y.to(device) == 1).float(); loss = loss + cfg["training"]["auxiliary_weight"] * nn.functional.binary_cross_entropy_with_logits(outputs["stemi_logits"], target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        yv, pv = predict(model, loaders["val"], device); metrics = multiclass_metrics(yv, pv); metrics.update(epoch=epoch, loss=float(np.mean(losses))); history.append(metrics)
        score = metrics["macro_auprc"]
        print(f"epoch={epoch} loss={metrics['loss']:.4f} val_macro_auprc={score:.4f}")
        if score > best: best, stale, best_state = score, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= cfg["training"]["patience"]: break
    model.load_state_dict(best_state); results={}
    for split, loader in loaders.items():
        y, prob = predict(model, loader, device); results[split] = multiclass_metrics(y, prob)
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_columns": feature_columns, "config": cfg}, out / "best_model.pt")
    (out / "metrics.json").write_text(json.dumps({"history": history, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()

