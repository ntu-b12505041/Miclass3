from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miclass3.data import load_metadata
from miclass3.labels import build_label_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable PTB-XL three-class proxy labels.")
    parser.add_argument("--data-dir", default="data/ptbxl")
    parser.add_argument("--morphology-csv", help="Optional per-ECG J-point/LBBB evidence, keyed by ecg_id.")
    parser.add_argument("--out", default="data/label_manifest.csv")
    args = parser.parse_args()
    meta, scp = load_metadata(ROOT / args.data_dir)
    morphology = pd.read_csv(args.morphology_csv) if args.morphology_csv else None
    manifest = build_label_manifest(meta, scp, morphology)
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(manifest["label"].value_counts(dropna=False).to_string())
    print(f"Wrote {output}")


if __name__ == "__main__": main()

