from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miclass3.data import load_metadata
from miclass3.delineation import extract_morphology_features
from miclass3.labels import build_label_manifest


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _parse_folds(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _parse_ecg_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    path = _project_path(value)
    if path.exists():
        return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract P-QRS-T/J-point morphology from PTB-XL records500.")
    parser.add_argument("--data-dir", default="data/ptbxl", help="PTB-XL root containing ptbxl_database.csv and records500/.")
    parser.add_argument("--out", default="data/morphology_features.csv", help="Record-level morphology CSV.")
    parser.add_argument("--beats-out", default="data/morphology_beats.csv", help="Beat-level landmark CSV. Use empty string to skip.")
    parser.add_argument("--label-out", default="data/label_manifest.csv", help="Optional label manifest written after morphology extraction.")
    parser.add_argument("--skip-labels", action="store_true", help="Only write morphology outputs, not label_manifest.csv.")
    parser.add_argument("--folds", help="Comma-separated PTB-XL strat_fold values to process, e.g. 9,10.")
    parser.add_argument("--ecg-ids", help="Comma-separated ECG IDs or a text file with one ECG ID per line.")
    parser.add_argument("--max-records", type=int, help="Debug limit after fold/ID filtering.")
    parser.add_argument("--lbbb-source", choices=["either", "raw", "scp"], default="either")
    parser.add_argument("--resume", action="store_true", help="Skip ECG IDs already present in --out.")
    args = parser.parse_args()

    data_dir = _project_path(args.data_dir)
    meta, scp = load_metadata(data_dir)
    folds = _parse_folds(args.folds)
    if folds is not None:
        meta = meta[meta["strat_fold"].isin(folds)].copy()
    ecg_ids = _parse_ecg_ids(args.ecg_ids)
    if ecg_ids is not None:
        meta = meta[meta["ecg_id"].isin(ecg_ids)].copy()
    if args.max_records:
        meta = meta.head(args.max_records).copy()

    out_path = _project_path(args.out)
    beats_path = _project_path(args.beats_out) if args.beats_out else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if beats_path:
        beats_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[int] = set()
    if args.resume and out_path.exists():
        completed = set(pd.read_csv(out_path, usecols=["ecg_id"])["ecg_id"].astype(int))

    feature_rows: list[dict[str, object]] = []
    beat_rows: list[dict[str, object]] = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="extract morphology"):
        ecg_id = int(row["ecg_id"])
        if ecg_id in completed:
            continue
        try:
            import wfdb

            signal, fields = wfdb.rdsamp(str(data_dir / row["filename_hr"]))
            fs = int(fields.get("fs", 500))
            features, beats = extract_morphology_features(
                signal,
                fs=fs,
                age=None if pd.isna(row.get("age")) else float(row.get("age")),
                sex=row.get("sex"),
                metadata_row=row.to_dict(),
                lbbb_source=args.lbbb_source,
            )
            features.update({"ecg_id": ecg_id, "filename_hr": row["filename_hr"]})
            for beat in beats:
                beat.update({"ecg_id": ecg_id, "filename_hr": row["filename_hr"]})
            feature_rows.append(features)
            beat_rows.extend(beats)
        except Exception as exc:  # keep the run auditable instead of losing the record silently
            feature_rows.append(
                {
                    "ecg_id": ecg_id,
                    "filename_hr": row.get("filename_hr", ""),
                    "standard_stemi": False,
                    "lbbb": False,
                    "lbbb_raw": False,
                    "lbbb_scp": False,
                    "modified_sgarbossa_positive": False,
                    "max_st_j_mv": float("nan"),
                    "max_st_j60_mv": float("nan"),
                    "max_st_s_ratio": float("nan"),
                    "qrs_duration_ms": float("nan"),
                    "num_detected_beats": 0,
                    "num_beats_used": 0,
                    "morphology_quality": "load_error",
                    "morphology_error": f"{type(exc).__name__}: {exc}",
                }
            )

    new_features = pd.DataFrame(feature_rows)
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        features_df = pd.concat([existing, new_features], ignore_index=True)
        features_df = features_df.drop_duplicates("ecg_id", keep="last").sort_values("ecg_id")
    else:
        features_df = new_features.sort_values("ecg_id") if not new_features.empty else new_features
    features_df.to_csv(out_path, index=False)

    if beats_path:
        new_beats = pd.DataFrame(beat_rows)
        if args.resume and beats_path.exists() and not new_beats.empty:
            existing_beats = pd.read_csv(beats_path)
            beats_df = pd.concat([existing_beats, new_beats], ignore_index=True)
            beats_df = beats_df.drop_duplicates(["ecg_id", "beat_index"], keep="last").sort_values(["ecg_id", "beat_index"])
        else:
            beats_df = new_beats.sort_values(["ecg_id", "beat_index"]) if not new_beats.empty else new_beats
        beats_df.to_csv(beats_path, index=False)

    summary = {
        "records_written": int(len(features_df)),
        "new_records_processed": int(len(new_features)),
        "standard_stemi": int(features_df.get("standard_stemi", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "lbbb": int(features_df.get("lbbb", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "modified_sgarbossa_positive": int(features_df.get("modified_sgarbossa_positive", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "low_or_failed_quality": int((features_df.get("morphology_quality", pd.Series(dtype=str)) != "ok").sum()),
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    if beats_path:
        print(f"Wrote {beats_path}")

    if not args.skip_labels and args.label_out:
        manifest = build_label_manifest(meta, scp, features_df)
        label_out = _project_path(args.label_out)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(label_out, index=False)
        print(manifest["label"].value_counts(dropna=False).to_string())
        print(f"Wrote {label_out}")


if __name__ == "__main__":
    main()
