from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# The cascade Stage-2 uses five safe global morphology measurements plus
# per-lead ST-J, ST-J60, S-depth and QRS-polarity columns.  The direct trainer's
# `extended` feature profile also considers metadata/timing columns and the
# label-rule field `modified_sgarbossa_positive`; remove those columns here so
# both architectures receive the same morphology evidence for this comparison.
DROP_FEATURE_COLUMNS = (
    "modified_sgarbossa_positive",
    "age",
    "sex",
    "median_heart_rate_bpm",
    "median_rr_ms",
    "num_detected_beats",
    "num_beats_used",
    "median_p_peak_rel_ms",
    "median_q_peak_rel_ms",
    "median_s_peak_rel_ms",
    "median_qrs_onset_rel_ms",
    "median_qrs_offset_rel_ms",
    "median_t_peak_rel_ms",
)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def restore_filename_hr(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Ensure the direct trainer can locate each PTB-XL records500 waveform."""
    if "filename_hr" in frame.columns and frame["filename_hr"].notna().all():
        return frame

    metadata_path = data_dir / "ptbxl_database.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"manifest is missing filename_hr and metadata was not found at {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path, usecols=["ecg_id", "filename_hr"])
    if "filename_hr" in frame.columns:
        frame = frame.drop(columns=["filename_hr"])
    frame = frame.merge(metadata, on="ecg_id", how="left", validate="many_to_one")

    missing = int(frame["filename_hr"].isna().sum())
    if missing:
        raise ValueError(f"{missing} records could not be matched to PTB-XL filename_hr")

    print("Restored filename_hr from ptbxl_database.csv")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a direct-classifier manifest with cascade-matched safe lead-aware morphology inputs."
    )
    parser.add_argument("--input", default="data/label_manifest_custom65_common.csv")
    parser.add_argument("--out", default="data/label_manifest_custom65_common_directmatched.csv")
    parser.add_argument("--data-dir", default="data/ptbxl")
    args = parser.parse_args()

    input_path = project_path(args.input)
    out_path = project_path(args.out)
    data_dir = project_path(args.data_dir)
    frame = pd.read_csv(input_path)

    dropped = [column for column in DROP_FEATURE_COLUMNS if column in frame.columns]
    frame = frame.drop(columns=dropped)
    frame = restore_filename_hr(frame, data_dir)

    required = ["ecg_id", "strat_fold", "label_id", "label_tier", "filename_hr"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required manifest columns: {missing}")

    prefixes = ("st_j_", "st_j60_", "s_depth_", "qrs_polarity_")
    lead_columns = sorted(
        column for column in frame.columns if any(column.startswith(prefix) for prefix in prefixes)
    )
    safe_globals = [
        column
        for column in ("max_st_j_mv", "max_st_j60_mv", "max_st_s_ratio", "qrs_duration_ms", "lbbb")
        if column in frame.columns
    ]
    selected = safe_globals + lead_columns

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Records: {len(frame)}")
    print(f"Dropped comparison-confounding feature columns: {dropped}")
    print(f"Safe morphology columns available to direct model: {len(selected)}")
    print(", ".join(selected))


if __name__ == "__main__":
    main()
