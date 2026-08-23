from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def eligible(frame: pd.DataFrame) -> pd.Series:
    required = {"ecg_id", "label", "label_id", "label_tier", "strat_fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    return frame["label_tier"].ne("excluded") & frame["label_id"].notna()


def split_name(fold: int) -> str:
    if fold in range(1, 9):
        return "train"
    if fold == 9:
        return "val"
    if fold == 10:
        return "test"
    return f"fold_{fold}"


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts()
    return {
        "non_mi": int(counts.get("non_mi", 0)),
        "stemi_proxy": int(counts.get("stemi_proxy", 0)),
        "nstemi_proxy": int(counts.get("nstemi_proxy", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched Custom/NeuroKit manifests containing exactly the same eligible ECG IDs. "
            "Each backend keeps its own STEMI/NSTEMI proxy label so delineation-induced label changes remain measurable."
        )
    )
    parser.add_argument("--custom-manifest", default="data/label_manifest_custom65.csv")
    parser.add_argument("--neurokit-manifest", default="data/label_manifest_neurokit.csv")
    parser.add_argument("--custom-out", default="data/label_manifest_custom65_common.csv")
    parser.add_argument("--neurokit-out", default="data/label_manifest_neurokit65_common.csv")
    parser.add_argument("--summary-out", default="data/delineation_common_cohort_summary.json")
    args = parser.parse_args()

    custom = pd.read_csv(project_path(args.custom_manifest), low_memory=False)
    neurokit = pd.read_csv(project_path(args.neurokit_manifest), low_memory=False)

    if custom["ecg_id"].duplicated().any() or neurokit["ecg_id"].duplicated().any():
        raise ValueError("each manifest must contain exactly one row per ecg_id")

    custom_valid = custom.loc[eligible(custom)].copy()
    neurokit_valid = neurokit.loc[eligible(neurokit)].copy()

    common_ids = sorted(set(custom_valid["ecg_id"].astype(int)) & set(neurokit_valid["ecg_id"].astype(int)))
    if not common_ids:
        raise ValueError("no common eligible ECG IDs were found")

    common_set = set(common_ids)
    custom_common = custom_valid[custom_valid["ecg_id"].astype(int).isin(common_set)].copy().sort_values("ecg_id")
    neurokit_common = neurokit_valid[neurokit_valid["ecg_id"].astype(int).isin(common_set)].copy().sort_values("ecg_id")

    custom_ids = custom_common["ecg_id"].astype(int).tolist()
    neurokit_ids = neurokit_common["ecg_id"].astype(int).tolist()
    if custom_ids != neurokit_ids:
        raise RuntimeError("common manifests are not aligned by ecg_id")

    custom_index = custom_common.set_index("ecg_id")
    neurokit_index = neurokit_common.set_index("ecg_id")
    if not custom_index["strat_fold"].astype(int).equals(neurokit_index["strat_fold"].astype(int)):
        raise RuntimeError("strat_fold differs between manifests for at least one common ECG")

    custom_mi = custom_index["label"].ne("non_mi")
    neurokit_mi = neurokit_index["label"].ne("non_mi")
    if not custom_mi.equals(neurokit_mi):
        mismatch = int((custom_mi != neurokit_mi).sum())
        raise RuntimeError(f"MI/non-MI status differs for {mismatch} common ECGs; comparison is not controlled")

    custom_out = project_path(args.custom_out)
    neurokit_out = project_path(args.neurokit_out)
    summary_out = project_path(args.summary_out)
    for path in (custom_out, neurokit_out, summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    custom_common.to_csv(custom_out, index=False)
    neurokit_common.to_csv(neurokit_out, index=False)

    comparison = pd.DataFrame(
        {
            "strat_fold": custom_index["strat_fold"].astype(int),
            "custom_label": custom_index["label"].astype(str),
            "neurokit_label": neurokit_index["label"].astype(str),
        },
        index=custom_index.index,
    )
    comparison["split"] = comparison["strat_fold"].map(split_name)
    comparison["label_changed"] = comparison["custom_label"] != comparison["neurokit_label"]

    split_summary: dict[str, object] = {}
    for name, subset in comparison.groupby("split", sort=False):
        ids = subset.index
        split_summary[name] = {
            "n_records": int(len(ids)),
            "custom_counts": class_counts(custom_index.loc[ids].reset_index()),
            "neurokit_counts": class_counts(neurokit_index.loc[ids].reset_index()),
            "label_changes": int(subset["label_changed"].sum()),
        }

    transition = pd.crosstab(comparison["custom_label"], comparison["neurokit_label"])
    summary = {
        "custom_eligible_records": int(len(custom_valid)),
        "neurokit_eligible_records": int(len(neurokit_valid)),
        "common_records": int(len(common_ids)),
        "custom_only_eligible": int(len(custom_valid) - len(common_ids)),
        "neurokit_only_eligible": int(len(neurokit_valid) - len(common_ids)),
        "label_changes_on_common_cohort": int(comparison["label_changed"].sum()),
        "splits": split_summary,
        "label_transition_matrix": {
            str(row): {str(col): int(transition.loc[row, col]) for col in transition.columns}
            for row in transition.index
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {custom_out}")
    print(f"Wrote {neurokit_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
