from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3"

# All features below are available from the input ECG/morphology extractor or
# routine PTB-XL metadata at inference time. In particular, do not add
# `label`, `label_id`, `label_reason`, or SCP diagnostic codes here: those are
# target-side information and would invalidate the held-out evaluation.
CORE_MORPHOLOGY_FEATURES = (
    "max_st_j_mv",
    "max_st_j60_mv",
    "max_st_s_ratio",
    "qrs_duration_ms",
)
LABEL_ROUTING_FEATURES = ("lbbb", "modified_sgarbossa_positive")
EXTENDED_MORPHOLOGY_BASE_FEATURES = (
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
LEAD_MORPHOLOGY_PREFIXES = ("st_j_", "st_j60_", "s_depth_", "qrs_polarity_")


def select_model_features(
    frame: pd.DataFrame,
    profile: str = "core",
    include_label_routing_flags: bool = False,
) -> list[str]:
    """Select auditable, non-target-leaking morphology model inputs.

    ``extended`` adds lead-specific ST/J, S-wave, QRS-polarity, timing, age,
    and sex information. Direct routing flags used to construct the proxy
    label (LBBB and modified-Sgarbossa positivity) are excluded by default;
    they can be restored only for a pre-specified ablation. Label outputs and
    diagnostic SCP statements are always excluded. The same column order is
    used for every fold.
    """
    profile = str(profile).lower()
    if profile not in {"core", "extended"}:
        raise ValueError("feature profile must be 'core' or 'extended'")
    requested = list(CORE_MORPHOLOGY_FEATURES)
    if include_label_routing_flags:
        requested.extend(LABEL_ROUTING_FEATURES)
    if profile == "extended":
        requested.extend(EXTENDED_MORPHOLOGY_BASE_FEATURES)
        requested.extend(
            sorted(
                column
                for column in frame.columns
                if any(column.startswith(prefix) for prefix in LEAD_MORPHOLOGY_PREFIXES)
            )
        )
    # dict.fromkeys preserves deterministic first appearance while removing
    # aliases that may be present in a custom manifest.
    return [column for column in dict.fromkeys(requested) if column in frame.columns]


def normalize_ecg_waveform(signal: np.ndarray, mode: str = "per_lead_zscore") -> np.ndarray:
    """Normalize a samples-by-leads ECG without changing its temporal shape.

    ``per_lead_zscore`` is the historical baseline. ``lead_centered_global_scale``
    uses one robust scale for the entire 12-lead record, preserving amplitude
    relationships between leads that can be informative for ST morphology.
    """
    x = np.asarray(signal, dtype=np.float32).T
    mode = str(mode).lower()
    if mode == "per_lead_zscore":
        return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    if mode == "lead_centered_global_scale":
        centered = x - np.nanmedian(x, axis=-1, keepdims=True)
        scale = float(np.nanmedian(np.nanstd(centered, axis=-1)))
        return np.clip(centered / max(scale, 1e-6), -12.0, 12.0).astype(np.float32)
    raise ValueError("waveform normalization must be 'per_lead_zscore' or 'lead_centered_global_scale'")


@dataclass(frozen=True)
class FeatureTransform:
    """Training-fold-only robust transform for morphology fusion features.

    Continuous morphology values have very different units (mV versus ms), so
    passing their raw values to a shared linear layer makes optimization depend
    unnecessarily on their scale. Binary evidence fields deliberately remain
    0/1. Missingness flags are appended so median imputation cannot silently
    look like a measured value.
    """

    columns: tuple[str, ...]
    impute_values: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    binary_mask: np.ndarray
    clip: float = 6.0
    add_missing_indicators: bool = True

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        columns: Sequence[str],
        clip: float = 6.0,
        add_missing_indicators: bool = True,
    ) -> "FeatureTransform":
        columns = tuple(columns)
        if not columns:
            empty = np.empty(0, dtype=np.float32)
            return cls(columns, empty, empty, empty, np.empty(0, dtype=bool), clip, add_missing_indicators)

        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        impute_values = np.zeros(len(columns), dtype=np.float64)
        centers = np.zeros(len(columns), dtype=np.float64)
        scales = np.ones(len(columns), dtype=np.float64)
        binary_mask = np.zeros(len(columns), dtype=bool)
        for index in range(len(columns)):
            observed = values[np.isfinite(values[:, index]), index]
            if not observed.size:
                continue
            unique = np.unique(observed)
            binary_mask[index] = bool(np.isin(unique, (0.0, 1.0)).all())
            impute_values[index] = float(np.median(observed))
            if binary_mask[index]:
                # Preserve any ablation-only binary inputs as exact 0/1 values.
                continue
            q1, q3 = np.quantile(observed, (0.25, 0.75))
            robust_scale = float((q3 - q1) / 1.349)
            if not np.isfinite(robust_scale) or robust_scale < 1e-6:
                robust_scale = float(np.std(observed))
            scales[index] = robust_scale if np.isfinite(robust_scale) and robust_scale >= 1e-6 else 1.0
            centers[index] = impute_values[index]
        return cls(
            columns=columns,
            impute_values=impute_values.astype(np.float32),
            centers=centers.astype(np.float32),
            scales=scales.astype(np.float32),
            binary_mask=binary_mask,
            clip=float(clip),
            add_missing_indicators=bool(add_missing_indicators),
        )

    @property
    def output_dim(self) -> int:
        return len(self.columns) * (2 if self.add_missing_indicators else 1)

    def transform(self, values: Sequence[object] | np.ndarray) -> np.ndarray:
        values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)
        if values.shape != self.impute_values.shape:
            raise ValueError(f"expected {len(self.columns)} morphology features, got {len(values)}")
        missing = ~np.isfinite(values)
        values = np.where(missing, self.impute_values, values)
        transformed = (values - self.centers) / self.scales
        transformed[self.binary_mask] = values[self.binary_mask]
        transformed = np.clip(transformed, -self.clip, self.clip).astype(np.float32)
        if self.add_missing_indicators:
            transformed = np.concatenate([transformed, missing.astype(np.float32)])
        return transformed

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "impute_values": self.impute_values.tolist(),
            "centers": self.centers.tolist(),
            "scales": self.scales.tolist(),
            "binary_mask": self.binary_mask.astype(bool).tolist(),
            "clip": self.clip,
            "add_missing_indicators": self.add_missing_indicators,
        }


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=(10, 120))
    response.raise_for_status()
    destination.write_bytes(response.content)


def load_metadata(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    for filename in ("ptbxl_database.csv", "scp_statements.csv"):
        download(f"{PTBXL_URL}/{filename}", root / filename)
    meta = pd.read_csv(root / "ptbxl_database.csv")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    return meta, pd.read_csv(root / "scp_statements.csv", index_col=0)


class PTBXL500Dataset:
    """Lazy records500 loader.  PTB-XL's default lead order is retained."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_dir: str | Path,
        feature_columns: list[str] | None = None,
        feature_transform: FeatureTransform | None = None,
        waveform_normalization: str = "per_lead_zscore",
        auxiliary_columns: list[str] | None = None,
    ):
        self.records = manifest.reset_index(drop=True)
        self.root = Path(data_dir)
        self.feature_columns = feature_columns or []
        self.feature_transform = feature_transform or FeatureTransform.fit(
            self.records, self.feature_columns, add_missing_indicators=False
        )
        self.waveform_normalization = waveform_normalization
        self.auxiliary_columns = auxiliary_columns or []
        if tuple(self.feature_columns) != self.feature_transform.columns:
            raise ValueError("feature_columns must match the fitted feature transform")

    @property
    def feature_dim(self) -> int:
        return self.feature_transform.output_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import wfdb
        import torch

        row = self.records.iloc[index]
        signal, _ = wfdb.rdsamp(str(self.root / row["filename_hr"]))
        x = normalize_ecg_waveform(signal, self.waveform_normalization)
        features = self.feature_transform.transform(row.reindex(self.feature_columns).to_numpy())
        batch = (torch.from_numpy(x), torch.tensor(int(row["label_id"])), torch.from_numpy(features))
        if not self.auxiliary_columns:
            return batch
        auxiliary = (
            pd.to_numeric(row.reindex(self.auxiliary_columns), errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        return (*batch, torch.from_numpy(auxiliary))
