from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from torch.utils.data import Dataset

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3"


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


class PTBXL500Dataset(Dataset):
    """Lazy records500 loader.  PTB-XL's default lead order is retained."""

    def __init__(self, manifest: pd.DataFrame, data_dir: str | Path, feature_columns: list[str] | None = None):
        self.records = manifest.reset_index(drop=True)
        self.root = Path(data_dir)
        self.feature_columns = feature_columns or []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import wfdb
        import torch

        row = self.records.iloc[index]
        signal, _ = wfdb.rdsamp(str(self.root / row["filename_hr"]))
        x = signal.T.astype(np.float32)
        x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
        features = np.nan_to_num(row.reindex(self.feature_columns).to_numpy(dtype=np.float32))
        return torch.from_numpy(x), torch.tensor(int(row["label_id"])), torch.from_numpy(features)

