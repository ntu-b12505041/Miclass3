from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from . import CLASS_TO_ID


@dataclass(frozen=True)
class LabelDecision:
    label: str
    label_id: int | None
    tier: str
    reason: str


def parse_scp_codes(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, str):
        return {str(k): float(v) for k, v in ast.literal_eval(value).items()}
    return {}


def mi_scp_codes(scp: pd.DataFrame) -> set[str]:
    mask = scp["diagnostic_class"].fillna("").str.upper().eq("MI")
    if "diagnostic" in scp:
        mask &= scp["diagnostic"].fillna(0).astype(float).astype(bool)
    return set(scp.index[mask].astype(str))


def _is_old_stage(value: object) -> bool:
    return "old" in str(value).lower()


def _truthy(row: Mapping[str, object], name: str) -> bool:
    value = row.get(name, False)
    if pd.isna(value):
        return False
    return bool(value)


def label_record(
    row: Mapping[str, object], mi_codes: set[str], old_mi_policy: str = "exclude"
) -> LabelDecision:
    """Create the agreed ECG-only proxy label.

    STEMI proxy requires either standard contiguous-lead J-point elevation or a
    positive modified-Sgarbossa assessment for LBBB.  Every other non-old MI
    SCP statement is NSTEMI-proxy by definition; it is not a clinical NSTEMI.
    """
    codes = parse_scp_codes(row.get("scp_codes", {}))
    has_mi = any(code in mi_codes and score > 0 for code, score in codes.items())
    if not has_mi:
        return LabelDecision("non_mi", CLASS_TO_ID["non_mi"], "silver", "no_mi_scp")

    stages = (row.get("infarction_stadium1", ""), row.get("infarction_stadium2", ""), row.get("infarction_stage", ""))
    if any(_is_old_stage(stage) for stage in stages) and old_mi_policy == "exclude":
        return LabelDecision("exclude_old_mi", None, "excluded", "old_infarction_stage")

    standard_stemi = _truthy(row, "standard_stemi")
    lbbb = _truthy(row, "lbbb")
    modified_sgarbossa = _truthy(row, "modified_sgarbossa_positive")
    if lbbb:
        if modified_sgarbossa:
            return LabelDecision("stemi_proxy", CLASS_TO_ID["stemi_proxy"], "silver", "lbbb_modified_sgarbossa")
        return LabelDecision("nstemi_proxy", CLASS_TO_ID["nstemi_proxy"], "silver", "lbbb_without_modified_sgarbossa")
    if standard_stemi:
        return LabelDecision("stemi_proxy", CLASS_TO_ID["stemi_proxy"], "silver", "contiguous_j_point_st_elevation")
    return LabelDecision("nstemi_proxy", CLASS_TO_ID["nstemi_proxy"], "silver", "mi_scp_without_stemi_pattern")


def build_label_manifest(
    metadata: pd.DataFrame, scp: pd.DataFrame, morphology: pd.DataFrame | None = None,
    old_mi_policy: str = "exclude",
) -> pd.DataFrame:
    """Merge morphology evidence and emit an auditable label manifest."""
    table = metadata.copy()
    if morphology is not None:
        if "ecg_id" not in morphology:
            raise ValueError("morphology table must contain ecg_id")
        table = table.merge(morphology, on="ecg_id", how="left", validate="one_to_one")
    for col in ("standard_stemi", "lbbb", "modified_sgarbossa_positive"):
        if col not in table:
            table[col] = False
    codes = mi_scp_codes(scp)
    decisions = [label_record(row, codes, old_mi_policy) for _, row in table.iterrows()]
    table["label"] = [d.label for d in decisions]
    table["label_id"] = [d.label_id for d in decisions]
    table["label_tier"] = [d.tier for d in decisions]
    table["label_reason"] = [d.reason for d in decisions]
    table["mi_scp_codes"] = [
        ";".join(sorted(code for code, score in parse_scp_codes(v).items() if code in codes and score > 0))
        for v in table["scp_codes"]
    ]
    return table
