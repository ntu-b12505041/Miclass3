from __future__ import annotations

from collections.abc import Iterable

import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONTIGUOUS_GROUPS = (("II", "III", "aVF"), ("I", "aVL", "V5", "V6"), ("V1", "V2", "V3", "V4"))


def stemi_threshold_mv(lead: str, sex: str | None, age: float | None) -> float:
    if lead not in {"V2", "V3"}:
        return 0.1
    if str(sex).lower().startswith("f") or sex in {1, "1"}:
        return 0.15
    return 0.25 if age is not None and age < 40 else 0.20


def standard_stemi_from_jpoints(jpoint_mv: dict[str, float], sex: str | None, age: float | None) -> bool:
    """Require elevation at the J point in two leads within a contiguous group."""
    positive = {lead for lead, value in jpoint_mv.items() if value >= stemi_threshold_mv(lead, sex, age)}
    return any(sum(lead in positive for lead in group) >= 2 for group in CONTIGUOUS_GROUPS)


def modified_sgarbossa_positive(
    st_j_mv: dict[str, float], s_depth_mv: dict[str, float], qrs_polarity: dict[str, int]
) -> bool:
    """Smith-modified Sgarbossa: concordance or discordant STE/S >= 0.25.

    qrs_polarity is +1 for dominant positive QRS and -1 for dominant negative.
    A clinical reviewer must verify LBBB before this rule is used.
    """
    for lead, st in st_j_mv.items():
        polarity = qrs_polarity.get(lead, 0)
        if polarity > 0 and st >= 0.1:
            return True
        if lead in {"V1", "V2", "V3"} and polarity < 0 and st <= -0.1:
            return True
        depth = s_depth_mv.get(lead, 0.0)
        if polarity < 0 and st >= 0.1 and depth > 0 and st / depth >= 0.25:
            return True
    return False


def robust_median(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.median(values)) if values else float("nan")
