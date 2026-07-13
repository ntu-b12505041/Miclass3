from __future__ import annotations

import ast
from typing import Mapping

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .morphology import LEADS, modified_sgarbossa_positive, standard_stemi_from_jpoints


LEAD_INDEX = {lead: index for index, lead in enumerate(LEADS)}
LATERAL_LEADS = ("I", "aVL", "V5", "V6")


def _safe_filter(signal: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = fs / 2
    high_hz = min(high_hz, nyquist - 1)
    if low_hz <= 0 or high_hz <= low_hz or signal.shape[0] < fs:
        return signal.copy()
    sos = butter(3, (low_hz / nyquist, high_hz / nyquist), btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _window_median(signal: np.ndarray, start: int, stop: int) -> np.ndarray | None:
    start = max(0, int(start))
    stop = min(signal.shape[0], int(stop))
    if stop <= start:
        return None
    return np.nanmedian(signal[start:stop], axis=0)


def _lead_window_peak(signal: np.ndarray, lead_index: int, start: int, stop: int, baseline: float) -> int | None:
    start = max(0, int(start))
    stop = min(signal.shape[0], int(stop))
    if stop <= start:
        return None
    segment = signal[start:stop, lead_index] - baseline
    if segment.size == 0 or np.all(~np.isfinite(segment)):
        return None
    return start + int(np.nanargmax(np.abs(segment)))


def _lead_window_minimum(signal: np.ndarray, lead_index: int, start: int, stop: int, baseline: float) -> int | None:
    start = max(0, int(start))
    stop = min(signal.shape[0], int(stop))
    if stop <= start:
        return None
    segment = signal[start:stop, lead_index] - baseline
    if segment.size == 0 or np.all(~np.isfinite(segment)):
        return None
    return start + int(np.nanargmin(segment))


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Detect R peaks using a multi-lead QRS energy envelope.

    This is intentionally dependency-light: PTB-XL records are 10-second,
    12-lead WFDB traces, so a vector envelope across all leads is stable
    enough for morphology extraction and easy to audit.
    """
    qrs_band = _safe_filter(signal, fs, 5.0, 25.0)
    energy = np.sqrt(np.nanmean(np.square(qrs_band), axis=1))
    envelope = _moving_average(energy, int(0.08 * fs))
    finite = envelope[np.isfinite(envelope)]
    if finite.size == 0:
        return np.array([], dtype=int)
    mad = np.median(np.abs(finite - np.median(finite)))
    threshold = max(float(np.percentile(finite, 82)), float(np.median(finite) + 2.5 * mad))
    peaks, _ = find_peaks(envelope, height=threshold, distance=int(0.28 * fs), prominence=max(mad, 1e-6))
    if peaks.size == 0:
        peaks, _ = find_peaks(envelope, height=np.percentile(finite, 90), distance=int(0.28 * fs))
    refined: list[int] = []
    for peak in peaks:
        start = max(0, peak - int(0.08 * fs))
        stop = min(signal.shape[0], peak + int(0.08 * fs) + 1)
        local = energy[start:stop]
        if local.size:
            refined.append(start + int(np.nanargmax(local)))
    return np.array(sorted(set(refined)), dtype=int)


def locate_qrs_bounds(signal: np.ndarray, fs: int, r_peak: int) -> tuple[int, int]:
    qrs_band = _safe_filter(signal, fs, 5.0, 35.0)
    energy = np.sqrt(np.nanmean(np.square(qrs_band), axis=1))
    envelope = _moving_average(energy, int(0.025 * fs))
    left = max(0, r_peak - int(0.16 * fs))
    right = min(signal.shape[0] - 1, r_peak + int(0.20 * fs))
    local = envelope[left : right + 1]
    if local.size == 0:
        return max(0, r_peak - int(0.05 * fs)), min(signal.shape[0] - 1, r_peak + int(0.06 * fs))
    noise = float(np.nanpercentile(local, 20))
    peak = float(envelope[r_peak])
    threshold = noise + 0.12 * max(peak - noise, 1e-9)
    hold = max(2, int(0.012 * fs))

    onset = left
    for index in range(r_peak, left + hold, -1):
        if np.all(envelope[index - hold : index] <= threshold):
            onset = index
            break

    offset = right
    for index in range(r_peak, right - hold):
        if np.all(envelope[index : index + hold] <= threshold):
            offset = index
            break
    return int(onset), int(offset)


def _baseline_for_beat(signal: np.ndarray, fs: int, qrs_onset: int, r_peak: int) -> tuple[np.ndarray, str]:
    pr = _window_median(signal, qrs_onset - int(0.08 * fs), qrs_onset - int(0.02 * fs))
    if pr is not None:
        return pr, "PR"
    tp = _window_median(signal, r_peak - int(0.32 * fs), r_peak - int(0.22 * fs))
    if tp is not None:
        return tp, "TP"
    fallback = _window_median(signal, max(0, r_peak - int(0.20 * fs)), r_peak)
    if fallback is not None:
        return fallback, "preQRS"
    return np.zeros(signal.shape[1], dtype=float), "zero"


def _polarity_and_s_depth(signal: np.ndarray, onset: int, offset: int, baseline: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    polarity: dict[str, int] = {}
    s_depth: dict[str, float] = {}
    stop = min(signal.shape[0], offset + 1)
    segment = signal[max(0, onset) : stop] - baseline
    for lead in LEADS:
        values = segment[:, LEAD_INDEX[lead]]
        if values.size == 0:
            polarity[lead] = 0
            s_depth[lead] = 0.0
            continue
        high = float(np.nanmax(values))
        low = float(np.nanmin(values))
        polarity[lead] = 1 if abs(high) >= abs(low) else -1
        s_depth[lead] = max(0.0, -low)
    return polarity, s_depth


def _dominant_polarity(values: list[int]) -> int:
    values = [value for value in values if value != 0]
    if not values:
        return 0
    return 1 if sum(value > 0 for value in values) >= sum(value < 0 for value in values) else -1


def _metadata_has_lbbb(row: Mapping[str, object] | None) -> bool:
    if row is None:
        return False
    codes = row.get("scp_codes", {})
    if isinstance(codes, str):
        try:
            codes = ast.literal_eval(codes)
        except (SyntaxError, ValueError):
            codes = {}
    if not isinstance(codes, Mapping):
        return False
    return any("LBBB" in str(code).upper() and float(score) > 0 for code, score in codes.items())


def _raw_lbbb(qrs_duration_ms: float, polarities: dict[str, int]) -> bool:
    if not np.isfinite(qrs_duration_ms) or qrs_duration_ms < 120:
        return False
    v1_negative = polarities.get("V1", 0) < 0
    lateral_positive = sum(polarities.get(lead, 0) > 0 for lead in LATERAL_LEADS) >= 2
    return bool(v1_negative and lateral_positive)


def extract_morphology_features(
    signal: np.ndarray,
    fs: int,
    age: float | None = None,
    sex: str | int | None = None,
    metadata_row: Mapping[str, object] | None = None,
    lbbb_source: str = "either",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Extract ECG morphology features used by the MI proxy labeler.

    Returns one record-level feature dict plus beat-level landmarks. Signal is
    expected to be WFDB physical units in millivolts with shape
    ``(samples, 12)`` in the PTB-XL lead order.
    """
    if signal.ndim != 2 or signal.shape[1] != len(LEADS):
        raise ValueError(f"expected signal shape (samples, {len(LEADS)}), got {signal.shape}")

    r_peaks = detect_r_peaks(signal, fs)
    per_beat_rows: list[dict[str, object]] = []
    st_j_by_lead = {lead: [] for lead in LEADS}
    st_j60_by_lead = {lead: [] for lead in LEADS}
    s_depth_by_lead = {lead: [] for lead in LEADS}
    polarity_by_lead = {lead: [] for lead in LEADS}
    qrs_durations: list[float] = []
    rr_ms: list[float] = []

    if len(r_peaks) >= 2:
        rr_ms = (np.diff(r_peaks) / fs * 1000.0).tolist()

    for beat_index, r_peak in enumerate(r_peaks):
        if r_peak < int(0.32 * fs) or r_peak > signal.shape[0] - int(0.50 * fs):
            continue
        onset, offset = locate_qrs_bounds(signal, fs, int(r_peak))
        if offset <= onset or offset - onset > int(0.22 * fs):
            continue
        baseline, baseline_source = _baseline_for_beat(signal, fs, onset, int(r_peak))
        j_point = offset
        j60 = min(signal.shape[0] - 1, j_point + int(0.06 * fs))
        st_j = _window_median(signal, j_point - int(0.01 * fs), j_point + int(0.01 * fs) + 1)
        st_j60 = _window_median(signal, j60 - int(0.01 * fs), j60 + int(0.01 * fs) + 1)
        if st_j is None or st_j60 is None:
            continue
        st_j = st_j - baseline
        st_j60 = st_j60 - baseline
        polarity, s_depth = _polarity_and_s_depth(signal, onset, offset, baseline)
        q_peak = _lead_window_minimum(signal, LEAD_INDEX["II"], onset, int(r_peak), baseline[LEAD_INDEX["II"]])
        s_peak = _lead_window_minimum(signal, LEAD_INDEX["II"], int(r_peak), offset, baseline[LEAD_INDEX["II"]])
        p_peak = _lead_window_peak(signal, LEAD_INDEX["II"], onset - int(0.22 * fs), onset - int(0.06 * fs), baseline[LEAD_INDEX["II"]])
        t_peak = _lead_window_peak(signal, LEAD_INDEX["II"], offset + int(0.08 * fs), offset + int(0.42 * fs), baseline[LEAD_INDEX["II"]])
        qrs_duration_ms = (offset - onset) / fs * 1000.0
        qrs_durations.append(qrs_duration_ms)
        for lead in LEADS:
            idx = LEAD_INDEX[lead]
            st_j_by_lead[lead].append(float(st_j[idx]))
            st_j60_by_lead[lead].append(float(st_j60[idx]))
            s_depth_by_lead[lead].append(float(s_depth[lead]))
            polarity_by_lead[lead].append(int(polarity[lead]))
        per_beat_rows.append(
            {
                "beat_index": beat_index,
                "r_sample": int(r_peak),
                "q_peak_sample": q_peak,
                "s_peak_sample": s_peak,
                "qrs_onset_sample": int(onset),
                "qrs_offset_sample": int(offset),
                "j_point_sample": int(j_point),
                "p_peak_sample": p_peak,
                "t_peak_sample": t_peak,
                "qrs_duration_ms": qrs_duration_ms,
                "p_peak_rel_ms": None if p_peak is None else (p_peak - r_peak) / fs * 1000.0,
                "q_peak_rel_ms": None if q_peak is None else (q_peak - r_peak) / fs * 1000.0,
                "s_peak_rel_ms": None if s_peak is None else (s_peak - r_peak) / fs * 1000.0,
                "qrs_onset_rel_ms": (onset - r_peak) / fs * 1000.0,
                "qrs_offset_rel_ms": (offset - r_peak) / fs * 1000.0,
                "j_point_rel_ms": (j_point - r_peak) / fs * 1000.0,
                "t_peak_rel_ms": None if t_peak is None else (t_peak - r_peak) / fs * 1000.0,
                "baseline_source": baseline_source,
            }
        )

    median_st_j = {
        lead: float(np.nanmedian(values)) if values else float("nan")
        for lead, values in st_j_by_lead.items()
    }
    median_st_j60 = {
        lead: float(np.nanmedian(values)) if values else float("nan")
        for lead, values in st_j60_by_lead.items()
    }
    median_s_depth = {
        lead: float(np.nanmedian(values)) if values else 0.0
        for lead, values in s_depth_by_lead.items()
    }
    median_polarity = {
        lead: _dominant_polarity(values)
        for lead, values in polarity_by_lead.items()
    }
    qrs_duration_ms = float(np.nanmedian(qrs_durations)) if qrs_durations else float("nan")
    raw_lbbb = _raw_lbbb(qrs_duration_ms, median_polarity)
    scp_lbbb = _metadata_has_lbbb(metadata_row)
    if lbbb_source == "raw":
        lbbb = raw_lbbb
    elif lbbb_source == "scp":
        lbbb = scp_lbbb
    else:
        lbbb = raw_lbbb or scp_lbbb
    standard_stemi = False if lbbb else standard_stemi_from_jpoints(median_st_j, sex, age)
    modified_positive = bool(lbbb and modified_sgarbossa_positive(median_st_j, median_s_depth, median_polarity))
    finite_st = [value for value in median_st_j.values() if np.isfinite(value)]
    finite_st60 = [value for value in median_st_j60.values() if np.isfinite(value)]
    finite_ratio = [
        median_st_j[lead] / median_s_depth[lead]
        for lead in LEADS
        if np.isfinite(median_st_j[lead]) and median_st_j[lead] > 0 and median_s_depth[lead] > 0
    ]
    used = len(qrs_durations)
    feature_row: dict[str, object] = {
        "standard_stemi": bool(standard_stemi),
        "lbbb": bool(lbbb),
        "lbbb_raw": bool(raw_lbbb),
        "lbbb_scp": bool(scp_lbbb),
        "modified_sgarbossa_positive": modified_positive,
        "max_st_j_mv": float(np.nanmax(finite_st)) if finite_st else float("nan"),
        "max_st_j60_mv": float(np.nanmax(finite_st60)) if finite_st60 else float("nan"),
        "max_st_s_ratio": float(np.nanmax(finite_ratio)) if finite_ratio else 0.0,
        "qrs_duration_ms": qrs_duration_ms,
        "num_detected_beats": int(len(r_peaks)),
        "num_beats_used": int(used),
        "median_rr_ms": float(np.nanmedian(rr_ms)) if rr_ms else float("nan"),
        "median_heart_rate_bpm": float(60000.0 / np.nanmedian(rr_ms)) if rr_ms else float("nan"),
        "morphology_quality": "ok" if used >= 3 else "low_beat_count",
    }
    for name in ("p_peak_rel_ms", "q_peak_rel_ms", "s_peak_rel_ms", "qrs_onset_rel_ms", "qrs_offset_rel_ms", "j_point_rel_ms", "t_peak_rel_ms"):
        values = [row[name] for row in per_beat_rows if row[name] is not None]
        feature_row[f"median_{name}"] = float(np.nanmedian(values)) if values else float("nan")
    for lead in LEADS:
        safe = lead.replace("a", "a")
        feature_row[f"st_j_{safe}_mv"] = median_st_j[lead]
        feature_row[f"st_j60_{safe}_mv"] = median_st_j60[lead]
        feature_row[f"s_depth_{safe}_mv"] = median_s_depth[lead]
        feature_row[f"qrs_polarity_{safe}"] = median_polarity[lead]

    return feature_row, per_beat_rows
