import inspect

import numpy as np

from miclass3.delineation import _metadata_has_lbbb, _select_lbbb, extract_morphology_features
from miclass3.morphology import LEADS


def _gaussian(length, center, width):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def synthetic_stemi_record(fs=500, seconds=10):
    n = fs * seconds
    signal = np.zeros((n, len(LEADS)), dtype=float)
    elevated = {LEADS.index("II"), LEADS.index("III"), LEADS.index("aVF")}
    for r in range(fs, n - fs, fs):
        signal += 0.03 * _gaussian(n, r - int(0.18 * fs), 18)[:, None]
        for lead in range(len(LEADS)):
            polarity = -1.0 if LEADS[lead] == "aVR" else 1.0
            signal[:, lead] += polarity * 1.0 * _gaussian(n, r, 7)
            signal[:, lead] -= polarity * 0.25 * _gaussian(n, r + 22, 8)
            signal[:, lead] += 0.15 * _gaussian(n, r + int(0.28 * fs), 35)
        start = r + int(0.06 * fs)
        stop = r + int(0.18 * fs)
        signal[start:stop, list(elevated)] += 0.18
    return signal


def test_extract_morphology_promotes_contiguous_st_elevation():
    features, beats = extract_morphology_features(synthetic_stemi_record(), fs=500, age=60, sex="male")
    assert features["num_beats_used"] >= 5
    assert features["morphology_quality"] == "ok"
    assert features["standard_stemi"] is True
    assert features["lbbb"] is False
    assert features["max_st_j_mv"] >= 0.1
    assert {"r_sample", "q_peak_sample", "s_peak_sample", "qrs_onset_sample", "j_point_sample", "t_peak_sample"} <= set(beats[0])


def test_ecgdeli_backend_uses_external_fiducials_without_changing_label_rules():
    fs = 500
    signal = synthetic_stemi_record(fs=fs)
    fiducials = []
    for beat_index, r in enumerate(range(fs, fs * 9, fs)):
        fiducials.append(
            {
                "beat_index": beat_index,
                "r_sample": r,
                "qrs_onset_sample": r - 25,
                "qrs_offset_sample": r + 40,
                "p_peak_sample": r - 90,
                "q_peak_sample": r - 12,
                "s_peak_sample": r + 22,
                "t_peak_sample": r + 140,
            }
        )

    features, beats = extract_morphology_features(
        signal,
        fs=fs,
        age=60,
        sex="male",
        backend="ecgdeli",
        external_fiducials=fiducials,
    )

    assert features["requested_delineation_backend"] == "ecgdeli"
    assert features["delineation_backend"] == "ecgdeli"
    assert features["standard_stemi"] is True
    assert beats[0]["fiducial_source"] == "ecgdeli"


def test_primary_lbbb_route_uses_positive_scp_code_only():
    assert inspect.signature(extract_morphology_features).parameters["lbbb_source"].default == "scp"
    assert _metadata_has_lbbb({"scp_codes": {"LBBB": 100}})
    assert not _metadata_has_lbbb({"scp_codes": {"LBBB": 0, "IMI": 100}})
    assert _select_lbbb(raw_lbbb=True, scp_lbbb=False) is False
    assert _select_lbbb(raw_lbbb=True, scp_lbbb=False, source="either") is True
    assert _select_lbbb(raw_lbbb=False, scp_lbbb=True, source="scp") is True
