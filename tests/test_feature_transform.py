import numpy as np
import pandas as pd

from miclass3.data import FeatureTransform, normalize_ecg_waveform, select_model_features


def test_feature_transform_uses_training_statistics_and_preserves_binary_flags():
    train = pd.DataFrame(
        {
            "st_j": [0.0, 0.1, 0.2, 0.3],
            "qrs_ms": [80.0, 90.0, 100.0, 110.0],
            "lbbb": [0, 0, 1, 0],
        }
    )
    transform = FeatureTransform.fit(train, ["st_j", "qrs_ms", "lbbb"])
    values = transform.transform([0.1, np.nan, 1])

    assert transform.output_dim == 6
    assert values[2] == 1.0
    assert values[3] == 0.0
    assert values[4] == 1.0
    assert values[5] == 0.0
    assert np.isfinite(values).all()


def test_extended_feature_profile_keeps_ecg_inputs_and_excludes_target_columns():
    frame = pd.DataFrame(
        {
            "max_st_j_mv": [0.1], "lbbb": [0], "age": [62], "sex": [1],
            "st_j_V1_mv": [0.02], "label": ["stemi_proxy"], "label_id": [1],
            "scp_codes": ["{'IMI': 100.0}"],
        }
    )
    features = select_model_features(frame, "extended")
    assert {"max_st_j_mv", "lbbb", "age", "sex", "st_j_V1_mv"} <= set(features)
    assert "label" not in features and "label_id" not in features and "scp_codes" not in features


def test_global_waveform_normalization_preserves_relative_lead_amplitude():
    signal = np.column_stack([np.array([-1.0, 0.0, 1.0]), np.array([-2.0, 0.0, 2.0])])
    normalized = normalize_ecg_waveform(signal, "lead_centered_global_scale")
    assert np.isclose(np.ptp(normalized[0]) / np.ptp(normalized[1]), 0.5)
