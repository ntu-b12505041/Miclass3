import numpy as np

from miclass3.cascade import (
    calibrate_soft_cascade,
    compose_calibrated_soft_cascade,
    compose_hard_cascade,
    compose_soft_cascade,
)


def test_soft_cascade_probabilities_sum_to_one():
    p_mi = np.array([0.1, 0.8, 0.9])
    p_stemi_given_mi = np.array([0.7, 0.75, 0.2])
    probabilities = compose_soft_cascade(p_mi, p_stemi_given_mi)
    assert probabilities.shape == (3, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[0], [0.9, 0.07, 0.03])


def test_hard_cascade_respects_stage_gate():
    p_mi = np.array([0.2, 0.8, 0.8])
    p_stemi_given_mi = np.array([0.9, 0.7, 0.3])
    probabilities, predictions = compose_hard_cascade(
        p_mi,
        p_stemi_given_mi,
        mi_threshold=0.5,
        stemi_threshold=0.5,
    )
    assert predictions.tolist() == [0, 1, 2]
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_calibrated_soft_cascade_preserves_probability_simplex():
    p_mi = np.array([0.15, 0.55, 0.85])
    p_stemi = np.array([0.2, 0.8, 0.4])
    probabilities = compose_calibrated_soft_cascade(
        p_mi,
        p_stemi,
        mi_logit_bias=0.4,
        stemi_logit_bias=-0.2,
    )
    assert probabilities.shape == (3, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_soft_calibration_uses_validation_labels_and_respects_recall_floor():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    p_mi = np.array([0.10, 0.20, 0.45, 0.60, 0.75, 0.65, 0.75, 0.85])
    p_stemi = np.array([0.20, 0.40, 0.55, 0.60, 0.75, 0.20, 0.25, 0.35])
    summary, candidates = calibrate_soft_cascade(
        y,
        p_mi,
        p_stemi,
        minimum_stemi_recall=0.5,
        bias_min=-0.5,
        bias_max=0.5,
        bias_step=0.5,
    )
    assert summary["candidate_count"] == 9
    assert summary["constraint_satisfied"]
    assert summary["stemi_recall"] >= 0.5
    assert len(candidates) == 9
