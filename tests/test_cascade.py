import numpy as np

from miclass3.cascade import compose_hard_cascade, compose_soft_cascade


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
