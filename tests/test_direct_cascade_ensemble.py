import numpy as np

from miclass3.direct_cascade_ensemble import (
    apply_class_biases,
    blend_probabilities,
    cascade_soft_probabilities,
)


def test_cascade_soft_probabilities_are_normalized():
    p_mi = np.array([0.2, 0.8])
    p_stemi = np.array([0.25, 0.75])
    prob = cascade_soft_probabilities(p_mi, p_stemi)
    assert prob.shape == (2, 3)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)
    np.testing.assert_allclose(prob[0], [0.8, 0.05, 0.15])


def test_blend_endpoints_and_biases():
    direct = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    cascade = np.array([[0.6, 0.1, 0.3], [0.1, 0.7, 0.2]])
    np.testing.assert_allclose(blend_probabilities(direct, cascade, 1.0), direct)
    np.testing.assert_allclose(blend_probabilities(direct, cascade, 0.0), cascade)

    mixed = blend_probabilities(direct, cascade, 0.5)
    biased = apply_class_biases(mixed, stemi_bias=0.5, nstemi_bias=-0.5)
    np.testing.assert_allclose(biased.sum(axis=1), 1.0)
    assert biased[0, 1] > mixed[0, 1]
