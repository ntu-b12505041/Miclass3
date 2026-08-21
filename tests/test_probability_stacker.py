import numpy as np

from miclass3.probability_stacker import (
    FEATURE_NAMES,
    apply_class_biases,
    build_stacker_features,
    model_probabilities,
)


def test_stacker_features_have_expected_shape_and_are_finite():
    p_mi = np.array([0.1, 0.5, 0.9])
    p_stemi = np.array([0.2, 0.5, 0.8])
    aux_non = np.array([0.8, 0.3, 0.1])
    aux_stemi = np.array([0.1, 0.3, 0.7])
    aux_nstemi = np.array([0.1, 0.4, 0.2])
    x = build_stacker_features(p_mi, p_stemi, aux_non, aux_stemi, aux_nstemi)
    assert x.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(x).all()


def test_model_probabilities_and_class_biases_are_normalized():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    mean = np.zeros(2)
    scale = np.ones(2)
    coef = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    intercept = np.zeros(3)
    p = model_probabilities(x, mean, scale, coef, intercept)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)

    biased = apply_class_biases(p, stemi_bias=0.5, nstemi_bias=-0.5)
    np.testing.assert_allclose(biased.sum(axis=1), 1.0)
    assert biased[1, 1] > p[1, 1]
