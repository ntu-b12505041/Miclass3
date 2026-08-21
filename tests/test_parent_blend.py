import numpy as np

from miclass3.parent_blend import blend_parent_probability


def test_parent_blend_alpha_one_keeps_primary_probability():
    primary = np.array([0.1, 0.4, 0.9])
    aux_non_mi = np.array([0.7, 0.2, 0.1])
    blended = blend_parent_probability(primary, aux_non_mi, 1.0)
    np.testing.assert_allclose(blended, primary)


def test_parent_blend_uses_auxiliary_mi_probability():
    primary = np.array([0.2, 0.8])
    aux_non_mi = np.array([0.2, 0.8])
    # aux MI = [0.8, 0.2]; with equal weights both become 0.5.
    blended = blend_parent_probability(primary, aux_non_mi, 0.5)
    np.testing.assert_allclose(blended, np.array([0.5, 0.5]))
