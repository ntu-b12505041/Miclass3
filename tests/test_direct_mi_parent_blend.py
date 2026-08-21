import numpy as np

from miclass3.direct_mi_parent_blend import blend_parent_log_odds


def test_parent_blend_endpoints_reproduce_inputs():
    stage1 = np.array([0.1, 0.5, 0.9])
    direct = np.array([0.2, 0.6, 0.8])
    np.testing.assert_allclose(blend_parent_log_odds(stage1, direct, 1.0), stage1, atol=1e-6)
    np.testing.assert_allclose(blend_parent_log_odds(stage1, direct, 0.0), direct, atol=1e-6)


def test_parent_logodds_blend_is_between_agreeing_inputs():
    stage1 = np.array([0.2, 0.8])
    direct = np.array([0.4, 0.9])
    blended = blend_parent_log_odds(stage1, direct, 0.5)
    assert 0.2 < blended[0] < 0.4
    assert 0.8 < blended[1] < 0.9
    assert np.isfinite(blended).all()
