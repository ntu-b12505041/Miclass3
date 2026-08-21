import numpy as np

from miclass3.threeway_parent_blend import blend_three_parent_log_odds


def test_threeway_parent_blend_reproduces_each_endpoint():
    p1 = np.array([0.1, 0.5, 0.9])
    p2 = np.array([0.2, 0.6, 0.8])
    p3 = np.array([0.3, 0.4, 0.7])
    np.testing.assert_allclose(blend_three_parent_log_odds(p1, p2, p3, 1.0, 0.0), p1, atol=1e-6)
    np.testing.assert_allclose(blend_three_parent_log_odds(p1, p2, p3, 0.0, 1.0), p2, atol=1e-6)
    np.testing.assert_allclose(blend_three_parent_log_odds(p1, p2, p3, 0.0, 0.0), p3, atol=1e-6)


def test_threeway_parent_blend_is_finite_and_bounded():
    p1 = np.array([0.01, 0.99])
    p2 = np.array([0.20, 0.80])
    p3 = np.array([0.40, 0.60])
    blended = blend_three_parent_log_odds(p1, p2, p3, 0.4, 0.3)
    assert np.isfinite(blended).all()
    assert ((blended > 0.0) & (blended < 1.0)).all()
