import numpy as np

from miclass3.direct_aux_hierarchy import direct_aux_hierarchy_predict, sigmoid


def test_sigmoid_is_stable_and_bounded():
    values = np.array([-1000.0, 0.0, 1000.0])
    prob = sigmoid(values)
    assert np.isfinite(prob).all()
    assert 0.0 <= prob[0] < 1e-6
    assert prob[1] == 0.5
    assert 1.0 - 1e-6 < prob[2] <= 1.0


def test_direct_aux_hierarchy_routes_parent_then_subtype():
    p_mi = np.array([0.2, 0.8, 0.8])
    p_stemi = np.array([0.9, 0.9, 0.1])
    pred = direct_aux_hierarchy_predict(
        p_mi,
        p_stemi,
        mi_threshold=0.5,
        stemi_threshold=0.5,
    )
    np.testing.assert_array_equal(pred, np.array([0, 1, 2]))
