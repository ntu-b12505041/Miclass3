import numpy as np

from miclass3.strict80 import hierarchical_predict, search_strict80_thresholds, six_metric_summary


def test_hierarchical_predict_and_six_metrics_perfect_case():
    y = np.array([0, 0, 1, 1, 2, 2])
    p_mi = np.array([0.05, 0.10, 0.90, 0.85, 0.88, 0.80])
    p_stemi = np.array([0.40, 0.70, 0.95, 0.90, 0.05, 0.10])
    pred = hierarchical_predict(p_mi, p_stemi, 0.5, 0.5)
    assert np.array_equal(pred, y)
    summary, cm = six_metric_summary(y, pred, target=0.80)
    assert cm.trace() == 6
    assert summary["minimum_of_six"] == 1.0
    assert summary["all_six_strictly_above_target"] is True


def test_threshold_search_finds_strict80_feasible_solution():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    p_mi = np.array([0.1] * 10 + [0.9] * 10 + [0.85] * 10)
    p_stemi = np.array([0.5] * 10 + [0.9] * 10 + [0.1] * 10)
    best, _ = search_strict80_thresholds(
        y,
        p_mi,
        p_stemi,
        target=0.80,
        coarse_step=0.1,
        fine_radius=0.05,
        fine_step=0.01,
    )
    assert best.summary["minimum_of_six"] == 1.0
    assert best.summary["all_six_strictly_above_target"] is True
