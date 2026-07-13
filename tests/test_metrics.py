import numpy as np

from miclass3.metrics import classification_tables, predict_labels


def test_classification_tables_include_common_metrics_and_all_classes():
    truth = np.array([0, 1, 2, 1, 0, 2])
    probabilities = np.array([
        [.9, .05, .05], [.1, .8, .1], [.1, .1, .8], [.1, .7, .2], [.2, .7, .1], [.2, .2, .6],
    ])
    overall, report, matrix = classification_tables(truth, probabilities)
    assert {"accuracy", "balanced_accuracy", "macro_auroc", "macro_auprc", "macro_f1", "stemi_recall"} <= overall.keys()
    assert report["class"].tolist() == ["non_mi", "stemi_proxy", "nstemi_proxy"]
    assert matrix.to_numpy().sum() == len(truth)


def test_stemi_threshold_keeps_three_class_decision_rule():
    probabilities = np.array([
        [.45, .40, .15],
        [.20, .30, .50],
        [.05, .20, .75],
    ])
    assert predict_labels(probabilities, stemi_threshold=0.35).tolist() == [1, 2, 2]
