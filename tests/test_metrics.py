import numpy as np

from miclass3.metrics import classification_tables


def test_classification_tables_include_common_metrics_and_all_classes():
    truth = np.array([0, 1, 2, 1, 0, 2])
    probabilities = np.array([
        [.9, .05, .05], [.1, .8, .1], [.1, .1, .8], [.1, .7, .2], [.2, .7, .1], [.2, .2, .6],
    ])
    overall, report, matrix = classification_tables(truth, probabilities)
    assert {"accuracy", "balanced_accuracy", "macro_auroc", "macro_auprc", "macro_f1", "stemi_recall"} <= overall.keys()
    assert report["class"].tolist() == ["non_mi", "stemi_proxy", "nstemi_proxy"]
    assert matrix.to_numpy().sum() == len(truth)
