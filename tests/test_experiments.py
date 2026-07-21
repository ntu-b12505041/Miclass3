from miclass3.experiments import build_validation_leaderboard, metric_floor


def test_metric_floor_requires_every_requested_metric():
    metrics = {"accuracy": 0.9, "macro_f1": 0.9}
    assert metric_floor(metrics, ("accuracy", "macro_f1")) == 0.9
    assert metric_floor(metrics, ("accuracy", "balanced_accuracy")) == float("-inf")


def test_leaderboard_prioritizes_the_weakest_required_validation_metric():
    trials = [
        {
            "trial": "high_auprc_low_recall",
            "metrics": {
                "accuracy": 0.90, "balanced_accuracy": 0.80, "macro_auroc": 0.92,
                "macro_auprc": 0.95, "macro_f1": 0.89, "stemi_recall": 0.80,
            },
        },
        {
            "trial": "all_rounder",
            "metrics": {
                "accuracy": 0.88, "balanced_accuracy": 0.87, "macro_auroc": 0.91,
                "macro_auprc": 0.88, "macro_f1": 0.86, "stemi_recall": 0.87,
            },
        },
    ]
    board = build_validation_leaderboard(trials, target=0.85)
    assert board.loc[0, "trial"] == "all_rounder"
    assert bool(board.loc[0, "passes_target"])
    assert not bool(board.loc[1, "passes_target"])
