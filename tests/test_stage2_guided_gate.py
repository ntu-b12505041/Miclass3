import numpy as np

from miclass3.stage2_guided_gate import stage2_guided_predict


def test_stage2_guided_rescue_only_lowers_parent_gate_for_nstemi_like_cases():
    p_mi = np.array([0.60, 0.31, 0.31, 0.10])
    p_stemi = np.array([0.80, 0.10, 0.40, 0.10])

    pred, rescue = stage2_guided_predict(
        p_mi,
        p_stemi,
        mi_threshold=0.324,
        stemi_threshold=0.276,
        rescue_floor=0.28,
        rescue_stemi_max=0.20,
    )

    # Normal MI route remains unchanged and Stage2 calls STEMI.
    assert pred[0] == 1
    assert not rescue[0]
    # Borderline parent miss + strong Stage2 NSTEMI signal is rescued.
    assert pred[1] == 2
    assert rescue[1]
    # Borderline but Stage2 is not sufficiently NSTEMI-like: stays non-MI.
    assert pred[2] == 0
    assert not rescue[2]
    # Far below the parent boundary: stays non-MI even if Stage2 is NSTEMI-like.
    assert pred[3] == 0
    assert not rescue[3]
