import numpy as np

from miclass3.nstemi_rescue import nstemi_rescue_predict


def test_rescue_only_changes_borderline_aux_nstemi_cases():
    p_mi = np.array([0.60, 0.31, 0.10, 0.31])
    p_stemi = np.array([0.80, 0.20, 0.20, 0.80])
    aux_non = np.array([0.10, 0.20, 0.20, 0.20])
    aux_nstemi = np.array([0.10, 0.70, 0.70, 0.70])

    pred = nstemi_rescue_predict(
        p_mi,
        p_stemi,
        aux_non,
        aux_nstemi,
        mi_threshold=0.324,
        stemi_threshold=0.276,
        rescue_margin=0.30,
        rescue_floor=0.28,
    )

    # Normal MI path -> STEMI.
    assert pred[0] == 1
    # Borderline Stage1-negative + NSTEMI-like auxiliary -> rescued, Stage2 says NSTEMI.
    assert pred[1] == 2
    # Too far below the parent boundary -> remains non-MI.
    assert pred[2] == 0
    # Rescued record still goes through Stage2, which can call STEMI.
    assert pred[3] == 1
