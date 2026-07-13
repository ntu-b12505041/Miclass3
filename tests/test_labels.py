import pandas as pd

from miclass3.labels import build_label_manifest


def test_proxy_labels_keep_old_mi_out_of_nstemi():
    meta = pd.DataFrame([
        {"ecg_id": 1, "scp_codes": {"NORM": 100}},
        {"ecg_id": 2, "scp_codes": {"IMI": 100}, "standard_stemi": True},
        {"ecg_id": 3, "scp_codes": {"IMI": 100}},
        {"ecg_id": 4, "scp_codes": {"IMI": 100}, "infarction_stadium1": "old"},
    ])
    scp = pd.DataFrame({"diagnostic_class": ["NORM", "MI"], "diagnostic": [1, 1]}, index=["NORM", "IMI"])
    out = build_label_manifest(meta, scp)
    assert out.label.tolist() == ["non_mi", "stemi_proxy", "nstemi_proxy", "exclude_old_mi"]


def test_lbbb_needs_modified_sgarbossa_not_standard_rule():
    meta = pd.DataFrame([{"ecg_id": 1, "scp_codes": {"IMI": 100}, "lbbb": True, "standard_stemi": True}])
    scp = pd.DataFrame({"diagnostic_class": ["MI"], "diagnostic": [1]}, index=["IMI"])
    assert build_label_manifest(meta, scp).iloc[0].label == "nstemi_proxy"


def test_failed_morphology_is_excluded_instead_of_forced_to_nstemi():
    meta = pd.DataFrame([{"ecg_id": 1, "scp_codes": {"IMI": 100}, "morphology_quality": "load_error"}])
    scp = pd.DataFrame({"diagnostic_class": ["MI"], "diagnostic": [1]}, index=["IMI"])
    out = build_label_manifest(meta, scp)
    assert out.iloc[0].label == "exclude_morphology_failed"
    assert pd.isna(out.iloc[0].label_id)
