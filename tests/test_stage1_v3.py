import pytest

torch = pytest.importorskip("torch")

from miclass3.stage1_v3 import Stage1V3MorphologyFusion


def test_stage1_v3_emits_binary_and_auxiliary_logits():
    model = Stage1V3MorphologyFusion(feature_dim=106)
    x = torch.randn(2, 12, 5000)
    features = torch.randn(2, 106)
    out = model(x, features)
    assert out["class_logits"].shape == (2, 2)
    assert out["subtype_logits"].shape == (2, 3)
