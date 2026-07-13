import pytest

torch = pytest.importorskip("torch")

from miclass3.models import make_model


def test_all_recommended_models_emit_three_classes():
    x = torch.randn(2, 12, 5000)
    for name in ("seresnet", "inceptiontime", "morphology_fusion"):
        model = make_model(name, feature_dim=6)
        out = model(x, torch.randn(2, 6))
        assert out["class_logits"].shape == (2, 3)
