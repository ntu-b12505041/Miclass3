import pytest


torch = pytest.importorskip("torch")

from miclass3.stage1_v2 import Stage1V2MultiScaleSE


def test_stage1_v2_emits_binary_and_auxiliary_logits():
    model = Stage1V2MultiScaleSE(width=16, kernels=(11, 25, 49), bottleneck=8)
    x = torch.randn(2, 12, 5000)
    output = model(x)

    assert output["class_logits"].shape == (2, 2)
    assert output["subtype_logits"].shape == (2, 3)
