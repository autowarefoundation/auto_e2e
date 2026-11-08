"""BEVFormer V2 T8 temporal fusion contracts."""

from __future__ import annotations

import pytest
import torch

from model_components.view_fusion.bevformer_v2_t8 import BEVFormerV2T8TemporalFusion


def test_t8_fusion_preserves_order_and_current_gradient():
    torch.manual_seed(7)
    fusion = BEVFormerV2T8TemporalFusion(
        embed_dim=4,
        inter_channels=32,
        num_layers=3,
    ).eval()
    history = [
        torch.randn(2, 4, 3, 2)
        for _ in range(7)
    ]
    current = torch.randn(2, 4, 3, 2, requires_grad=True)

    output = fusion([*history, current])
    reordered = fusion([
        history[1],
        history[0],
        *history[2:],
        current,
    ])
    output.square().mean().backward()

    assert output.shape == current.shape
    assert not torch.equal(output, reordered)
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in fusion.parameters()
    )


def test_t8_fusion_rejects_incomplete_history():
    fusion = BEVFormerV2T8TemporalFusion(
        embed_dim=4,
        inter_channels=32,
    )

    with pytest.raises(ValueError, match="8 ordered"):
        fusion([torch.zeros(1, 4, 2, 2) for _ in range(7)])


