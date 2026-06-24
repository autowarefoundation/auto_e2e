"""Tests for the World Action Model (slow branch, WG 2026-06-24 agreement).

Verifies the agreed contract: per-frame embedding -> rolling FIFO buffer ->
visual_history (history_len*frame_embed_dim = 896), future prediction + JEPA
loss in training (frozen target, stop-gradient), and inference returning only
the history vector.
"""

import torch
import torch.nn as nn

from model_components.world_action_model import (
    FrameEncoder,
    RollingHistoryBuffer,
    WorldActionModel,
)

CH = 8  # mock backbone channels (small for speed)


class _MockBackbone(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, CH, 3, padding=1)

    def forward(self, x):
        return [self.conv(x)]  # list of feature maps, like the real backbone


def _wam(device, **kw):
    return WorldActionModel(_MockBackbone(), feature_channels=CH,
                            frame_embed_dim=224, history_len=4,
                            num_future_steps=4, **kw).to(device)


def _frames(B, n, device):
    return torch.randn(B, n, 3, 16, 16, device=device)


def test_visual_history_dim_is_896(device):
    m = _wam(device)
    assert m.visual_history_dim == 896  # 4 * 224
    vh = m(_frames(2, 4, device))        # inference: only the history vector
    assert vh.shape == (2, 896)


def test_frame_encoder_shape(device):
    enc = FrameEncoder(_MockBackbone(), feature_channels=CH,
                       frame_embed_dim=224).to(device)
    assert enc(torch.randn(2, 3, 16, 16, device=device)).shape == (2, 224)


def test_training_returns_history_prediction_and_loss(device):
    m = _wam(device)
    vh, predicted, loss = m(_frames(2, 4, device), future_frames=_frames(2, 4, device))
    assert vh.shape == (2, 896)
    assert len(predicted) == 4 and all(p.shape == (2, 224) for p in predicted)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_jepa_gradient_flows_to_online_not_target(device):
    m = _wam(device)
    _vh, _pred, loss = m(_frames(2, 4, device), future_frames=_frames(2, 4, device))
    loss.backward()
    assert any(p.grad is not None for p in m.future_predictor.parameters())
    assert any(p.grad is not None for p in m.encoder.parameters()), \
        "online encoder must receive gradient"
    assert all(p.grad is None for p in m.target.parameters()), \
        "frozen JEPA target must NOT receive gradient"


def test_inference_returns_only_visual_history(device):
    m = _wam(device)
    out = m(_frames(1, 4, device))
    assert isinstance(out, torch.Tensor) and out.shape == (1, 896)


def test_configurable_horizons(device):
    m = WorldActionModel(_MockBackbone(), feature_channels=CH, frame_embed_dim=32,
                         history_len=3, num_future_steps=2).to(device)
    vh, pred, _loss = m(_frames(2, 3, device), future_frames=_frames(2, 2, device))
    assert vh.shape == (2, 96)  # 3 * 32
    assert len(pred) == 2


class TestRollingHistoryBuffer:
    def test_fifo_keeps_last_n_and_dim(self, device):
        buf = RollingHistoryBuffer(history_len=4)
        for _ in range(6):
            buf.push(torch.randn(2, 224, device=device))
        vh = buf.visual_history()
        assert vh.shape == (2, 896)  # 4 * 224, oldest dropped

    def test_left_pads_before_full(self, device):
        buf = RollingHistoryBuffer(history_len=4)
        buf.push(torch.ones(1, 224, device=device))
        vh = buf.visual_history()
        assert vh.shape == (1, 896)
        # first 3 slots zero-padded, last slot is the pushed frame
        assert torch.all(vh[:, : 3 * 224] == 0) and torch.all(vh[:, 3 * 224:] == 1)

    def test_fifo_order_first_in_first_out(self, device):
        buf = RollingHistoryBuffer(history_len=2)
        buf.push(torch.full((1, 224), 1.0, device=device))
        buf.push(torch.full((1, 224), 2.0, device=device))
        buf.push(torch.full((1, 224), 3.0, device=device))  # evicts the "1"
        vh = buf.visual_history()
        assert torch.all(vh[:, :224] == 2.0) and torch.all(vh[:, 224:] == 3.0)
