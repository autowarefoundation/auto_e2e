"""Tests for the World Action Model (slow branch, WG 2026-06-24 agreement).

Verifies the per-tick API Zain specified in auto_e2e.py (lines 67-76):
``visual_embedding, future_state_pred = WorldActionModel(frame, visual_history)``,
an external rolling FIFO buffer (size N=4) forming the Encoded Visual History
(N*frame_embed_dim = 896), and the JEPA loss (frozen target, stop-gradient)
computed separately via ``jepa_loss``.
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


def _frame(B, device):
    return torch.randn(B, 3, 16, 16, device=device)


def _window(B, n, device):
    return torch.randn(B, n, 3, 16, 16, device=device)


def test_visual_history_dim_is_896(device):
    m = _wam(device)
    assert m.visual_history_dim == 896  # 4 * 224
    vh = m.encode_history(_window(2, 4, device))  # windowed encode -> [B, 896]
    assert vh.shape == (2, 896)


def test_frame_encoder_shape(device):
    enc = FrameEncoder(_MockBackbone(), feature_channels=CH,
                       frame_embed_dim=224).to(device)
    assert enc(_frame(2, device)).shape == (2, 224)


def test_forward_per_tick_returns_embedding_and_future(device):
    """Zain's API: visual_embedding, future_state_pred = WAM(frame, visual_history)."""
    m = _wam(device)
    vh = torch.randn(2, 896, device=device)              # current buffer state
    emb, future = m(_frame(2, device), visual_history=vh)
    assert emb.shape == (2, 224)                          # pushed to the buffer
    assert len(future) == 4 and all(f.shape == (2, 224) for f in future)


def test_forward_without_history_has_no_future(device):
    """Inference / first ticks: no visual_history -> future_state_pred is None."""
    m = _wam(device)
    emb, future = m(_frame(2, device))
    assert emb.shape == (2, 224) and future is None


def test_jepa_loss_grad_to_online_not_target(device):
    m = _wam(device)
    vh = torch.randn(2, 896, device=device)
    _emb, future = m(_frame(2, device), visual_history=vh)
    loss = m.jepa_loss(future, _window(2, 4, device))    # vs frozen target
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in m.future_predictor.parameters())
    assert all(p.grad is None for p in m.target.parameters()), \
        "frozen JEPA target must NOT receive gradient"


def test_configurable_horizons(device):
    m = WorldActionModel(_MockBackbone(), feature_channels=CH, frame_embed_dim=32,
                         history_len=3, num_future_steps=2).to(device)
    assert m.visual_history_dim == 96  # 3 * 32
    emb, future = m(_frame(2, device), visual_history=torch.randn(2, 96, device=device))
    assert emb.shape == (2, 32) and len(future) == 2


class TestRollingHistoryBuffer:
    def test_fifo_keeps_last_n_and_dim(self, device):
        buf = RollingHistoryBuffer(history_len=4)
        for _ in range(6):
            buf.push(torch.randn(2, 224, device=device))
        assert buf.visual_history().shape == (2, 896)  # 4*224, oldest dropped

    def test_left_pads_before_full(self, device):
        buf = RollingHistoryBuffer(history_len=4)
        buf.push(torch.ones(1, 224, device=device))
        vh = buf.visual_history()
        assert vh.shape == (1, 896)
        assert torch.all(vh[:, : 3 * 224] == 0) and torch.all(vh[:, 3 * 224:] == 1)

    def test_fifo_order_first_in_first_out(self, device):
        buf = RollingHistoryBuffer(history_len=2)
        for v in (1.0, 2.0, 3.0):
            buf.push(torch.full((1, 224), v, device=device))  # 1.0 evicted
        vh = buf.visual_history()
        assert torch.all(vh[:, :224] == 2.0) and torch.all(vh[:, 224:] == 3.0)


def test_online_loop_buffer_then_reactive_shape(device):
    """End-to-end online pattern from Zain's auto_e2e wiring: per tick encode ->
    push to buffer -> the buffer is the visual_history for the reactive planner."""
    m = _wam(device)
    buf = RollingHistoryBuffer(history_len=4)
    vh = None
    for _ in range(5):  # 5 ticks
        emb, _future = m(_frame(1, device), visual_history=vh)
        buf.push(emb)
        vh = buf.visual_history()
    assert vh.shape == (1, 896)  # ready to feed Reactive_E2E


class _MockBackbone4(nn.Module):
    """4-stage mock matching the real backbone, for the full AutoE2E path."""

    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone_channels = 1440
        self._st = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, 3, 1, 1), nn.AdaptiveAvgPool2d(64)),
            nn.Sequential(nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32)),
            nn.Sequential(nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16)),
            nn.Sequential(nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8)),
        ])

    def forward(self, x):
        outs, h = [], x
        for s in self._st:
            h = s(h)
            outs.append(h)
        return outs


class TestAutoE2EWorldModelWiring:
    """Zain's auto_e2e.py wiring (lines 67-76): WAM -> buffer -> reactive; return
    trajectory (infer) / (trajectory, future_state_pred) (train)."""

    def _build(self, device):
        from unittest.mock import patch
        from model_components.auto_e2e import AutoE2E
        with patch("model_components.reactive_e2e.Backbone", _MockBackbone4):
            return AutoE2E(num_views=6, view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                           enable_world_model=True,
                           world_model_kwargs={"feature_channels": 768}).to(device)

    def _inputs(self, device):
        return (torch.randn(2, 6, 3, 256, 256, device=device),  # camera_tiles
                torch.randn(2, 3, 256, 256, device=device),      # map_input
                torch.zeros(2, 896, device=device),              # legacy visual_history
                torch.randn(2, 256, device=device))              # egomotion

    def test_world_model_is_built(self, device):
        m = self._build(device)
        assert m.World_Action_Model_E2E is not None
        assert m.visual_history_buffer is not None

    def test_infer_returns_trajectory_only(self, device):
        m = self._build(device)
        m.reset_visual_history()
        cam, mp, vh, ego = self._inputs(device)
        out = m(cam, mp, vh, ego, mode="infer")
        traj = out[0] if isinstance(out, tuple) else out
        assert traj.shape == (2, 128) and torch.isfinite(traj).all()

    def test_train_returns_trajectory_and_future_state(self, device):
        m = self._build(device)
        m.reset_visual_history()
        cam, mp, vh, ego = self._inputs(device)
        tgt = torch.randn(2, 128, device=device)
        m(cam, mp, vh, ego, mode="train", trajectory_target=tgt)        # tick 1 fills buffer
        out = m(cam, mp, vh, ego, mode="train", trajectory_target=tgt)  # tick 2
        assert isinstance(out, tuple) and len(out) == 2
        traj, future_state_pred = out
        traj0 = traj[0] if isinstance(traj, tuple) else traj
        assert traj0.shape == (2, 128)
        assert future_state_pred is not None and len(future_state_pred) == 4

    def test_default_world_model_off_unchanged(self, device):
        """Default (no World Model) keeps the reactive-only behaviour."""
        from unittest.mock import patch
        from model_components.auto_e2e import AutoE2E
        with patch("model_components.reactive_e2e.Backbone", _MockBackbone4):
            m = AutoE2E(num_views=6, view_fusion_kwargs={"bev_h": 8, "bev_w": 8}).to(device)
        assert m.World_Action_Model_E2E is None
        cam, mp, vh, ego = self._inputs(device)
        out = m(cam, mp, vh, ego, mode="infer")
        assert not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], list))
