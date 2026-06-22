"""Unit tests for the optional Bezier trajectory planner.

These tests exercise the planner in isolation (no backbone) and verify:
  - the output contract matches TrajectoryPlanner (trajectory + ego_hidden),
  - the Bezier expansion yields low-jerk control-signal profiles by
    construction (much smoother than raw per-step regression),
  - the registry resolves "bezier" and rejects unknown names,
  - shapes are configurable and gradients flow to all parameters.
"""

import pytest
import torch
import torch.nn as nn

from model_components.trajectory_planning import (
    BasePlanner,
    BezierPlanner,
    build_planner,
)


class _MockBackbone(nn.Module):
    """Minimal stand-in for Backbone (4 channels-first feature maps),
    self-contained so this test file does not depend on conftest imports."""

    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone_channels = 1440
        self._stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, 3, 1, 1), nn.AdaptiveAvgPool2d(64)),
            nn.Sequential(nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32)),
            nn.Sequential(nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16)),
            nn.Sequential(nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8)),
        ])

    def forward(self, image):
        outs, x = [], image
        for stage in self._stages:
            x = stage(x)
            outs.append(x)
        return outs


EMBED_DIM = 256
EGO_DIM = 256
VIS_DIM = 896


def _make_inputs(batch_size, device, h=8, w=8):
    bev = torch.randn(batch_size, EMBED_DIM, h, w, device=device)
    visual_history = torch.randn(batch_size, VIS_DIM, device=device)
    egomotion_history = torch.randn(batch_size, EGO_DIM, device=device)
    return bev, visual_history, egomotion_history


def test_output_contract_shapes(device):
    planner = BezierPlanner(embed_dim=EMBED_DIM).to(device)
    bev, vis, ego = _make_inputs(4, device)
    trajectory, ego_hidden = planner(bev, vis, ego)
    # 64 timesteps x 2 signals (accel, curvature)
    assert trajectory.shape == (4, 128)
    assert ego_hidden.shape == (4, EMBED_DIM)
    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(ego_hidden).all()


def test_is_base_planner_subclass():
    assert issubclass(BezierPlanner, BasePlanner)


def test_registry_builds_bezier(device):
    planner = build_planner("bezier", embed_dim=EMBED_DIM).to(device)
    assert isinstance(planner, BezierPlanner)
    bev, vis, ego = _make_inputs(2, device)
    trajectory, ego_hidden = planner(bev, vis, ego)
    assert trajectory.shape == (2, 128)


def test_registry_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown planner"):
        build_planner("does_not_exist")


def test_resolution_invariant_to_bev_size(device):
    """Planner must accept any BEV spatial resolution (global pooling)."""
    planner = BezierPlanner(embed_dim=EMBED_DIM).to(device)
    for h, w in [(8, 8), (45, 30), (7, 7)]:
        bev, vis, ego = _make_inputs(2, device, h=h, w=w)
        trajectory, _ = planner(bev, vis, ego)
        assert trajectory.shape == (2, 128)


def test_configurable_timesteps_and_signals(device):
    planner = BezierPlanner(
        embed_dim=EMBED_DIM, num_timesteps=32, num_signals=3, num_controls=4
    ).to(device)
    bev, vis, ego = _make_inputs(2, device)
    trajectory, _ = planner(bev, vis, ego)
    assert trajectory.shape == (2, 32 * 3)


def test_invalid_num_controls():
    with pytest.raises(ValueError):
        BezierPlanner(num_controls=1)
    with pytest.raises(ValueError):
        BezierPlanner(num_timesteps=4, num_controls=8)


def test_bernstein_basis_partition_of_unity(device):
    """Bernstein basis rows must sum to 1 (partition of unity)."""
    planner = BezierPlanner(embed_dim=EMBED_DIM, num_timesteps=64,
                            num_controls=5).to(device)
    row_sums = planner.bernstein_basis.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_smoothness_lower_jerk_than_raw_regression(device):
    """The Bezier profile must have far lower step-to-step variance (jerk)
    than an unconstrained per-step regression of the same length, regardless
    of (random) weights — this is the architectural guarantee."""
    torch.manual_seed(0)
    planner = BezierPlanner(embed_dim=EMBED_DIM, num_timesteps=64,
                            num_signals=2, num_controls=5).to(device)
    bev, vis, ego = _make_inputs(8, device)
    trajectory, _ = planner(bev, vis, ego)
    traj = trajectory.view(8, 64, 2)

    # First-difference variance per signal channel (jerk proxy).
    diffs = traj[:, 1:, :] - traj[:, :-1, :]
    bezier_var = diffs.var().item()

    # Raw baseline: unconstrained per-step values of comparable magnitude.
    raw = torch.randn(8, 64, 2, device=device) * traj.std()
    raw_diffs = raw[:, 1:, :] - raw[:, :-1, :]
    raw_var = raw_diffs.var().item()

    assert bezier_var < raw_var * 0.1, (
        f"Bezier jerk {bezier_var:.3e} not << raw jerk {raw_var:.3e}"
    )


def test_autoe2e_with_bezier_planner_end_to_end(device):
    """AutoE2E must accept planner='bezier' and keep the (trajectory,
    ego_hidden, future) 3-tuple contract intact."""
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E
    from model_components.trajectory_planning import BezierPlanner

    with patch("model_components.auto_e2e.Backbone", _MockBackbone):
        model = AutoE2E(num_views=8, fusion_mode="concat",
                        planner_mode="bezier").to(device)

    assert isinstance(model.TrajectoryPlanner, BezierPlanner)

    x = torch.randn(2, 8, 3, 256, 256, device=device)
    map_input = torch.randn(2, 3, 256, 256, device=device)
    vis = torch.randn(2, 896, device=device)
    ego = torch.randn(2, 256, device=device)
    trajectory, ego_hidden, future = model(x, map_input, vis, ego, mode="infer")

    assert trajectory.shape == (2, 128)
    assert ego_hidden.shape == (2, 256)
    assert future is None  # infer mode returns None for future visual features


def test_autoe2e_with_bezier_planner_train_mode(device):
    """Full forward pass in train mode: AutoE2E -> BezierPlanner.compute_planner_loss
    returns a SCALAR loss, future features are produced, and gradients flow back
    through the planner. This exercises the actual layers end-to-end."""
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E

    with patch("model_components.auto_e2e.Backbone", _MockBackbone):
        model = AutoE2E(num_views=8, fusion_mode="concat",
                        planner_mode="bezier").to(device)

    x = torch.randn(2, 8, 3, 256, 256, device=device)
    map_input = torch.randn(2, 3, 256, 256, device=device)
    vis = torch.randn(2, 896, device=device)
    ego = torch.randn(2, 256, device=device)
    target = torch.randn(2, 128, device=device)

    loss, ego_hidden, future = model(
        x, map_input, vis, ego, mode="train", trajectory_target=target
    )

    assert loss.ndim == 0, "train mode must return a scalar planner loss"
    assert torch.isfinite(loss) and loss.item() >= 0.0
    assert ego_hidden.shape == (2, 256)
    assert future is not None  # train mode produces future visual features

    loss.backward()
    assert any(p.grad is not None for p in model.TrajectoryPlanner.parameters()), \
        "planner must receive gradient from compute_planner_loss"


def test_compute_planner_loss_delegates_to_shared_loss(device):
    """compute_planner_loss must use the shared TrajectoryImitationLoss
    (losses/ module), not an inline loss, and honour the (loss, ego_hidden)
    BasePlanner contract."""
    from model_components.losses.trajectory_loss import TrajectoryImitationLoss

    planner = BezierPlanner(embed_dim=EMBED_DIM).to(device)
    assert isinstance(planner.trajectory_loss, TrajectoryImitationLoss)

    bev, vis, ego = _make_inputs(2, device)
    target = torch.randn(2, 128, device=device)
    loss, ego_hidden = planner.compute_planner_loss(bev, vis, ego, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert ego_hidden.shape == (2, EMBED_DIM)
    loss.backward()  # loss must be differentiable through the planner


def test_autoe2e_default_planner_unchanged(device):
    """Default planner must remain the autoregressive TrajectoryPlanner."""
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E
    from model_components.trajectory_planning import GRUPlanner as TrajectoryPlanner

    with patch("model_components.auto_e2e.Backbone", _MockBackbone):
        model = AutoE2E(num_views=8, fusion_mode="concat").to(device)
    assert isinstance(model.TrajectoryPlanner, TrajectoryPlanner)


def test_gradients_flow_to_all_parameters(device):
    planner = BezierPlanner(embed_dim=EMBED_DIM).to(device)
    bev, vis, ego = _make_inputs(2, device)
    trajectory, ego_hidden = planner(bev, vis, ego)
    (trajectory.pow(2).mean() + ego_hidden.pow(2).mean()).backward()
    for name, p in planner.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
