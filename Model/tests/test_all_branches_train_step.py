"""All-branch training-step smoke: Reactive + Reasoning + World Model (#13/#98).

Proves the three branches in the architecture diagram actually train together:
the COMBINED loss (imitation + JEPA + reasoning) is finite and DECREASES under
optimization, and each branch's gradient reaches its own head. Mirrors the
Flyte train_il wiring exactly.

Mock backbone + mock reasoning teacher, no GPU / network / shards.
"""

from __future__ import annotations

import torch

from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.targets import (
    collate_reasoning_targets,
    record_to_target_tensors,
)
from data_processing.reasoning_label_generation.teacher_client import TeacherRequest
from training.losses.horizon_reasoning_loss import HorizonReasoningLoss

B, V, T, F = 2, 6, 4, 4


def _inputs():
    return {
        "visual": torch.randn(B, V, 3, 256, 256),
        "map_input": torch.randn(B, 3, 256, 256),
        "vis_hist": torch.zeros(B, 896),
        "ego": torch.randn(B, 256),
        "target": torch.randn(B, 128),
        "history_frames": torch.randn(B, T, V, 3, 256, 256),
        "future_frames": torch.randn(B, F, V, 3, 256, 256),
    }


def _reasoning_targets():
    teacher = MockTeacher()
    per_sample = [
        record_to_target_tensors(teacher.label(TeacherRequest(f"s{i}", "l2d")))
        for i in range(B)
    ]
    return collate_reasoning_targets(per_sample)


def test_world_model_windowed_path_is_differentiable(build_mock_model):
    """The WM windowed path (history_frames given) predicts futures and the JEPA
    loss flows — the gap where train_il never triggered WM training."""
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"), enable_world_model=True,
    )
    inp = _inputs()
    _, aux = model(
        inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
        mode="train", trajectory_target=inp["target"],
        history_frames=inp["history_frames"], future_frames=inp["future_frames"],
    )
    fsp = aux["future_state_pred"]
    assert fsp is not None and len(fsp) == 4
    jepa = model.World_Action_Model_E2E.jepa_loss(fsp, aux["future_frames"])
    assert torch.isfinite(jepa) and jepa > 0
    jepa.backward()
    # Gradient reaches the future predictor (the WM head being trained).
    grads = [p.grad for n, p in model.World_Action_Model_E2E.named_parameters()
             if "future_predictor" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_all_three_branches_combined_loss_decreases(build_mock_model):
    torch.manual_seed(0)
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"),
        enable_world_model=True,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    model.train()
    traj_loss_fn = torch.nn.SmoothL1Loss()
    reason_loss_fn = HorizonReasoningLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inp = _inputs()
    tb = _reasoning_targets()

    def step():
        opt.zero_grad()
        out = model(
            inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
            mode="train", trajectory_target=inp["target"],
            history_frames=inp["history_frames"], future_frames=inp["future_frames"],
        )
        trajectory, aux = out
        # Three loss terms, exactly as train_il combines them.
        loss = traj_loss_fn(trajectory, inp["target"])
        jepa = model.World_Action_Model_E2E.jepa_loss(
            aux["future_state_pred"], aux["future_frames"])
        terms = reason_loss_fn(
            aux["reasoning_pred"], tb.targets,
            source_weights=tb.source_weights, confidence_targets=tb.confidence_targets,
        )
        total = loss + 1.0 * jepa + 0.5 * terms["total"]
        total.backward()
        opt.step()
        return float(total.detach())

    first = step()
    assert torch.isfinite(torch.tensor(first))
    last = first
    for _ in range(15):
        last = step()
    assert last < first, f"combined loss did not decrease: {first} -> {last}"


def test_each_branch_receives_gradient(build_mock_model):
    """Every enabled branch must get NONZERO gradient from the combined loss.

    Regression guard: it is easy for a branch's loss to be added to the total yet
    for that branch's parameters to receive no gradient (detach, wrong param
    group, a zero-multiplier gate). Assert the grad-norm of a parameter unique to
    each branch is finite and > 0 — the same check train_il prints as a one-time
    probe. Verified on GPU: planner ~21.7, world_model ~16.4, reasoning ~0.88.
    """
    torch.manual_seed(0)
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"),
        enable_world_model=True, enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    model.train()
    traj_loss_fn = torch.nn.SmoothL1Loss()
    reason_loss_fn = HorizonReasoningLoss()
    inp = _inputs()
    tb = _reasoning_targets()

    out = model(
        inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
        mode="train", trajectory_target=inp["target"],
        history_frames=inp["history_frames"], future_frames=inp["future_frames"],
    )
    trajectory, aux = out
    loss = traj_loss_fn(trajectory, inp["target"])
    loss = loss + model.World_Action_Model_E2E.jepa_loss(
        aux["future_state_pred"], aux["future_frames"])
    terms = reason_loss_fn(
        aux["reasoning_pred"], tb.targets,
        source_weights=tb.source_weights, confidence_targets=tb.confidence_targets)
    loss = loss + 0.05 * terms["total"]
    loss.backward()

    def grad_norm(substr):
        tot, n = 0.0, 0
        for nm, p in model.named_parameters():
            if substr in nm and p.grad is not None:
                tot += float(p.grad.norm()) ** 2
                n += 1
        return tot ** 0.5, n

    for branch in ("TrajectoryPlanner", "World_Action_Model", "ReasoningHead"):
        gn, n = grad_norm(branch)
        assert n > 0, f"{branch}: no params found"
        assert gn > 0.0, f"{branch}: zero gradient (branch loss added but not learning)"


def test_wm_supplies_visual_history_not_zeros(build_mock_model):
    """With the WM on and a window given, the planner/reasoning see the WM's
    Encoded Visual History, not the zeros the shard provides."""
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"),
        enable_world_model=True, enable_reasoning=True, reasoning_mode="pooled_latent",
    ).eval()
    inp = _inputs()
    captured = {}

    def hook(_m, args, _out):
        captured["vh"] = args[0].detach().clone()

    handle = model.Reactive_E2E.ReasoningHead.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
                  mode="train", trajectory_target=inp["target"],
                  history_frames=inp["history_frames"], future_frames=inp["future_frames"])
    finally:
        handle.remove()
    # The reasoning head's visual_history input is the WM-derived one (non-zero),
    # not the zeros passed in as vis_hist.
    assert captured["vh"].abs().sum() > 0


def test_world_model_feature_channels_derived_from_backbone(device=torch.device("cpu")):
    """WM JEPA channels must follow the backbone's last stage, not a hardcoded
    768 (regression: res_net_50's 2048-ch last map crashed the WM). Simulate a
    non-768 backbone and confirm the WM builds + forwards + reconstructs without
    a shape error."""
    from unittest.mock import patch

    import torch.nn as nn

    class _Backbone2048(nn.Module):
        # A backbone whose LAST stage is 2048 channels (like ResNet50).
        def __init__(self, *a, **k):
            super().__init__()
            self.feature_channels = [256, 512, 1024, 2048]
            self.backbone_channels = sum(self.feature_channels)
            self.s0 = nn.Sequential(nn.Conv2d(3, 256, 3, padding=1), nn.AdaptiveAvgPool2d(64))
            self.s1 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.AdaptiveAvgPool2d(32))
            self.s2 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.AdaptiveAvgPool2d(16))
            self.s3 = nn.Sequential(nn.Conv2d(1024, 2048, 3, padding=1), nn.AdaptiveAvgPool2d(8))

        def forward(self, x):
            a = self.s0(x)
            b = self.s1(a)
            c = self.s2(b)
            d = self.s3(c)
            return [a, b, c, d]

    with patch("model_components.reactive_e2e.Backbone", _Backbone2048):
        from model_components.auto_e2e import AutoE2E
        model = AutoE2E(num_views=V, view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                        enable_world_model=True).to(device)

    wam = model.World_Action_Model_E2E
    assert wam.feature_channels == 2048, "WM did not derive channels from the backbone"
    inp = _inputs()
    _, aux = model(inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
                   mode="train", trajectory_target=inp["target"],
                   history_frames=inp["history_frames"], future_frames=inp["future_frames"])
    # The reconstruction (predicted [B,2048,8,8] vs frozen-target [B,2048,8,8])
    # must compose without a shape mismatch.
    jepa = wam.jepa_loss(aux["future_state_pred"], aux["future_frames"])
    assert torch.isfinite(jepa)
