"""World-Model JEPA must NOT reshape the shared trajectory backbone (#13).

The WM shares the trajectory branch's image backbone (AutoE2E passes the same
instance to ReactiveE2E and to WorldActionModel). If the JEPA future-feature
loss backprops into that shared backbone, it pulls the representation toward
"predict future features" and away from "features the planner regresses a
trajectory from" — empirically flooring the trajectory loss at ~0.845 vs the
0.36 an imitation-only run reaches, regardless of batch size.

FrameEncoder.detach_backbone (default True) stop-gradients the backbone so JEPA
trains only the WM's own proj / aggregator / future_predictor. These tests pin
that contract: JEPA gradient must be ZERO on the shared backbone yet NONZERO on
the WM's predictor. Mock backbone, CPU, no shards.
"""

from __future__ import annotations

import torch

B, V, T, F = 2, 6, 4, 4


def _window():
    return torch.randn(B, T, V, 3, 256, 256), torch.randn(B, F, V, 3, 256, 256)


def test_jepa_grad_does_not_reach_shared_backbone(build_mock_model):
    """The JEPA loss alone must leave the SHARED backbone's grads at None/zero —
    otherwise it competes with the trajectory loss for the backbone weights."""
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"), enable_world_model=True)
    model.train()
    history_frames, future_frames = _window()

    _, aux = model(
        torch.randn(B, V, 3, 256, 256), torch.randn(B, 3, 256, 256),
        torch.zeros(B, 896), torch.randn(B, 256),
        mode="train", trajectory_target=torch.randn(B, 128),
        history_frames=history_frames, future_frames=future_frames)

    # JEPA loss ONLY (no trajectory term) — isolate what JEPA's grad touches.
    jepa = model.World_Action_Model_E2E.jepa_loss(
        aux["future_state_pred"], aux["future_frames"])
    jepa.backward()

    # The shared backbone lives under Reactive_E2E.Backbone. No parameter there
    # may carry gradient from the JEPA loss.
    backbone = model.Reactive_E2E.Backbone
    offenders = [n for n, p in backbone.named_parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not offenders, (
        f"JEPA gradient leaked into the shared backbone: {offenders[:5]}")


def test_jepa_still_trains_wm_predictor(build_mock_model):
    """Decoupling the backbone must NOT starve the WM: JEPA must still train the
    WM's own future_predictor / proj (the parts that are supposed to learn)."""
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"), enable_world_model=True)
    model.train()
    history_frames, future_frames = _window()

    _, aux = model(
        torch.randn(B, V, 3, 256, 256), torch.randn(B, 3, 256, 256),
        torch.zeros(B, 896), torch.randn(B, 256),
        mode="train", trajectory_target=torch.randn(B, 128),
        history_frames=history_frames, future_frames=future_frames)
    jepa = model.World_Action_Model_E2E.jepa_loss(
        aux["future_state_pred"], aux["future_frames"])
    jepa.backward()

    wam = model.World_Action_Model_E2E
    def _gn(module):
        tot = 0.0
        for p in module.parameters():
            if p.grad is not None:
                tot += float(p.grad.norm()) ** 2
        return tot ** 0.5

    assert _gn(wam.future_predictor) > 0, "JEPA did not train the future predictor"
    assert _gn(wam.encoder.proj) > 0, "JEPA did not train the WM's frame projection"


def test_default_detach_backbone_is_on():
    """The decoupling is the safe default (the shared-backbone case is the norm):
    a FrameEncoder built without the flag must detach."""
    import torch.nn as nn
    from model_components.world_action_model import FrameEncoder

    class _BB(nn.Module):
        def forward(self, x):
            return [torch.randn(x.shape[0], 768, 8, 8, requires_grad=True)]

    enc = FrameEncoder(_BB(), feature_channels=768, frame_embed_dim=224, num_views=1)
    assert enc.detach_backbone is True
