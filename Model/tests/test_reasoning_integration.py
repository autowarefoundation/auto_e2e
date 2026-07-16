"""AutoE2E ↔ reasoning-branch integration (issue #98, v2).

Mock backbone, no GPU / network. Covers:
    * enable_reasoning=False → byte-identical to the reactive baseline;
    * enable_reasoning=True at init → trajectory unchanged vs baseline up to
      numerical tolerance (zero-init coupling, the whole point of R7);
    * train mode returns (trajectory, aux) with aux["reasoning_pred"] a
      HorizonReasoningPrediction carrying 5-horizon logits + confidence;
    * inference mode returns only the trajectory even with reasoning on;
    * both pooled_latent and horizon_cross_attention modes wire end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from model_components.reasoning.types import HorizonReasoningPrediction

NUM_VIEWS = 7


def _inputs(B, device):
    return (
        torch.randn(B, NUM_VIEWS, 3, 256, 256, device=device),
        torch.randn(B, 3, 256, 256, device=device),
        torch.randn(B, 896, device=device),
        torch.randn(B, 256, device=device),
    )


def test_reasoning_off_is_baseline(build_mock_model, device):
    torch.manual_seed(0)
    base = build_mock_model(num_views=NUM_VIEWS, device=device).eval()
    torch.manual_seed(0)
    with_off = build_mock_model(
        num_views=NUM_VIEWS, device=device, enable_reasoning=False
    ).eval()
    cam, mp, vh, ego = _inputs(2, device)
    with torch.no_grad():
        a = base(cam, mp, vh, ego, mode="infer")
        b = with_off(cam, mp, vh, ego, mode="infer")
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("mode", ["pooled_latent", "horizon_cross_attention"])
def test_reasoning_on_is_noop_at_init(build_mock_model, device, mode):
    # Zero-init coupling: on ONE reasoning-on instance, running the head must
    # give the same trajectory as bypassing it (the coupling residual is 0 at
    # init). Comparing two SEPARATELY-built models would be wrong — building the
    # head consumes RNG and shifts the planner's weights.
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode=mode,
    ).eval()
    cam, mp, vh, ego = _inputs(2, device)
    with torch.no_grad():
        active = model(cam, mp, vh, ego, mode="infer")
        head = model.Reactive_E2E.ReasoningHead
        model.Reactive_E2E.ReasoningHead = None      # bypass the branch
        try:
            bypassed = model(cam, mp, vh, ego, mode="infer")
        finally:
            model.Reactive_E2E.ReasoningHead = head
    assert torch.allclose(active, bypassed, atol=1e-5), f"{mode} not no-op at init"


def test_train_mode_returns_reasoning_pred(build_mock_model, device):
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    cam, mp, vh, ego = _inputs(2, device)
    out = model(cam, mp, vh, ego, mode="train",
                trajectory_target=torch.randn(2, 128, device=device))
    assert isinstance(out, tuple) and len(out) == 2
    traj, aux = out
    assert traj.shape == (2, 128)
    pred = aux["reasoning_pred"]
    assert isinstance(pred, HorizonReasoningPrediction)
    assert pred.horizon_tokens.shape == (2, 5, 256)
    assert pred.confidence_logits.shape == (2, 5, 1)
    assert pred.cause_logits.shape[1] == 5


def test_infer_mode_returns_only_trajectory(build_mock_model, device):
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="horizon_cross_attention",
    ).eval()
    cam, mp, vh, ego = _inputs(2, device)
    with torch.no_grad():
        out = model(cam, mp, vh, ego, mode="infer")
    assert torch.is_tensor(out) and out.shape == (2, 128)
