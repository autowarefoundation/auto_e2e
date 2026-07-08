"""Tests for zero-init reasoning→planner coupling (issue #98, R7).

Synthetic tensors, no GPU / network. Covers, for both Bezier and Flow-matching:
    * reasoning_mode="none" is byte-identical to no reasoning input;
    * pooled_latent and horizon_cross_attention are NO-OP at init (alpha=0),
      i.e. the trajectory equals the reasoning-off trajectory up to numerical
      tolerance;
    * after the gate is pushed off zero, the trajectory changes (coupling live);
    * the ReasoningCoupling module rejects an unknown mode.
"""

from __future__ import annotations

import pytest
import torch

from model_components.trajectory_planning.bezier_planner import BezierPlanner
from model_components.trajectory_planning.flow_matching_planner import FlowMatchingPlanner
from model_components.trajectory_planning.reasoning_coupling import (
    REASONING_MODES,
    ReasoningCoupling,
)

B, EMBED, HZ = 3, 256, 5


def _inputs(planner):
    bev = torch.randn(B, EMBED, 8, 8)
    vis = torch.randn(B, 896)
    ego = torch.randn(B, 256)
    latent = torch.randn(B, EMBED)
    tokens = torch.randn(B, HZ, EMBED)
    return bev, vis, ego, latent, tokens


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="reasoning_mode"):
        ReasoningCoupling(EMBED, mode="bogus")


def test_coupling_modes_constant():
    assert REASONING_MODES == ("none", "pooled_latent", "horizon_cross_attention")


@pytest.mark.parametrize("mode", ["pooled_latent", "horizon_cross_attention"])
def test_bezier_noop_at_init(mode):
    torch.manual_seed(0)
    planner = BezierPlanner(reasoning_mode=mode).eval()
    bev, vis, ego, latent, tokens = _inputs(planner)
    with torch.no_grad():
        base = planner(bev, vis, ego)
        coupled = planner(
            bev, vis, ego,
            reasoning_latent=latent, reasoning_horizon_tokens=tokens,
        )
    assert torch.allclose(base, coupled, atol=1e-6), f"{mode} not a no-op at init"


@pytest.mark.parametrize("mode", ["pooled_latent", "horizon_cross_attention"])
def test_bezier_active_after_gate_opens(mode):
    torch.manual_seed(0)
    planner = BezierPlanner(reasoning_mode=mode).eval()
    bev, vis, ego, latent, tokens = _inputs(planner)
    with torch.no_grad():
        planner.reasoning_coupling.alpha.fill_(1.0)  # open the gate
        # also perturb reason_proj's zero-init final layer so the residual is nonzero
        planner.reasoning_coupling.reason_proj[-1].weight.normal_()
        base = planner(bev, vis, ego)
        coupled = planner(
            bev, vis, ego,
            reasoning_latent=latent, reasoning_horizon_tokens=tokens,
        )
    assert not torch.allclose(base, coupled, atol=1e-5)


@pytest.mark.parametrize("mode", ["pooled_latent", "horizon_cross_attention"])
def test_flow_matching_noop_at_init(mode):
    torch.manual_seed(0)
    planner = FlowMatchingPlanner(reasoning_mode=mode, num_inference_steps=3).eval()
    bev, vis, ego, latent, tokens = _inputs(planner)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    with torch.no_grad():
        base = planner(bev, vis, ego, generator=g1)
        coupled = planner(
            bev, vis, ego, generator=g2,
            reasoning_latent=latent, reasoning_horizon_tokens=tokens,
        )
    assert torch.allclose(base, coupled, atol=1e-6), f"{mode} not a no-op at init"


def test_bezier_none_mode_ignores_reasoning():
    torch.manual_seed(0)
    planner = BezierPlanner(reasoning_mode="none").eval()
    bev, vis, ego, latent, tokens = _inputs(planner)
    with torch.no_grad():
        base = planner(bev, vis, ego)
        with_inputs = planner(
            bev, vis, ego,
            reasoning_latent=latent, reasoning_horizon_tokens=tokens,
        )
    assert torch.allclose(base, with_inputs, atol=1e-7)
