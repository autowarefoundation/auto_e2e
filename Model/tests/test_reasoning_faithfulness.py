"""Tests for reasoning faithfulness / intervention eval (issue #98, R14).

Mock backbone, no GPU / network. Covers:
    * reasoning_intervention_delta uses the current AutoE2E ABI and is ~0 at init
      (zero-init coupling → bypassing the head does not move the trajectory);
    * it raises without a reasoning head;
    * horizon interventions (zero_all / zero_1s / shuffle) run and are ~0 at init;
    * an unknown intervention is rejected;
    * the rolling buffer is restored (no state leak) — run twice, same delta.
"""

from __future__ import annotations

import pytest
import torch

from evaluation.faithfulness import (
    horizon_intervention_delta,
    reasoning_intervention_delta,
)

NUM_VIEWS = 7


def _inputs(B=2, device=None):
    # Inputs must live on the SAME device as the model (the fixture may build on
    # CUDA). The faithfulness eval helpers don't move inputs, so a CPU input into
    # a CUDA model raises "Input type ... and weight type ... should be the same".
    return tuple(
        t.to(device) for t in (
            torch.randn(B, NUM_VIEWS, 3, 256, 256),
            torch.randn(B, 3, 256, 256),
            torch.randn(B, 896),
            torch.randn(B, 256),
        )
    )


def test_intervention_delta_zero_at_init(build_mock_model, device):
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    out = reasoning_intervention_delta(model, *_inputs(device=device))
    assert out["trajectory_l2"] == pytest.approx(0.0, abs=1e-5)


def test_intervention_requires_reasoning(build_mock_model, device):
    model = build_mock_model(num_views=NUM_VIEWS, device=device)
    with pytest.raises(ValueError, match="enable_reasoning=True"):
        reasoning_intervention_delta(model, *_inputs(device=device))


@pytest.mark.parametrize("intervention", ["zero_all", "zero_1s", "zero_2s", "shuffle"])
def test_horizon_interventions_zero_at_init(build_mock_model, device, intervention):
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="horizon_cross_attention",
    )
    out = horizon_intervention_delta(model, *_inputs(device=device), intervention=intervention)
    # Coupling gate is zero at init → any horizon-token perturbation is a no-op.
    assert out["trajectory_l2"] == pytest.approx(0.0, abs=1e-5)


def test_unknown_intervention_rejected(build_mock_model, device):
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="horizon_cross_attention",
    )
    with pytest.raises(ValueError, match="unknown intervention"):
        horizon_intervention_delta(model, *_inputs(device=device), intervention="bogus")


def test_intervention_does_not_leak_buffer_state(build_mock_model, device):
    # World Model on so there is a rolling buffer to (not) leak.
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
        enable_world_model=True,
    )
    inp = _inputs(device=device)
    d1 = reasoning_intervention_delta(model, *inp)
    d2 = reasoning_intervention_delta(model, *inp)
    assert d1["trajectory_l2"] == pytest.approx(d2["trajectory_l2"], abs=1e-6)
