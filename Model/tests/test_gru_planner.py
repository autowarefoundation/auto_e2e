"""Contracts for the GRU planner."""

from __future__ import annotations

import torch

from model_components.trajectory_planning import GRUPlanner


def test_planner_decodes_every_step_and_trains_every_parameter():
    planner = GRUPlanner(
        embed_dim=8,
        num_timesteps=4,
        num_signals=2,
        num_points=2,
        egomotion_dim=4,
    )
    bev = torch.randn(2, 8, 4, 4, requires_grad=True)
    egomotion = torch.randn(2, 4, requires_grad=True)

    controls = planner(bev, egomotion)
    controls.square().mean().backward()

    assert controls.shape == (2, 4 * 2)
    assert bev.grad is not None
    assert egomotion.grad is not None
    assert all(parameter.grad is not None for parameter in planner.parameters())
