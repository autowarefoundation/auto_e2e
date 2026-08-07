"""Numerical stability contracts for the Reactive GRU planner."""

from __future__ import annotations

import torch

from model_components.trajectory_planning import GRUPlanner


def test_lookup_query_stops_only_recurrent_coordinate_feedback():
    planner = GRUPlanner(
        embed_dim=8,
        num_timesteps=4,
        num_points=2,
        egomotion_dim=4,
        visual_history_dim=6,
    )
    hidden = torch.randn(2, 8, requires_grad=True)
    ego_query = planner.ego_query.weight

    planner._lookup_query(hidden, ego_query).sum().backward()

    assert hidden.grad is None
    assert ego_query.grad is not None
    assert torch.equal(
        ego_query.grad,
        torch.full_like(ego_query, 2.0),
    )


def test_stabilized_lookup_keeps_planner_parameters_trainable():
    planner = GRUPlanner(
        embed_dim=8,
        num_timesteps=4,
        num_points=2,
        egomotion_dim=4,
        visual_history_dim=6,
    )
    bev = torch.randn(2, 8, 4, 4, requires_grad=True)
    visual_history = torch.randn(2, 6, requires_grad=True)
    egomotion = torch.randn(2, 4, requires_grad=True)

    planner(bev, visual_history, egomotion).square().mean().backward()

    assert bev.grad is not None
    assert visual_history.grad is not None
    assert egomotion.grad is not None
    assert planner.reference_point.weight.grad is not None
    assert planner.sampling_offsets.weight.grad is not None
    assert planner.attention_weights.weight.grad is not None
    assert planner.ego_query.weight.grad is not None
    assert planner.gru.weight_hh_l0.grad is not None
