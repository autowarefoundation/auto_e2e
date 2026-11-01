"""Tests for the GRPO update: basis, sampling, advantages, anchor and a toy policy.

    python -m pytest experiment/closed-loop/rl/test_grpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo import (  # noqa: E402
    group_advantages,
    grpo_loss,
    low_frequency_basis,
    plan_distance,
    sample_plans,
)

SIGMA = torch.tensor([0.5, 0.01])  # acceleration [m/s^2], curvature [1/m]


def test_basis_is_orthonormal_and_starts_constant() -> None:
    basis = low_frequency_basis(64, 4)
    assert basis.shape == (64, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-5)
    assert torch.allclose(basis[:, 0], torch.full((64,), 1 / 8.0))


def test_samples_differ_only_in_the_low_frequency_directions() -> None:
    basis = low_frequency_basis(64, 4)
    mu = torch.randn(64, 2)
    plans, z = sample_plans(mu, basis, SIGMA, 8, torch.Generator().manual_seed(0))
    assert plans.shape == (8, 64, 2) and z.shape == (8, 4, 2)
    residual = plans - mu
    projected = torch.einsum("tk,gkc->gtc", basis, torch.einsum("tk,gtc->gkc", basis, residual))
    assert torch.allclose(residual, projected, atol=1e-5)
    assert z[..., 1].std() < z[..., 0].std()  # per-signal scales


def test_group_advantages_are_centred_not_scaled() -> None:
    advantages = group_advantages(torch.tensor([1.0, 2.0, 3.0, 6.0]))
    assert torch.allclose(advantages, torch.tensor([-2.0, -1.0, 0.0, 3.0]))
    assert not group_advantages(torch.full((4,), -3.0)).any()


def test_gradient_moves_the_mean_towards_the_better_sample() -> None:
    basis = low_frequency_basis(64, 4)
    mu = torch.zeros(64, 2, requires_grad=True)
    z = torch.zeros(2, 4, 2)
    z[0, 1, 1] = SIGMA[1]  # sample 0: a slow curvature sweep
    z[1, 1, 1] = -SIGMA[1]
    loss, anchor = grpo_loss(mu, mu.detach(), z, torch.tensor([1.0, -1.0]), basis, SIGMA, 0.1)
    assert anchor == 0.0
    loss.backward()
    step = -mu.grad
    # Natural gradient: the step is the better sample's offset, in plan units.
    assert torch.allclose(step, basis @ z[0], atol=1e-7)


def test_natural_gradient_treats_both_signals_alike() -> None:
    basis = low_frequency_basis(64, 4)
    mu = torch.zeros(64, 2, requires_grad=True)
    z = torch.zeros(2, 4, 2)
    z[0, 1] = SIGMA  # one sigma in both signals
    z[1, 1] = -SIGMA
    loss, _ = grpo_loss(mu, mu.detach(), z, torch.tensor([1.0, -1.0]), basis, SIGMA, 0.0)
    loss.backward()
    coefficient = basis.T @ -mu.grad
    assert torch.allclose(coefficient[1] / SIGMA, torch.ones(2))


def test_anchor_covers_high_frequencies() -> None:
    basis = low_frequency_basis(64, 4)
    mu_ref = torch.zeros(64, 2)
    zigzag = low_frequency_basis(64, 64)[:, -1]  # highest DCT frequency, outside span(B)
    assert (basis.T @ zigzag).abs().max() < 1e-5
    mu = torch.stack([torch.zeros(64), 0.001 * zigzag], dim=1).requires_grad_()
    loss, anchor = grpo_loss(mu, mu_ref, torch.zeros(4, 4, 2), torch.zeros(4), basis, SIGMA, 1.0)
    assert anchor == pytest.approx(plan_distance(mu.detach(), mu_ref, SIGMA))
    assert anchor == pytest.approx((0.001 / 0.01) ** 2 / 2)
    loss.backward()
    assert torch.allclose(-mu.grad, mu_ref - mu.detach())  # pulled straight back


def test_grpo_steers_a_toy_policy_to_the_rewarded_plan() -> None:
    """A free mean plan learns an acceleration and a curvature sweep only the reward knows."""
    torch.manual_seed(0)
    basis = low_frequency_basis(64, 4)
    target = basis @ torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.02], [0.0, 0.0]])
    mu = torch.nn.Parameter(torch.zeros(64, 2))
    reference = mu.detach().clone()
    optimizer = torch.optim.SGD([mu], lr=1.0)
    generator = torch.Generator().manual_seed(1)

    def error() -> torch.Tensor:
        return (basis.T @ (mu.detach() - target)).norm(dim=0) / SIGMA

    start = error()
    for _ in range(300):
        plans, z = sample_plans(mu, basis, SIGMA, 8, generator)
        # Bounded in (0, 1], like the driving reward.
        rewards = torch.exp(-(((plans - target) / SIGMA) ** 2).sum(dim=(1, 2)) / 32.0)
        loss, _ = grpo_loss(mu, reference, z, group_advantages(rewards), basis, SIGMA, 1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = error()
    assert (end < 0.2 * start).all()
