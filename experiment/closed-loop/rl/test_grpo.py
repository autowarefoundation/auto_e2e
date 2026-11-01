"""Tests for the GRPO update: basis, sampling, advantages and a toy policy.

    python -m pytest experiment/closed-loop/rl/test_grpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo import group_advantages, grpo_loss, low_frequency_basis, sample_plans  # noqa: E402

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


def test_group_advantages_are_standardised() -> None:
    advantages = group_advantages(torch.tensor([1.0, 2.0, 3.0, 6.0]))
    assert advantages.mean().abs() < 1e-6
    assert advantages.std(unbiased=False) == pytest.approx(1.0, abs=1e-4)
    assert not group_advantages(torch.full((4,), -3.0)).any()


def test_gradient_moves_the_mean_towards_the_better_sample() -> None:
    basis = low_frequency_basis(64, 4)
    mu = torch.zeros(64, 2, requires_grad=True)
    z = torch.zeros(2, 4, 2)
    z[0, 1, 1] = SIGMA[1]  # sample 0: a slow curvature sweep
    z[1, 1, 1] = -SIGMA[1]
    loss, kl = grpo_loss(mu, mu.detach(), z, torch.tensor([1.0, -1.0]), basis, SIGMA, 0.1)
    assert kl == 0.0
    loss.backward()
    step = -mu.grad
    assert (basis.T @ step)[1, 1] > 0  # along sample 0
    other = (basis.T @ step)[[0, 2, 3]]
    assert other.abs().max() < 1e-4 * (basis.T @ step)[1, 1]


def test_kl_pulls_the_mean_back_to_the_reference() -> None:
    basis = low_frequency_basis(64, 4)
    mu_ref = torch.zeros(64, 2)
    mu = (basis[:, :1] @ torch.tensor([[0.0, 0.02]])).requires_grad_()
    loss, kl = grpo_loss(mu, mu_ref, torch.zeros(4, 4, 2), torch.zeros(4), basis, SIGMA, 1.0)
    assert kl == pytest.approx(0.5 * (0.02 / 0.01) ** 2)
    loss.backward()
    assert (basis.T @ mu.grad)[0, 1] > 0  # descent lowers the offset


def test_grpo_steers_a_toy_policy_to_the_rewarded_plan() -> None:
    """A free mean plan learns a curvature sweep that only the reward knows.

    The reward sees curvature only, so acceleration gets pure-noise gradients;
    plain SGD leaves it in place.
    """
    torch.manual_seed(0)
    basis = low_frequency_basis(64, 4)
    target = basis @ torch.tensor([[0.0, 0.0], [0.0, 0.03], [0.0, 0.0], [0.0, 0.0]])
    mu = torch.nn.Parameter(torch.zeros(64, 2))
    reference = mu.detach().clone()
    optimizer = torch.optim.SGD([mu], lr=3e-6)
    generator = torch.Generator().manual_seed(1)

    def error() -> torch.Tensor:
        return (basis.T @ (mu.detach() - target)).norm(dim=0)

    start = error()
    for _ in range(300):
        plans, z = sample_plans(mu, basis, SIGMA, 8, generator)
        rewards = -(plans[..., 1] - target[..., 1]).norm(dim=1)
        loss, _ = grpo_loss(mu, reference, z, group_advantages(rewards), basis, SIGMA, 1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = error()
    assert end[1] < 0.1 * start[1]
    assert end[0] < 1e-3
