"""GRPO over AutoE2E's control plan with low-frequency Gaussian exploration.

The GRU planner is deterministic: it outputs a mean plan ``mu`` of 64
[acceleration, curvature] steps. The policy adds noise only in a few
low-frequency directions of each signal, ``a = mu + B z`` with ``B`` the
first ``K`` orthonormal DCT-II vectors over the 64 steps and
``z ~ N(0, sigma_c^2)`` per signal ``c``. The sampled plans therefore differ by
smooth trajectory-level changes, not by step-to-step jitter (#211, after the
structured low-frequency perturbations of Qwen-Drive).

Within ``mu + span(B)`` the policy is an isotropic Gaussian in ``z``, so for a
sample ``z_g`` drawn around the current ``mu``

    log pi(a_g) = -|z_g - B^T (mu_theta - mu)|^2 / (2 sigma^2) + const,

whose value is the sampling density and whose gradient moves ``mu_theta``
towards high-advantage samples. The KL to the imitation policy has the closed
form ``|B^T (mu_theta - mu_ref)|^2 / (2 sigma^2)``.

GRPO (Shao et al., 2024, arXiv:2402.03300) normalises the rewards of a group
of samples drawn for the same state and needs no value network. With one
gradient step per batch of samples the importance ratio is 1, so the clipped
surrogate reduces to the advantage-weighted log-likelihood used here.
"""

from __future__ import annotations

import math

import torch


def low_frequency_basis(num_steps: int, num_basis: int) -> torch.Tensor:
    """``[num_steps, num_basis]`` orthonormal DCT-II vectors, constant first."""
    steps = torch.arange(num_steps, dtype=torch.float64) + 0.5
    frequencies = torch.arange(num_basis, dtype=torch.float64)
    basis = torch.cos(math.pi * steps[:, None] * frequencies[None, :] / num_steps)
    basis[:, 0] /= math.sqrt(2.0)
    return (basis * math.sqrt(2.0 / num_steps)).float()


def sample_plans(
    mu: torch.Tensor,
    basis: torch.Tensor,
    sigma: torch.Tensor,
    group_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``group_size`` plans ``[G, T, C]`` around ``mu [T, C]`` and their ``z [G, K, C]``."""
    noise = torch.randn(
        (group_size, basis.shape[1], mu.shape[-1]), generator=generator, dtype=mu.dtype
    ).to(mu.device)
    z = noise * sigma.to(mu.device)
    return mu.detach() + torch.einsum("tk,gkc->gtc", basis.to(mu.device), z), z


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rewards of one group standardised within the group (zero when all are equal)."""
    centred = rewards - rewards.mean()
    return centred / (rewards.std(unbiased=False) + eps)


def grpo_loss(
    mu_theta: torch.Tensor,
    mu_ref: torch.Tensor,
    z: torch.Tensor,
    advantages: torch.Tensor,
    basis: torch.Tensor,
    sigma: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss for one state's group, and the KL to the reference policy.

    ``mu_theta [T, C]`` carries the gradient; the samples ``z`` were drawn
    around ``mu_theta.detach()``.
    """
    basis = basis.to(mu_theta.device)
    sigma = sigma.to(mu_theta.device)
    shift = basis.T @ (mu_theta - mu_theta.detach())
    log_prob = -(((z - shift) / sigma) ** 2).sum(dim=(1, 2)) / 2.0
    kl = (((basis.T @ (mu_theta - mu_ref)) / sigma) ** 2).sum() / 2.0
    return -(advantages.to(log_prob.device) * log_prob).mean() + kl_weight * kl, kl.detach()


__all__ = ["group_advantages", "grpo_loss", "low_frequency_basis", "sample_plans"]
