"""GRPO over AutoE2E's control plan with low-frequency Gaussian exploration.

The GRU planner is deterministic: it outputs a mean plan ``mu`` of 64
[acceleration, curvature] steps. The policy adds noise only in a few
low-frequency directions of each signal, ``a = mu + B z`` with ``B`` the
first ``K`` orthonormal DCT-II vectors over the 64 steps and
``z ~ N(0, sigma_c^2)`` per signal ``c``. The sampled plans therefore differ by
smooth trajectory-level changes, not by step-to-step jitter. #211 asks for
this kind of structured exploration and describes low-frequency perturbations
in Qwen-Drive; this module does not follow a published recipe.

Within ``mu + span(B)`` the policy is an isotropic Gaussian in ``z``. Its
log-likelihood gradient with respect to the mean is ``B z / sigma^2``; the
Fisher information of the mean is ``1 / sigma^2``, so the natural gradient
is ``B z``: move the mean towards the sample, in the units of the plan. The
surrogate below has exactly that gradient with respect to the mean plan, so
the plan-space step does not favour the signal with the smaller ``sigma``.
It is natural in the mean plan only: the planner's parameters receive
``J^T B z``, with ``J`` the planner's Jacobian, and the trust region in
``grpo_step.py`` bounds the change that step causes.

The anchor to the imitation planner covers all 64 steps, so high-frequency
drift is held back as well. Its value, and the trust region of the update,
use the distance ``|(mu_a - mu_b) / sigma|^2 / 2`` (the KL of two policies
with these means, were the noise on every step). Its gradient is
preconditioned like the policy term: the KL's gradient is
``(mu_theta - mu_ref) / sigma^2`` and its natural gradient ``mu_theta - mu_ref``,
which is what the raw squared distance in ``grpo_loss`` gives. Both terms of
the loss are therefore in the same preconditioned units, and the line search
in ``grpo_step.py`` lowers that preconditioned surrogate.

GRPO (Shao et al., 2024, arXiv:2402.03300) standardises the rewards of a
group of samples drawn for the same state and needs no value network. With
one update per batch of samples the importance ratio is 1, so its clipped
surrogate reduces to the advantage-weighted likelihood.
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


def plan_distance(mu_a: torch.Tensor, mu_b: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """``|(mu_a - mu_b) / sigma|^2 / 2`` over every step of two ``[T, C]`` mean plans."""
    return (((mu_a - mu_b) / sigma.to(mu_a.device)) ** 2).sum() / 2.0


def grpo_loss(
    mu_theta: torch.Tensor,
    mu_ref: torch.Tensor,
    z: torch.Tensor,
    advantages: torch.Tensor,
    basis: torch.Tensor,
    sigma: torch.Tensor,
    anchor_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural-gradient GRPO loss for one state's group, and the anchor distance.

    ``mu_theta [T, C]`` carries the gradient; the samples ``z`` were drawn
    around ``mu_theta.detach()``.
    """
    shift = basis.to(mu_theta.device).T @ (mu_theta - mu_theta.detach())
    surrogate = (z * shift).sum(dim=(1, 2))
    pull = ((mu_theta - mu_ref) ** 2).sum() / 2.0
    loss = -(advantages.to(surrogate.device) * surrogate).mean() + anchor_weight * pull
    return loss, plan_distance(mu_theta.detach(), mu_ref, sigma)


__all__ = [
    "group_advantages",
    "grpo_loss",
    "low_frequency_basis",
    "plan_distance",
    "sample_plans",
]
