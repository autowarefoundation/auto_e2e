"""Shared differentiable rollout for AutoE2E control trajectories."""

from __future__ import annotations

import math

import torch


ROLLOUT_POLICY_VERSION = "semi_implicit_unicycle_v1"


def _assert_tensor(predicate: torch.Tensor, message: str) -> None:
    if predicate.device.type == "cuda":
        torch._assert_async(predicate, message)
    elif not bool(predicate.item()):
        raise ValueError(message)


def integrate_controls_torch(
    controls: torch.Tensor,
    initial_speed: torch.Tensor,
    *,
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate ``(acceleration, curvature)`` with float32 unicycle dynamics."""
    if controls.ndim == 2:
        if controls.shape[1] % 2:
            raise ValueError("flattened controls must have an even width")
        controls = controls.reshape(controls.shape[0], -1, 2)
    if controls.ndim != 3 or controls.shape[2] != 2:
        raise ValueError("controls must have shape [B,T,2] or [B,2T]")
    if controls.shape[1] <= 0:
        raise ValueError("controls must contain at least one timestep")
    if initial_speed.shape != (controls.shape[0],):
        raise ValueError("initial_speed must have shape [B]")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    _assert_tensor(
        torch.isfinite(controls).all(),
        "controls contain non-finite values",
    )
    _assert_tensor(
        torch.isfinite(initial_speed).all(),
        "initial_speed contains non-finite values",
    )
    _assert_tensor(
        (initial_speed >= 0).all(),
        "initial_speed must be non-negative",
    )

    with torch.autocast(device_type=controls.device.type, enabled=False):
        controls_f32 = controls.to(dtype=torch.float32)
        speed = initial_speed.to(
            device=controls.device,
            dtype=torch.float32,
        )
        heading = torch.zeros_like(speed)
        x = torch.zeros_like(speed)
        y = torch.zeros_like(speed)
        positions = []
        headings = []
        speeds = []
        for step in range(controls_f32.shape[1]):
            acceleration = controls_f32[:, step, 0]
            curvature = controls_f32[:, step, 1]
            speed = torch.clamp_min(speed + acceleration * dt, 0.0)
            heading = heading + speed * curvature * dt
            x = x + speed * torch.cos(heading) * dt
            y = y + speed * torch.sin(heading) * dt
            positions.append(torch.stack((x, y), dim=-1))
            headings.append(heading)
            speeds.append(speed)

        return (
            torch.stack(positions, dim=1),
            torch.stack(headings, dim=1),
            torch.stack(speeds, dim=1),
        )
