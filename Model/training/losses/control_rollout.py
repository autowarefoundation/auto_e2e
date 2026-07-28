"""Shared differentiable rollout for AutoE2E control trajectories."""

from __future__ import annotations

import math

import torch


ROLLOUT_POLICY_VERSION = "semi_implicit_unicycle_v1"


def _assert_tensor(predicate: torch.Tensor, message: str) -> None:
    if not bool(predicate.detach().to(device="cpu").item()):
        raise ValueError(message)


def _integrate_controls_f32(
    controls: torch.Tensor,
    initial_speed: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speed = initial_speed
    heading = torch.zeros_like(speed)
    x = torch.zeros_like(speed)
    y = torch.zeros_like(speed)
    positions = []
    headings = []
    speeds = []
    for step in range(controls.shape[1]):
        speed = torch.clamp_min(
            speed + controls[:, step, 0] * dt,
            0.0,
        )
        heading = heading + speed * controls[:, step, 1] * dt
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


_compiled_integrate_controls_f32 = torch.compile(
    _integrate_controls_f32,
    fullgraph=True,
    dynamic=False,
)


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
    with torch.autocast(device_type=controls.device.type, enabled=False):
        controls_f32 = controls.to(dtype=torch.float32)
        initial_speed_f32 = initial_speed.to(
            device=controls.device,
            dtype=torch.float32,
        )
        integration = (
            _compiled_integrate_controls_f32
            if controls.device.type == "cuda"
            else _integrate_controls_f32
        )
        return integration(controls_f32, initial_speed_f32, dt)
