"""Shared differentiable rollout for AutoE2E control trajectories."""

from __future__ import annotations

import math

import torch


ROLLOUT_POLICY_VERSION = "semi_implicit_unicycle_v1"


def _assert_tensor(predicate: torch.Tensor, message: str) -> None:
    if not bool(predicate.detach().to(device="cpu").item()):
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
    with torch.autocast(device_type=controls.device.type, enabled=False):
        controls_f32 = controls.to(dtype=torch.float32)
        initial_speed_f32 = initial_speed.to(
            device=controls.device,
            dtype=torch.float32,
        )
        speed_delta = torch.cumsum(
            controls_f32[:, :, 0] * dt,
            dim=1,
        )
        # Lindley's recurrence is the vector form of clamp_min at every step.
        running_min = torch.cummin(speed_delta, dim=1).values
        reset_floor = torch.minimum(
            running_min,
            -initial_speed_f32.unsqueeze(1),
        )
        speeds = speed_delta - reset_floor
        headings = torch.cumsum(
            speeds * controls_f32[:, :, 1] * dt,
            dim=1,
        )
        positions = torch.stack(
            (
                torch.cumsum(speeds * torch.cos(headings) * dt, dim=1),
                torch.cumsum(speeds * torch.sin(headings) * dt, dim=1),
            ),
            dim=-1,
        )
        return positions, headings, speeds
