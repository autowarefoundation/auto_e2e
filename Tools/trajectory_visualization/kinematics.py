"""Canonical control contract and trajectory integration for AOVL reports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ControlContract:
    horizon_steps: int
    signal_names: tuple[str, ...]
    sampling_interval_s: float
    acceleration_unit: str
    curvature_unit: str
    speed_unit: str
    coordinate_frame: str

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if self.signal_names != ("acceleration", "curvature"):
            raise ValueError(
                "only acceleration/curvature controls are supported"
            )
        if (
            not math.isfinite(self.sampling_interval_s)
            or self.sampling_interval_s <= 0
        ):
            raise ValueError("sampling_interval_s must be finite and positive")
        if self.acceleration_unit != "m/s^2":
            raise ValueError("acceleration_unit must be m/s^2")
        if self.curvature_unit != "1/m":
            raise ValueError("curvature_unit must be 1/m")
        if self.speed_unit != "m/s":
            raise ValueError("speed_unit must be m/s")
        if self.coordinate_frame != "x_forward_y_left":
            raise ValueError(
                "coordinate_frame must be x_forward_y_left"
            )

    def manifest(self) -> dict[str, Any]:
        values = asdict(self)
        values["signal_names"] = list(self.signal_names)
        return values


AOVL_V1_CONTROL_CONTRACT = ControlContract(
    horizon_steps=64,
    signal_names=("acceleration", "curvature"),
    sampling_interval_s=0.1,
    acceleration_unit="m/s^2",
    curvature_unit="1/m",
    speed_unit="m/s",
    coordinate_frame="x_forward_y_left",
)


def curvature_sign_for_dataset(dataset: str) -> int:
    """Match the browser's L2D compass-heading correction."""
    return -1 if dataset in {"l2d", "yaak-ai/L2D"} else 1


def integrate_controls(
    controls: np.ndarray,
    v0: float,
    *,
    contract: ControlContract = AOVL_V1_CONTROL_CONTRACT,
    curvature_sign: int = 1,
) -> np.ndarray:
    """Integrate controls into metric points in the contract coordinate frame."""
    values = np.asarray(controls)
    expected = (contract.horizon_steps, len(contract.signal_names))
    if values.shape != expected:
        raise ValueError(f"controls must have shape {expected}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("controls must be floating point")
    if not np.isfinite(values).all() or not math.isfinite(v0):
        raise ValueError("controls and v0 must be finite")
    if curvature_sign not in {-1, 1}:
        raise ValueError("curvature_sign must be -1 or 1")

    points = np.empty((contract.horizon_steps, 2), dtype=np.float64)
    speed = float(v0)
    heading = 0.0
    x_forward = 0.0
    y_left = 0.0
    dt = contract.sampling_interval_s
    for index, (acceleration, curvature) in enumerate(values):
        speed = max(0.0, speed + float(acceleration) * dt)
        heading += curvature_sign * float(curvature) * speed * dt
        x_forward += speed * math.cos(heading) * dt
        y_left += speed * math.sin(heading) * dt
        points[index] = (x_forward, y_left)
    return points
