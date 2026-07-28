"""Rollout-aligned planner loss terms for AutoE2E control outputs."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
)
from training.losses.control_rollout import integrate_controls_torch
from training.losses.route_consistency_loss import (
    _outside_distance,
    _sample_field,
    ego_points_to_grid,
)


ROLLOUT_ALIGNED_LOSS_VERSION = "rollout_aligned_planner_v1"


def _structured_controls(controls: torch.Tensor) -> torch.Tensor:
    if controls.ndim == 2:
        if controls.shape[1] % 2:
            raise ValueError("flattened controls must have an even width")
        controls = controls.reshape(controls.shape[0], -1, 2)
    if controls.ndim != 3 or controls.shape[2] != 2:
        raise ValueError("controls must have shape [B,T,2] or [B,2T]")
    return controls


def _huber_distance(distance_m: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(
        distance_m,
        torch.zeros_like(distance_m),
        beta=1.0,
        reduction="none",
    )


def _barrier(values: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.relu(values.abs() / threshold - 1.0).square()


def comfort_excess_per_sample(
    predicted_controls: torch.Tensor,
    target_controls: torch.Tensor,
    predicted_speeds: torch.Tensor,
    target_speeds: torch.Tensor,
    *,
    dt: float = 0.1,
    jerk_threshold_mps3: float = 4.13,
    lateral_acceleration_threshold_mps2: float = 4.89,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target-relative comfort, jerk, and lateral peak excess."""
    predicted = _structured_controls(predicted_controls)
    target = _structured_controls(target_controls)
    if target.shape != predicted.shape:
        raise ValueError("predicted and target control shapes differ")
    if predicted.shape[1] < 2:
        raise ValueError("comfort loss requires at least two timesteps")
    if predicted_speeds.shape != predicted.shape[:2]:
        raise ValueError("predicted speeds must match control timesteps")
    if target_speeds.shape != target.shape[:2]:
        raise ValueError("target speeds must match control timesteps")
    with torch.autocast(
        device_type=predicted.device.type,
        enabled=False,
    ):
        predicted = predicted.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        predicted_speeds = predicted_speeds.to(dtype=torch.float32)
        target_speeds = target_speeds.to(dtype=torch.float32)
        predicted_jerk = (
            predicted[:, 1:, 0] - predicted[:, :-1, 0]
        ) / dt
        target_jerk = (
            target[:, 1:, 0] - target[:, :-1, 0]
        ) / dt
        predicted_lateral = (
            predicted_speeds.square() * predicted[:, :, 1]
        )
        target_lateral = target_speeds.square() * target[:, :, 1]

        predicted_jerk_peak = _barrier(
            predicted_jerk,
            jerk_threshold_mps3,
        ).amax(dim=1)
        target_jerk_peak = _barrier(
            target_jerk,
            jerk_threshold_mps3,
        ).amax(dim=1)
        predicted_lateral_peak = _barrier(
            predicted_lateral,
            lateral_acceleration_threshold_mps2,
        ).amax(dim=1)
        target_lateral_peak = _barrier(
            target_lateral,
            lateral_acceleration_threshold_mps2,
        ).amax(dim=1)
        jerk_excess = torch.relu(
            predicted_jerk_peak - target_jerk_peak.detach()
        )
        lateral_excess = torch.relu(
            predicted_lateral_peak - target_lateral_peak.detach()
        )
    return (
        0.5 * (jerk_excess + lateral_excess),
        jerk_excess,
        lateral_excess,
    )


def _footprint_corners(
    positions: torch.Tensor,
    headings: torch.Tensor,
    *,
    length_m: float,
    width_m: float,
) -> torch.Tensor:
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [B,T,2]")
    if headings.shape != positions.shape[:2]:
        raise ValueError("headings must match position timesteps")
    offsets = positions.new_tensor(
        [
            [length_m / 2.0, width_m / 2.0],
            [length_m / 2.0, -width_m / 2.0],
            [-length_m / 2.0, width_m / 2.0],
            [-length_m / 2.0, -width_m / 2.0],
        ]
    )
    cosine = torch.cos(headings).unsqueeze(-1)
    sine = torch.sin(headings).unsqueeze(-1)
    local_x = offsets[:, 0].view(1, 1, 4)
    local_y = offsets[:, 1].view(1, 1, 4)
    rotated = torch.stack(
        (
            cosine * local_x - sine * local_y,
            sine * local_x + cosine * local_y,
        ),
        dim=-1,
    )
    return positions.unsqueeze(2) + rotated


def _footprint_outside_distance(
    field: torch.Tensor,
    positions: torch.Tensor,
    headings: torch.Tensor,
    geometry: NavigationRasterGeometry,
    *,
    length_m: float,
    width_m: float,
) -> torch.Tensor:
    corners = _footprint_corners(
        positions,
        headings,
        length_m=length_m,
        width_m=width_m,
    )
    batch_size, timesteps, corner_count, _ = corners.shape
    flattened = corners.reshape(batch_size, timesteps * corner_count, 2)
    sampled = _sample_field(
        field,
        ego_points_to_grid(flattened, geometry),
        mode="bilinear",
        padding_mode="border",
    )
    sampled = sampled + _outside_distance(flattened, geometry)
    return sampled.reshape(
        batch_size,
        timesteps,
        corner_count,
    ).amax(dim=2)


class RolloutAlignedLoss(nn.Module):
    """Compute rollout and target-relative comfort/map planner terms."""

    def __init__(
        self,
        *,
        geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
        dt: float = 0.1,
        jerk_threshold_mps3: float = 4.13,
        lateral_acceleration_threshold_mps2: float = 4.89,
        footprint_length_m: float = 4.8,
        footprint_width_m: float = 2.0,
        map_tolerance_m: float | None = None,
    ) -> None:
        super().__init__()
        values = (
            dt,
            jerk_threshold_mps3,
            lateral_acceleration_threshold_mps2,
            footprint_length_m,
            footprint_width_m,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("rollout-aligned loss values must be positive")
        if map_tolerance_m is None:
            map_tolerance_m = 0.5 * geometry.meters_per_pixel
        if map_tolerance_m < 0.0:
            raise ValueError("map_tolerance_m must be non-negative")
        self.geometry = geometry
        self.dt = dt
        self.jerk_threshold_mps3 = jerk_threshold_mps3
        self.lateral_acceleration_threshold_mps2 = (
            lateral_acceleration_threshold_mps2
        )
        self.footprint_length_m = footprint_length_m
        self.footprint_width_m = footprint_width_m
        self.map_tolerance_m = map_tolerance_m

    def metadata(self) -> dict[str, object]:
        return {
            "version": ROLLOUT_ALIGNED_LOSS_VERSION,
            "dt": self.dt,
            "position_target_source": "packed_logged_xy",
            "rollout": {
                "path_weight": 0.75,
                "final_weight": 0.25,
                "huber_delta_m": 1.0,
            },
            "comfort": {
                "jerk_threshold_mps3": self.jerk_threshold_mps3,
                "lateral_acceleration_threshold_mps2": (
                    self.lateral_acceleration_threshold_mps2
                ),
                "comparison": "target_relative_peak_excess",
            },
            "map": {
                "footprint_length_m": self.footprint_length_m,
                "footprint_width_m": self.footprint_width_m,
                "target_relative_tolerance_m": self.map_tolerance_m,
                "raster_tolerance_pixels": (
                    self.map_tolerance_m
                    / self.geometry.meters_per_pixel
                ),
            },
        }

    def _comfort(
        self,
        predicted_controls: torch.Tensor,
        target_controls: torch.Tensor,
        predicted_speeds: torch.Tensor,
        target_speeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return comfort_excess_per_sample(
            predicted_controls,
            target_controls,
            predicted_speeds,
            target_speeds,
            dt=self.dt,
            jerk_threshold_mps3=self.jerk_threshold_mps3,
            lateral_acceleration_threshold_mps2=(
                self.lateral_acceleration_threshold_mps2
            ),
        )

    def _region_excess(
        self,
        field: torch.Tensor,
        predicted_positions: torch.Tensor,
        predicted_headings: torch.Tensor,
        target_positions: torch.Tensor,
        target_headings: torch.Tensor,
    ) -> torch.Tensor:
        predicted_distance = _footprint_outside_distance(
            field,
            predicted_positions,
            predicted_headings,
            self.geometry,
            length_m=self.footprint_length_m,
            width_m=self.footprint_width_m,
        )
        target_distance = _footprint_outside_distance(
            field,
            target_positions,
            target_headings,
            self.geometry,
            length_m=self.footprint_length_m,
            width_m=self.footprint_width_m,
        )
        return torch.relu(
            predicted_distance
            - target_distance.detach()
            - self.map_tolerance_m
        ).mean(dim=1)

    def forward(
        self,
        predicted_controls: torch.Tensor,
        target_controls: torch.Tensor,
        initial_speed: torch.Tensor,
        logged_positions: torch.Tensor,
        route_supervision: Mapping[str, torch.Tensor],
        map_valid: torch.Tensor,
        route_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted = _structured_controls(predicted_controls)
        target = _structured_controls(target_controls)
        if target.shape != predicted.shape:
            raise ValueError("predicted and target control shapes differ")
        batch_size = predicted.shape[0]
        if batch_size == 0:
            raise ValueError("rollout-aligned loss requires a non-empty batch")
        if initial_speed.shape != (batch_size,):
            raise ValueError("initial_speed must have shape [B]")
        if map_valid.shape != (batch_size,):
            raise ValueError("map_valid must have shape [B]")
        if route_valid.shape != (batch_size,):
            raise ValueError("route_valid must have shape [B]")
        if logged_positions.shape != (
            batch_size,
            predicted.shape[1],
            2,
        ):
            raise ValueError(
                "logged_positions must match control shape [B,T,2]"
            )
        with torch.autocast(
            device_type=predicted.device.type,
            enabled=False,
        ):
            logged_positions = logged_positions.to(
                device=predicted.device,
                dtype=torch.float32,
            )
            if not torch.isfinite(logged_positions).all():
                raise ValueError("logged_positions must be finite")

        (
            predicted_positions,
            predicted_headings,
            predicted_speeds,
        ) = integrate_controls_torch(
            predicted,
            initial_speed,
            dt=self.dt,
        )
        (
            _,
            target_headings,
            target_speeds,
        ) = integrate_controls_torch(
            target,
            initial_speed,
            dt=self.dt,
        )

        position_error = torch.linalg.vector_norm(
            predicted_positions - logged_positions,
            dim=2,
        )
        path_per_sample = _huber_distance(position_error).mean(dim=1)
        final_per_sample = _huber_distance(position_error[:, -1])
        rollout_per_sample = (
            0.75 * path_per_sample + 0.25 * final_per_sample
        )

        (
            comfort_per_sample,
            jerk_per_sample,
            lateral_per_sample,
        ) = self._comfort(
            predicted,
            target,
            predicted_speeds,
            target_speeds,
        )

        required_fields = {
            "distance_to_corridor_m",
            "distance_to_drivable_m",
            "available",
            "drivable_available",
        }
        missing = required_fields - set(route_supervision)
        if missing:
            raise ValueError(
                "navigation supervision is missing fields: "
                f"{sorted(missing)}"
            )
        expected_field_shape = (
            batch_size,
            self.geometry.height_px,
            self.geometry.width_px,
        )
        fields = {}
        for name in (
            "distance_to_corridor_m",
            "distance_to_drivable_m",
        ):
            field = route_supervision[name].to(
                device=predicted.device,
                dtype=torch.float32,
            )
            if field.shape != expected_field_shape:
                raise ValueError(
                    f"{name} differs from navigation geometry"
                )
            fields[name] = field
        artifact_available = route_supervision["available"].to(
            device=predicted.device,
            dtype=torch.bool,
        )
        if artifact_available.shape != (batch_size,):
            raise ValueError(
                "navigation supervision availability must have shape [B]"
            )
        drivable_artifact_available = route_supervision[
            "drivable_available"
        ].to(
            device=predicted.device,
            dtype=torch.bool,
        )
        if drivable_artifact_available.shape != (batch_size,):
            raise ValueError(
                "drivable supervision availability must have shape [B]"
            )

        route_active = (
            route_valid.to(device=predicted.device, dtype=torch.bool)
            & artifact_available
        )
        drivable_active = (
            map_valid.to(device=predicted.device, dtype=torch.bool)
            & artifact_available
            & drivable_artifact_available
        )
        for name, active in (
            ("distance_to_corridor_m", route_active),
            ("distance_to_drivable_m", drivable_active),
        ):
            if not torch.isfinite(fields[name][active]).all():
                raise ValueError(f"active {name} must be finite")
        route_field = torch.where(
            route_active[:, None, None],
            fields["distance_to_corridor_m"],
            torch.zeros_like(fields["distance_to_corridor_m"]),
        )
        drivable_field = torch.where(
            drivable_active[:, None, None],
            fields["distance_to_drivable_m"],
            torch.zeros_like(fields["distance_to_drivable_m"]),
        )
        route_per_sample = self._region_excess(
            route_field,
            predicted_positions,
            predicted_headings,
            logged_positions,
            target_headings,
        )
        drivable_per_sample = self._region_excess(
            drivable_field,
            predicted_positions,
            predicted_headings,
            logged_positions,
            target_headings,
        )
        route_weight = route_active.to(dtype=torch.float32)
        drivable_weight = drivable_active.to(dtype=torch.float32)
        map_term_count = route_weight + drivable_weight
        route_contribution = torch.where(
            route_active,
            route_per_sample,
            torch.zeros_like(route_per_sample),
        )
        drivable_contribution = torch.where(
            drivable_active,
            drivable_per_sample,
            torch.zeros_like(drivable_per_sample),
        )
        map_per_sample = (
            route_contribution + drivable_contribution
        ) / map_term_count.clamp_min(1.0)
        map_available = map_term_count > 0.0
        constraint_per_sample = (
            comfort_per_sample
            + map_per_sample
        ) / (
            1.0 + map_available.to(dtype=torch.float32)
        )
        zero = predicted.reshape(-1)[0].to(dtype=torch.float32) * 0.0

        def active_mean(
            values: torch.Tensor,
            active: torch.Tensor,
        ) -> torch.Tensor:
            weights = active.to(dtype=values.dtype)
            active_values = torch.where(
                active,
                values,
                torch.zeros_like(values),
            )
            return (
                active_values.sum()
                / weights.sum().clamp_min(1.0)
                + zero
            )

        return {
            "rollout": rollout_per_sample.mean(),
            "path": path_per_sample.mean(),
            "final": final_per_sample.mean(),
            "constraint": constraint_per_sample.mean(),
            "comfort": comfort_per_sample.mean(),
            "jerk": jerk_per_sample.mean(),
            "lateral_acceleration": lateral_per_sample.mean(),
            "map": active_mean(map_per_sample, map_available),
            "route": active_mean(route_per_sample, route_active),
            "drivable": active_mean(
                drivable_per_sample,
                drivable_active,
            ),
            "map_sample_count": map_available.sum(),
            "route_sample_count": route_active.sum(),
            "drivable_sample_count": drivable_active.sum(),
        }
