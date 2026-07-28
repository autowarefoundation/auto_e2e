"""Differentiable selected-route consistency objective for control outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
)
from navigation.supervision import ROUTE_SUPERVISION_ARTIFACT_VERSION
from training.losses.control_rollout import integrate_controls_torch


@dataclass(frozen=True)
class RouteConsistencyWeights:
    corridor: float = 1.0
    branch: float = 2.0
    destination: float = 0.5
    heading: float = 0.25

    def __post_init__(self) -> None:
        if any(
            value < 0.0
            for value in (
                self.corridor,
                self.branch,
                self.destination,
                self.heading,
            )
        ):
            raise ValueError("route consistency weights must be non-negative")

    def metadata(self) -> dict[str, float]:
        return {
            "corridor": self.corridor,
            "branch": self.branch,
            "destination": self.destination,
            "heading": self.heading,
        }


def ego_points_to_grid(
    points_xy_m: torch.Tensor,
    geometry: NavigationRasterGeometry,
) -> torch.Tensor:
    """Convert ego-FLU points to an align_corners=False sampling grid."""
    if points_xy_m.ndim != 3 or points_xy_m.shape[-1] != 2:
        raise ValueError("points_xy_m must have shape [B,T,2]")
    x_forward = points_xy_m[..., 0]
    y_left = points_xy_m[..., 1]
    row = (
        (geometry.x_max_m - x_forward) / geometry.meters_per_pixel
        - 0.5
    )
    col = (
        (geometry.y_max_m - y_left) / geometry.meters_per_pixel
        - 0.5
    )
    normalized_col = (
        2.0 * (col + 0.5) / geometry.width_px - 1.0
    )
    normalized_row = (
        2.0 * (row + 0.5) / geometry.height_px - 1.0
    )
    return torch.stack((normalized_col, normalized_row), dim=-1)


def _sample_field(
    field: torch.Tensor,
    grid: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError("route supervision fields must have shape [B,H,W]")
    if field.shape[0] != grid.shape[0]:
        raise ValueError("field and grid batch sizes differ")
    sampled = F.grid_sample(
        field.unsqueeze(1),
        grid.unsqueeze(2),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    return sampled[:, 0, :, 0]


def _inside_geometry(
    points_xy_m: torch.Tensor,
    geometry: NavigationRasterGeometry,
) -> torch.Tensor:
    x_forward = points_xy_m[..., 0]
    y_left = points_xy_m[..., 1]
    return (
        (x_forward >= geometry.x_min_m)
        & (x_forward <= geometry.x_max_m)
        & (y_left >= geometry.y_min_m)
        & (y_left <= geometry.y_max_m)
    )


def _outside_distance(
    points_xy_m: torch.Tensor,
    geometry: NavigationRasterGeometry,
) -> torch.Tensor:
    x_forward = points_xy_m[..., 0]
    y_left = points_xy_m[..., 1]
    dx = (
        torch.relu(geometry.x_min_m - x_forward)
        + torch.relu(x_forward - geometry.x_max_m)
    )
    dy = (
        torch.relu(geometry.y_min_m - y_left)
        + torch.relu(y_left - geometry.y_max_m)
    )
    return torch.hypot(dx, dy)


def _masked_sample_mean(
    values: torch.Tensor,
    point_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_mask = point_mask.to(dtype=torch.bool)
    sample_mask = sample_mask.to(dtype=torch.bool)
    active_samples = sample_mask & point_mask.any(dim=1)
    weights = point_mask.to(dtype=values.dtype)
    per_sample = (values * weights).sum(dim=1) / weights.sum(
        dim=1
    ).clamp_min(1.0)
    active_weights = active_samples.to(dtype=values.dtype)
    value = (per_sample * active_weights).sum() / active_weights.sum(
    ).clamp_min(1.0)
    return value + zero, active_samples.sum()


class RouteConsistencyLoss(nn.Module):
    """Penalize selected-route violations after integrating control outputs."""

    def __init__(
        self,
        *,
        geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
        temporal_decay: float = 0.99,
        target_compliance_threshold: float = 0.90,
        weights: RouteConsistencyWeights | None = None,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < temporal_decay <= 1.0:
            raise ValueError("temporal_decay must be in (0, 1]")
        if not 0.0 <= target_compliance_threshold <= 1.0:
            raise ValueError(
                "target_compliance_threshold must be in [0, 1]"
            )
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.geometry = geometry
        self.target_compliance_threshold = target_compliance_threshold
        self.weights = weights or RouteConsistencyWeights()
        self.dt = dt
        self.temporal_decay = temporal_decay

    def metadata(self) -> dict[str, object]:
        return {
            "artifact_version": ROUTE_SUPERVISION_ARTIFACT_VERSION,
            "target_compliance_threshold": (
                self.target_compliance_threshold
            ),
            "term_weights": self.weights.metadata(),
            "temporal_decay": self.temporal_decay,
            "temporal_weight_normalization": "mean_one",
        }

    def forward(
        self,
        predicted_controls: torch.Tensor,
        target_controls: torch.Tensor,
        initial_speed: torch.Tensor,
        route_supervision: Mapping[str, torch.Tensor],
        route_valid: torch.Tensor,
        route_intersection: torch.Tensor,
        *,
        route_quality_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = predicted_controls.shape[0]
        if target_controls.shape[0] != batch_size:
            raise ValueError("predicted and target batch sizes differ")
        if route_valid.shape != (batch_size,):
            raise ValueError("route_valid must have shape [B]")
        if route_intersection.shape != (batch_size,):
            raise ValueError("route_intersection must have shape [B]")
        if route_quality_valid is None:
            route_quality_valid = torch.ones_like(
                route_valid,
                dtype=torch.bool,
            )
        if route_quality_valid.shape != (batch_size,):
            raise ValueError("route_quality_valid must have shape [B]")

        predicted_positions, predicted_headings, predicted_speeds = (
            integrate_controls_torch(
                predicted_controls,
                initial_speed,
                dt=self.dt,
            )
        )
        target_positions, _, _ = integrate_controls_torch(
            target_controls,
            initial_speed,
            dt=self.dt,
        )
        timestep_count = predicted_positions.shape[1]
        if target_positions.shape[1] != timestep_count:
            raise ValueError("predicted and target horizons differ")

        required = {
            "distance_to_corridor_m",
            "route_heading_sin",
            "route_heading_cos",
            "route_heading_valid",
            "destination_xy_m",
            "destination_visible",
            "available",
        }
        missing = required - set(route_supervision)
        if missing:
            raise ValueError(
                f"route supervision is missing fields: {sorted(missing)}"
            )
        distance_field = route_supervision[
            "distance_to_corridor_m"
        ].to(
            device=predicted_controls.device,
            dtype=predicted_controls.dtype,
        )
        expected_raster_shape = (
            batch_size,
            self.geometry.height_px,
            self.geometry.width_px,
        )
        if distance_field.shape != expected_raster_shape:
            raise ValueError(
                "route distance field differs from navigation geometry"
            )
        available = route_supervision["available"].to(
            device=predicted_controls.device,
            dtype=torch.bool,
        )
        if available.shape != (batch_size,):
            raise ValueError("route supervision availability must have shape [B]")

        target_grid = ego_points_to_grid(
            target_positions,
            self.geometry,
        )
        target_inside = _inside_geometry(
            target_positions,
            self.geometry,
        )
        target_corridor = _sample_field(
            (distance_field <= self.geometry.meters_per_pixel * 0.5).to(
                dtype=predicted_controls.dtype
            ),
            target_grid,
            mode="nearest",
            padding_mode="zeros",
        ) > 0.5
        in_bounds_count = target_inside.sum(dim=1)
        compliance = (
            (target_corridor & target_inside).sum(dim=1).to(
                dtype=predicted_controls.dtype
            )
            / in_bounds_count.clamp_min(1).to(
                dtype=predicted_controls.dtype
            )
        )
        candidate = (
            route_valid.to(
                device=predicted_controls.device,
                dtype=torch.bool,
            )
            & available
            & route_quality_valid.to(
                device=predicted_controls.device,
                dtype=torch.bool,
            )
        )
        eligible = (
            candidate
            & (in_bounds_count > 0)
            & (compliance >= self.target_compliance_threshold)
        )

        zero = predicted_controls.sum() * 0.0
        prediction_grid = ego_points_to_grid(
            predicted_positions,
            self.geometry,
        )
        sampled_distance = _sample_field(
            distance_field,
            prediction_grid,
        )
        outside_distance = _outside_distance(
            predicted_positions,
            self.geometry,
        )
        temporal_weights = self.temporal_decay ** torch.arange(
            timestep_count,
            device=predicted_controls.device,
            dtype=predicted_controls.dtype,
        )
        temporal_weights = temporal_weights / temporal_weights.mean()
        corridor_values = (
            F.smooth_l1_loss(
                sampled_distance / 10.0,
                torch.zeros_like(sampled_distance),
                reduction="none",
            )
            + outside_distance / 10.0
        ) * temporal_weights.unsqueeze(0)
        corridor, corridor_count = _masked_sample_mean(
            corridor_values,
            torch.ones_like(corridor_values, dtype=torch.bool),
            eligible,
            zero,
        )

        late_start = timestep_count // 2
        branch_values = F.smooth_l1_loss(
            sampled_distance[:, late_start:]
            / self.geometry.route_corridor_width_m,
            torch.zeros_like(sampled_distance[:, late_start:]),
            reduction="none",
        )
        branch, branch_count = _masked_sample_mean(
            branch_values,
            torch.ones_like(branch_values, dtype=torch.bool),
            eligible
            & route_intersection.to(
                device=predicted_controls.device,
                dtype=torch.bool,
            ),
            zero,
        )

        destination = route_supervision["destination_xy_m"].to(
            device=predicted_controls.device,
            dtype=predicted_controls.dtype,
        )
        if destination.shape != (batch_size, 2):
            raise ValueError("route destination must have shape [B,2]")
        destination_visible = route_supervision[
            "destination_visible"
        ].to(
            device=predicted_controls.device,
            dtype=torch.bool,
        )
        predicted_terminal_distance = torch.linalg.vector_norm(
            predicted_positions[:, -1] - destination,
            dim=1,
        )
        target_terminal_distance = torch.linalg.vector_norm(
            target_positions[:, -1] - destination,
            dim=1,
        )
        destination_values = (
            torch.relu(
                predicted_terminal_distance
                - target_terminal_distance
                - 1.0
            )
            / 10.0
        ).unsqueeze(1)
        destination_term, destination_count = _masked_sample_mean(
            destination_values,
            torch.ones_like(destination_values, dtype=torch.bool),
            eligible & destination_visible,
            zero,
        )

        heading_sin = _sample_field(
            route_supervision["route_heading_sin"].to(
                device=predicted_controls.device,
                dtype=predicted_controls.dtype,
            ),
            prediction_grid,
        )
        heading_cos = _sample_field(
            route_supervision["route_heading_cos"].to(
                device=predicted_controls.device,
                dtype=predicted_controls.dtype,
            ),
            prediction_grid,
        )
        heading_valid = _sample_field(
            route_supervision["route_heading_valid"].to(
                device=predicted_controls.device,
                dtype=predicted_controls.dtype,
            ),
            prediction_grid,
            mode="nearest",
            padding_mode="zeros",
        ) > 0.5
        heading_values = 1.0 - (
            torch.cos(predicted_headings) * heading_cos
            + torch.sin(predicted_headings) * heading_sin
        )
        heading_mask = (
            heading_valid
            & (predicted_speeds >= 1.0)
            & _inside_geometry(predicted_positions, self.geometry)
        )
        heading, heading_count = _masked_sample_mean(
            heading_values,
            heading_mask,
            eligible,
            zero,
        )

        total = (
            self.weights.corridor * corridor
            + self.weights.branch * branch
            + self.weights.destination * destination_term
            + self.weights.heading * heading
        )
        return {
            "total": total,
            "corridor": corridor,
            "branch": branch,
            "destination": destination_term,
            "heading": heading,
            "candidate_count": candidate.sum(),
            "eligible_count": eligible.sum(),
            "compliance_rejected_count": (candidate & ~eligible).sum(),
            "corridor_active_count": corridor_count,
            "branch_active_count": branch_count,
            "destination_active_count": destination_count,
            "heading_active_count": heading_count,
            "target_compliance_sum": (
                compliance * candidate.to(dtype=compliance.dtype)
            ).sum(),
        }
