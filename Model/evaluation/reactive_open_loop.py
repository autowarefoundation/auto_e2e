"""Open-loop safety, navigation, and comfort metrics for Reactive planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch

from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MapChannel,
    NavigationRasterGeometry,
    RouteChannel,
)


REACTIVE_OPEN_LOOP_METRICS_VERSION: Final = "reactive_open_loop_metrics_v1"
EGO_FOOTPRINT_LENGTH_M: Final = 4.8
EGO_FOOTPRINT_WIDTH_M: Final = 2.0
COMFORT_THRESHOLDS: Final = {
    "lon_accel_max": 2.40,
    "lon_accel_min": -4.05,
    "lat_accel": 4.89,
    "yaw_rate": 0.95,
    "yaw_accel": 1.93,
    "lon_jerk": 4.13,
    "mag_jerk": 8.37,
}
OPEN_LOOP_STATISTIC_NAMES: Final = (
    "comfort_supported_samples",
    "comfort_violation_samples",
    "comfort_lon_accel_violation_samples",
    "comfort_lat_accel_violation_samples",
    "comfort_yaw_rate_violation_samples",
    "comfort_yaw_accel_violation_samples",
    "comfort_lon_jerk_violation_samples",
    "comfort_mag_jerk_violation_samples",
    "comfort_max_lon_accel_sum",
    "comfort_min_lon_accel_sum",
    "comfort_max_lat_accel_sum",
    "comfort_max_yaw_rate_sum",
    "comfort_max_yaw_accel_sum",
    "comfort_max_lon_jerk_sum",
    "comfort_max_mag_jerk_sum",
    "drivable_supported_samples",
    "drivable_compliant_steps",
    "drivable_total_steps",
    "drivable_success_samples",
    "route_supported_samples",
    "route_compliant_steps",
    "route_total_steps",
    "route_success_samples",
    "route_progress_m_sum",
    "route_progress_ratio_sum",
    "route_progress_supported_samples",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _footprint_inside_mask(
    mask: torch.Tensor,
    positions: torch.Tensor,
    headings: torch.Tensor,
    geometry: NavigationRasterGeometry,
    *,
    length_m: float = EGO_FOOTPRINT_LENGTH_M,
    width_m: float = EGO_FOOTPRINT_WIDTH_M,
) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError("footprint mask must have shape [B,H,W]")
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must have shape [B,T,2]")
    if headings.shape != positions.shape[:2]:
        raise ValueError("headings must have shape [B,T]")
    if mask.shape[0] != positions.shape[0] or mask.shape[1:] != (
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError("footprint mask differs from navigation geometry")

    corners = positions.new_tensor([
        [length_m / 2.0, width_m / 2.0],
        [length_m / 2.0, -width_m / 2.0],
        [-length_m / 2.0, width_m / 2.0],
        [-length_m / 2.0, -width_m / 2.0],
    ])
    cosine = torch.cos(headings)[..., None]
    sine = torch.sin(headings)[..., None]
    corner_x = (
        positions[..., 0, None]
        + cosine * corners[:, 0]
        - sine * corners[:, 1]
    )
    corner_y = (
        positions[..., 1, None]
        + sine * corners[:, 0]
        + cosine * corners[:, 1]
    )
    rows = torch.round(
        (geometry.x_max_m - corner_x) / geometry.meters_per_pixel
        - 0.5
    ).to(torch.long)
    columns = torch.round(
        (geometry.y_max_m - corner_y) / geometry.meters_per_pixel
        - 0.5
    ).to(torch.long)
    in_bounds = (
        (rows >= 0)
        & (rows < geometry.height_px)
        & (columns >= 0)
        & (columns < geometry.width_px)
    )
    flat_indices = (
        rows.clamp(0, geometry.height_px - 1) * geometry.width_px
        + columns.clamp(0, geometry.width_px - 1)
    )
    sampled = torch.gather(
        mask.to(dtype=torch.bool).reshape(mask.shape[0], -1),
        1,
        flat_indices.reshape(mask.shape[0], -1),
    ).reshape_as(flat_indices)
    return (in_bounds & sampled).all(dim=2)


def _route_progress(
    predicted_terminal: torch.Tensor,
    target_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted_terminal.ndim != 2 or predicted_terminal.shape[1] != 2:
        raise ValueError("predicted terminal positions must have shape [B,2]")
    if target_xy.ndim != 3 or target_xy.shape[2] != 2:
        raise ValueError("target trajectory must have shape [B,T,2]")
    if target_xy.shape[0] != predicted_terminal.shape[0]:
        raise ValueError("route progress batch dimensions differ")

    origin = target_xy.new_zeros((target_xy.shape[0], 1, 2))
    starts = torch.cat((origin, target_xy[:, :-1]), dim=1)
    segments = target_xy - starts
    lengths = torch.linalg.vector_norm(segments, dim=2)
    squared_lengths = segments.square().sum(dim=2)
    offset = predicted_terminal[:, None, :] - starts
    projection = torch.where(
        squared_lengths > 1e-8,
        (offset * segments).sum(dim=2) / squared_lengths.clamp_min(1e-8),
        torch.zeros_like(squared_lengths),
    ).clamp(0.0, 1.0)
    projected = starts + projection[..., None] * segments
    distance_squared = (
        predicted_terminal[:, None, :] - projected
    ).square().sum(dim=2)
    distance_squared = torch.where(
        lengths > 1e-4,
        distance_squared,
        torch.full_like(distance_squared, float("inf")),
    )
    best_segment = distance_squared.argmin(dim=1)
    cumulative_start = torch.cumsum(lengths, dim=1) - lengths
    progress = torch.gather(
        cumulative_start + projection * lengths,
        1,
        best_segment[:, None],
    ).squeeze(1)
    target_length = lengths.sum(dim=1)
    supported = target_length > 1.0
    ratio = torch.where(
        supported,
        progress / target_length.clamp_min(1e-6),
        torch.zeros_like(progress),
    ).clamp(0.0, 1.0)
    return progress, ratio, supported


def reactive_open_loop_statistics(
    controls: torch.Tensor,
    predicted_xy: torch.Tensor,
    predicted_headings: torch.Tensor,
    predicted_speeds: torch.Tensor,
    target_xy: torch.Tensor,
    complete: torch.Tensor,
    map_context: torch.Tensor,
    map_valid: torch.Tensor,
    route_mask: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    dt: float = 0.1,
) -> torch.Tensor:
    """Return additive statistics for distributed open-loop evaluation."""
    if controls.ndim != 3 or controls.shape[2] != 2:
        raise ValueError("controls must have shape [B,T,2]")
    expected_trajectory = controls.shape[:2]
    if (
        predicted_xy.shape != (*expected_trajectory, 2)
        or predicted_headings.shape != expected_trajectory
        or predicted_speeds.shape != expected_trajectory
        or target_xy.shape != (*expected_trajectory, 2)
        or complete.shape != (controls.shape[0],)
    ):
        raise ValueError("open-loop trajectory tensors differ in shape")
    if map_context.shape != (
        controls.shape[0],
        len(MapChannel),
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError("map context differs from navigation geometry")
    if route_mask.shape != (
        controls.shape[0],
        len(RouteChannel),
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError("route mask differs from navigation geometry")
    if map_valid.shape != complete.shape or route_valid.shape != complete.shape:
        raise ValueError("map and route validity must have shape [B]")

    complete = complete.to(dtype=torch.bool)
    statistics = controls.new_zeros(
        len(OPEN_LOOP_STATISTIC_NAMES),
        dtype=torch.float64,
    )
    values = {
        name: statistics[index]
        for index, name in enumerate(OPEN_LOOP_STATISTIC_NAMES)
    }

    comfort_supported = complete
    acceleration = controls[..., 0]
    curvature = controls[..., 1]
    lateral_acceleration = predicted_speeds.square() * curvature
    yaw_rate = predicted_speeds * curvature
    longitudinal_jerk = torch.diff(acceleration, dim=1) / dt
    lateral_jerk = torch.diff(lateral_acceleration, dim=1) / dt
    yaw_acceleration = torch.diff(yaw_rate, dim=1) / dt
    magnitude_jerk = torch.hypot(longitudinal_jerk, lateral_jerk)
    comfort_peaks = {
        "comfort_max_lon_accel_sum": acceleration.max(dim=1).values,
        "comfort_min_lon_accel_sum": acceleration.min(dim=1).values,
        "comfort_max_lat_accel_sum": lateral_acceleration.abs().max(
            dim=1
        ).values,
        "comfort_max_yaw_rate_sum": yaw_rate.abs().max(dim=1).values,
        "comfort_max_yaw_accel_sum": yaw_acceleration.abs().max(
            dim=1
        ).values,
        "comfort_max_lon_jerk_sum": longitudinal_jerk.abs().max(
            dim=1
        ).values,
        "comfort_max_mag_jerk_sum": magnitude_jerk.max(dim=1).values,
    }
    comfort_component_violations = {
        "comfort_lon_accel_violation_samples": (
            (comfort_peaks["comfort_max_lon_accel_sum"]
             > COMFORT_THRESHOLDS["lon_accel_max"])
            | (comfort_peaks["comfort_min_lon_accel_sum"]
               < COMFORT_THRESHOLDS["lon_accel_min"])
        ),
        "comfort_lat_accel_violation_samples": (
            comfort_peaks["comfort_max_lat_accel_sum"]
            > COMFORT_THRESHOLDS["lat_accel"]
        ),
        "comfort_yaw_rate_violation_samples": (
            comfort_peaks["comfort_max_yaw_rate_sum"]
            > COMFORT_THRESHOLDS["yaw_rate"]
        ),
        "comfort_yaw_accel_violation_samples": (
            comfort_peaks["comfort_max_yaw_accel_sum"]
            > COMFORT_THRESHOLDS["yaw_accel"]
        ),
        "comfort_lon_jerk_violation_samples": (
            comfort_peaks["comfort_max_lon_jerk_sum"]
            > COMFORT_THRESHOLDS["lon_jerk"]
        ),
        "comfort_mag_jerk_violation_samples": (
            comfort_peaks["comfort_max_mag_jerk_sum"]
            > COMFORT_THRESHOLDS["mag_jerk"]
        ),
    }
    comfort_violation = torch.stack(
        tuple(comfort_component_violations.values()),
        dim=1,
    ).any(dim=1)
    values["comfort_supported_samples"] += comfort_supported.sum()
    values["comfort_violation_samples"] += (
        comfort_violation & comfort_supported
    ).sum()
    for name, per_sample in comfort_component_violations.items():
        values[name] += (per_sample & comfort_supported).sum()
    for name, per_sample in comfort_peaks.items():
        values[name] += per_sample[comfort_supported].sum()

    route_valid_samples = complete & route_valid.to(dtype=torch.bool)
    drivable_supported = complete & map_valid.to(dtype=torch.bool)
    drivable_inside = _footprint_inside_mask(
        map_context[:, MapChannel.DRIVABLE_AREA] >= 0.5,
        predicted_xy,
        predicted_headings,
        geometry,
    )
    values["drivable_supported_samples"] += drivable_supported.sum()
    values["drivable_compliant_steps"] += (
        drivable_inside & drivable_supported[:, None]
    ).sum()
    values["drivable_total_steps"] += (
        drivable_supported.sum() * controls.shape[1]
    )
    values["drivable_success_samples"] += (
        drivable_inside.all(dim=1) & drivable_supported
    ).sum()

    route_inside = _footprint_inside_mask(
        route_mask[:, RouteChannel.SELECTED_CORRIDOR] >= 0.5,
        predicted_xy,
        predicted_headings,
        geometry,
    )
    values["route_supported_samples"] += route_valid_samples.sum()
    values["route_compliant_steps"] += (
        route_inside & route_valid_samples[:, None]
    ).sum()
    values["route_total_steps"] += (
        route_valid_samples.sum() * controls.shape[1]
    )
    values["route_success_samples"] += (
        route_inside.all(dim=1) & route_valid_samples
    ).sum()
    progress_m, progress_ratio, progress_available = _route_progress(
        predicted_xy[:, -1],
        target_xy,
    )
    progress_supported = route_valid_samples & progress_available
    values["route_progress_m_sum"] += progress_m[progress_supported].sum()
    values["route_progress_ratio_sum"] += (
        progress_ratio[progress_supported].sum()
    )
    values["route_progress_supported_samples"] += (
        progress_supported.sum()
    )

    return statistics


def open_loop_metrics_from_statistics(
    statistics: torch.Tensor,
) -> dict[str, float | str]:
    """Convert globally reduced additive statistics to report metrics."""
    if statistics.shape != (len(OPEN_LOOP_STATISTIC_NAMES),):
        raise ValueError("open-loop statistics have invalid shape")
    raw: Mapping[str, float] = {
        name: float(statistics[index].item())
        for index, name in enumerate(OPEN_LOOP_STATISTIC_NAMES)
    }
    comfort_count = raw["comfort_supported_samples"]
    drivable_count = raw["drivable_supported_samples"]
    route_count = raw["route_supported_samples"]
    progress_count = raw["route_progress_supported_samples"]
    return {
        "open_loop_metrics_version": REACTIVE_OPEN_LOOP_METRICS_VERSION,
        "comfort_metrics_available": float(comfort_count > 0.0),
        "comfort_supported_samples": comfort_count,
        "comfort_violation_rate": _safe_ratio(
            raw["comfort_violation_samples"],
            comfort_count,
        ),
        "comfort_lon_accel_violation_rate": _safe_ratio(
            raw["comfort_lon_accel_violation_samples"],
            comfort_count,
        ),
        "comfort_lat_accel_violation_rate": _safe_ratio(
            raw["comfort_lat_accel_violation_samples"],
            comfort_count,
        ),
        "comfort_yaw_rate_violation_rate": _safe_ratio(
            raw["comfort_yaw_rate_violation_samples"],
            comfort_count,
        ),
        "comfort_yaw_accel_violation_rate": _safe_ratio(
            raw["comfort_yaw_accel_violation_samples"],
            comfort_count,
        ),
        "comfort_lon_jerk_violation_rate": _safe_ratio(
            raw["comfort_lon_jerk_violation_samples"],
            comfort_count,
        ),
        "comfort_mag_jerk_violation_rate": _safe_ratio(
            raw["comfort_mag_jerk_violation_samples"],
            comfort_count,
        ),
        "comfortable_rate": _safe_ratio(
            comfort_count - raw["comfort_violation_samples"],
            comfort_count,
        ),
        "comfort_mean_max_lon_accel_mps2": _safe_ratio(
            raw["comfort_max_lon_accel_sum"],
            comfort_count,
        ),
        "comfort_mean_min_lon_accel_mps2": _safe_ratio(
            raw["comfort_min_lon_accel_sum"],
            comfort_count,
        ),
        "comfort_mean_max_lat_accel_mps2": _safe_ratio(
            raw["comfort_max_lat_accel_sum"],
            comfort_count,
        ),
        "comfort_mean_max_yaw_rate_radps": _safe_ratio(
            raw["comfort_max_yaw_rate_sum"],
            comfort_count,
        ),
        "comfort_mean_max_yaw_accel_radps2": _safe_ratio(
            raw["comfort_max_yaw_accel_sum"],
            comfort_count,
        ),
        "comfort_mean_max_lon_jerk_mps3": _safe_ratio(
            raw["comfort_max_lon_jerk_sum"],
            comfort_count,
        ),
        "comfort_mean_max_jerk_mps3": _safe_ratio(
            raw["comfort_max_mag_jerk_sum"],
            comfort_count,
        ),
        "drivable_metrics_available": float(drivable_count > 0.0),
        "drivable_supported_samples": drivable_count,
        "drivable_area_compliance_rate": _safe_ratio(
            raw["drivable_compliant_steps"],
            raw["drivable_total_steps"],
        ),
        "drivable_area_success_rate": _safe_ratio(
            raw["drivable_success_samples"],
            drivable_count,
        ),
        "route_corridor_metrics_available": float(route_count > 0.0),
        "route_corridor_supported_samples": route_count,
        "route_corridor_compliance_rate": _safe_ratio(
            raw["route_compliant_steps"],
            raw["route_total_steps"],
        ),
        "route_corridor_success_rate": _safe_ratio(
            raw["route_success_samples"],
            route_count,
        ),
        "route_progress_proxy_metrics_available": float(
            progress_count > 0.0
        ),
        "route_progress_proxy_supported_samples": progress_count,
        "route_progress_proxy_m": _safe_ratio(
            raw["route_progress_m_sum"],
            progress_count,
        ),
        "route_progress_proxy_ratio": _safe_ratio(
            raw["route_progress_ratio_sum"],
            progress_count,
        ),
    }
