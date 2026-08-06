"""Per-sample logged-XY metrics for rollout-aligned checkpoint selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
    RouteChannel,
)
from training.losses.control_rollout import integrate_controls_torch
from training.losses.rollout_aligned_loss import (
    _footprint_outside_distance,
    comfort_excess_per_sample,
)


ROLLOUT_VALIDATION_VERSION = "rollout_validation_v2"
ROLLOUT_VALIDATION_HORIZON_STEPS = 30


def build_rollout_validation_records(
    predicted_controls: torch.Tensor,
    target_controls: torch.Tensor,
    initial_speeds_mps: torch.Tensor,
    logged_xy_m: torch.Tensor | np.ndarray,
    route_supervision: Mapping[str, torch.Tensor],
    map_valid: torch.Tensor,
    route_valid: torch.Tensor,
    sample_uids: Sequence[str],
    split_group_uids: Sequence[str],
    *,
    route_mask: torch.Tensor | None = None,
    route_intersections: Sequence[bool] | None = None,
    geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
    footprint_length_m: float = 4.8,
    footprint_width_m: float = 2.0,
) -> list[dict[str, object]]:
    """Build one complete selector record per validation sample."""
    predicted = predicted_controls.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    target = target_controls.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if predicted.ndim == 2:
        predicted = predicted.reshape(predicted.shape[0], -1, 2)
    if target.ndim == 2:
        target = target.reshape(target.shape[0], -1, 2)
    if (
        predicted.ndim != 3
        or predicted.shape[2] != 2
        or target.shape != predicted.shape
    ):
        raise ValueError(
            "predicted and target controls must share shape [B,T,2]"
        )
    batch_size, timestep_count, _ = predicted.shape
    if timestep_count < ROLLOUT_VALIDATION_HORIZON_STEPS:
        raise ValueError(
            "rollout validation requires at least 30 timesteps"
        )
    evaluation_slice = slice(0, ROLLOUT_VALIDATION_HORIZON_STEPS)
    predicted_evaluation = predicted[:, evaluation_slice]
    target_evaluation = target[:, evaluation_slice]
    speeds = initial_speeds_mps.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if speeds.shape != (batch_size,):
        raise ValueError("initial speeds must have shape [B]")
    logged = torch.as_tensor(
        logged_xy_m,
        dtype=torch.float32,
        device="cpu",
    )
    if logged.shape != (batch_size, timestep_count, 2):
        raise ValueError(
            "logged XY must match the control batch and horizon"
        )
    if (
        len(sample_uids) != batch_size
        or len(split_group_uids) != batch_size
    ):
        raise ValueError("validation identities must match batch size")
    if any(not str(uid) for uid in sample_uids):
        raise ValueError("validation sample UIDs must be non-empty")
    if any(not str(uid) for uid in split_group_uids):
        raise ValueError("validation group UIDs must be non-empty")
    intersections = (
        [False] * batch_size
        if route_intersections is None
        else [bool(value) for value in route_intersections]
    )
    if len(intersections) != batch_size:
        raise ValueError("route intersections must match batch size")

    with torch.no_grad():
        (
            predicted_xy,
            predicted_headings,
            predicted_speeds,
        ) = integrate_controls_torch(predicted_evaluation, speeds)
        _, target_headings, target_speeds = integrate_controls_torch(
            target_evaluation,
            speeds,
        )
        comfort, _, _ = comfort_excess_per_sample(
            predicted_evaluation,
            target_evaluation,
            predicted_speeds,
            target_speeds,
        )
        fields = {}
        expected_field_shape = (
            batch_size,
            geometry.height_px,
            geometry.width_px,
        )
        for name in (
            "distance_to_corridor_m",
            "distance_to_drivable_m",
        ):
            field = route_supervision[name].detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            if field.shape != expected_field_shape:
                raise ValueError(
                    f"{name} differs from validation geometry"
                )
            fields[name] = field
        available = route_supervision["available"].detach().to(
            device="cpu",
            dtype=torch.bool,
        )
        if available.shape != (batch_size,):
            raise ValueError(
                "navigation supervision availability must have shape [B]"
            )
        drivable_available = route_supervision[
            "drivable_available"
        ].detach().to(
            device="cpu",
            dtype=torch.bool,
        )
        if drivable_available.shape != (batch_size,):
            raise ValueError(
                "drivable supervision availability must have shape [B]"
            )
        map_available = (
            map_valid.detach().to(device="cpu", dtype=torch.bool)
            & available
            & drivable_available
        )
        route_available = (
            route_valid.detach().to(device="cpu", dtype=torch.bool)
            & available
        )
        sampling_fields = {}
        for name, active in (
            ("distance_to_corridor_m", route_available),
            ("distance_to_drivable_m", map_available),
        ):
            if not torch.isfinite(fields[name][active]).all():
                raise ValueError(f"active {name} must be finite")
            sampling_fields[name] = torch.where(
                active[:, None, None],
                fields[name],
                torch.zeros_like(fields[name]),
            )
        if route_mask is None:
            routes = torch.zeros(
                (
                    batch_size,
                    len(RouteChannel),
                    geometry.height_px,
                    geometry.width_px,
                ),
                dtype=torch.float32,
            )
            routes[:, RouteChannel.SELECTED_CORRIDOR] = (
                sampling_fields["distance_to_corridor_m"] <= 0.0
            )
        else:
            routes = route_mask.detach().to(
                device="cpu",
                dtype=torch.float32,
            )
        if routes.shape != (
            batch_size,
            len(RouteChannel),
            geometry.height_px,
            geometry.width_px,
        ):
            raise ValueError(
                "route mask differs from validation geometry"
            )
        predicted_drivable_distance = _footprint_outside_distance(
            sampling_fields["distance_to_drivable_m"],
            predicted_xy,
            predicted_headings,
            geometry,
            length_m=footprint_length_m,
            width_m=footprint_width_m,
        )
        target_drivable_distance = _footprint_outside_distance(
            sampling_fields["distance_to_drivable_m"],
            logged[:, evaluation_slice],
            target_headings,
            geometry,
            length_m=footprint_length_m,
            width_m=footprint_width_m,
        )
        predicted_route_distance = _footprint_outside_distance(
            sampling_fields["distance_to_corridor_m"],
            predicted_xy,
            predicted_headings,
            geometry,
            length_m=footprint_length_m,
            width_m=footprint_width_m,
        )
        target_route_distance = _footprint_outside_distance(
            sampling_fields["distance_to_corridor_m"],
            logged[:, evaluation_slice],
            target_headings,
            geometry,
            length_m=footprint_length_m,
            width_m=footprint_width_m,
        )

    logged_evaluation = logged[:, evaluation_slice]
    errors = torch.linalg.vector_norm(
        predicted_xy - logged_evaluation,
        dim=2,
    )
    raster_tolerance_m = 0.5 * geometry.meters_per_pixel
    predicted_drivable_inside = (
        predicted_drivable_distance <= raster_tolerance_m
    )
    target_drivable_inside = (
        target_drivable_distance <= raster_tolerance_m
    )
    predicted_route_inside = (
        predicted_route_distance <= raster_tolerance_m
    )
    target_route_inside = (
        target_route_distance <= raster_tolerance_m
    )
    destination = route_supervision["destination_xy_m"].detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    destination_visible = route_supervision[
        "destination_visible"
    ].detach().to(device="cpu", dtype=torch.bool)
    if destination.shape != (batch_size, 2):
        raise ValueError("destination_xy_m must have shape [B,2]")
    if destination_visible.shape != (batch_size,):
        raise ValueError(
            "destination_visible must have shape [B]"
        )

    records = []
    for index in range(batch_size):
        offroad_excess = None
        predicted_offroad = None
        target_offroad = None
        if bool(map_available[index]):
            predicted_offroad = float(
                (~predicted_drivable_inside[index]).float().mean()
            )
            target_offroad = float(
                (~target_drivable_inside[index]).float().mean()
            )
            offroad_excess = max(
                0.0,
                predicted_offroad - target_offroad,
            )

        route_gap = None
        wrong_branch_excess = None
        destination_error = None
        predicted_compliance = None
        target_compliance = None
        if bool(route_available[index]):
            from evaluation.navigation_metrics import (
                _mask_values_at_positions,
            )

            predicted_compliance = float(
                predicted_route_inside[index].float().mean()
            )
            target_compliance = float(
                target_route_inside[index].float().mean()
            )
            route_gap = max(
                0.0,
                target_compliance - predicted_compliance,
            )
            corridor = (
                routes[index, RouteChannel.SELECTED_CORRIDOR].numpy()
                > 0.0
            )
            predicted_center_on_route = _mask_values_at_positions(
                corridor,
                predicted_xy[index].numpy(),
                geometry,
            )
            target_center_on_route = _mask_values_at_positions(
                corridor,
                logged_evaluation[index].numpy(),
                geometry,
            )
            if intersections[index] and bool(target_center_on_route[-1]):
                wrong_branch_excess = float(
                    not bool(predicted_center_on_route[-1])
                )
        if (
            bool(available[index])
            and bool(destination_visible[index])
        ):
            predicted_terminal = float(torch.linalg.vector_norm(
                predicted_xy[index, -1] - destination[index]
            ))
            target_terminal = float(torch.linalg.vector_norm(
                logged_evaluation[index, -1] - destination[index]
            ))
            destination_error = abs(
                predicted_terminal - target_terminal
            )

        records.append({
            "sample_uid": str(sample_uids[index]),
            "split_group_uid": str(split_group_uids[index]),
            "ade_3s_m": float(errors[index].mean()),
            "fde_3s_m": float(errors[index, -1]),
            "comfort_excess": float(comfort[index]),
            "offroad_excess": offroad_excess,
            "route_gap": route_gap,
            "wrong_branch_excess": wrong_branch_excess,
            "destination_error_m": destination_error,
            "diagnostic_predicted_offroad_rate": predicted_offroad,
            "diagnostic_target_offroad_rate": target_offroad,
            "diagnostic_predicted_route_compliance": (
                predicted_compliance
            ),
            "diagnostic_target_route_compliance": target_compliance,
            "diagnostic_raster_tolerance_m": raster_tolerance_m,
        })
    return records
