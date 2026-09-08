import pytest
import torch

from evaluation.reactive_open_loop import (
    open_loop_metrics_from_statistics,
    reactive_open_loop_statistics,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MapChannel,
    RouteChannel,
)
from training.losses.control_rollout import integrate_controls_torch


def _inputs():
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    controls = torch.zeros(1, 64, 2)
    speed = torch.tensor([5.0])
    positions, headings, speeds = integrate_controls_torch(controls, speed)
    map_context = torch.zeros(
        1,
        len(MapChannel),
        geometry.height_px,
        geometry.width_px,
    )
    map_context[:, MapChannel.DRIVABLE_AREA] = 1.0
    route_mask = torch.zeros(
        1,
        len(RouteChannel),
        geometry.height_px,
        geometry.width_px,
    )
    route_mask[:, RouteChannel.SELECTED_CORRIDOR] = 1.0
    return {
        "controls": controls,
        "predicted_xy": positions,
        "predicted_headings": headings,
        "predicted_speeds": speeds,
        "target_xy": positions.clone(),
        "complete": torch.ones(1, dtype=torch.bool),
        "map_context": map_context,
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_mask": route_mask,
        "route_valid": torch.ones(1, dtype=torch.bool),
    }


def test_open_loop_metrics_report_safe_straight_rollout():
    metrics = open_loop_metrics_from_statistics(
        reactive_open_loop_statistics(**_inputs())
    )

    assert metrics["comfortable_rate"] == 1.0
    assert metrics["drivable_area_compliance_rate"] == 1.0
    assert metrics["drivable_area_success_rate"] == 1.0
    assert metrics["route_corridor_compliance_rate"] == 1.0
    assert metrics["route_progress_proxy_ratio"] == pytest.approx(1.0)


def test_open_loop_metrics_detect_offroad_and_discomfort():
    inputs = _inputs()
    controls = inputs["controls"].clone()
    controls[:, ::2, 0] = 6.0
    controls[:, 1::2, 0] = -6.0
    positions, headings, speeds = integrate_controls_torch(
        controls,
        torch.tensor([5.0]),
    )
    inputs.update({
        "controls": controls,
        "predicted_xy": positions,
        "predicted_headings": headings,
        "predicted_speeds": speeds,
    })
    inputs["map_context"][:, MapChannel.DRIVABLE_AREA] = 0.0

    metrics = open_loop_metrics_from_statistics(
        reactive_open_loop_statistics(**inputs)
    )

    assert metrics["comfort_violation_rate"] == 1.0
    assert metrics["comfort_lon_accel_violation_rate"] == 1.0
    assert metrics["comfort_lon_jerk_violation_rate"] == 1.0
    assert metrics["drivable_area_compliance_rate"] == 0.0
    assert metrics["drivable_area_success_rate"] == 0.0


def test_open_loop_metrics_reject_navigation_geometry_mismatch():
    inputs = _inputs()
    inputs["map_context"] = torch.zeros(1, len(MapChannel), 3, 5)
    inputs["route_mask"] = torch.zeros(1, len(RouteChannel), 3, 5)

    with pytest.raises(ValueError, match="navigation geometry"):
        reactive_open_loop_statistics(**inputs)

