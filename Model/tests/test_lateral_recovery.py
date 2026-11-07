"""Tests for the lateral recovery augmentation and its feature losses (#221)."""

import math

import pytest
import torch

from navigation.geometry import MapChannel, NavigationRasterGeometry
from training.lateral_recovery import (
    LateralShift,
    MAX_OFFSET_M,
    MAX_YAW_RAD,
    _frenet_slope,
    _history_camera_targets,
    _shift_map_context,
    _warp_images,
    aligned_clean_planner_bev,
    apply_lateral_recovery,
    ego_shift_transform,
    feature_consistency_kl,
    preserved_batch_norm_statistics,
    recovery_trajectory,
    sample_lateral_shift,
    shift_egomotion_history,
)

_GEOMETRY = NavigationRasterGeometry(
    geometry_id="lateral-recovery-test",
    height_px=36,
    width_px=24,
    meters_per_pixel=0.5,
    x_min_m=-6.0,
    x_max_m=12.0,
    y_min_m=-6.0,
    y_max_m=6.0,
    ego_anchor_row=23.5,
    ego_anchor_col=11.5,
    matching_pc_range=(-6.0, -6.0, -5.0, 12.0, 6.0, 3.0),
    matching_bev_h=36,
    matching_bev_w=24,
    route_corridor_width_m=3.5,
    destination_marker_radius_m=2.0,
    route_rear_clip_m=10.0,
)


def _shift(offset, yaw, active=None):
    offset = torch.as_tensor(offset, dtype=torch.float32).reshape(-1)
    yaw = torch.as_tensor(yaw, dtype=torch.float32).reshape(-1)
    if active is None:
        active = torch.ones_like(offset, dtype=torch.bool)
    return LateralShift(active=active, offset_m=offset, yaw_rad=yaw)


def _camera(yaw_deg, image_size=64, forward=1.5, height=1.6):
    """Ego-to-pixel matrix of a pinhole camera looking along ego yaw."""
    yaw = math.radians(yaw_deg)
    # Camera axes in ego: x right, y down, z forward.
    forward_axis = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
    right_axis = torch.tensor([math.sin(yaw), -math.cos(yaw), 0.0])
    down_axis = torch.tensor([0.0, 0.0, -1.0])
    rotation = torch.stack([right_axis, down_axis, forward_axis])
    centre = torch.tensor([forward, 0.0, height])
    extrinsic = torch.cat([rotation, (-rotation @ centre)[:, None]], dim=1)
    focal = image_size / 2.0
    intrinsic = torch.tensor([
        [focal, 0.0, image_size / 2.0],
        [0.0, focal, image_size / 2.0],
        [0.0, 0.0, 1.0],
    ])
    return intrinsic @ extrinsic


def _project(matrix, point):
    pixel = matrix @ torch.cat([point, point.new_ones(1)])
    return pixel[:2] / pixel[2]


def _straight_history(speed=10.0):
    signals = torch.zeros(1, 64, 4)
    signals[..., 0] = speed
    return signals.reshape(1, 256)


def _straight_future(step_m=1.0):
    x = step_m * torch.arange(1, 65, dtype=torch.float32)
    return torch.stack([x, torch.zeros_like(x)], dim=-1)[None]


def test_projection_shift_sees_shifted_points_where_original_saw_them():
    shift = _shift([0.7], [0.06])
    transform = ego_shift_transform(shift.offset_m, shift.yaw_rad)[0]
    camera = _camera(0.0)
    point_in_shifted = torch.tensor([9.0, -1.0, 0.3])
    point_in_original = (
        transform @ torch.cat([point_in_shifted, torch.ones(1)])
    )[:3]
    shifted_camera = camera @ transform
    assert torch.allclose(
        _project(shifted_camera, point_in_shifted),
        _project(camera, point_in_original),
        atol=1e-4,
    )


def test_image_warp_moves_a_ground_mark_to_the_shifted_camera_pixel():
    size = 64
    camera = _camera(0.0, image_size=size)
    shift = _shift([0.8], [0.05])
    to_shifted = torch.linalg.inv(
        ego_shift_transform(shift.offset_m, shift.yaw_rad)[0]
    )
    mark = torch.tensor([8.0, 1.0, 0.0])
    source_pixel = _project(camera, mark)
    rows = torch.arange(size, dtype=torch.float32) + 0.5
    v, u = torch.meshgrid(rows, rows, indexing="ij")
    blob = torch.exp(
        -((u - source_pixel[0]) ** 2 + (v - source_pixel[1]) ** 2) / 2.0
    )
    image = blob.expand(1, 3, size, size).clone()
    warped = _warp_images(image, camera[None], (camera @ to_shifted)[None])
    expected = _project(camera @ to_shifted, mark)
    peak = torch.nonzero(warped[0, 0] == warped[0, 0].max())[0].float() + 0.5
    assert torch.allclose(peak.flip(0), expected, atol=1.0)


def test_image_warp_with_identity_target_is_identity():
    camera = _camera(30.0)
    image = torch.rand(1, 3, 64, 64)
    warped = _warp_images(image, camera[None], camera[None])
    assert torch.allclose(warped, image, atol=1e-4)


def test_history_cameras_follow_the_past_displacement():
    rig = torch.stack([_camera(yaw) for yaw in (0, 55, -55, 110, 180, -110)])
    rows = torch.tensor([29, 34, 39, 44, 49, 54, 59])
    # Rows end 0.1 s before now, so a straight 10 m/s history puts row r at
    # x = -(64 - r) m.
    past_from_current = torch.eye(4).repeat(7, 1, 1)
    past_from_current[:, 0, 3] = (64 - rows).float()
    history = (rig[None] @ past_from_current[:, None])[None]
    lateral = torch.linspace(0.0, 0.6, 64)[None]
    heading = torch.linspace(0.0, 0.04, 64)[None]
    displaced = _history_camera_targets(
        history,
        _straight_history().reshape(1, 64, 4),
        lateral,
        heading,
    )
    moved_from_past = torch.linalg.inv(
        ego_shift_transform(lateral[0, rows], heading[0, rows])
    )
    expected = rig[None] @ (moved_from_past @ past_from_current)[:, None]
    assert torch.allclose(displaced[0], expected, rtol=1e-4, atol=1e-2)


def test_recovery_target_starts_at_the_shifted_ego_and_rejoins_the_path():
    future = _straight_future(step_m=0.05)
    shift = _shift([0.8], [0.06])
    target = recovery_trajectory(future, shift, torch.zeros(1))[0]
    first = target[0]
    # The first point is 5 cm ahead of the shifted ego along its heading.
    assert torch.allclose(first, torch.tensor([0.05, 0.0]), atol=2e-3)
    cos, sin = math.cos(0.06), math.sin(0.06)
    relative = future[0] - torch.tensor([0.0, 0.8])
    logged_in_shifted = torch.stack([
        cos * relative[:, 0] + sin * relative[:, 1],
        -sin * relative[:, 0] + cos * relative[:, 1],
    ], dim=-1)
    assert torch.allclose(target[24:], logged_in_shifted[24:], atol=1e-5)
    lateral_error = (target - logged_in_shifted).norm(dim=-1)
    assert bool((lateral_error[1:24] <= lateral_error[:23] + 1e-5).all())


def test_zero_shift_leaves_target_and_history_unchanged():
    future = torch.randn(2, 64, 2).cumsum(dim=1)
    history = torch.randn(2, 256)
    shift = _shift([0.0, 0.0], [0.0, 0.0])
    assert torch.equal(
        recovery_trajectory(future, shift, torch.zeros(2)),
        future,
    )
    shifted_history, _, _ = shift_egomotion_history(history, shift)
    assert torch.equal(shifted_history, history)


def test_history_drifts_into_the_shifted_pose():
    shift = _shift([0.8], [0.05])
    history, lateral, heading = shift_egomotion_history(
        _straight_history(),
        shift,
    )
    signals = history.reshape(64, 4)
    # The last row is 0.1 s (1 m) before the shifted current pose.
    assert lateral[0, -1].item() == pytest.approx(
        0.8 - math.tan(0.05),
        abs=5e-3,
    )
    assert heading[0, -1].item() == pytest.approx(0.05, abs=3e-3)
    turned = signals[:, 2].sum().item() * 0.1
    assert turned == pytest.approx(0.05, abs=0.01)
    assert torch.equal(signals[:40], _straight_history().reshape(64, 4)[:40])


def test_sampling_respects_probability_bounds_and_eligibility():
    future = torch.cat([_straight_future(), torch.zeros(1, 64, 2)])
    history = torch.cat([_straight_history(), _straight_history(0.0)])
    batch = {
        "trajectory_xy_m": future,
        "trajectory_valid": torch.ones(2, 64, dtype=torch.bool),
        "egomotion_history": history,
    }
    generator = torch.Generator().manual_seed(0)
    assert not bool(sample_lateral_shift(batch, torch.zeros(2, dtype=torch.bool), generator).active.any())
    shift = sample_lateral_shift(batch, torch.ones(2, dtype=torch.bool), generator)
    assert shift.active.tolist() == [True, False]
    assert abs(shift.offset_m[0].item()) <= MAX_OFFSET_M
    assert abs(shift.yaw_rad[0].item()) <= MAX_YAW_RAD
    assert shift.offset_m[1].item() == 0.0


def _camera_batch(batch_size=2, size=16):
    rig = torch.stack([_camera(yaw, image_size=size) for yaw in (0, 60, -60)])
    motion = torch.eye(4).repeat(7, 1, 1)
    motion[:, 0, 3] = -torch.arange(7, 0, -1, dtype=torch.float32) * 5.0
    history = rig[None] @ motion[:, None]
    return {
        "visual_tiles": torch.rand(batch_size, 3, 3, size, size),
        "camera_history_tiles": torch.rand(batch_size, 7, 3, 3, size, size),
        "camera_projection_matrix": rig.expand(batch_size, -1, -1, -1).clone(),
        "autobev_depth": torch.rand(batch_size, 3, size // 4, size // 4) + 1.0,
        "autobev_boxes": [torch.tensor([[10.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.1, 5.0, 0.0]])] * batch_size,
        "camera_history_projection_matrix": history.expand(
            batch_size, -1, -1, -1, -1
        ).clone(),
        "map_context": torch.rand(batch_size, 14, 450, 300),
        "route_mask": torch.rand(batch_size, 2, 450, 300),
        "bev_segmentation_target": torch.rand(batch_size, 2, 220, 160) > 0.5,
        "bev_segmentation_valid": torch.ones(
            batch_size, 2, 220, 160, dtype=torch.bool
        ),
        "egomotion_history": torch.cat(
            [_straight_history()] * batch_size
        ),
        "initial_speed_mps": torch.full((batch_size,), 10.0),
        "trajectory_xy_m": torch.cat([_straight_future()] * batch_size),
        "trajectory_valid": torch.ones(batch_size, 64, dtype=torch.bool),
    }


@pytest.mark.parametrize("method", ["projection", "image_warp"])
def test_inactive_samples_are_bitwise_unchanged(method):
    batch = _camera_batch()
    shift = _shift([0.6, 0.0], [0.04, 0.0], torch.tensor([True, False]))
    out = apply_lateral_recovery(
        batch,
        shift,
        method=method,
    )
    boxes = batch.pop("autobev_boxes")
    # An inactive sample's boxes go through the identity rotation: equal up to rounding.
    assert torch.allclose(out["autobev_boxes"][1], boxes[1])
    assert not torch.allclose(out["autobev_boxes"][0], boxes[0])
    for key, value in batch.items():
        assert torch.equal(out[key][1], value[1]), key
    changed = [
        key for key, value in batch.items()
        if not torch.equal(out[key][0], value[0])
    ]
    common = {
        "egomotion_history",
        "trajectory_xy_m",
        "map_context",
        "route_mask",
    }
    if method == "projection":
        expected = common | {
            "camera_projection_matrix",
            "camera_history_projection_matrix",
        }
    else:
        expected = common | {
            "visual_tiles",
            "camera_history_tiles",
            "autobev_depth",
            "camera_history_projection_matrix",
        }
    assert expected <= set(changed)
    if method == "image_warp":
        assert torch.equal(
            out["camera_projection_matrix"],
            batch["camera_projection_matrix"],
        )


def test_raster_mark_moves_to_its_shifted_cell():
    raster = torch.zeros(1, 14, 36, 24)
    # Ego point (x=4.25 m, y=1.25 m) is row 15, column 9.
    raster[0, 0, 15, 9] = 1.0
    shifted = _shift_map_context(raster, _shift([1.0], [0.0]), _GEOMETRY)
    # One metre to the left moves the point one metre right in E': two columns.
    peak = torch.nonzero(shifted[0, 0] == 1.0)
    assert peak.tolist() == [[15, 11]]


def test_map_traffic_direction_rotates_into_the_shifted_frame():
    raster = torch.zeros(1, 14, 36, 24)
    raster[0, MapChannel.TRAFFIC_DIRECTION_SIN] = 0.5
    raster[0, MapChannel.TRAFFIC_DIRECTION_COS] = 1.0
    raster[0, MapChannel.TRAFFIC_DIRECTION_VALID] = 1.0
    shifted = _shift_map_context(raster, _shift([0.0], [math.radians(5.0)]), _GEOMETRY)
    cell = shifted[0, :, 18, 12]
    assert cell[MapChannel.TRAFFIC_DIRECTION_SIN].item() == pytest.approx(
        (math.sin(math.radians(-5.0)) + 1.0) / 2.0, abs=1e-5
    )
    assert cell[MapChannel.TRAFFIC_DIRECTION_COS].item() == pytest.approx(
        (math.cos(math.radians(-5.0)) + 1.0) / 2.0, abs=1e-5
    )


def test_heading_offset_matches_the_yaw_on_curved_paths():
    offset = torch.tensor([1.0])
    yaw = torch.tensor([math.radians(5.0)])
    curvature = torch.tensor([0.1])
    slope = _frenet_slope(offset, yaw, curvature)
    heading = torch.atan2(slope, 1.0 - curvature * offset)
    assert heading.item() == pytest.approx(yaw.item(), abs=1e-6)


def test_planner_bev_alignment_uses_its_own_resolution_and_full_support():
    shift = _shift([0.1], [0.0])
    # The planner BEV covers the AutoBEV extent, here at a coarser 24x16 grid.
    teacher = torch.zeros(1, 2, 24, 16)
    teacher[:, 0] = 4.0
    aligned, mask = aligned_clean_planner_bev(teacher, shift)
    assert aligned.shape == teacher.shape
    assert bool(mask.any())
    loss = feature_consistency_kl(teacher.clone(), aligned, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_teacher_forward_keeps_batch_norm_statistics():
    norm = torch.nn.BatchNorm2d(3).train()
    before = [norm.running_mean.clone(), norm.num_batches_tracked.clone()]
    with torch.no_grad(), preserved_batch_norm_statistics(norm):
        norm(torch.randn(4, 3, 5, 5) + 2.0)
    assert torch.equal(norm.running_mean, before[0])
    assert torch.equal(norm.num_batches_tracked, before[1])
    norm(torch.randn(4, 3, 5, 5) + 2.0)
    assert not torch.equal(norm.running_mean, before[0])


def test_consistency_kl_trains_only_the_shifted_branch():
    clean = torch.randn(2, 4, 36, 24, requires_grad=True)
    shift = _shift([0.5, 0.0], [0.03, 0.0], torch.tensor([True, False]))
    aligned, inside = aligned_clean_planner_bev(clean, shift)
    # The trainer compares only the shifted samples (distortion_mix.aligned_teacher_bev).
    mask = inside & shift.active[:, None, None]
    assert bool(inside[1].any()) and not bool(mask[1].any())
    assert torch.allclose(aligned[1], clean.detach()[1], atol=1e-5)
    same = feature_consistency_kl(aligned.clone(), aligned, mask)
    assert same.item() == pytest.approx(0.0, abs=1e-6)
    shifted = torch.randn(2, 4, 36, 24, requires_grad=True)
    loss = feature_consistency_kl(shifted, aligned, mask)
    loss.backward()
    assert loss.item() > 0.0
    assert clean.grad is None
    assert shifted.grad[0].abs().sum() > 0.0
    assert torch.equal(shifted.grad[1], torch.zeros_like(shifted.grad[1]))
