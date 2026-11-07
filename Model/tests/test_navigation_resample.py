import math

import torch

from model_components.autobev.geometry import AUTOBEV_GEOMETRY
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY, MAP_CHANNEL_COUNT, MapChannel
from navigation.resample import compose_poses, resample_map_context, resample_route, sampling_grid


def _raster_with_point(x_m: float, y_m: float, channel: int = MapChannel.CROSSWALK) -> torch.Tensor:
    raster = torch.zeros(1, MAP_CHANNEL_COUNT, 450, 300)
    row, col = AUTOE2E_NAVIGATION_GEOMETRY.ego_to_pixel(torch.tensor([[x_m, y_m]]).numpy())[0]
    raster[0, channel, int(round(row)) - 2 : int(round(row)) + 3, int(round(col)) - 2 : int(round(col)) + 3] = 1.0
    return raster


def _peak(bev: torch.Tensor) -> tuple[float, float]:
    index = int(bev.flatten().argmax())
    row, col = divmod(index, bev.shape[-1])
    x, y = AUTOBEV_GEOMETRY.pixel_to_ego(torch.tensor([[row, col]], dtype=torch.float64).numpy())[0]
    return float(x), float(y)


def test_zero_pose_keeps_world_position():
    raster = _raster_with_point(20.0, 5.0)
    out = resample_map_context(raster, torch.zeros(1, 3), AUTOBEV_GEOMETRY)
    assert out.shape == (1, MAP_CHANNEL_COUNT, 220, 160)
    x, y = _peak(out[0, MapChannel.CROSSWALK])
    assert abs(x - 20.0) <= 0.5 and abs(y - 5.0) <= 0.5


def test_translated_crop_moves_map_the_other_way():
    # A crop frame 2 m ahead of the raster frame sees the point 2 m closer.
    raster = _raster_with_point(20.0, 0.0)
    out = resample_map_context(raster, torch.tensor([[2.0, 0.0, 0.0]]), AUTOBEV_GEOMETRY)
    x, _ = _peak(out[0, MapChannel.CROSSWALK])
    assert abs(x - 18.0) <= 0.5


def test_rotated_crop_rotates_points_and_directions():
    raster = _raster_with_point(20.0, 0.0)
    valid = int(MapChannel.TRAFFIC_DIRECTION_VALID)
    raster[0, valid] = 1.0
    raster[0, int(MapChannel.TRAFFIC_DIRECTION_SIN)] = 0.5  # sin 0
    raster[0, int(MapChannel.TRAFFIC_DIRECTION_COS)] = 1.0  # cos 1: heading along +x
    yaw = math.radians(10.0)
    out = resample_map_context(raster, torch.tensor([[0.0, 0.0, yaw]]), AUTOBEV_GEOMETRY)
    x, y = _peak(out[0, MapChannel.CROSSWALK])
    # The point at (20, 0) in M lies at R(-yaw) (20, 0) in F.
    assert abs(x - 20.0 * math.cos(yaw)) <= 0.6 and abs(y + 20.0 * math.sin(yaw)) <= 0.6
    sin = 2.0 * out[0, int(MapChannel.TRAFFIC_DIRECTION_SIN), 110, 80] - 1.0
    assert abs(float(sin) - math.sin(-yaw)) < 1e-3


def test_route_and_grid_shapes():
    route = torch.rand(2, 2, 450, 300)
    assert resample_route(route, torch.zeros(2, 3), AUTOBEV_GEOMETRY).shape == (2, 2, 220, 160)
    assert sampling_grid(torch.zeros(2, 3), AUTOBEV_GEOMETRY, margin_cells=4).shape == (2, 228, 168, 2)


def test_compose_poses_is_se2_product():
    first = torch.tensor([1.0, 2.0, math.pi / 2])
    second = torch.tensor([3.0, 0.0, 0.1])
    out = compose_poses(first, second)
    assert torch.allclose(out, torch.tensor([1.0, 5.0, math.pi / 2 + 0.1]), atol=1e-6)
