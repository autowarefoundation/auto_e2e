from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from data_parsing.kit_scenes.navigation import KitScenesSceneNavigation
from navigation.contracts import Maneuver
from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
    MapChannel,
    RouteChannel,
)


class _RecordingRasterizer:
    def __init__(self):
        self.render_calls = []
        self.warp_calls = []

    def render(self, navigation_map, route, pose):
        self.render_calls.append(pose)
        return {"render_timestamp_ns": pose.timestamp_ns}

    def warp(self, raster, pose):
        self.warp_calls.append((raster, pose))
        return {
            **raster,
            "sample_timestamp_ns": pose.timestamp_ns,
        }


def _scene():
    scene = object.__new__(KitScenesSceneNavigation)
    scene.timestamps = np.arange(11, dtype=np.int64) * 100_000_000
    scene.positions = np.column_stack(
        [np.arange(11, dtype=np.float64), np.zeros(11)]
    )
    scene.yaws = np.zeros(11, dtype=np.float64)
    scene.navigation_map = object()
    scene.route = object()
    scene.rasterizer = _RecordingRasterizer()
    scene._anchor_cache = {}
    return scene


def test_anchor_grid_never_selects_a_future_pose():
    scene = _scene()

    assert [scene.anchor_index(index) for index in range(11)] == [
        0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10
    ]
    for frame_idx in range(11):
        anchor_idx = scene.anchor_index(frame_idx)
        assert scene.timestamps[anchor_idx] <= scene.timestamps[frame_idx]


def test_samples_share_one_render_and_apply_per_sample_warp():
    scene = _scene()

    first = scene.raster_for_frame(6)
    second = scene.raster_for_frame(7)
    anchor = scene.raster_for_frame(5)

    assert first["render_timestamp_ns"] == 500_000_000
    assert first["sample_timestamp_ns"] == 600_000_000
    assert second["sample_timestamp_ns"] == 700_000_000
    assert anchor == {"render_timestamp_ns": 500_000_000}
    assert len(scene.rasterizer.render_calls) == 1
    assert len(scene.rasterizer.warp_calls) == 2


def test_route_semantics_use_lane_sequence_and_semantic_raster_only():
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    scene = object.__new__(KitScenesSceneNavigation)
    scene.positions = np.asarray([[0.0, 0.0]], dtype=np.float64)
    scene.route = SimpleNamespace(
        valid=True,
        lane_sequence=(
            SimpleNamespace(
                lane_id="lane-straight",
                centerline_enu_m=np.asarray(
                    [[-10.0, 0.0], [10.0, 0.0]]
                ),
                maneuver=Maneuver.STRAIGHT,
            ),
            SimpleNamespace(
                lane_id="lane-left",
                centerline_enu_m=np.asarray(
                    [[10.0, 0.0], [20.0, 5.0], [25.0, 15.0]]
                ),
                maneuver=Maneuver.LEFT,
            ),
        ),
    )
    scene.rasterizer = SimpleNamespace(geometry=geometry)
    map_context = np.zeros(
        (MAP_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    route_mask = np.zeros(
        (ROUTE_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.uint8,
    )
    row, col = np.rint(
        geometry.ego_to_pixel(np.asarray([[20.0, 5.0]]))[0]
    ).astype(int)
    map_context[MapChannel.INTERSECTION, row, col] = 1.0
    route_mask[RouteChannel.SELECTED_CORRIDOR, row, col] = 1
    route_mask[RouteChannel.DESTINATION, row, col] = 1
    raster = SimpleNamespace(
        map_context=map_context,
        route_mask=route_mask,
    )

    semantics = scene.route_semantics(0, raster)

    assert semantics["route_maneuver"] == "left"
    assert semantics["route_intersection"] is True
    assert semantics["destination_visible"] is True
    assert semantics["current_route_lane_id"] == "lane-straight"


def test_invalid_route_semantics_are_explicitly_unknown():
    scene = object.__new__(KitScenesSceneNavigation)
    scene.route = SimpleNamespace(valid=False, lane_sequence=())

    semantics = scene.route_semantics(0, None)

    assert semantics["route_maneuver"] == "unknown"
    assert semantics["route_intersection"] is False
    assert semantics["destination_visible"] is False
