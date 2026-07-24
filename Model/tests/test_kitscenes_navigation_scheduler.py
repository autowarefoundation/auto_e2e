from __future__ import annotations

import numpy as np

from data_parsing.kit_scenes.navigation import KitScenesSceneNavigation


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
