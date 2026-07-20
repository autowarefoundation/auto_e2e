from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("kitscenes")

from data_parsing.kit_scenes.camera import (
    compute_camera_projection_matrices,
    load_camera_frame,
)
from data_parsing.kit_scenes.dataset import (
    KitScenesDataset,
    _heading_cw_from_north,
    _local_xy_to_absolute_utm,
    _utm32_to_wgs84,
)
from data_parsing.kit_scenes import dataset as dataset_module
from data_parsing.kit_scenes import map as map_module


def _dataset_stub(samples):
    dataset = object.__new__(KitScenesDataset)
    dataset._samples = list(samples)
    dataset._wm_num_frames = 4
    dataset._wm_stride = 10
    dataset.camera_names = ["front", "left"]
    dataset._scene_egomotion = {
        scene_id: np.zeros((200, 4), dtype=np.float32)
        for scene_id, _ in samples
    }
    return dataset


def test_sample_uid_is_stable_across_scene_subsets():
    scene = "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    first = _dataset_stub([("other", 64), (scene, 77)])
    second = _dataset_stub([(scene, 77), ("third", 64)])

    assert first.sample_uid(1) == second.sample_uid(0)
    assert first.sample_uid(1) == f"kitscenes-v1-{scene}-f000077"
    assert first.split_group_uid(1) == f"kitscenes-{scene}"
    assert first.frame_index(1) == 77


def test_world_model_ids_and_rows_never_leave_scene():
    scene = "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    dataset = _dataset_stub([(scene, 100)])

    index = dataset.window_frame_ids(0)
    identifiers = [
        frame_id
        for step in index["history"] + index["future"]
        for frame_id in step
    ]
    assert len(identifiers) == 16
    assert all(f"kitscenes-v1-{scene}-r" in value for value in identifiers)
    assert dataset.window_rows(0) == [
        (scene, 70),
        (scene, 80),
        (scene, 90),
        (scene, 100),
        (scene, 110),
        (scene, 120),
        (scene, 130),
        (scene, 140),
    ]


def test_heading_conversion_uses_absolute_yaw_not_yaw_rate():
    assert _heading_cw_from_north(0.0) == pytest.approx(90.0)
    assert _heading_cw_from_north(np.pi / 2) == pytest.approx(0.0)
    assert _heading_cw_from_north(np.pi) == pytest.approx(270.0)


def test_map_rasterizer_queries_with_scene_local_pose(monkeypatch, tmp_path):
    class _SceneMap:
        utm_origin = np.array([456_789.0, 5_432_100.0])

        def __init__(self):
            self.query_centers = []

        def get_lanelets_in_roi(self, center, radius):
            self.query_centers.append(np.asarray(center).copy())
            return []

        def get_stop_lines(self):
            return []

    scene_map = _SceneMap()
    monkeypatch.setattr(map_module, "_cached_scene_map", lambda _: scene_map)

    tile = map_module.generate_bev_map_tile(
        tmp_path,
        ego_x=2917.7171,
        ego_y=-3280.8901,
        canvas_size=32,
    )

    np.testing.assert_allclose(
        scene_map.query_centers,
        [[2917.7171, -3280.8901]],
    )
    assert tile.shape == (32, 32, 3)


def test_map_loader_omits_only_degenerate_lanelets(tmp_path):
    scene_path = tmp_path / "scene"
    maps_path = scene_path / "maps"
    maps_path.mkdir(parents=True)
    map_path = maps_path / "map.osm"
    map_path.write_text(
        """\
<osm version="0.6">
  <way id="valid-left"><nd ref="1"/><nd ref="2"/></way>
  <way id="valid-right"><nd ref="3"/><nd ref="4"/></way>
  <way id="degenerate-left"><nd ref="5"/></way>
  <relation id="valid-lanelet">
    <member type="way" ref="valid-left" role="left"/>
    <member type="way" ref="valid-right" role="right"/>
    <tag k="type" v="lanelet"/>
  </relation>
  <relation id="degenerate-lanelet">
    <member type="way" ref="degenerate-left" role="left"/>
    <member type="way" ref="valid-right" role="right"/>
    <tag k="type" v="lanelet"/>
  </relation>
  <relation id="regulatory-element">
    <tag k="type" v="regulatory_element"/>
  </relation>
</osm>
"""
    )

    sanitized_path = map_module._map_without_degenerate_lanelets(scene_path)

    assert sanitized_path != map_path
    sanitized_ids = {
        relation.attrib["id"]
        for relation in ET.parse(sanitized_path).getroot().findall("relation")
    }
    assert sanitized_ids == {"valid-lanelet", "regulatory-element"}
    original_ids = {
        relation.attrib["id"]
        for relation in ET.parse(map_path).getroot().findall("relation")
    }
    assert original_ids == {
        "valid-lanelet",
        "degenerate-lanelet",
        "regulatory-element",
    }


def test_map_loader_preserves_valid_map_path(tmp_path):
    scene_path = tmp_path / "scene"
    maps_path = scene_path / "maps"
    maps_path.mkdir(parents=True)
    map_path = maps_path / "map.osm"
    map_path.write_text(
        """\
<osm version="0.6">
  <way id="left"><nd ref="1"/><nd ref="2"/></way>
  <way id="right"><nd ref="3"/><nd ref="4"/></way>
  <relation id="lanelet">
    <member type="way" ref="left" role="left"/>
    <member type="way" ref="right" role="right"/>
    <tag k="type" v="lanelet"/>
  </relation>
</osm>
"""
    )

    assert map_module._map_without_degenerate_lanelets(scene_path) == map_path


def test_geographic_output_adds_map_origin_once(monkeypatch, tmp_path):
    origin_utm = np.array([456114.5958622605, 5427629.2039247155])
    monkeypatch.setattr(
        dataset_module,
        "_cached_scene_map",
        lambda _: SimpleNamespace(utm_origin=origin_utm),
    )
    positions_local = np.array([
        [0.0, 0.0],
        [2917.7171, -3280.8901],
    ])

    positions_utm = _local_xy_to_absolute_utm(tmp_path, positions_local)
    np.testing.assert_allclose(
        positions_utm,
        positions_local + origin_utm,
    )
    np.testing.assert_allclose(
        _utm32_to_wgs84(positions_utm),
        [
            [49.0, 8.4],
            [48.97068824389647, 8.440219548828917],
        ],
        rtol=0.0,
        atol=1e-10,
    )


def test_geographic_output_requires_map_origin(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_module, "_cached_scene_map", lambda _: None)

    with pytest.raises(ValueError, match="no loadable map origin"):
        _local_xy_to_absolute_utm(tmp_path, np.zeros((1, 2)))


class _CameraLoader:
    def __init__(self):
        self.images = {
            "front": np.full((10, 20, 3), 10, dtype=np.uint8),
            "left": np.full((12, 16, 3), 20, dtype=np.uint8),
        }

    def get_camera_image(self, name, frame_idx):
        assert frame_idx == 3
        return self.images[name]

    def get_camera_calibration(self, name):
        width = self.images[name].shape[1]
        height = self.images[name].shape[0]
        return SimpleNamespace(
            image_size=(width, height),
            intrinsic=np.array([
                [100.0, 0.0, width / 2],
                [0.0, 100.0, height / 2],
                [0.0, 0.0, 1.0],
            ]),
            extrinsic=np.eye(4),
        )


def test_camera_pipeline_resizes_without_normalizing_and_scales_intrinsics():
    loader = _CameraLoader()
    frames = load_camera_frame(
        loader,
        3,
        camera_names=["front", "left"],
        image_size=32,
    )
    assert frames.shape == (2, 3, 32, 32)
    assert frames.dtype == torch.uint8
    assert int(frames[0, 0, 0, 0]) == 10
    assert int(frames[1, 0, 0, 0]) == 20

    projection = compute_camera_projection_matrices(
        loader,
        camera_names=["front", "left"],
        image_size=32,
    )
    assert projection.shape == (2, 3, 4)
    assert projection[0, 0, 0] == pytest.approx(160.0)
    assert projection[0, 1, 1] == pytest.approx(320.0)
    assert projection[0, 0, 2] == pytest.approx(16.0)
    assert projection[0, 1, 2] == pytest.approx(16.0)


def test_numeric_contract_emits_current_plus_64_gps_points():
    scene = "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    dataset = _dataset_stub([(scene, 64)])
    signals = np.zeros((200, 4), dtype=np.float32)
    dataset._scene_egomotion[scene] = signals
    dataset._scene_latlon = {
        scene: np.column_stack([
            49.0 + np.arange(200) / 100000,
            8.0 + np.arange(200) / 100000,
        ])
    }
    dataset._scene_yaws = {scene: np.zeros(200, dtype=np.float64)}
    dataset._scene_timestamps_ns = {
        scene: np.arange(200, dtype=np.int64) * 100_000_000
    }

    ego, target, pose, gps = dataset.numeric_for(0)

    assert ego.shape == (256,)
    assert target.shape == (128,)
    assert gps.shape == (65, 2)
    assert pose["latitude_deg"] == pytest.approx(gps[0, 0])
    assert pose["longitude_deg"] == pytest.approx(gps[0, 1])
    assert pose["heading_deg_cw_from_north"] == pytest.approx(90.0)


def test_dataset_geo_iterators_preserve_scene_and_sample_identity():
    scene = "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    dataset = _dataset_stub([(scene, 1), (scene, 2)])
    dataset._scene_ids = [scene]
    dataset._scene_latlon = {
        scene: np.array([
            [49.0, 8.0],
            [49.0001, 8.0001],
            [49.0002, 8.0002],
        ])
    }
    dataset._scene_yaws = {
        scene: np.array([0.0, np.pi / 2, np.pi], dtype=np.float64)
    }
    dataset._scene_timestamps_ns = {
        scene: np.array([10, 20, 30], dtype=np.int64)
    }

    assert dataset.episode_indices() == [scene]
    path = dataset.episode_path(scene)
    np.testing.assert_array_equal(path[:, :2], dataset._scene_latlon[scene])
    np.testing.assert_allclose(path[:, 2], [90.0, 0.0, 270.0])

    records = list(dataset.sample_pose_records())
    assert [record["episode_id"] for record in records] == [scene, scene]
    assert records[0]["sample_uid"] == (
        f"kitscenes-v1-{scene}-f000001"
    )
    assert records[1]["timestamp_ns"] == 30
