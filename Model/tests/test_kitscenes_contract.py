from __future__ import annotations

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
)


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
