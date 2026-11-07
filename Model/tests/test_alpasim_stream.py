"""Tests for the AutoE2E AlpaSim driver plugin.

The unit tests check the rig profiles, the egomotion and projection math, the
parsed model inputs and the call into ``AutoE2E.forward``. The ``integration``
tests run a real model; the checkpoint test needs a stage 2 checkpoint:

    AUTOE2E_CHECKPOINT=/path/to/checkpoint.pt
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "alpasim_driver"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

pytest.importorskip("alpasim_driver")
pytest.importorskip("alpasim_utils")

from alpasim_driver.models.base import (  # noqa: E402
    ModelInputValidationError,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig  # noqa: E402
from alpasim_grpc.v0 import common_pb2  # noqa: E402
from alpasim_autoe2e.config import (  # noqa: E402
    CALIBRATION_DIR,
    find_rig_profile,
    load_rig_profile,
)
from alpasim_autoe2e.parser import (  # noqa: E402
    CONTEXT_FRAMES,
    MAP_POSE_SIGMA,
    egomotion_from_poses,
    history_projections,
    pose_trajectory,
)
from alpasim_autoe2e.plugin import (  # noqa: E402
    MOCK_CHECKPOINT,
    UNTRAINED_CHECKPOINT,
    AutoE2EDriver,
)
from data_processing.autobev_members import crop_params  # noqa: E402
from training.losses.control_rollout import integrate_controls_torch  # noqa: E402

KIT_CAMERAS = [
    "camera_base_front_center",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear_left",
    "camera_ring_rear",
    "camera_ring_rear_right",
]
KIT_SLOTS = ["front", "front_left", "front_right", "rear_left", "rear", "rear_right"]
NUPLAN_CAMERAS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_B0", "CAM_R2"]
NUPLAN_NUREC_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_front_tele_30fov",
    "camera_rear_right_70fov",
]
T0_US = 10_000_000


def _pose(timestamp_us: int, x: float, y: float, yaw: float) -> common_pb2.PoseAtTime:
    pose = common_pb2.PoseAtTime(timestamp_us=timestamp_us)
    pose.pose.vec.x, pose.pose.vec.y = x, y
    pose.pose.quat.z, pose.pose.quat.w = np.sin(yaw / 2), np.cos(yaw / 2)
    return pose


def _unicycle_poses(
    speed: float = 8.0,
    yaw_rate: float = 0.0,
    acceleration: float = 0.0,
    start_us: int = 0,
    t0_us: int = T0_US,
) -> list[common_pb2.PoseAtTime]:
    """Poses every 100 ms from ``start_us`` to ``t0_us``, integrated finely."""
    poses, x, y, yaw, v = [], 0.0, 0.0, 0.0, speed
    dt_us = 1_000
    for t_us in range(start_us, t0_us + 1, dt_us):
        if (t_us - start_us) % 100_000 == 0:
            poses.append(_pose(t_us, x, y, yaw))
        x += v * np.cos(yaw) * dt_us / 1e6
        y += v * np.sin(yaw) * dt_us / 1e6
        yaw += yaw_rate * dt_us / 1e6
        v += acceleration * dt_us / 1e6
    return poses


def _prediction_input(
    poses: list[common_pb2.PoseAtTime] | None = None,
    spacing_us: int = 500_000,
    cameras: list[str] = KIT_CAMERAS,
    colors: dict[str, int] | None = None,
) -> PredictionInput:
    images = {}
    for index, camera in enumerate(cameras):
        value = (colors or {}).get(camera, 20 * index)
        images[camera] = [
            (T0_US - (CONTEXT_FRAMES - 1 - k) * spacing_us, np.full((57, 88, 3), value, np.uint8))
            for k in range(CONTEXT_FRAMES)
        ]
    return PredictionInput(
        camera_images=images,
        command=0,
        speed=8.0,
        acceleration=0.0,
        ego_pose_history=poses if poses is not None else _unicycle_poses(),
        inference_seed=0,
        previous_plan=None,
        route=None,
    )


class _ControlsModel(torch.nn.Module):
    """Stands in for AutoE2E: binds the call to ``ReactiveE2E.forward`` and returns fixed controls."""

    def __init__(self, acceleration: float = 0.0, curvature: float = 0.0) -> None:
        super().__init__()
        from model_components.reactive_e2e import ReactiveE2E

        self.signature = inspect.signature(ReactiveE2E.forward)
        self.controls = torch.tensor([acceleration, curvature]).repeat(64)
        self.calls: list[inspect.BoundArguments] = []

    def forward(self, **inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.calls.append(self.signature.bind(self, **inputs))
        return self.controls.unsqueeze(0), {}


@pytest.fixture
def driver() -> AutoE2EDriver:
    return AutoE2EDriver(MOCK_CHECKPOINT, KIT_CAMERAS)


# --- rig profiles -------------------------------------------------------------


def test_kit_profile_is_selected_by_its_cameras() -> None:
    profile = find_rig_profile(list(reversed(KIT_CAMERAS)))
    assert profile.name == "kit_scenes"
    assert profile.dataset_name == "KIT-MRT/KITScenes-Multimodal"
    assert [c.logical_id for c in profile.ordered_by_slots(KIT_SLOTS)] == KIT_CAMERAS


def test_unknown_rig_is_rejected() -> None:
    with pytest.raises(ValueError, match="No calibration profile"):
        find_rig_profile(["camera_front_wide_120fov"])


def test_profile_override_from_environment(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "kit_scenes_revision_b.json"
    shutil.copy(CALIBRATION_DIR / "kit_scenes.json", override)
    monkeypatch.setenv("AUTOE2E_RIG_PROFILE", str(override))
    assert find_rig_profile(KIT_CAMERAS).name == "kit_scenes_revision_b"


def test_nuplan_profile_export_rectifies_the_camera_table(tmp_path: Path, monkeypatch) -> None:
    """Export a profile from a nuPlan-style ``camera`` table and drive with it."""
    cv2 = pytest.importorskip("cv2")
    import pickle
    import sqlite3
    import subprocess

    channels = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_B0", "CAM_R2"]
    intrinsic = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
    distortion = np.array([-0.35, 0.17, -0.002, 0.0005, -0.05])
    database = tmp_path / "2021.05.12.22.00.38_veh-35_01008_01518.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE log (vehicle_name TEXT)")
        connection.execute("INSERT INTO log VALUES ('veh-35')")
        connection.execute(
            "CREATE TABLE camera (channel TEXT, translation BLOB, rotation BLOB, "
            "intrinsic BLOB, distortion BLOB, width INTEGER, height INTEGER)"
        )
        for index, channel in enumerate(channels):
            yaw = np.radians(60.0 * index)
            # OpenCV camera axes (z forward, x right, y down) turned by ``yaw``.
            rotation = np.array(
                [[np.sin(yaw), 0.0, np.cos(yaw)], [-np.cos(yaw), 0.0, np.sin(yaw)], [0.0, -1.0, 0.0]]
            )
            w = np.sqrt(1.0 + np.trace(rotation)) / 2.0
            quaternion = [
                w,
                (rotation[2, 1] - rotation[1, 2]) / (4 * w),
                (rotation[0, 2] - rotation[2, 0]) / (4 * w),
                (rotation[1, 0] - rotation[0, 1]) / (4 * w),
            ]
            connection.execute(
                "INSERT INTO camera VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    channel,
                    pickle.dumps([1.5, 0.1 * index, 1.6]),
                    pickle.dumps(quaternion),
                    pickle.dumps(intrinsic),
                    pickle.dumps(distortion),
                    1920,
                    1080,
                ),
            )
    output = tmp_path / "nuplan.json"
    subprocess.run(
        [sys.executable, str(_PLUGIN_DIR / "tools" / "export_nuplan_calibration.py"), str(database), str(output)],
        check=True,
    )

    profile = load_rig_profile(output)
    assert profile.dataset_name == "nuplan/nuplan-v1.1"
    assert [c.slot for c in profile.cameras] == KIT_SLOTS
    assert [c.logical_id for c in profile.cameras] == channels
    rectified, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (1920, 1080), 0.0, (1920, 1080))
    assert np.allclose(profile.cameras[0].intrinsic, rectified)
    assert np.allclose(profile.cameras[1].camera_to_reference[:3, 3], [1.5, 0.1, 1.6])
    assert np.allclose(profile.cameras[1].camera_to_reference[:3, 2], [0.5, np.sqrt(3) / 2, 0.0])

    monkeypatch.setenv("AUTOE2E_RIG_PROFILE", str(output))
    nuplan_driver = AutoE2EDriver(MOCK_CHECKPOINT, channels)
    inputs = nuplan_driver.parser.parse(_prediction_input(cameras=channels)).model_inputs
    assert inputs["images"].shape == (1, 6, 3, 768, 1344)


def test_nuplan_curvature_rule_zeroes_and_clips() -> None:
    from alpasim_autoe2e.config import EgomotionRule

    rule = EgomotionRule(
        curvature_speed_floor_mps=0.5, curvature_zero_below_mps=0.5, curvature_limit=0.5
    )
    speed = np.array([0.2, 1.0, 1.0, 10.0])
    yaw_rate = np.array([0.1, 0.2, 2.0, 0.5])
    assert np.allclose(rule.curvature(yaw_rate, speed), [0.0, 0.2, 0.5, 0.05])
    assert np.allclose(EgomotionRule().curvature(yaw_rate, speed), [0.5, 0.2, 2.0, 0.05])


@pytest.mark.parametrize("frame_wh", [(876, 568), (438, 284)])
def test_projection_maps_into_the_autobev_crop(frame_wh: tuple[int, int]) -> None:
    camera = load_rig_profile(CALIBRATION_DIR / "kit_scenes.json").cameras[1]
    projection = camera.projection(frame_wh, np.eye(4))
    width, height = camera.image_size_wh
    sx, sy, _, _, left, top = crop_params(*frame_wh)
    center = camera.camera_to_reference[:, 3]
    assert np.allclose(projection @ center, 0.0, atol=1e-9)
    on_axis = camera.camera_to_reference @ np.array([0.0, 0.0, 10.0, 1.0])
    pixel = projection @ on_axis
    expected = [
        camera.intrinsic[0, 2] * frame_wh[0] / width * sx - left,
        camera.intrinsic[1, 2] * frame_wh[1] / height * sy - top,
    ]
    assert np.allclose(pixel[:2] / pixel[2], expected)


# --- egomotion ----------------------------------------------------------------


def test_egomotion_straight_constant_speed() -> None:
    egomotion = egomotion_from_poses(pose_trajectory(_unicycle_poses(speed=8.0)), T0_US)
    assert egomotion.shape == (64, 4)
    assert np.allclose(egomotion[:, 0], 8.0, atol=1e-3)
    assert np.allclose(egomotion[:, 1:], 0.0, atol=1e-3)


def test_egomotion_constant_turn() -> None:
    poses = _unicycle_poses(speed=8.0, yaw_rate=0.2)
    egomotion = egomotion_from_poses(pose_trajectory(poses), T0_US)
    assert np.allclose(egomotion[:, 2], 0.2, atol=1e-3)
    assert np.allclose(egomotion[:, 3], 0.2 / 8.0, atol=1e-3)


def test_egomotion_rows_end_one_step_before_t0() -> None:
    poses = _unicycle_poses(speed=1.0, acceleration=0.5)
    egomotion = egomotion_from_poses(pose_trajectory(poses), T0_US)
    assert egomotion[-1, 0] == pytest.approx(1.0 + 0.5 * 9.9, abs=1e-2)
    assert egomotion[0, 0] == pytest.approx(1.0 + 0.5 * 3.6, abs=1e-2)
    assert np.allclose(egomotion[1:-1, 1], 0.5, atol=1e-2)


# --- projections --------------------------------------------------------------


def test_history_projection_of_a_stationary_ego_is_the_current_one() -> None:
    cameras = find_rig_profile(KIT_CAMERAS).ordered_by_slots(KIT_SLOTS)
    frame_wh = [c.image_size_wh for c in cameras]
    pose = np.eye(4)
    history = history_projections(cameras, frame_wh, pose, [pose] * 7)
    current = np.stack([c.projection(size, np.eye(4)) for c, size in zip(cameras, frame_wh)])
    assert history.shape == (7, 6, 3, 4)
    assert np.allclose(history, current[None], atol=1e-3)


def test_history_projection_follows_ego_motion() -> None:
    cameras = find_rig_profile(KIT_CAMERAS).ordered_by_slots(KIT_SLOTS)
    frame_wh = [c.image_size_wh for c in cameras]
    now, past = np.eye(4), np.eye(4)
    now[0, 3] = 5.0  # the ego moved 5 m forward since the past frame
    history = history_projections(cameras, frame_wh, now, [past])
    point_now = np.array([20.0, 1.0, 0.5, 1.0])
    point_past = point_now + np.array([5.0, 0.0, 0.0, 0.0])
    expected = cameras[0].projection(frame_wh[0], np.eye(4)) @ point_past
    actual = history[0, 0] @ point_now
    assert np.allclose(actual[:2] / actual[2], expected[:2] / expected[2], atol=1e-3)


# --- parsing ------------------------------------------------------------------


def test_parse_builds_the_autoe2e_inputs(driver: AutoE2EDriver) -> None:
    parsed = driver.parser.parse(_prediction_input())
    inputs = parsed.model_inputs
    assert inputs["images"].shape == (1, 6, 3, 768, 1344)
    assert inputs["history_images"].shape == (1, 7, 6, 3, 768, 1344)
    assert inputs["projection"].shape == (1, 6, 3, 4)
    assert inputs["history_projection"].shape == (1, 7, 6, 3, 4)
    assert inputs["egomotion_history"].shape == (1, 256)
    assert inputs["map_context"].shape == (1, 14, 450, 300)
    assert inputs["route_mask"].shape == (1, 2, 450, 300)
    assert not inputs["map_valid"].any() and not inputs["route_valid"].any()
    assert torch.equal(inputs["map_pose"], torch.zeros(1, 3))
    expected = torch.diag(torch.tensor(MAP_POSE_SIGMA) ** 2)
    assert torch.allclose(inputs["map_pose_covariance"][0], expected)
    assert inputs["map_matching"].all()
    assert parsed.t0_us == T0_US
    assert parsed.initial_speed_mps == pytest.approx(8.0, abs=1e-3)
    assert np.allclose(parsed.pose_local_to_rig_t0.vec3[:2], [80.0, 0.0], atol=1e-3)


def test_parse_orders_cameras_by_model_slot(driver: AutoE2EDriver) -> None:
    colors = {camera: 40 * index for index, camera in enumerate(KIT_CAMERAS)}
    shuffled = KIT_CAMERAS[3:] + KIT_CAMERAS[:3]
    inputs = driver.parser.parse(_prediction_input(cameras=shuffled, colors=colors)).model_inputs
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    red = inputs["images"][0, :, 0].mean(dim=(1, 2)) * std[0] + mean[0]
    assert torch.allclose(red * 255, torch.tensor([40.0 * i for i in range(6)]), atol=1.0)


def test_parse_centre_crops_after_scaling_to_cover(driver: AutoE2EDriver) -> None:
    """A frame wider than 1344x768 keeps its centre columns and loses its sides."""
    prediction_input = _prediction_input()
    frame = np.zeros((768, 2000, 3), np.uint8)
    frame[:, :328] = 255  # cut away: (2000 - 1344) / 2 = 328 columns per side
    frame[:, 1000] = 255  # survives at column 1000 - 328
    prediction_input.camera_images[KIT_CAMERAS[0]][-1] = (T0_US, frame)
    image = driver.parser.parse(prediction_input).model_inputs["images"][0, 0, 0]
    mean, std = 0.485, 0.229
    column = (image * std + mean).mean(dim=0)
    assert torch.allclose(column[:300], torch.zeros(300), atol=1e-3)
    assert column[1000 - 328] > 0.9


def test_parse_passes_the_raw_egomotion(driver: AutoE2EDriver) -> None:
    poses = _unicycle_poses(speed=1.0, acceleration=0.5)
    history = driver.parser.parse(_prediction_input(poses)).model_inputs["egomotion_history"]
    expected = egomotion_from_poses(pose_trajectory(poses), T0_US)
    assert torch.equal(history.reshape(64, 4), torch.from_numpy(expected))


def test_parse_rejects_wrong_frame_spacing(driver: AutoE2EDriver) -> None:
    with pytest.raises(ModelInputValidationError, match="500000 us apart"):
        driver.parser.parse(_prediction_input(spacing_us=100_000))


def test_parse_rejects_a_short_pose_history(driver: AutoE2EDriver) -> None:
    poses = _unicycle_poses(start_us=T0_US - 5_000_000)
    with pytest.raises(ModelInputValidationError, match="force_gt_duration_us"):
        driver.parser.parse(_prediction_input(poses))


def test_parse_rejects_a_missing_camera(driver: AutoE2EDriver) -> None:
    prediction_input = _prediction_input()
    del prediction_input.camera_images["camera_ring_rear"]
    with pytest.raises(ModelInputValidationError, match="camera_ring_rear"):
        driver.parser.parse(prediction_input)


# --- driver -------------------------------------------------------------------


def test_from_config_rejects_another_context_length() -> None:
    cfg = dataclasses.replace(ModelConfig(), checkpoint_path=MOCK_CHECKPOINT)
    with pytest.raises(ValueError, match="context_length"):
        AutoE2EDriver.from_config(cfg, torch.device("cpu"), KIT_CAMERAS, 1, 10)
    driver = AutoE2EDriver.from_config(cfg, torch.device("cpu"), KIT_CAMERAS, 8, 10)
    assert driver.camera_ids == KIT_CAMERAS
    assert driver.context_length == CONTEXT_FRAMES


def test_missing_checkpoint_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        AutoE2EDriver("/nonexistent/checkpoint.pt", KIT_CAMERAS)


def test_mock_mode_drives_straight(driver: AutoE2EDriver) -> None:
    prediction = driver.predict(_prediction_input())
    assert prediction.candidate_positions.shape == (1, 64, 3)
    assert np.allclose(prediction.selected_positions[:, 1:], 0.0)
    assert prediction.selected_positions[-1, 0] == pytest.approx(8.0 * 6.4)


def test_predict_calls_autoe2e_forward_by_name(driver: AutoE2EDriver) -> None:
    driver.model = _ControlsModel()
    prediction = driver.predict(_prediction_input())

    call = driver.model.calls[0].arguments
    assert set(call) == set(driver.model.signature.parameters) - {"image_projection"}
    assert call["history_images"].shape == (1, 7, 6, 3, 768, 1344)
    assert call["history_projection"].shape == (1, 7, 6, 3, 4)

    # Zero controls keep the initial speed: 0.8 m per 100 ms step.
    assert np.allclose(prediction.selected_positions[:, 0], 0.8 * np.arange(1, 65), atol=1e-3)
    assert prediction.model_t0_us == T0_US
    assert prediction.waypoint_timestamps_us[0] == T0_US + 100_000
    assert prediction.waypoint_timestamps_us[-1] == T0_US + 6_400_000
    assert np.allclose(prediction.pose_local_to_rig_t0.vec3[:2], [80.0, 0.0], atol=1e-3)


def test_predict_integrates_controls_like_training(driver: AutoE2EDriver) -> None:
    driver.model = _ControlsModel(acceleration=0.5, curvature=0.02)
    prediction = driver.predict(_prediction_input())
    positions, headings, _ = integrate_controls_torch(
        torch.tensor([0.5, 0.02]).repeat(1, 64), torch.tensor([8.0])
    )
    assert np.allclose(prediction.selected_positions[:, :2], positions[0].numpy(), atol=1e-2)
    yaw = np.arctan2(
        prediction.selected_rotations[:, 1, 0], prediction.selected_rotations[:, 0, 0]
    )
    assert np.allclose(yaw, headings[0].numpy(), atol=1e-4)


@pytest.mark.parametrize(
    ("driver_name", "runtime_name", "cameras"),
    [
        ("autoe2e", "autoe2e_configs", KIT_CAMERAS),
        ("autoe2e_nuplan", "autoe2e_nuplan_configs", NUPLAN_NUREC_CAMERAS),
    ],
)
def test_driver_config_fits_the_alpasim_schemas(
    driver_name: str, runtime_name: str, cameras: list[str]
) -> None:
    configs = _PLUGIN_DIR / "alpasim_autoe2e" / "configs" / "driver"
    base_cfg = yaml.safe_load((configs / "autoe2e.yaml").read_text())
    driver_cfg = yaml.safe_load((configs / f"{driver_name}.yaml").read_text())
    runtime_cfg = yaml.safe_load((configs / f"{runtime_name}.yaml").read_text())
    model_fields = {field.name for field in dataclasses.fields(ModelConfig)}
    assert set(base_cfg["model"]) <= model_fields
    assert base_cfg["inference"]["context_length"] == CONTEXT_FRAMES
    assert sorted(driver_cfg["inference"]["use_cameras"]) == sorted(cameras)
    find_rig_profile(driver_cfg["inference"]["use_cameras"])
    simulation = runtime_cfg["runtime"]["simulation_config"]
    assert sorted(camera["logical_id"] for camera in simulation["cameras"]) == sorted(cameras)
    for camera in simulation["cameras"]:
        spacing = camera["frame_interval_us"] * base_cfg["inference"]["subsample_factor"]
        assert spacing == 500_000
    assert simulation["force_gt_duration_us"] >= 7_000_000


def test_nuplan_extra_cameras_match_the_profile() -> None:
    """The rendered nuPlan rig must be the calibration the plugin projects with."""
    sys.path.insert(0, str(_PLUGIN_DIR / "tools"))
    from profile_to_extra_cameras import extra_cameras, renamed

    nuplan = json.loads((CALIBRATION_DIR / "nuplan.json").read_text())
    nurec = json.loads((CALIBRATION_DIR / "nuplan_nurec.json").read_text())
    assert nurec == renamed(nuplan, dict(zip(NUPLAN_CAMERAS, NUPLAN_NUREC_CAMERAS)))
    runtime_cfg = yaml.safe_load(
        (_PLUGIN_DIR / "alpasim_autoe2e" / "configs" / "driver" / "autoe2e_nuplan_configs.yaml").read_text()
    )
    assert runtime_cfg["runtime"]["extra_cameras"] == extra_cameras(nurec)
    assert find_rig_profile(NUPLAN_NUREC_CAMERAS).name == "nuplan_nurec"


# --- integration --------------------------------------------------------------


@pytest.mark.integration
def test_untrained_model_predicts() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    driver = AutoE2EDriver(UNTRAINED_CHECKPOINT, KIT_CAMERAS, device=device)
    prediction = driver.predict(_prediction_input())
    assert np.isfinite(prediction.selected_positions).all()


@pytest.mark.integration
@pytest.mark.skipif("AUTOE2E_CHECKPOINT" not in os.environ, reason="needs AUTOE2E_CHECKPOINT")
@pytest.mark.parametrize(
    "cameras", [KIT_CAMERAS, NUPLAN_CAMERAS, NUPLAN_NUREC_CAMERAS], ids=["kit_scenes", "nuplan", "nuplan_nurec"]
)
def test_checkpoint_predicts(cameras: list[str]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    driver = AutoE2EDriver(os.environ["AUTOE2E_CHECKPOINT"], cameras, device=device)
    prediction = driver.predict(_prediction_input(cameras=cameras))
    assert np.isfinite(prediction.selected_positions).all()

