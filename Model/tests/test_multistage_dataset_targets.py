"""Dataset adapters for the nuPlan -> L2D training sequence."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sqlite3
import sys
import tarfile
import types

import numpy as np
import pytest
from PIL import Image

from data_parsing.nuplan.packing import (
    NUPLAN_CAMERA_CHANNELS,
    NUPLAN_CAMERA_SLOTS,
    NUPLAN_CAMERA_SYNC_TOLERANCE_US,
    NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M,
    NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US,
    NuPlanCameraBundle,
    _camera_rows,
    _nuplan_point_cloud_xyz,
    _quaternion_transform,
    _rectify_camera_rows,
    camera_visibility_from_projection_matrices,
    nuplan_reactive_sample_members,
    pack_nuplan_reactive_scenarios,
)
import data_parsing.nuplan.targets as nuplan_targets
from data_parsing.pre_extracted import _is_validation_group
from data_processing.autobev_members import (
    crop_projection,
    decode_boxes,
    decode_npy,
    decode_stride_depth,
    encode_camera,
    manifest_contract,
)
from data_processing.dataset_snapshot import split_bucket
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
    decode_bev_segmentation,
    decode_bev_segmentation_stats,
    decode_trajectory_xy,
)
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, IMAGE_H, IMAGE_W
from navigation.artifacts import decode_array
from navigation.geometry import NavigationRasterGeometry
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
)



def _geometry() -> NavigationRasterGeometry:
    return NavigationRasterGeometry(
        geometry_id="test-multistage-v1",
        height_px=40,
        width_px=20,
        meters_per_pixel=1.0,
        x_min_m=-10.0,
        x_max_m=30.0,
        y_min_m=-10.0,
        y_max_m=10.0,
        ego_anchor_row=29.5,
        ego_anchor_col=9.5,
        matching_pc_range=(-10.0, -10.0, -5.0, 30.0, 10.0, 3.0),
        matching_bev_h=40,
        matching_bev_w=20,
        route_corridor_width_m=3.5,
        destination_marker_radius_m=2.0,
        route_rear_clip_m=10.0,
)


def test_nuplan_history_uses_distinct_strictly_ordered_frames(tmp_path):
    database = tmp_path / "log.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE ego_pose (
                token BLOB PRIMARY KEY,
                x REAL, y REAL, z REAL,
                qw REAL, qx REAL, qy REAL, qz REAL
            );
            CREATE TABLE lidar_pc (
                token BLOB PRIMARY KEY,
                ego_pose_token BLOB,
                timestamp INTEGER
            );
            CREATE TABLE camera (
                token BLOB PRIMARY KEY,
                channel TEXT,
                model TEXT,
                translation BLOB,
                rotation BLOB,
                intrinsic BLOB,
                distortion BLOB,
                width INTEGER,
                height INTEGER
            );
            CREATE TABLE image (
                filename_jpg TEXT,
                timestamp INTEGER,
                camera_token BLOB,
                ego_pose_token BLOB
            );
            """
        )
        pose_token = b"pose"
        connection.execute(
            "INSERT INTO ego_pose VALUES (?, 0, 0, 0, 1, 0, 0, 0)",
            (pose_token,),
        )
        lidar_token = bytes.fromhex("aa")
        reference_timestamp = 4_000_000
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?)",
            (lidar_token, pose_token, reference_timestamp),
        )
        for index, channel in enumerate(NUPLAN_CAMERA_CHANNELS):
            camera_token = f"camera-{index}".encode("ascii")
            connection.execute(
                "INSERT INTO camera VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    camera_token,
                    channel,
                    "pinhole",
                    b"",
                    b"",
                    b"",
                    b"",
                    1920,
                    1080,
                ),
            )
            timestamps = tuple(
                index * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                for index in range(1, 9)
            )
            for timestamp in timestamps:
                connection.execute(
                    "INSERT INTO image VALUES (?, ?, ?, ?)",
                    (
                        f"{channel}-{timestamp}.jpg",
                        timestamp,
                        camera_token,
                        pose_token,
                    ),
                )
        connection.commit()
    finally:
        connection.close()

    _, current, history = _camera_rows(
        str(database),
        lidar_token.hex(),
    )

    assert len(current) == len(NUPLAN_CAMERA_CHANNELS)
    assert {
        int(row["timestamp"]) for row in current.values()
    } == {reference_timestamp}
    assert len(history) == REACTIVE_BEVFORMER_HISTORY_FRAMES
    assert all(
        set(frame) == set(NUPLAN_CAMERA_CHANNELS)
        for frame in history
    )
    assert {
        int(row["timestamp"]) for row in history[0].values()
    } == {500_000}
    for history_index, frame in enumerate(history):
        target_timestamp = (
            reference_timestamp
            + REACTIVE_BEVFORMER_FRAME_OFFSETS[history_index]
            * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        )
        assert all(
            abs(int(row["timestamp"]) - target_timestamp)
            <= NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US
            for row in frame.values()
        )
        assert {
            int(row["timestamp"]) for row in frame.values()
        } == {target_timestamp}
    for channel in NUPLAN_CAMERA_CHANNELS:
        channel_timestamps = [
            int(frame[channel]["timestamp"]) for frame in history
        ]
        assert channel_timestamps == sorted(set(channel_timestamps))


def test_nuplan_current_camera_rejects_stale_fallback(tmp_path):
    database = tmp_path / "log.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE ego_pose (
                token BLOB PRIMARY KEY,
                x REAL, y REAL, z REAL,
                qw REAL, qx REAL, qy REAL, qz REAL
            );
            CREATE TABLE lidar_pc (
                token BLOB PRIMARY KEY,
                ego_pose_token BLOB,
                timestamp INTEGER
            );
            CREATE TABLE camera (
                token BLOB PRIMARY KEY,
                channel TEXT,
                model TEXT,
                translation BLOB,
                rotation BLOB,
                intrinsic BLOB,
                distortion BLOB,
                width INTEGER,
                height INTEGER
            );
            CREATE TABLE image (
                filename_jpg TEXT,
                timestamp INTEGER,
                camera_token BLOB,
                ego_pose_token BLOB
            );
            """
        )
        pose_token = b"pose"
        connection.execute(
            "INSERT INTO ego_pose VALUES (?, 0, 0, 0, 1, 0, 0, 0)",
            (pose_token,),
        )
        lidar_token = bytes.fromhex("aa")
        reference_timestamp = 4_000_000
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?)",
            (lidar_token, pose_token, reference_timestamp),
        )
        for index, channel in enumerate(NUPLAN_CAMERA_CHANNELS):
            camera_token = f"camera-{index}".encode("ascii")
            connection.execute(
                "INSERT INTO camera VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    camera_token,
                    channel,
                    "pinhole",
                    b"",
                    b"",
                    b"",
                    b"",
                    1920,
                    1080,
                ),
            )
            timestamp = (
                reference_timestamp
                - NUPLAN_CAMERA_SYNC_TOLERANCE_US
                - 1
                if index == 0
                else reference_timestamp
            )
            connection.execute(
                "INSERT INTO image VALUES (?, ?, ?, ?)",
                (
                    f"{channel}-{timestamp}.jpg",
                    timestamp,
                    camera_token,
                    pose_token,
                ),
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="missing required cameras"):
        _camera_rows(str(database), lidar_token.hex())




def _fake_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        getOptimalNewCameraMatrix=lambda intrinsic, *_args: (
            intrinsic.copy(),
            None,
        ),
        undistort=lambda image, *_args: image,
    ))


def _camera_row(image_name, *, size, rotation, timestamp, x):
    return {
        "distortion": pickle.dumps(np.zeros(5, dtype=np.float64)),
        "filename_jpg": image_name,
        "height": size,
        "intrinsic": pickle.dumps(np.eye(3, dtype=np.float64)),
        "model": "DesignCore D3CM-IMX390",
        "qw": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "rotation": pickle.dumps(rotation),
        "timestamp": timestamp,
        "translation": pickle.dumps(np.zeros(3, dtype=np.float64)),
        "width": size,
        "x": x,
        "y": 0.0,
        "z": 0.0,
    }


def test_history_camera_projection_uses_current_ego_reference(
    tmp_path,
    monkeypatch,
):
    _fake_cv2(monkeypatch)
    image_path = tmp_path / "history.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    identity_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    row = _camera_row(
        image_path.name,
        size=4,
        rotation=identity_quaternion,
        timestamp=6_500_000,
        x=8.0,
    )
    reference_pose = _quaternion_transform(
        [10.0, 0.0, 0.0],
        identity_quaternion,
    )

    _, matrices, camera_from_reference, metadata = _rectify_camera_rows(
        {channel: row for channel in NUPLAN_CAMERA_CHANNELS},
        sensor_root=str(tmp_path),
        reference_pose=reference_pose,
        reference_timestamp=10_000_000,
    )

    image_from_current = np.eye(4, dtype=np.float64)
    image_from_current[0, 3] = 2.0
    expected = crop_projection(np.eye(3), 4, 4) @ image_from_current[:3]
    np.testing.assert_allclose(matrices[0], expected, rtol=1e-6)
    np.testing.assert_allclose(camera_from_reference[0], image_from_current)
    assert {
        camera["sensor_model"] for camera in metadata
    } == {"DesignCore D3CM-IMX390"}


def test_nuplan_camera_rectification_rejects_non_caltech_distortion(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace())
    image_path = tmp_path / "invalid-distortion.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    row = {
        "distortion": pickle.dumps(np.zeros(4, dtype=np.float64)),
        "filename_jpg": image_path.name,
        "height": 4,
        "intrinsic": pickle.dumps(np.eye(3, dtype=np.float64)),
        "model": "DesignCore D3CM-IMX390",
        "width": 4,
    }

    with pytest.raises(ValueError, match="CAM_F0 distortion is invalid"):
        _rectify_camera_rows(
            {channel: row for channel in NUPLAN_CAMERA_CHANNELS},
            sensor_root=str(tmp_path),
            reference_pose=np.eye(4, dtype=np.float64),
            reference_timestamp=10_000_000,
        )


def test_nuplan_cameras_are_cropped_to_the_autobev_size(
    tmp_path,
    monkeypatch,
):
    _fake_cv2(monkeypatch)
    image_path = tmp_path / "front.jpg"
    pixels = np.arange(16 * 9 * 3, dtype=np.uint8).reshape(9, 16, 3)
    Image.fromarray(pixels).save(image_path, quality=95)
    row = _camera_row(
        image_path.name,
        size=0,
        rotation=np.asarray([1.0, 0.0, 0.0, 0.0]),
        timestamp=10_000_000,
        x=0.0,
    )
    row.update({"width": 16, "height": 9})

    jpegs, matrices, _, metadata = _rectify_camera_rows(
        {channel: row for channel in NUPLAN_CAMERA_CHANNELS},
        sensor_root=str(tmp_path),
        reference_pose=np.eye(4, dtype=np.float64),
        reference_timestamp=10_000_000,
    )

    with Image.open(image_path) as source:
        expected = encode_camera(np.asarray(source.convert("RGB")))
    assert jpegs == [expected] * len(NUPLAN_CAMERA_CHANNELS)
    with Image.open(io.BytesIO(jpegs[0])) as image:
        assert image.size == (IMAGE_W, IMAGE_H)
    np.testing.assert_allclose(
        metadata[0]["cropped_intrinsic"],
        crop_projection(np.eye(3), 16, 9),
    )
    assert matrices.shape == (len(NUPLAN_CAMERA_CHANNELS), 3, 4)


def test_nuplan_projection_inverts_sensor_to_ego_yaw(
    tmp_path,
    monkeypatch,
):
    _fake_cv2(monkeypatch)
    image_path = tmp_path / "yaw.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    root_half = np.sqrt(0.5)
    row = _camera_row(
        image_path.name,
        size=4,
        rotation=np.asarray([root_half, 0.0, 0.0, root_half]),
        timestamp=10_000_000,
        x=0.0,
    )

    _, matrices, _, _ = _rectify_camera_rows(
        {channel: row for channel in NUPLAN_CAMERA_CHANNELS},
        sensor_root=str(tmp_path),
        reference_pose=np.eye(4, dtype=np.float64),
        reference_timestamp=10_000_000,
    )

    camera_from_ego = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    expected = crop_projection(np.eye(3), 4, 4) @ camera_from_ego[:3]
    np.testing.assert_allclose(matrices[0], expected, atol=1e-4)


class _Velocity:
    x = 5.0
    y = 0.0


class _EgoState:
    def __init__(self, x: float, y: float, heading: float = 0.0):
        self.rear_axle = types.SimpleNamespace(
            x=x,
            y=y,
            heading=heading,
        )
        self.dynamic_car_state = types.SimpleNamespace(
            rear_axle_velocity_2d=_Velocity()
        )


def _tracked_vehicle(center_x: float = 5.0) -> types.SimpleNamespace:
    """A 2 m square vehicle box at (center_x, 0) in global coordinates."""
    coords = [
        (center_x - 1.0, -1.0),
        (center_x + 1.0, -1.0),
        (center_x + 1.0, 1.0),
        (center_x - 1.0, 1.0),
        (center_x - 1.0, -1.0),
    ]
    polygon = types.SimpleNamespace(
        is_empty=False,
        geom_type="Polygon",
        exterior=types.SimpleNamespace(coords=coords),
    )
    return types.SimpleNamespace(
        tracked_object_type=types.SimpleNamespace(name="VEHICLE"),
        box=types.SimpleNamespace(
            geometry=polygon,
            center=types.SimpleNamespace(x=center_x, y=0.0, heading=0.0),
            length=2.0,
            width=2.0,
            height=1.5,
        ),
        velocity=types.SimpleNamespace(x=3.0, y=0.0),
    )


class _Scenario:
    def __init__(self):
        self.future_calls = 0
        self.states = [
            _EgoState(float(index), 0.0)
            for index in range(65)
        ]
        self.map_api = types.SimpleNamespace(map_name="us-nv-las-vegas-strip")
        self._detections = types.SimpleNamespace(
            tracked_objects=[_tracked_vehicle()]
        )

    def get_ego_state_at_iteration(self, iteration):
        return self.states[iteration]

    def get_ego_future_trajectory(
        self,
        iteration,
        *,
        time_horizon,
        num_samples,
    ):
        assert time_horizon == pytest.approx(6.4)
        self.future_calls += 1
        return iter(self.states[iteration + 1:iteration + 1 + num_samples])

    def get_tracked_objects_at_iteration(self, iteration):
        return self._detections

    def get_mission_goal(self):
        return types.SimpleNamespace(x=20.0, y=0.0)


def test_nuplan_future_pose_target_is_current_ego_relative():
    scenario = _Scenario()
    xy, valid, speed = nuplan_targets.future_trajectory_xy(scenario)

    assert xy.shape == (64, 2)
    assert np.array_equal(xy[:3, 0], np.asarray([1.0, 2.0, 3.0]))
    assert np.count_nonzero(xy[:, 1]) == 0
    assert valid.all()
    assert speed == pytest.approx(5.0)


def _patch_map(monkeypatch):
    static_polygon = np.asarray(
        [[-5.0, -4.0], [25.0, -4.0], [25.0, 4.0], [-5.0, 4.0]]
    )
    map_polygons = {
        "drivable_area": [static_polygon],
        "intersection": [],
        "crosswalk": [],
        "stop_line": [],
    }
    map_available = {
        "drivable_area": True,
        "lane_boundary": True,
        "intersection": True,
        "crosswalk": True,
        "stop_line": True,
    }
    monkeypatch.setattr(
        nuplan_targets,
        "_map_layer_polygons",
        lambda *_args: (map_polygons, map_available),
    )
    lane_boundary = np.asarray([[-5.0, -2.0], [25.0, -2.0]])
    lane_calls = []

    def lane_features(*args, **kwargs):
        lane_calls.append((args[2].geometry_id, kwargs))
        return [lane_boundary], [lane_boundary]

    monkeypatch.setattr(nuplan_targets, "_lane_features", lane_features)
    monkeypatch.setattr(
        nuplan_targets,
        "_route_polygons",
        lambda _scenario: [static_polygon],
    )
    return lane_calls


def _bev_geometry() -> NavigationRasterGeometry:
    """A coarser label grid inside ``_geometry``, like AutoBEV in the 0.4 m raster."""
    return NavigationRasterGeometry(
        geometry_id="test-multistage-bev-v1",
        height_px=20,
        width_px=10,
        meters_per_pixel=1.0,
        x_min_m=-5.0,
        x_max_m=15.0,
        y_min_m=-5.0,
        y_max_m=5.0,
        ego_anchor_row=14.5,
        ego_anchor_col=4.5,
        matching_pc_range=(-5.0, -5.0, -0.5, 15.0, 5.0, 3.5),
        matching_bev_h=20,
        matching_bev_w=10,
        route_corridor_width_m=3.5,
        destination_marker_radius_m=2.0,
        route_rear_clip_m=10.0,
    )


def _ring(radius: float) -> np.ndarray:
    """LiDAR returns on a circle 0.5 m above the rear axle."""
    angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    return np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full_like(angles, 0.5),
    ])


def test_nuplan_full_target_builder_uses_current_annotations(
    monkeypatch,
):
    scenario = _Scenario()
    geometry = _geometry()
    bev_geometry = _bev_geometry()
    lane_calls = _patch_map(monkeypatch)
    # One return inside the vehicle box keeps it observed.
    points = np.concatenate([_ring(14.0), [[5.0, 0.0, 0.5]]])

    targets = nuplan_targets.build_nuplan_reactive_targets(
        scenario,
        geometry=geometry,
        bev_geometry=bev_geometry,
        camera_visibility=np.ones((20, 10), dtype=np.bool_),
        lidar_points_ego=points,
    )

    assert targets.bev_segmentation.shape == (8, 20, 10)
    assert targets.bev_segmentation[0].max() == pytest.approx(1.0)
    assert targets.bev_segmentation[1].max() == pytest.approx(1.0)
    assert not np.array_equal(
        targets.bev_segmentation[0],
        targets.bev_segmentation[1],
    )
    vehicle = targets.bev_segmentation[5]
    assert vehicle.max() == pytest.approx(1.0)
    assert targets.bev_segmentation_valid[5][vehicle > 0.5].all()
    assert targets.map_context.shape == (14, 40, 20)
    assert targets.route_target.shape == (2, 40, 20)
    assert targets.route_channel_valid.tolist() == [True, True]
    assert targets.route_target[1].max() == pytest.approx(1.0)
    assert scenario.future_calls == 1
    assert lane_calls == [
        (geometry.geometry_id, {}),
        (bev_geometry.geometry_id, {"include_connectors": False}),
    ]

    members = nuplan_targets.nuplan_reactive_target_members(
        targets,
        geometry=geometry,
        metadata={"scenario_token": "scenario-1"},
    )
    trajectory_xy, trajectory_valid = decode_trajectory_xy(
        members["trajectory_xy.npz"]
    )
    bev_target, bev_valid = decode_bev_segmentation(
        members["bev_segmentation.npz"]
    )
    bev_stats = decode_bev_segmentation_stats(
        members["bev_segmentation_stats.json"]
    )
    navigation_metadata = json.loads(members["navigation_meta.json"])
    assert trajectory_xy.shape == (64, 2)
    assert trajectory_valid.all()
    assert bev_target.shape == (8, 20, 10)
    assert bev_valid.shape == bev_target.shape
    assert tuple(BEV_SEGMENTATION_CLASSES) == (
        "drivable_area",
        "lane_boundary",
        "intersection",
        "crosswalk",
        "stop_line",
        "vehicle",
        "vulnerable_road_user",
        "other_obstacle",
    )
    assert bev_stats["positive_cell_count"][0] > (
        bev_stats["positive_cell_count"][1]
    )
    assert decode_array(members["map_semantic.npz"]).shape == (14, 40, 20)
    assert decode_array(members["route_mask.npz"]).shape == (2, 40, 20)
    assert navigation_metadata["map_source"] == "nuplan_native"
    assert navigation_metadata["scenario_token"] == "scenario-1"


def test_nuplan_box_without_lidar_return_is_unknown_for_objects(
    monkeypatch,
):
    scenario = _Scenario()
    _patch_map(monkeypatch)
    # Returns reach past the box, but none lies inside it; a road return
    # under the box does not count as seeing it.
    points = np.concatenate([_ring(14.0), [[5.0, 0.0, -0.2]]])

    targets = nuplan_targets.build_nuplan_reactive_targets(
        scenario,
        geometry=_geometry(),
        bev_geometry=_bev_geometry(),
        camera_visibility=np.ones((20, 10), dtype=np.bool_),
        lidar_points_ego=points,
    )

    vehicle_cells = targets.bev_segmentation[5] > 0.5
    assert vehicle_cells.any()
    for class_index in (5, 6, 7):
        assert not targets.bev_segmentation_valid[class_index][
            vehicle_cells
        ].any()
    # Map classes stay valid over the unseen box.
    assert targets.bev_segmentation_valid[0][vehicle_cells].all()
    assert targets.bev_segmentation_valid[5].any()


def test_nuplan_boxes_are_in_the_current_rear_axle_frame():
    tracked = _tracked_vehicle(center_x=15.0)
    tracked.tracked_object_type = types.SimpleNamespace(name="CZONE_SIGN")

    boxes, labels = nuplan_targets.nuplan_boxes(
        [tracked],
        (10.0, 0.0, np.pi / 2),
    )

    np.testing.assert_allclose(
        boxes,
        [[0.0, -5.0, 0.75, 2.0, 2.0, 1.5, -np.pi / 2, 0.0, -3.0]],
        atol=1e-5,
    )
    assert labels.tolist() == [3]


def test_nuplan_bev_builder_rejects_visibility_on_another_grid(monkeypatch):
    _patch_map(monkeypatch)
    with pytest.raises(ValueError, match="camera visibility"):
        nuplan_targets.build_nuplan_reactive_targets(
            _Scenario(),
            geometry=_geometry(),
            bev_geometry=_bev_geometry(),
            camera_visibility=np.ones((40, 20), dtype=np.bool_),
            lidar_points_ego=np.zeros((1, 3)),
        )


def test_nuplan_visibility_helpers_use_metric_geometry():
    geometry = _geometry()
    # Camera looks along ego +X with image coordinates centered at (10, 10).
    projection = np.asarray([[
        [0.0, -1.0, 0.0, 10.0],
        [0.0, 0.0, -1.0, 10.0],
        [1.0, 0.0, 0.0, 0.0],
    ]])
    visibility = camera_visibility_from_projection_matrices(
        projection,
        image_width=20,
        image_height=20,
        geometry=geometry,
    )
    assert visibility.shape == (40, 20)
    x_grid, _ = geometry.pixel_center_grids()
    assert visibility[x_grid > 0.0].any()
    assert not visibility[x_grid < 0.0].any()
    assert NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M == (-4.0, -2.0, 0.0, 2.0)

    lidar = nuplan_targets.lidar_observability_from_points(
        np.asarray([[5.0, 0.0, 0.0], [8.0, 1.0, 0.0]]),
        geometry=geometry,
        angular_bins=360,
    )
    assert lidar.shape == visibility.shape
    assert lidar.any()
    assert not lidar[0].any()


def test_nuplan_point_cloud_layout_rejects_points_first():
    channels_first = np.arange(6 * 100, dtype=np.float64).reshape(6, 100)

    xyz = _nuplan_point_cloud_xyz(channels_first)

    assert xyz.shape == (100, 3)
    np.testing.assert_array_equal(xyz, channels_first[:3].T)
    with pytest.raises(ValueError, match="channels-first"):
        _nuplan_point_cloud_xyz(channels_first.T)


def _camera_bundle() -> NuPlanCameraBundle:
    jpeg = encode_camera(np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8))
    # Every camera looks along ego +X from the origin.
    camera_from_ego = np.asarray([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    intrinsic = np.asarray([
        [500.0, 0.0, IMAGE_W / 2],
        [0.0, 500.0, IMAGE_H / 2],
        [0.0, 0.0, 1.0],
    ])
    projection = np.repeat(
        (intrinsic @ camera_from_ego[:3])[None],
        len(NUPLAN_CAMERA_CHANNELS),
        axis=0,
    ).astype(np.float32)
    return NuPlanCameraBundle(
        jpegs=(jpeg,) * len(NUPLAN_CAMERA_CHANNELS),
        projection=projection,
        camera_from_ego=np.repeat(
            camera_from_ego[None],
            len(NUPLAN_CAMERA_CHANNELS),
            axis=0,
        ),
        history_jpegs=((jpeg,) * len(NUPLAN_CAMERA_CHANNELS),)
        * REACTIVE_BEVFORMER_HISTORY_FRAMES,
        history_projection=np.repeat(
            projection[None],
            REACTIVE_BEVFORMER_HISTORY_FRAMES,
            axis=0,
        ),
        metadata={
            "camera_order": list(NUPLAN_CAMERA_CHANNELS),
            "camera_slots": list(NUPLAN_CAMERA_SLOTS),
            "cameras": [
                {
                    "channel": channel,
                    "image_time_offset_us": (
                        NUPLAN_CAMERA_SYNC_TOLERANCE_US
                        if index == 0
                        else 0
                    ),
                }
                for index, channel in enumerate(NUPLAN_CAMERA_CHANNELS)
            ],
            "image_wh": [IMAGE_W, IMAGE_H],
            "rectification_policy": "test",
            "reference_lidar_timestamp_us": 4_000_000,
            "temporal_camera_timestamps_us": [
                [
                    4_000_000
                    + offset * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                    for _ in NUPLAN_CAMERA_CHANNELS
                ]
                for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]
            ],
            "temporal_frame_interval_us": (
                REACTIVE_BEVFORMER_FRAME_INTERVAL_US
            ),
            "temporal_frame_offsets": list(
                REACTIVE_BEVFORMER_FRAME_OFFSETS
            ),
        },
    )


def _packing_scenario(map_name: str = "us-nv-las-vegas-strip") -> _Scenario:
    scenario = _Scenario()
    scenario.log_name = "log-a.db"
    scenario.token = "token-a"
    scenario.map_version = "nuplan-maps-v1.0"
    scenario.map_api = types.SimpleNamespace(map_name=map_name)
    scenario.get_ego_past_trajectory = (
        lambda _iteration, *, time_horizon, num_samples: iter(
            [_EgoState(float(index - 64), 0.0) for index in range(64)]
        )
    )
    return scenario


def _fake_targets() -> nuplan_targets.NuPlanReactiveTargets:
    return nuplan_targets.NuPlanReactiveTargets(
        trajectory_xy_m=np.zeros((64, 2), dtype=np.float32),
        trajectory_valid=np.ones(64, dtype=np.bool_),
        initial_speed_mps=5.0,
        map_context=np.zeros((14, 450, 300), dtype=np.float32),
        map_valid=True,
        bev_segmentation=np.zeros(
            (8, AUTOBEV_GEOMETRY.height_px, AUTOBEV_GEOMETRY.width_px),
            dtype=np.float32,
        ),
        bev_segmentation_valid=np.ones(
            (8, AUTOBEV_GEOMETRY.height_px, AUTOBEV_GEOMETRY.width_px),
            dtype=np.bool_,
        ),
        route_target=np.zeros((2, 450, 300), dtype=np.float32),
        route_channel_valid=np.ones(2, dtype=np.bool_),
    )


def test_nuplan_sample_members_carry_the_autobev_contract():
    target_calls = []

    def target_builder(*_args, **kwargs):
        target_calls.append(kwargs)
        return _fake_targets()

    # One return 10 m ahead on the optical axis of every camera.
    points = np.asarray([[10.0, 0.0, 0.0]])
    sample_uid, split_group_uid, members = (
        nuplan_reactive_sample_members(
            _packing_scenario(),
            source_revision="nuplan-v1.1-test",
            camera_bundle=_camera_bundle(),
            lidar_points_ego=points,
            target_builder=target_builder,
        )
    )

    assert sample_uid.startswith("nuplan-")
    assert split_group_uid.startswith("nuplan-log-")
    (call,) = target_calls
    assert call["camera_visibility"].shape == (
        AUTOBEV_GEOMETRY.height_px,
        AUTOBEV_GEOMETRY.width_px,
    )
    np.testing.assert_array_equal(call["lidar_points_ego"], points)
    assert decode_npy(members["projection.npy"]).shape == (6, 3, 4)
    assert decode_npy(members["history_projection.npy"]).shape == (
        7,
        6,
        3,
        4,
    )
    assert "front_camera_fpn.jpg" not in members
    assert sum(name.endswith(".jpg") for name in members) == 6 * 8
    assert "hist_6_cam_5.jpg" in members
    depth = decode_stride_depth(members["depth.npz"])
    assert depth.shape == (6, IMAGE_H // 8, IMAGE_W // 8)
    assert depth[:, IMAGE_H // 16, IMAGE_W // 16] == pytest.approx(10.0)
    boxes, labels = decode_boxes(members["boxes.npz"])
    assert boxes.shape == (1, 9)
    assert labels.tolist() == [0]
    assert json.loads(members["meta.json"])["map_name"] == (
        "us-nv-las-vegas-strip"
    )


def test_nuplan_pittsburgh_samples_get_a_held_out_split_group():
    _, split_group_uid, _ = nuplan_reactive_sample_members(
        _packing_scenario("us-pa-pittsburgh-hazelwood"),
        source_revision="nuplan-v1.1-test",
        camera_bundle=_camera_bundle(),
        lidar_points_ego=np.zeros((1, 3)),
        target_builder=lambda *_args, **_kwargs: _fake_targets(),
    )

    assert split_group_uid.startswith("heldout-nuplan-log-")
    assert split_bucket(split_group_uid) == 0
    assert all(
        _is_validation_group(split_group_uid, fraction)
        for fraction in (0.1, 0.2, 0.5)
    )


def test_nuplan_packer_emits_log_grouped_immutable_shards(
    tmp_path,
):
    def sample_builder(raw_scenario, *, iteration, source_revision):
        return nuplan_reactive_sample_members(
            raw_scenario,
            iteration=iteration,
            source_revision=source_revision,
            camera_bundle=_camera_bundle(),
            lidar_points_ego=np.zeros((1, 3)),
            target_builder=lambda *_args, **_kwargs: _fake_targets(),
        )

    manifest = pack_nuplan_reactive_scenarios(
        [_packing_scenario()],
        tmp_path,
        source_revision="nuplan-v1.1-test",
        map_version="nuplan-maps-v1.0",
        sample_builder=sample_builder,
    )

    assert manifest["total_samples"] == 1
    assert manifest["bev_statistics_count"] == 1
    assert (
        manifest["bev_taxonomy_version"]
        == BEV_SEGMENTATION_TAXONOMY_VERSION
    )
    assert manifest["split_policy"] == "log_level_hash_bucket"
    assert manifest["navigation_geometry"] == (
        nuplan_targets.AUTOE2E_NAVIGATION_GEOMETRY.contract()
    )
    assert manifest["sample_kind"] == "planner"
    assert manifest["image_size"] == [IMAGE_W, IMAGE_H]
    assert {
        key: manifest[key] for key in manifest_contract()
    } == manifest_contract()
    assert "front_camera_index" not in manifest
    assert manifest["max_camera_time_offset_us"] == (
        NUPLAN_CAMERA_SYNC_TOLERANCE_US
    )
    assert manifest["max_history_camera_time_offset_us"] == 0
    assert manifest["history_camera_spread_max_us"] == 100_000
    assert (
        manifest["distinct_history_frame_count"]
        == REACTIVE_BEVFORMER_HISTORY_FRAMES
    )
    tar_path = tmp_path / manifest["shard_names"][0]
    assert manifest["shard_sha256"][tar_path.name] == hashlib.sha256(
        tar_path.read_bytes()
    ).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(tar_path.read_bytes())) as archive:
        names = archive.getnames()
        assert sum(name.endswith(".jpg") for name in names) == (
            len(NUPLAN_CAMERA_CHANNELS)
            * (REACTIVE_BEVFORMER_HISTORY_FRAMES + 1)
        )
        for suffix in (
            ".trajectory_xy.npz",
            ".bev_segmentation.npz",
            ".projection.npy",
            ".history_projection.npy",
            ".depth.npz",
            ".boxes.npz",
        ):
            assert any(name.endswith(suffix) for name in names)
        meta_name = next(name for name in names if name.endswith(".meta.json"))
        metadata = json.load(archive.extractfile(meta_name))
        assert metadata["split_group_uid"].startswith("nuplan-log-")
