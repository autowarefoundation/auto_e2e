"""Dataset adapters for the nuPlan -> L2D training sequence."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import pickle
import sqlite3
from pathlib import Path
import sys
import tarfile
import types

import networkx as nx
import numpy as np
import pytest
from PIL import Image

import data_parsing.l2d.navigation as l2d_navigation
import data_parsing.l2d.osm_graph_builder as osm_graph_builder
from data_parsing.l2d.osm_graph_builder import (
    L2D_OSM_GRAPH_ADAPTER_VERSION,
    OSMWayRecord,
    encode_l2d_osm_graph_snapshot,
)
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
    lidar_observability_from_points,
    pack_nuplan_reactive_scenarios,
)
import data_parsing.nuplan.targets as nuplan_targets
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
    decode_bev_segmentation,
    decode_bev_segmentation_stats,
    decode_trajectory_xy,
)
from navigation.artifacts import decode_array
from navigation.geometry import NavigationRasterGeometry
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
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


def test_history_camera_projection_uses_current_ego_reference(
    tmp_path,
    monkeypatch,
):
    fake_cv2 = types.SimpleNamespace(
        getOptimalNewCameraMatrix=lambda intrinsic, *_args: (
            intrinsic.copy(),
            None,
        ),
        undistort=lambda image, *_args: image,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    image_path = tmp_path / "history.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    identity_quaternion = np.asarray(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    row = {
        "distortion": pickle.dumps(np.zeros(5, dtype=np.float64)),
        "filename_jpg": image_path.name,
        "height": 4,
        "intrinsic": pickle.dumps(np.eye(3, dtype=np.float64)),
        "model": "DesignCore D3CM-IMX390",
        "qw": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "rotation": pickle.dumps(identity_quaternion),
        "timestamp": 6_500_000,
        "translation": pickle.dumps(np.zeros(3, dtype=np.float64)),
        "width": 4,
        "x": 8.0,
        "y": 0.0,
        "z": 0.0,
    }
    reference_pose = _quaternion_transform(
        [10.0, 0.0, 0.0],
        identity_quaternion,
    )

    _, matrices, front_matrix, metadata = _rectify_camera_rows(
        {channel: row for channel in NUPLAN_CAMERA_CHANNELS},
        sensor_root=str(tmp_path),
        reference_pose=reference_pose,
        reference_timestamp=10_000_000,
        image_size=REACTIVE_CAMERA_IMAGE_SIZE,
        native_front=False,
    )

    scaled_intrinsic = np.eye(3, dtype=np.float64)
    scaled_intrinsic[0] *= REACTIVE_CAMERA_IMAGE_SIZE / 4
    scaled_intrinsic[1] *= REACTIVE_CAMERA_IMAGE_SIZE / 4
    image_from_current = np.eye(4, dtype=np.float64)
    image_from_current[0, 3] = 2.0
    expected = scaled_intrinsic @ image_from_current[:3]
    np.testing.assert_allclose(matrices[0], expected)
    assert front_matrix is None
    assert {
        camera["sensor_model"] for camera in metadata
    } == {"DesignCore D3CM-IMX390"}


def test_nuplan_camera_rectification_rejects_non_caltech_distortion(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(),
    )
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
            image_size=REACTIVE_CAMERA_IMAGE_SIZE,
            native_front=False,
        )


def test_nuplan_projection_inverts_sensor_to_ego_yaw(
    tmp_path,
    monkeypatch,
):
    fake_cv2 = types.SimpleNamespace(
        getOptimalNewCameraMatrix=lambda intrinsic, *_args: (
            intrinsic.copy(),
            None,
        ),
        undistort=lambda image, *_args: image,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    image_path = tmp_path / "yaw.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    root_half = np.sqrt(0.5)
    sensor_to_ego_yaw_90 = np.asarray(
        [root_half, 0.0, 0.0, root_half],
        dtype=np.float64,
    )
    row = {
        "distortion": pickle.dumps(np.zeros(5, dtype=np.float64)),
        "filename_jpg": image_path.name,
        "height": 4,
        "intrinsic": pickle.dumps(np.eye(3, dtype=np.float64)),
        "model": "DesignCore D3CM-IMX390",
        "qw": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "rotation": pickle.dumps(sensor_to_ego_yaw_90),
        "timestamp": 10_000_000,
        "translation": pickle.dumps(np.zeros(3, dtype=np.float64)),
        "width": 4,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }

    _, matrices, _, _ = _rectify_camera_rows(
        {channel: row for channel in NUPLAN_CAMERA_CHANNELS},
        sensor_root=str(tmp_path),
        reference_pose=np.eye(4, dtype=np.float64),
        reference_timestamp=10_000_000,
        image_size=REACTIVE_CAMERA_IMAGE_SIZE,
        native_front=False,
    )

    scaled_intrinsic = np.eye(3, dtype=np.float64)
    scaled_intrinsic[0] *= REACTIVE_CAMERA_IMAGE_SIZE / 4
    scaled_intrinsic[1] *= REACTIVE_CAMERA_IMAGE_SIZE / 4
    camera_from_ego = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    expected = scaled_intrinsic @ camera_from_ego[:3]
    np.testing.assert_allclose(matrices[0], expected, atol=1e-12)


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


class _Scenario:
    def __init__(self):
        self.future_calls = 0
        self.states = [
            _EgoState(float(index), 0.0)
            for index in range(65)
        ]
        polygon = types.SimpleNamespace(
            is_empty=False,
            geom_type="Polygon",
            exterior=types.SimpleNamespace(
                coords=[
                    (4.0, -1.0),
                    (6.0, -1.0),
                    (6.0, 1.0),
                    (4.0, 1.0),
                    (4.0, -1.0),
                ]
            ),
        )
        box = types.SimpleNamespace(
            geometry=polygon,
        )
        tracked = types.SimpleNamespace(
            tracked_object_type=types.SimpleNamespace(name="VEHICLE"),
            box=box,
        )
        self._detections = types.SimpleNamespace(
            tracked_objects=[tracked]
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


def test_nuplan_full_target_builder_uses_current_annotations(
    monkeypatch,
):
    scenario = _Scenario()
    geometry = _geometry()
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
    lane_boundary = np.asarray(
        [[-5.0, -2.0], [25.0, -2.0]],
        dtype=np.float64,
    )
    lane_calls = []

    def lane_features(*_args, **kwargs):
        lane_calls.append(kwargs)
        return [lane_boundary], [lane_boundary]

    monkeypatch.setattr(
        nuplan_targets,
        "_lane_features",
        lane_features,
    )
    monkeypatch.setattr(
        nuplan_targets,
        "_route_polygons",
        lambda _scenario: [static_polygon],
    )
    visibility = np.ones(
        (geometry.height_px, geometry.width_px),
        dtype=np.bool_,
    )

    targets = nuplan_targets.build_nuplan_reactive_targets(
        scenario,
        geometry=geometry,
        camera_visibility=visibility,
        lidar_observability=visibility,
    )

    assert targets.bev_segmentation.shape == (8, 40, 20)
    assert targets.bev_segmentation[0].max() == pytest.approx(1.0)
    assert targets.bev_segmentation[1].max() == pytest.approx(1.0)
    assert not np.array_equal(
        targets.bev_segmentation[0],
        targets.bev_segmentation[1],
    )
    assert targets.bev_segmentation[5].max() == pytest.approx(1.0)
    assert targets.route_target.shape == (2, 40, 20)
    assert targets.route_channel_valid.tolist() == [True, True]
    assert targets.route_target[1].max() == pytest.approx(1.0)
    assert scenario.future_calls == 1
    assert lane_calls == [
        {},
        {"include_connectors": False},
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
    navigation_metadata = json.loads(
        members["navigation_meta.json"]
    )
    assert trajectory_xy.shape == (64, 2)
    assert trajectory_valid.all()
    assert bev_target.shape == (8, 40, 20)
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


def test_nuplan_bev_builder_rejects_unknown_observability(monkeypatch):
    scenario = _Scenario()
    monkeypatch.setattr(
        nuplan_targets,
        "_map_layer_polygons",
        lambda *_args: ({}, {}),
    )
    with pytest.raises(ValueError, match="observability"):
        nuplan_targets.build_nuplan_reactive_targets(
            scenario,
            geometry=_geometry(),
        )


def test_l2d_navigation_uses_only_route_waypoints(monkeypatch):
    graph = nx.MultiDiGraph()
    base_lon = 8.0
    base_lat = 52.0
    for node in range(10):
        graph.add_node(
            node,
            x=base_lon,
            y=base_lat + node * 0.00001,
        )
    for node in range(9):
        graph.add_edge(node, node + 1, lanes=2)
    monkeypatch.setattr(
        l2d_navigation,
        "_map_match_waypoints",
        lambda *_args, **_kwargs: (list(range(10)), list(range(10))),
    )
    waypoints = np.asarray(
        [
            [base_lon, base_lat + node * 0.00001]
            for node in range(10)
        ],
        dtype=np.float64,
    )

    targets = l2d_navigation.build_l2d_navigation_targets(
        graph,
        waypoints,
        ego_lat=base_lat,
        ego_lon=base_lon,
        heading_deg_cw_from_north=0.0,
        geometry=_geometry(),
    )

    assert targets.map_context.shape == (14, 40, 20)
    assert targets.map_valid
    assert targets.route_target.shape == (2, 40, 20)
    assert targets.route_channel_valid.tolist() == [True, True]
    assert targets.route_target[1].max() == pytest.approx(1.0)
    assert targets.route_node_count == 10


def test_l2d_spatial_index_matches_full_graph_rasterization():
    graph = nx.MultiDiGraph()
    base_lon = 8.0
    base_lat = 52.0
    for node in range(10):
        graph.add_node(
            f"near-{node}",
            x=base_lon,
            y=base_lat + node * 0.00001,
        )
    for node in range(9):
        graph.add_edge(
            f"near-{node}",
            f"near-{node + 1}",
            key="0",
            length=1.1,
            lanes=2,
        )
    for node in range(101):
        graph.add_node(
            f"far-{node}",
            x=base_lon + 1.0,
            y=base_lat + node * 0.00001,
        )
    for node in range(100):
        graph.add_edge(
            f"far-{node}",
            f"far-{node + 1}",
            key="0",
            length=1.1,
            lanes=2,
        )
    waypoints = np.asarray(
        [
            [base_lon, base_lat + node * 0.00001]
            for node in range(10)
        ],
        dtype=np.float64,
    )
    spatial_index = l2d_navigation._build_graph_spatial_index(graph)

    full_targets = l2d_navigation.build_l2d_navigation_targets(
        graph,
        waypoints,
        ego_lat=base_lat,
        ego_lon=base_lon,
        heading_deg_cw_from_north=0.0,
        geometry=_geometry(),
    )
    indexed_targets = l2d_navigation.build_l2d_navigation_targets(
        graph,
        waypoints,
        ego_lat=base_lat,
        ego_lon=base_lon,
        heading_deg_cw_from_north=0.0,
        geometry=_geometry(),
        spatial_index=spatial_index,
    )
    candidate_edges = l2d_navigation._candidate_edge_refs(
        graph,
        spatial_index,
        ego_lat=base_lat,
        ego_lon=base_lon,
        geometry=_geometry(),
    )

    np.testing.assert_array_equal(
        indexed_targets.map_context,
        full_targets.map_context,
    )
    np.testing.assert_array_equal(
        indexed_targets.route_target,
        full_targets.route_target,
    )
    np.testing.assert_array_equal(
        indexed_targets.route_channel_valid,
        full_targets.route_channel_valid,
    )
    assert indexed_targets.matched_node_count == 10
    assert indexed_targets.route_node_count == 10
    assert len(candidate_edges) == 9
    assert len(candidate_edges) < graph.number_of_edges()


def test_l2d_osm_snapshot_encodes_common_navigation_members(tmp_path):
    base_lon = 8.0
    base_lat = 52.0
    payload = {
        "schema_version": "l2d_osm_graph_v1",
        "adapter_version": L2D_OSM_GRAPH_ADAPTER_VERSION,
        "source_artifact_sha256": "a" * 64,
        "source_date": "2026-08-01",
        "source_revision": "geofabrik-2026-08-01",
        "attribution": "OpenStreetMap contributors",
        "nodes": [
            {
                "id": str(index),
                "longitude_deg": base_lon,
                "latitude_deg": base_lat + index * 0.00001,
            }
            for index in range(10)
        ],
        "edges": [
            {
                "source": str(index),
                "destination": str(index + 1),
                "key": "0",
                "length_m": 1.1,
                "lanes": 2,
            }
            for index in range(9)
        ],
    }
    snapshot_path = tmp_path / "osm-graph.json"
    snapshot_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="ascii",
    )
    snapshot = l2d_navigation.load_l2d_osm_graph_snapshot(
        snapshot_path
    )
    assert snapshot.spatial_index.node_coordinates_lon_lat.shape == (10, 2)
    assert not snapshot.spatial_index.node_coordinates_lon_lat.flags.writeable
    assert len(snapshot.spatial_index.edge_refs) == 9
    waypoints = np.asarray(
        [
            [base_lon, base_lat + index * 0.00001]
            for index in range(10)
        ],
        dtype=np.float64,
    )
    members = l2d_navigation.l2d_reactive_navigation_members(
        snapshot,
        waypoints,
        {
            "latitude_deg": base_lat,
            "longitude_deg": base_lon,
            "heading_deg_cw_from_north": 0.0,
            "timestamp_ns": 1,
        },
        geometry=_geometry(),
    )

    metadata = json.loads(members["navigation_meta.json"])
    map_context = decode_array(members["map_semantic.npz"])
    route_mask = decode_array(members["route_mask.npz"])
    assert map_context.shape == (14, 40, 20)
    assert route_mask.shape == (2, 40, 20)
    assert metadata["map_source"] == "pinned_osm_graph"
    assert metadata["route_source"] == (
        "l2d_observation_state_waypoints"
    )
    assert metadata["osm_source_revision"] == "geofabrik-2026-08-01"
    assert metadata["osm_source_sha256"] == snapshot.source_sha256
    assert metadata["osm_source_artifact_sha256"] == "a" * 64
    assert metadata["osm_source_date"] == "2026-08-01"
    assert metadata["osm_adapter_version"] == (
        L2D_OSM_GRAPH_ADAPTER_VERSION
    )


def test_osm_graph_snapshot_encoding_is_order_independent():
    nodes = {
        "3": (8.0, 52.00002),
        "1": (8.0, 52.0),
        "2": (8.0, 52.00001),
    }
    ways = [
        OSMWayRecord(
            way_id="20",
            node_ids=("2", "3"),
            highway="residential",
            oneway="yes",
            lanes="1",
        ),
        OSMWayRecord(
            way_id="10",
            node_ids=("1", "2"),
            highway="primary",
            lanes="2",
            width_m="7 m",
        ),
    ]
    kwargs = {
        "source_revision": "geofabrik-2026-08-01",
        "source_date": "2026-08-01",
        "source_artifact_sha256": "b" * 64,
        "attribution": "OpenStreetMap contributors",
    }
    first = encode_l2d_osm_graph_snapshot(nodes, ways, **kwargs)
    second = encode_l2d_osm_graph_snapshot(
        dict(reversed(list(nodes.items()))),
        list(reversed(ways)),
        **kwargs,
    )

    assert first == second
    payload = json.loads(first)
    assert payload["adapter_version"] == L2D_OSM_GRAPH_ADAPTER_VERSION
    assert [node["id"] for node in payload["nodes"]] == ["1", "2", "3"]
    assert len(payload["edges"]) == 3


def test_osm_graph_snapshot_indexes_normalized_nodes_once():
    source = Path(osm_graph_builder.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "encode_l2d_osm_graph_snapshot"
    )
    index_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "normalized_nodes"
    ]
    way_loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "way"
    )

    assert len(index_calls) == 1
    assert index_calls[0].lineno < way_loop.lineno


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

    lidar = lidar_observability_from_points(
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


def test_nuplan_packer_emits_log_grouped_immutable_shards(
    tmp_path,
):
    geometry = nuplan_targets.AUTOE2E_NAVIGATION_GEOMETRY
    scenario = _Scenario()
    scenario.log_name = "log-a.db"
    scenario.token = "token-a"
    scenario.map_version = "nuplan-maps-v1.0"
    scenario.get_ego_past_trajectory = (
        lambda _iteration, *, time_horizon, num_samples: iter(
            [_EgoState(float(index - 64), 0.0) for index in range(64)]
        )
    )
    visibility = np.ones(
        (geometry.height_px, geometry.width_px),
        dtype=np.bool_,
    )
    projection_matrices = np.zeros(
        (len(NUPLAN_CAMERA_CHANNELS), 3, 4),
        dtype=np.float32,
    )
    projection_matrices[:, 0, 3] = REACTIVE_CAMERA_IMAGE_SIZE / 2
    projection_matrices[:, 1, 3] = REACTIVE_CAMERA_IMAGE_SIZE / 2
    projection_matrices[:, 2, 3] = 1.0
    front_projection_matrix = projection_matrices[
        REACTIVE_FRONT_CAMERA_INDEX:
        REACTIVE_FRONT_CAMERA_INDEX + 1
    ].copy()
    front_projection_matrix[:, :2] *= (
        REACTIVE_FRONT_CAMERA_IMAGE_SIZE / REACTIVE_CAMERA_IMAGE_SIZE
    )
    bundle = NuPlanCameraBundle(
        jpeg_by_channel={
            channel: b"\xff\xd8\xff\xd9"
            for channel in NUPLAN_CAMERA_CHANNELS
        },
        projection_matrices=projection_matrices,
        front_projection_matrix=front_projection_matrix,
        history_jpeg_by_frame=tuple(
            {
                channel: b"\xff\xd8\xff\xd9"
                for channel in NUPLAN_CAMERA_CHANNELS
            }
            for _ in range(REACTIVE_BEVFORMER_HISTORY_FRAMES)
        ),
        history_projection_matrices=np.repeat(
            projection_matrices[None],
            REACTIVE_BEVFORMER_HISTORY_FRAMES,
            axis=0,
        ),
        camera_visibility=visibility,
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
            "front_camera_image_size": REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
            "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
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
    target = nuplan_targets.NuPlanReactiveTargets(
        trajectory_xy_m=np.zeros((64, 2), dtype=np.float32),
        trajectory_valid=np.ones(64, dtype=np.bool_),
        initial_speed_mps=5.0,
        map_context=np.zeros((14, 450, 300), dtype=np.float32),
        map_valid=True,
        bev_segmentation=np.zeros((8, 450, 300), dtype=np.float32),
        bev_segmentation_valid=np.ones(
            (8, 450, 300),
            dtype=np.bool_,
        ),
        route_target=np.zeros((2, 450, 300), dtype=np.float32),
        route_channel_valid=np.ones(2, dtype=np.bool_),
    )

    def sample_builder(
        raw_scenario,
        *,
        iteration,
        image_size,
        source_revision,
    ):
        from data_parsing.nuplan.packing import (
            nuplan_reactive_sample_members,
        )

        return nuplan_reactive_sample_members(
            raw_scenario,
            iteration=iteration,
            image_size=image_size,
            source_revision=source_revision,
            camera_bundle=bundle,
            lidar_observability=visibility,
            target_builder=lambda *_args, **_kwargs: target,
        )

    manifest = pack_nuplan_reactive_scenarios(
        [scenario],
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
    assert manifest["navigation_geometry"] == geometry.contract()
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
        assert any(name.endswith(".trajectory_xy.npz") for name in names)
        assert any(name.endswith(".bev_segmentation.npz") for name in names)
        meta_name = next(name for name in names if name.endswith(".meta.json"))
        metadata = json.load(archive.extractfile(meta_name))
        assert metadata["split_group_uid"].startswith("nuplan-log-")
        calib_name = next(
            name for name in names if name.endswith(".calib.json")
        )
        calibration = json.load(archive.extractfile(calib_name))
        assert calibration["front_camera_index"] == (
            REACTIVE_FRONT_CAMERA_INDEX
        )
        assert calibration["front_camera_image_size"] == (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE
        )
        assert np.asarray(
            calibration["front_projection"]["matrix"]
        ).shape == (1, 3, 4)
        expected_front = np.asarray(
            calibration["projection"]["matrix"],
            dtype=np.float32,
        )[REACTIVE_FRONT_CAMERA_INDEX:
          REACTIVE_FRONT_CAMERA_INDEX + 1].copy()
        expected_front[:, :2] *= (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            / REACTIVE_CAMERA_IMAGE_SIZE
        )
        np.testing.assert_allclose(
            calibration["front_projection"]["matrix"],
            expected_front,
        )
        assert np.asarray(
            calibration["history_projection"]["matrix"]
        ).shape == (
            REACTIVE_BEVFORMER_HISTORY_FRAMES,
            len(NUPLAN_CAMERA_CHANNELS),
            3,
            4,
        )
        assert calibration["history_projection"]["reference_frame"] == (
            "current_ego"
        )
