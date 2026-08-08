"""Dataset adapters for the nuPlan -> L2D training sequence."""

from __future__ import annotations

import json
import hashlib
import io
import tarfile
import types

import networkx as nx
import numpy as np
import pytest

import data_parsing.l2d.navigation as l2d_navigation
from data_parsing.l2d.osm_graph_builder import (
    L2D_OSM_GRAPH_ADAPTER_VERSION,
    OSMWayRecord,
    encode_l2d_osm_graph_snapshot,
)
from data_parsing.nuplan.packing import (
    NUPLAN_CAMERA_CHANNELS,
    NuPlanCameraBundle,
    camera_visibility_from_projection_matrices,
    lidar_observability_from_points,
    pack_nuplan_reactive_scenarios,
)
import data_parsing.nuplan.targets as nuplan_targets
from data_processing.reactive_training_artifacts import (
    decode_bev_segmentation,
    decode_trajectory_xy,
)
from navigation.artifacts import decode_array
from navigation.geometry import NavigationRasterGeometry


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
        "lane_area": [static_polygon],
        "intersection": [],
        "crosswalk": [],
        "stop_line": [],
    }
    map_available = {
        "drivable_area": True,
        "lane_area": True,
        "intersection": True,
        "crosswalk": True,
        "stop_line": True,
    }
    monkeypatch.setattr(
        nuplan_targets,
        "_map_layer_polygons",
        lambda *_args: (map_polygons, map_available),
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
    assert targets.bev_segmentation[5].max() == pytest.approx(1.0)
    assert targets.route_target.shape == (2, 40, 20)
    assert targets.route_channel_valid.tolist() == [True, True]
    assert targets.route_target[1].max() == pytest.approx(1.0)
    assert scenario.future_calls == 1

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
    navigation_metadata = json.loads(
        members["navigation_meta.json"]
    )
    assert trajectory_xy.shape == (64, 2)
    assert trajectory_valid.all()
    assert bev_target.shape == (8, 40, 20)
    assert bev_valid.shape == bev_target.shape
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
    assert visibility.any()
    assert not visibility[-1].any()

    lidar = lidar_observability_from_points(
        np.asarray([[5.0, 0.0, 0.0], [8.0, 1.0, 0.0]]),
        geometry=geometry,
        angular_bins=360,
    )
    assert lidar.shape == visibility.shape
    assert lidar.any()
    assert not lidar[0].any()


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
    bundle = NuPlanCameraBundle(
        jpeg_by_channel={
            channel: b"\xff\xd8\xff\xd9"
            for channel in NUPLAN_CAMERA_CHANNELS
        },
        projection_matrices=np.zeros((8, 3, 4), dtype=np.float32),
        camera_visibility=visibility,
        metadata={
            "camera_order": list(NUPLAN_CAMERA_CHANNELS),
            "image_size": 256,
            "rectification_policy": "test",
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
    assert manifest["split_policy"] == "log_level_hash_bucket"
    assert manifest["navigation_geometry"] == geometry.contract()
    tar_path = tmp_path / manifest["shard_names"][0]
    assert manifest["shard_sha256"][tar_path.name] == hashlib.sha256(
        tar_path.read_bytes()
    ).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(tar_path.read_bytes())) as archive:
        names = archive.getnames()
        assert sum(name.endswith(".jpg") for name in names) == 8
        assert any(name.endswith(".trajectory_xy.npz") for name in names)
        assert any(name.endswith(".bev_segmentation.npz") for name in names)
        meta_name = next(name for name in names if name.endswith(".meta.json"))
        metadata = json.load(archive.extractfile(meta_name))
        assert metadata["split_group_uid"].startswith("nuplan-log-")
