from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from data_processing.odd_labeling.deterministic import (
    MAP_ROUTE_KEYS,
    _junction_type,
    _match_lane,
    _match_route_segment,
    label_kinematics,
    label_map_route,
)
from data_processing.odd_labeling.published_snapshot import (
    CAMERA_ROLES_BY_COUNT,
    CameraObject,
    CanonicalSceneEvidence,
    DatasetEvidenceAdapter,
    PublishedSceneDescriptor,
    PublishedSnapshotAdapter,
)
from data_processing.odd_labeling.schema import (
    CameraCapability,
    ChannelCapability,
    DatasetCapabilityManifest,
)
from navigation.artifacts import encode_scene_navigation
from navigation.contracts import (
    Destination,
    DirectedLaneField,
    Maneuver,
    MapFrame,
    NavigationMap,
    NavigationRoute,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _navigation() -> tuple[NavigationMap, NavigationRoute]:
    frame = MapFrame("fixture-enu", 49.0, 8.0, "local ENU")
    navigation_map = NavigationMap(
        map_version="fixture-map-v1",
        provider="fixture",
        frame=frame,
        bounds_enu_m=(-100.0, -100.0, 100.0, 100.0),
        layer_availability={"lane_centerlines": True},
        provenance={"source_sha256": "1" * 64},
    )
    segment = RouteLaneSegment(
        lane_id="lane-1",
        provider_segment_id="provider-lane-1",
        centerline_enu_m=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
    )
    route = NavigationRoute(
        route_id="route-1",
        revision=1,
        provider="fixture",
        timestamp_ns=0,
        valid_from_ns=0,
        map_version=navigation_map.map_version,
        frame=frame,
        lane_sequence=(segment,),
        destination=Destination(np.asarray([20.0, 0.0]), "fixture"),
        confidence=1.0,
        valid=True,
        quality=RouteQuality(1.0, 0.0, 0.0, 0.0, 0.0),
        estimated_destination=False,
        provenance=RouteProvenance(
            source_revision="fixture-source-v1",
            matcher_version="fixture-matcher-v1",
            matcher_config_sha256="2" * 64,
            map_sha256="3" * 64,
            trace_sha256="4" * 64,
        ),
    )
    return navigation_map, route


def test_local_map_and_route_matching_handle_exact_candidate_ties() -> None:
    centerline = np.asarray([[0.0, 0.0], [20.0, 0.0]])
    first_lane = DirectedLaneField(
        lane_id="duplicate-lane",
        centerline_enu_m=centerline,
        road_class="residential",
    )
    second_lane = replace(first_lane, centerline_enu_m=centerline.copy())
    first_segment = RouteLaneSegment(
        lane_id="duplicate-lane",
        provider_segment_id="provider-segment-a",
        centerline_enu_m=centerline,
    )
    second_segment = replace(
        first_segment,
        provider_segment_id="provider-segment-b",
        centerline_enu_m=centerline.copy(),
    )

    lane_match = _match_lane(
        np.asarray([5.0, 0.0]),
        0.0,
        (first_lane, second_lane),
    )
    route_match = _match_route_segment(
        np.asarray([5.0, 0.0]),
        0.0,
        (first_segment, second_segment),
    )

    assert lane_match is not None
    assert lane_match[0] is first_lane
    assert route_match is not None
    assert route_match[0] is first_segment


class _MemoryBody(io.BytesIO):
    pass


class _MemoryS3:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        payload = self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))]
        return {
            "Body": _MemoryBody(payload),
            "ContentLength": len(payload),
        }

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        prefix = str(kwargs["Prefix"])
        contents = [
            {"Key": key, "Size": len(payload)}
            for (object_bucket, key), payload in sorted(self.objects.items())
            if object_bucket == bucket and key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}


def _published_adapter() -> PublishedSnapshotAdapter:
    bucket = "fixture-bucket"
    source_prefix = "published/scene-1"
    scene_uid = "scene-1"
    path = np.asarray(
        [
            [49.0, 8.0, 0.0, 0.0],
            [49.000001, 8.0, 0.0, 100_000_000.0],
            [49.000002, 8.0, 0.0, 200_000_000.0],
        ],
        dtype="<f8",
    )
    navigation_map, route = _navigation()
    navigation_payload = encode_scene_navigation(navigation_map, route)
    quality_payload = _canonical_bytes({"route_valid": True})
    source_manifest = {
        "shard_names": ["scene-1.tar"],
        "navigation": {
            "scenes": [
                {
                    "scene_id": scene_uid,
                    "hashes": {
                        "scene_navigation.json": _sha256(navigation_payload),
                        "navigation_quality.json": _sha256(quality_payload),
                    },
                }
            ]
        },
    }
    source_payload = _canonical_bytes(source_manifest)
    dataset_manifest = {
        "status": "ready",
        "dataset": "published-fixture",
        "version": "v1",
        "source_revision": "5" * 40,
        "episodes": 1,
        "num_views": 6,
        "hz": 10.0,
        "geo": {
            "path_point_count": len(path),
            "timestamp_dtype": "int64_ns",
            "privacy": {"endpoint_exclusion_frames": 0},
        },
        "partitions": [
            {
                "partition_id": "partition-1",
                "sample_count": 1,
                "source_uri": f"s3://{bucket}/{source_prefix}",
                "source_manifest_sha256": _sha256(source_payload),
            }
        ],
    }
    dataset_payload = _canonical_bytes(dataset_manifest)
    dataset_key = "published/manifest.json"
    objects = {
        (bucket, dataset_key): dataset_payload,
        (bucket, f"{source_prefix}/manifest.json"): source_payload,
        (bucket, f"{source_prefix}/scene_navigation.json"): navigation_payload,
        (bucket, f"{source_prefix}/navigation_quality.json"): quality_payload,
        (
            bucket,
            f"{source_prefix}/geo/episode_paths/{scene_uid}.f64",
        ): path.tobytes(),
    }
    for frame_index in range(len(path)):
        for camera_index in range(6):
            key = (
                f"{source_prefix}/pool/scene-1-r{frame_index:06d}"
                f"-c{camera_index}.jpg"
            )
            objects[(bucket, key)] = b"fixture-jpeg"
    return PublishedSnapshotAdapter(
        _MemoryS3(objects),
        dataset_manifest_uri=f"s3://{bucket}/{dataset_key}",
        dataset_manifest_sha256=_sha256(dataset_payload),
    )


def _channel(
    availability: str,
    *,
    count: int = 0,
    missing_count: int = 0,
) -> ChannelCapability:
    if availability == "absent":
        return ChannelCapability(
            availability="absent",
            coverage_start_ns=None,
            coverage_end_ns=None,
            nominal_rate_hz=None,
            observed_count=0,
            missing_count=0,
            source_artifact_sha256=None,
        )
    return ChannelCapability(
        availability=availability,
        coverage_start_ns=0,
        coverage_end_ns=300_000_000,
        nominal_rate_hz=10.0,
        observed_count=count,
        missing_count=missing_count,
        source_artifact_sha256="6" * 64,
    )


@dataclass
class _SyntheticAdapter:
    capability_manifest: DatasetCapabilityManifest
    descriptor: PublishedSceneDescriptor
    scene: CanonicalSceneEvidence

    def describe_capabilities(self) -> DatasetCapabilityManifest:
        return self.capability_manifest

    def list_scenes(self) -> tuple[PublishedSceneDescriptor, ...]:
        return (self.descriptor,)

    def open_scene(self, scene_uid: str) -> CanonicalSceneEvidence:
        if scene_uid != self.descriptor.scene_uid:
            raise KeyError(f"unknown scene_uid: {scene_uid}")
        return self.scene


def _synthetic_adapter() -> _SyntheticAdapter:
    camera_roles = CAMERA_ROLES_BY_COUNT[6]
    partial_camera = _channel("partial", count=5, missing_count=13)
    manifest = DatasetCapabilityManifest(
        dataset_name="second-adapter-fixture",
        dataset_version="v2",
        dataset_manifest_sha256="7" * 64,
        source_revision="fixture-revision-v2",
        adapter_name="synthetic_second_adapter",
        adapter_version="synthetic_second_adapter_v1",
        scene_inventory_sha256="8" * 64,
        canonical_clock="scene_monotonic_ns",
        absolute_time_available=False,
        timezone_resolution_available=False,
        cameras=tuple(
            CameraCapability(
                camera_id=role,
                canonical_role=role,
                channel=partial_camera,
                frame_inventory_mode="capture_timeline",
            )
            for role in camera_roles
        ),
        channels={
            "map": _channel("complete", count=1),
            "route": _channel("complete", count=1),
            "gnss": _channel("complete", count=3),
            "ins": _channel("complete", count=3),
            "lidar": _channel("absent"),
            "object_tracks": _channel("absent"),
            "can": _channel("absent"),
        },
        coordinate_frames=("wgs84", "enu", "ego_flu"),
        known_limitations=("camera frames are intentionally missing",),
    )
    descriptor = PublishedSceneDescriptor(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        dataset_manifest_uri="s3://fixture/second/manifest.json",
        dataset_manifest_sha256=manifest.dataset_manifest_sha256,
        partition_id="partition-2",
        scene_uid="scene-2",
        source_uri="s3://fixture/second/scene-2",
        source_manifest_sha256="9" * 64,
        shard_name="scene-2.tar",
        camera_count=6,
        endpoint_exclusion_frames=0,
    )
    path = np.asarray(
        [
            [49.0, 8.0, 0.0, 0.0],
            [49.000001, 8.0, 0.0, 100_000_000.0],
            [49.000002, 8.0, 0.0, 200_000_000.0],
        ]
    )
    navigation_map, route = _navigation()
    cameras = tuple(
        CameraObject(
            frame_index=index,
            camera_index=0,
            camera_role="front_center",
            timestamp_ns=int(path[index, 3]),
            bucket="fixture",
            key=f"second/scene-2-r{index:06d}-c0.jpg",
            byte_size=12,
        )
        for index in range(len(path))
    )
    scene = CanonicalSceneEvidence(
        descriptor=descriptor,
        path_latlon_heading_timestamp=path,
        navigation_map=navigation_map,
        navigation_route=route,
        navigation_quality={"route_valid": True},
        camera_objects=cameras,
        capability_manifest=manifest,
    )
    return _SyntheticAdapter(manifest, descriptor, scene)


def _synthetic_adapter_without_navigation() -> _SyntheticAdapter:
    adapter = _synthetic_adapter()
    channels = dict(adapter.capability_manifest.channels)
    channels["map"] = _channel("absent")
    channels["route"] = _channel("absent")
    manifest = replace(adapter.capability_manifest, channels=channels)
    scene = replace(
        adapter.scene,
        navigation_map=None,
        navigation_route=None,
        navigation_quality={},
        capability_manifest=manifest,
    )
    return _SyntheticAdapter(manifest, adapter.descriptor, scene)


def _assert_adapter_conforms(adapter: DatasetEvidenceAdapter) -> None:
    capabilities = adapter.describe_capabilities()
    descriptors = adapter.list_scenes()

    assert descriptors
    assert descriptors == tuple(
        sorted(descriptors, key=lambda item: item.scene_uid)
    )
    assert len({item.scene_uid for item in descriptors}) == len(descriptors)
    assert set(capabilities.channels) == {
        "map",
        "route",
        "gnss",
        "ins",
        "lidar",
        "object_tracks",
        "can",
    }
    assert capabilities.semantic_sha256() == capabilities.semantic_sha256()

    known_camera_roles = {
        camera.canonical_role for camera in capabilities.cameras
    }
    for descriptor in descriptors:
        assert descriptor.dataset_name == capabilities.dataset_name
        assert descriptor.dataset_version == capabilities.dataset_version
        assert (
            descriptor.dataset_manifest_sha256
            == capabilities.dataset_manifest_sha256
        )
        scene = adapter.open_scene(descriptor.scene_uid)
        assert scene.scene_uid == descriptor.scene_uid
        assert scene.capability_manifest is not None
        assert (
            scene.capability_manifest.semantic_sha256()
            == capabilities.semantic_sha256()
        )
        timestamps = scene.path_latlon_heading_timestamp[:, 3].astype(np.int64)
        assert np.all(np.diff(timestamps) > 0)
        assert scene.start_timestamp_ns == int(timestamps[0])
        assert scene.end_timestamp_ns > int(timestamps[-1])
        camera_identities = [
            (item.frame_index, item.camera_index) for item in scene.camera_objects
        ]
        assert len(camera_identities) == len(set(camera_identities))
        assert all(
            item.camera_role in known_camera_roles
            and scene.start_timestamp_ns
            <= item.timestamp_ns
            < scene.end_timestamp_ns
            for item in scene.camera_objects
        )

    for channel in capabilities.channels.values():
        if channel.availability == "absent":
            assert channel.observed_count == 0
            assert channel.coverage_start_ns is None
            assert channel.coverage_end_ns is None

    with pytest.raises(KeyError, match="unknown scene_uid"):
        adapter.open_scene("unknown-scene")


@pytest.mark.parametrize(
    "adapter_factory",
    [_published_adapter, _synthetic_adapter],
    ids=["published-snapshot", "synthetic-second-adapter"],
)
def test_dataset_evidence_adapter_conformance(adapter_factory) -> None:
    _assert_adapter_conforms(adapter_factory())


@pytest.mark.parametrize(
    "adapter_factory",
    [_published_adapter, _synthetic_adapter],
    ids=["published-snapshot", "synthetic-second-adapter"],
)
def test_adapter_evidence_runs_shared_deterministic_labelers(
    adapter_factory,
) -> None:
    adapter = adapter_factory()
    scene = adapter.open_scene(adapter.list_scenes()[0].scene_uid)

    kinematics = label_kinematics(scene)
    map_route = label_map_route(scene)

    assert kinematics
    assert map_route
    assert all(
        observation.scene_uid == scene.scene_uid
        for observation in (*kinematics, *map_route)
    )
    assert {
        "odd.ego.speed_bin",
        "event.ego.motion_state",
        "event.ego.maneuver",
        "event.ego.strong_response",
    }.issubset({observation.key for observation in kinematics})
    assert "odd.route.action" in {
        observation.key for observation in map_route
    }


def test_missing_navigation_abstains_without_blocking_kinematics() -> None:
    adapter = _synthetic_adapter_without_navigation()
    _assert_adapter_conforms(adapter)
    scene = adapter.open_scene("scene-2")

    map_route = label_map_route(scene)
    kinematics = label_kinematics(scene)

    assert {observation.key for observation in map_route} == set(
        MAP_ROUTE_KEYS
    )
    assert all(
        observation.status == "unavailable"
        and observation.values == ()
        and observation.confidence == 1.0
        for observation in map_route
    )
    assert kinematics
    assert all(observation.status == "valid" for observation in kinematics)


def test_map_labels_ignore_scene_wide_route_quality() -> None:
    scene = _published_adapter().open_scene("scene-1")
    source_map = scene.navigation_map
    source_route = scene.navigation_route
    assert source_map is not None
    assert source_route is not None
    centerline = np.asarray([[0.0, -10.0], [0.0, 30.0]])
    lane = DirectedLaneField(
        lane_id="lane-1",
        centerline_enu_m=centerline,
        road_class="residential",
        lane_subtype="road",
        one_way=False,
        carriageway_id="carriageway-1",
        median_separated=False,
        barrier_separated=False,
        provider_attributes={"location": "urban"},
    )
    navigation_map = replace(
        source_map,
        directed_lane_fields=(lane,),
        layer_availability={"lane_topology": True},
    )
    route_segment = RouteLaneSegment(
        lane_id=lane.lane_id,
        provider_segment_id="provider-lane-1",
        centerline_enu_m=centerline,
        maneuver=Maneuver.LEFT,
    )
    route = replace(
        source_route,
        lane_sequence=(route_segment,),
        confidence=0.01,
        valid=False,
        estimated_destination=True,
        quality=RouteQuality(
            matched_pose_ratio=0.2,
            median_lateral_distance_m=0.1,
            p95_lateral_distance_m=40.0,
            median_heading_error_rad=0.1,
            p95_heading_error_rad=3.0,
            unresolved_discontinuities=2,
            failure_reasons=("unresolved_discontinuity",),
        ),
    )

    observations = label_map_route(
        replace(
            scene,
            navigation_map=navigation_map,
            navigation_route=route,
        )
    )
    by_key = {observation.key: observation for observation in observations}

    assert by_key["odd.road.type"].values == ("residential",)
    assert by_key["odd.road.context"].values == ("residential",)
    assert by_key["odd.road.directionality"].values == ("two_way",)
    assert by_key["odd.road.horizontal_geometry"].values == ("straight",)
    assert by_key["odd.road.junction_position"].values == ("midblock",)
    assert by_key["odd.road.lane_count_bin"].values == ("one",)
    assert by_key["odd.road.division"].values == ("undivided",)
    assert by_key["odd.route.action"].values == ("turn_left",)
    assert by_key["odd.route.action"].provenance[
        "intent_semantics"
    ] == "reconstructed_from_ego_trace"
    assert by_key["odd.route.action"].provenance[
        "route_confidence_global"
    ] == 0.01


def test_map_labels_use_declared_projection_and_undirected_geometry() -> None:
    scene = _synthetic_adapter().scene
    source_map = scene.navigation_map
    source_route = scene.navigation_route
    assert source_map is not None
    assert source_route is not None
    frame = MapFrame(
        "kitscenes-utm",
        49.01439,
        8.41722,
        "EPSG:32632 local ENU",
    )
    path = np.asarray(
        [
            [48.98504346, 8.45748267, 137.27752823, 0.0],
            [48.98488597, 8.45774490, 138.5, 100_000_000.0],
            [48.98472848, 8.45800713, 140.35287086, 200_000_000.0],
        ]
    )
    reverse_centerline = np.asarray(
        [
            [2958.85346758, -3319.40846500],
            [2939.79264236, -3301.76419203],
            [2920.73181714, -3284.11991905],
        ]
    )
    lane = DirectedLaneField(
        lane_id="lane-reverse",
        centerline_enu_m=reverse_centerline,
        road_class="residential",
        lane_subtype="road",
        one_way=False,
        provider_attributes={"location": "urban"},
    )
    navigation_map = replace(
        source_map,
        frame=frame,
        bounds_enu_m=(2900.0, -3340.0, 2980.0, -3260.0),
        directed_lane_fields=(lane,),
        layer_availability={"lane_topology": True},
    )
    route = replace(
        source_route,
        frame=frame,
        map_version=navigation_map.map_version,
        lane_sequence=(
            RouteLaneSegment(
                lane_id=lane.lane_id,
                provider_segment_id="provider-lane-reverse",
                centerline_enu_m=reverse_centerline,
                maneuver=Maneuver.LEFT,
            ),
        ),
        confidence=0.01,
        valid=False,
        estimated_destination=True,
        quality=replace(
            source_route.quality,
            unresolved_discontinuities=2,
            failure_reasons=("unresolved_discontinuity",),
        ),
    )

    observations = label_map_route(
        replace(
            scene,
            path_latlon_heading_timestamp=path,
            navigation_map=navigation_map,
            navigation_route=route,
        )
    )
    by_key = {observation.key: observation for observation in observations}

    geometry = by_key["odd.road.horizontal_geometry"]
    assert geometry.status == "valid"
    assert geometry.provenance["local_map_match"]["distance_m"] < 1.0
    assert (
        geometry.provenance["local_map_match"]["heading_semantics"]
        == "undirected_centerline_geometry"
    )
    assert geometry.provenance["route_valid_global"] is False
    assert geometry.provenance["unresolved_discontinuities_global"] == 2
    assert by_key["odd.road.junction_position"].status == "valid"
    assert by_key["odd.route.action"].status == "ambiguous"


def test_route_action_rejects_only_the_local_discontinuity() -> None:
    scene = _published_adapter().open_scene("scene-1")
    source_map = scene.navigation_map
    source_route = scene.navigation_route
    assert source_map is not None
    assert source_route is not None
    centerline = np.asarray([[0.0, -10.0], [0.0, 30.0]])
    lane = DirectedLaneField(
        lane_id="lane-1",
        centerline_enu_m=centerline,
        road_class="residential",
        lane_subtype="road",
    )
    navigation_map = replace(
        source_map,
        directed_lane_fields=(lane,),
        layer_availability={"lane_topology": True},
    )
    route = replace(
        source_route,
        lane_sequence=(
            RouteLaneSegment(
                lane_id=lane.lane_id,
                provider_segment_id="provider-lane-1",
                centerline_enu_m=centerline,
                maneuver=Maneuver.LEFT,
                connected_from_previous=False,
            ),
        ),
        confidence=0.9,
        valid=False,
        quality=replace(
            source_route.quality,
            unresolved_discontinuities=1,
            failure_reasons=("unresolved_discontinuity",),
        ),
    )

    observations = label_map_route(
        replace(
            scene,
            navigation_map=navigation_map,
            navigation_route=route,
        )
    )
    action = next(
        observation
        for observation in observations
        if observation.key == "odd.route.action"
    )

    assert action.status == "ambiguous"
    assert action.values == ()
    assert "local discontinuity" in action.provenance["reason"]


@pytest.mark.parametrize(
    ("successor_angles_deg", "expected", "bedrock_eligible"),
    [
        ((0.0, 90.0), "t_junction", False),
        ((60.0, 300.0), "y_junction", False),
        ((35.0, 285.0), None, True),
    ],
)
def test_three_arm_junction_classifier_limits_bedrock_to_t_y_boundary(
    successor_angles_deg,
    expected,
    bedrock_eligible,
) -> None:
    def line(angle_deg: float, start: float, end: float) -> np.ndarray:
        angle = np.radians(angle_deg)
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        return np.asarray([direction * start, direction * end])

    predecessor = DirectedLaneField(
        lane_id="predecessor",
        centerline_enu_m=np.asarray([[-20.0, 0.0], [-10.0, 0.0]]),
    )
    successors = tuple(
        DirectedLaneField(
            lane_id=f"successor-{index}",
            centerline_enu_m=line(angle, 0.0, 10.0),
        )
        for index, angle in enumerate(successor_angles_deg)
    )
    current = DirectedLaneField(
        lane_id="current",
        centerline_enu_m=np.asarray([[-10.0, 0.0], [0.0, 0.0]]),
        predecessor_lane_ids=(predecessor.lane_id,),
        successor_lane_ids=tuple(item.lane_id for item in successors),
        is_intersection=True,
    )
    lanes = {
        item.lane_id: item
        for item in (predecessor, current, *successors)
    }

    value, branch_count, eligible = _junction_type(
        current, "inside", lanes
    )

    assert value == expected
    assert branch_count == 3
    assert eligible is bedrock_eligible


def test_synthetic_adapter_preserves_missing_camera_frames() -> None:
    adapter = _synthetic_adapter()
    scene = adapter.open_scene("scene-2")

    assert len(scene.camera_objects) == 3
    assert adapter.describe_capabilities().cameras[0].channel.missing_count == 13
    with pytest.raises(ValueError, match="no complete multi-camera anchor"):
        scene.camera_anchors()


def test_adapters_declare_camera_inventory_semantics() -> None:
    published = _published_adapter().describe_capabilities()
    synthetic = _synthetic_adapter().describe_capabilities()

    assert published.adapter_version == "published_snapshot_v2"
    assert {
        camera.frame_inventory_mode for camera in published.cameras
    } == {"sampled_evidence"}
    assert {
        camera.frame_inventory_mode for camera in synthetic.cameras
    } == {"capture_timeline"}


def test_camera_anchors_include_trigger_context_deterministically() -> None:
    scene = _published_adapter().open_scene("scene-1")

    first = scene.camera_anchors(
        interval_s=10.0,
        maximum_anchors=3,
        trigger_timestamps_ns=(100_000_000,),
        trigger_context_s=0.0,
    )
    second = scene.camera_anchors(
        interval_s=10.0,
        maximum_anchors=3,
        trigger_timestamps_ns=(100_000_000,),
        trigger_context_s=0.0,
    )

    assert [anchor[0].timestamp_ns for anchor in first] == [
        0,
        100_000_000,
        200_000_000,
    ]
    assert first == second


def test_camera_anchor_cap_spreads_trigger_context_without_duplicates() -> None:
    scene = _published_adapter().open_scene("scene-1")

    anchors = scene.camera_anchors(
        interval_s=0.05,
        maximum_anchors=2,
        trigger_timestamps_ns=(100_000_000,),
        trigger_context_s=0.1,
    )

    timestamps = [anchor[0].timestamp_ns for anchor in anchors]
    assert timestamps == [0, 200_000_000]
    assert len(timestamps) == len(set(timestamps))


def test_camera_anchor_rejects_negative_trigger_timestamp() -> None:
    scene = _published_adapter().open_scene("scene-1")

    with pytest.raises(ValueError, match="non-negative"):
        scene.camera_anchors(trigger_timestamps_ns=(-1,))


def test_capability_manifest_round_trip_preserves_semantic_identity() -> None:
    original = _synthetic_adapter().describe_capabilities()

    restored = DatasetCapabilityManifest.from_json(
        json.dumps(original.to_dict())
    )

    assert restored.to_dict() == original.to_dict()
    assert restored.semantic_sha256() == original.semantic_sha256()
