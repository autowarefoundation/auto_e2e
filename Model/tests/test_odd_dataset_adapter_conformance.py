from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import numpy as np
import pytest

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


class _MemoryBody(io.BytesIO):
    pass


class _MemoryS3:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        payload = self.objects[(Bucket, Key)]
        return {
            "Body": _MemoryBody(payload),
            "ContentLength": len(payload),
        }

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
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


def test_synthetic_adapter_preserves_missing_camera_frames() -> None:
    adapter = _synthetic_adapter()
    scene = adapter.open_scene("scene-2")

    assert len(scene.camera_objects) == 3
    assert adapter.describe_capabilities().cameras[0].channel.missing_count == 13
    with pytest.raises(ValueError, match="no complete multi-camera anchor"):
        scene.camera_anchors()


def test_capability_manifest_round_trip_preserves_semantic_identity() -> None:
    original = _synthetic_adapter().describe_capabilities()

    restored = DatasetCapabilityManifest.from_json(
        json.dumps(original.to_dict())
    )

    assert restored.to_dict() == original.to_dict()
    assert restored.semantic_sha256() == original.semantic_sha256()
