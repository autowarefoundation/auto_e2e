"""Adapter for immutable published dataset snapshots and scene artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from navigation.artifacts import decode_scene_navigation
from navigation.contracts import NavigationMap, NavigationRoute


MAX_MANIFEST_BYTES = 8 << 20
MAX_NAVIGATION_BYTES = 32 << 20
MAX_PATH_BYTES = 16 << 20
POOL_KEY_RE = re.compile(r".+-r(?P<frame>[0-9]{6})-c(?P<camera>[0-9]+)\.jpg$")

CAMERA_ROLES_BY_COUNT = {
    6: (
        "front_center",
        "front_left",
        "front_right",
        "rear",
        "rear_left",
        "rear_right",
    ),
    7: (
        "front_center",
        "front",
        "front_left",
        "front_right",
        "rear",
        "rear_left",
        "rear_right",
    ),
}


class S3Client(Protocol):
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @classmethod
    def parse(cls, uri: str) -> "S3Location":
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"expected an s3:// URI, got {uri!r}")
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError(f"S3 URI has no object key: {uri!r}")
        return cls(bucket=parsed.netloc, key=key)


@dataclasses.dataclass(frozen=True)
class PublishedSceneDescriptor:
    dataset_name: str
    dataset_version: str
    dataset_manifest_uri: str
    dataset_manifest_sha256: str
    partition_id: str
    scene_uid: str
    source_uri: str
    source_manifest_sha256: str
    shard_name: str
    camera_count: int
    endpoint_exclusion_frames: int

    def to_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PublishedSceneDescriptor":
        return cls(**json.loads(payload))


@dataclasses.dataclass(frozen=True)
class CameraObject:
    frame_index: int
    camera_index: int
    camera_role: str
    timestamp_ns: int
    bucket: str
    key: str
    byte_size: int


@dataclasses.dataclass(frozen=True)
class CanonicalSceneEvidence:
    descriptor: PublishedSceneDescriptor
    path_latlon_heading_timestamp: np.ndarray
    navigation_map: NavigationMap
    navigation_route: NavigationRoute
    navigation_quality: dict[str, Any]
    camera_objects: tuple[CameraObject, ...]

    def __post_init__(self) -> None:
        path = np.asarray(self.path_latlon_heading_timestamp, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 4 or len(path) < 2:
            raise ValueError("scene path must have shape [N,4] with at least 2 rows")
        if not np.isfinite(path).all():
            raise ValueError("scene path contains non-finite values")
        timestamps = path[:, 3].astype(np.int64)
        if np.any(timestamps < 0) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("scene path timestamps must be strictly increasing")
        object.__setattr__(
            self,
            "path_latlon_heading_timestamp",
            np.ascontiguousarray(path, dtype=np.float64),
        )
        if not self.camera_objects:
            raise ValueError("scene has no camera objects")

    @property
    def scene_uid(self) -> str:
        return self.descriptor.scene_uid

    @property
    def start_timestamp_ns(self) -> int:
        return int(self.path_latlon_heading_timestamp[0, 3])

    @property
    def nominal_period_ns(self) -> int:
        deltas = np.diff(
            self.path_latlon_heading_timestamp[:, 3].astype(np.int64)
        )
        return max(1, int(np.median(deltas)))

    @property
    def end_timestamp_ns(self) -> int:
        return int(self.path_latlon_heading_timestamp[-1, 3]) + self.nominal_period_ns

    @property
    def distance_m(self) -> float:
        latlon = np.radians(self.path_latlon_heading_timestamp[:, :2])
        dlat = np.diff(latlon[:, 0])
        dlon = np.diff(latlon[:, 1])
        mean_lat = (latlon[:-1, 0] + latlon[1:, 0]) * 0.5
        east = dlon * np.cos(mean_lat)
        return float(6_371_008.8 * np.hypot(dlat, east).sum())

    def camera_anchors(
        self,
        *,
        interval_s: float = 2.0,
        maximum_anchors: int = 32,
    ) -> tuple[tuple[CameraObject, ...], ...]:
        if interval_s <= 0.0 or maximum_anchors <= 0:
            raise ValueError("camera anchor interval and cap must be positive")
        by_frame: dict[int, list[CameraObject]] = {}
        for camera in self.camera_objects:
            by_frame.setdefault(camera.frame_index, []).append(camera)
        complete = [
            (frame, tuple(sorted(items, key=lambda item: item.camera_index)))
            for frame, items in sorted(by_frame.items())
            if len(items) == self.descriptor.camera_count
        ]
        if not complete:
            raise ValueError("scene has no complete multi-camera anchor")

        interval_ns = max(1, int(interval_s * 1_000_000_000))
        selected: list[tuple[CameraObject, ...]] = []
        next_timestamp = complete[0][1][0].timestamp_ns
        for _, cameras in complete:
            timestamp = cameras[0].timestamp_ns
            if timestamp < next_timestamp and selected:
                continue
            selected.append(cameras)
            next_timestamp = timestamp + interval_ns
            if len(selected) >= maximum_anchors:
                break
        if selected[-1] != complete[-1][1] and len(selected) < maximum_anchors:
            selected.append(complete[-1][1])
        return tuple(selected)


def _read_object(
    client: S3Client,
    location: S3Location,
    *,
    maximum_bytes: int,
) -> bytes:
    response = client.get_object(Bucket=location.bucket, Key=location.key)
    advertised = int(response.get("ContentLength", 0) or 0)
    if advertised > maximum_bytes:
        raise ValueError(
            f"S3 object exceeds {maximum_bytes} bytes: "
            f"s3://{location.bucket}/{location.key}"
        )
    body = response["Body"]
    payload = body.read(maximum_bytes + 1)
    try:
        body.close()
    except AttributeError:
        pass
    if len(payload) > maximum_bytes:
        raise ValueError(
            f"S3 object exceeds {maximum_bytes} bytes: "
            f"s3://{location.bucket}/{location.key}"
        )
    return bytes(payload)


def _verify_sha256(payload: bytes, expected: str, *, name: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{name} SHA-256 differs: expected={expected} actual={actual}")


def _child_location(root_uri: str, relative: str) -> S3Location:
    root = S3Location.parse(root_uri)
    prefix = root.key.rstrip("/")
    return S3Location(root.bucket, f"{prefix}/{relative.lstrip('/')}")


def _source_manifest(
    client: S3Client,
    source_uri: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_object(
        client,
        _child_location(source_uri, "manifest.json"),
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    _verify_sha256(payload, expected_sha256, name="source manifest")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("source manifest must be an object")
    return value, payload


def resolve_scene_descriptors(
    client: S3Client,
    *,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
) -> tuple[PublishedSceneDescriptor, ...]:
    manifest_location = S3Location.parse(dataset_manifest_uri)
    payload = _read_object(
        client,
        manifest_location,
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    _verify_sha256(payload, dataset_manifest_sha256, name="dataset manifest")
    manifest = json.loads(payload)
    if manifest.get("status") != "ready":
        raise ValueError("dataset publication is not ready")
    dataset_name = str(manifest["dataset"])
    dataset_version = str(manifest["version"])
    camera_count = int(manifest["num_views"])
    if camera_count not in CAMERA_ROLES_BY_COUNT:
        raise ValueError(f"unsupported camera count: {camera_count}")
    endpoint_exclusion_frames = int(
        manifest.get("geo", {}).get("privacy", {}).get(
            "endpoint_exclusion_frames", 0
        )
    )

    descriptors: list[PublishedSceneDescriptor] = []
    seen_scenes: set[str] = set()
    for partition in manifest.get("partitions", []):
        if int(partition.get("sample_count", 0)) <= 0:
            continue
        source_uri = str(partition["source_uri"])
        source_manifest_sha256 = str(partition["source_manifest_sha256"])
        source, _ = _source_manifest(
            client, source_uri, source_manifest_sha256
        )
        scenes = source.get("navigation", {}).get("scenes", [])
        if len(scenes) != 1:
            raise ValueError(
                f"partition {partition['partition_id']} must own one scene"
            )
        scene_uid = str(scenes[0]["scene_id"])
        if not scene_uid or scene_uid in seen_scenes:
            raise ValueError(f"duplicate or empty scene uid: {scene_uid!r}")
        seen_scenes.add(scene_uid)
        shard_names = source.get("shard_names", [])
        if len(shard_names) != 1:
            raise ValueError(
                f"partition {partition['partition_id']} must own one shard"
            )
        descriptors.append(
            PublishedSceneDescriptor(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                dataset_manifest_uri=dataset_manifest_uri,
                dataset_manifest_sha256=dataset_manifest_sha256,
                partition_id=str(partition["partition_id"]),
                scene_uid=scene_uid,
                source_uri=source_uri,
                source_manifest_sha256=source_manifest_sha256,
                shard_name=str(shard_names[0]),
                camera_count=camera_count,
                endpoint_exclusion_frames=endpoint_exclusion_frames,
            )
        )

    descriptors.sort(key=lambda item: item.scene_uid)
    if len(descriptors) != int(manifest.get("episodes", -1)):
        raise ValueError(
            f"resolved {len(descriptors)} scenes but publication advertises "
            f"{manifest.get('episodes')}"
        )
    return tuple(descriptors)


def _list_pool_objects(
    client: S3Client,
    descriptor: PublishedSceneDescriptor,
    timestamps: np.ndarray,
) -> tuple[CameraObject, ...]:
    root = S3Location.parse(descriptor.source_uri)
    prefix = f"{root.key.rstrip('/')}/pool/"
    roles = CAMERA_ROLES_BY_COUNT[descriptor.camera_count]
    output: list[CameraObject] = []
    continuation: str | None = None
    while True:
        request: dict[str, Any] = {
            "Bucket": root.bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if continuation:
            request["ContinuationToken"] = continuation
        response = client.list_objects_v2(**request)
        for item in response.get("Contents", []):
            key = str(item["Key"])
            match = POOL_KEY_RE.fullmatch(key)
            if match is None:
                continue
            frame_index = int(match.group("frame"))
            camera_index = int(match.group("camera"))
            if camera_index >= descriptor.camera_count:
                raise ValueError(f"camera index exceeds manifest count: {key}")
            path_index = frame_index - descriptor.endpoint_exclusion_frames
            if path_index < 0 or path_index >= len(timestamps):
                continue
            output.append(
                CameraObject(
                    frame_index=frame_index,
                    camera_index=camera_index,
                    camera_role=roles[camera_index],
                    timestamp_ns=int(timestamps[path_index]),
                    bucket=root.bucket,
                    key=key,
                    byte_size=int(item.get("Size", 0)),
                )
            )
        if not response.get("IsTruncated"):
            break
        continuation = str(response.get("NextContinuationToken") or "")
        if not continuation:
            raise ValueError("truncated S3 listing has no continuation token")
    output.sort(key=lambda item: (item.frame_index, item.camera_index))
    return tuple(output)


def load_scene_evidence(
    client: S3Client,
    descriptor: PublishedSceneDescriptor,
) -> CanonicalSceneEvidence:
    source_manifest, _ = _source_manifest(
        client,
        descriptor.source_uri,
        descriptor.source_manifest_sha256,
    )
    scene_entries = source_manifest.get("navigation", {}).get("scenes", [])
    if (
        len(scene_entries) != 1
        or str(scene_entries[0].get("scene_id")) != descriptor.scene_uid
    ):
        raise ValueError("source manifest scene differs from descriptor")
    hashes = scene_entries[0].get("hashes", {})

    navigation_payload = _read_object(
        client,
        _child_location(descriptor.source_uri, "scene_navigation.json"),
        maximum_bytes=MAX_NAVIGATION_BYTES,
    )
    _verify_sha256(
        navigation_payload,
        str(hashes["scene_navigation.json"]),
        name="scene navigation",
    )
    navigation_map, navigation_route = decode_scene_navigation(
        navigation_payload
    )

    quality_payload = _read_object(
        client,
        _child_location(descriptor.source_uri, "navigation_quality.json"),
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    _verify_sha256(
        quality_payload,
        str(hashes["navigation_quality.json"]),
        name="navigation quality",
    )
    navigation_quality = json.loads(quality_payload)

    path_payload = _read_object(
        client,
        _child_location(
            descriptor.source_uri,
            f"geo/episode_paths/{descriptor.scene_uid}.f64",
        ),
        maximum_bytes=MAX_PATH_BYTES,
    )
    if len(path_payload) % (4 * np.dtype("<f8").itemsize) != 0:
        raise ValueError("scene path byte size is not divisible by four f64 values")
    path = np.frombuffer(path_payload, dtype="<f8").reshape(-1, 4).copy()
    timestamps = path[:, 3].astype(np.int64)
    cameras = _list_pool_objects(client, descriptor, timestamps)

    return CanonicalSceneEvidence(
        descriptor=descriptor,
        path_latlon_heading_timestamp=path,
        navigation_map=navigation_map,
        navigation_route=navigation_route,
        navigation_quality=navigation_quality,
        camera_objects=cameras,
    )


def object_bytes(client: S3Client, camera: CameraObject) -> bytes:
    if camera.byte_size <= 0 or camera.byte_size > 16 << 20:
        raise ValueError(f"camera object has invalid size: {camera.key}")
    return _read_object(
        client,
        S3Location(camera.bucket, camera.key),
        maximum_bytes=16 << 20,
    )
