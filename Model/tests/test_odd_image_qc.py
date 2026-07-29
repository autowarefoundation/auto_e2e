from __future__ import annotations

import io

import numpy as np
from PIL import Image

from data_processing.odd_labeling.image_qc import (
    DEPENDENT_QC_KEYS,
    CameraAnchor,
    CameraFrame,
    label_image_quality,
    load_camera_quality_inputs,
)
from data_processing.odd_labeling.published_snapshot import (
    CameraObject,
    CanonicalSceneEvidence,
    PublishedSceneDescriptor,
)
from data_processing.odd_labeling.schema import (
    CameraCapability,
    ChannelCapability,
    DatasetCapabilityManifest,
)


EARTH_RADIUS_M = 6_371_008.8


class _MemoryS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        payload = self.objects[Key]
        return {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
        }


def _jpeg(value: int) -> tuple[bytes, np.ndarray]:
    rgb = np.full((16, 16, 3), value, dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="JPEG")
    return output.getvalue(), rgb


def _scene(
    timestamps_ns: np.ndarray,
    *,
    distances_m: np.ndarray | None = None,
    present_indexes: tuple[int, ...] | None = None,
    payload_sizes: dict[int, int] | None = None,
    frame_inventory_mode: str | None = None,
) -> CanonicalSceneEvidence:
    if distances_m is None:
        distances_m = np.zeros(len(timestamps_ns), dtype=np.float64)
    if present_indexes is None:
        present_indexes = tuple(range(len(timestamps_ns)))
    payload_sizes = payload_sizes or {}
    descriptor = PublishedSceneDescriptor(
        dataset_name="image-qc-fixture",
        dataset_version="v1",
        dataset_manifest_uri="s3://fixture/manifest.json",
        dataset_manifest_sha256="1" * 64,
        partition_id="partition-1",
        scene_uid="scene-1",
        source_uri="s3://fixture/scene-1",
        source_manifest_sha256="2" * 64,
        shard_name="scene-1.tar",
        camera_count=1,
        endpoint_exclusion_frames=0,
    )
    latitudes = 49.0 + np.degrees(distances_m / EARTH_RADIUS_M)
    path = np.column_stack(
        (
            latitudes,
            np.full(len(timestamps_ns), 8.0),
            np.zeros(len(timestamps_ns)),
            timestamps_ns,
        )
    )
    camera_objects = tuple(
        CameraObject(
            frame_index=index,
            camera_index=0,
            camera_role="front_center",
            timestamp_ns=int(timestamps_ns[index]),
            bucket="fixture",
            key=f"frame-{index}.jpg",
            byte_size=payload_sizes.get(index, 1),
        )
        for index in present_indexes
    )
    capability_manifest = None
    if frame_inventory_mode is not None:
        camera_channel = ChannelCapability(
            availability="complete",
            coverage_start_ns=int(timestamps_ns[0]),
            coverage_end_ns=int(timestamps_ns[-1]) + 100_000_000,
            nominal_rate_hz=10.0,
            observed_count=len(present_indexes),
            missing_count=len(timestamps_ns) - len(present_indexes),
            source_artifact_sha256="3" * 64,
        )
        absent_channel = ChannelCapability(
            availability="absent",
            coverage_start_ns=None,
            coverage_end_ns=None,
            nominal_rate_hz=None,
            observed_count=0,
            missing_count=0,
            source_artifact_sha256=None,
        )
        capability_manifest = DatasetCapabilityManifest(
            dataset_name=descriptor.dataset_name,
            dataset_version=descriptor.dataset_version,
            dataset_manifest_sha256=descriptor.dataset_manifest_sha256,
            source_revision="fixture-source-v1",
            adapter_name="fixture",
            adapter_version="fixture-v1",
            scene_inventory_sha256="4" * 64,
            canonical_clock="scene_monotonic_ns",
            absolute_time_available=False,
            timezone_resolution_available=False,
            cameras=(
                CameraCapability(
                    camera_id="front_center",
                    canonical_role="front_center",
                    channel=camera_channel,
                    frame_inventory_mode=frame_inventory_mode,
                ),
            ),
            channels={
                "map": absent_channel,
                "route": absent_channel,
                "gnss": camera_channel,
                "ins": camera_channel,
                "lidar": absent_channel,
                "object_tracks": absent_channel,
                "can": absent_channel,
            },
            coordinate_frames=("ego_flu",),
        )
    return CanonicalSceneEvidence(
        descriptor=descriptor,
        path_latlon_heading_timestamp=path,
        navigation_map=None,
        navigation_route=None,
        navigation_quality={},
        camera_objects=camera_objects,
        capability_manifest=capability_manifest,
    )


def _frame(
    frame_index: int,
    timestamp_ns: int,
    *,
    value: int = 100,
    jpeg: bytes | None = None,
) -> CameraFrame:
    encoded, rgb = _jpeg(value)
    return CameraFrame(
        frame_index=frame_index,
        camera_index=0,
        camera_role="front_center",
        timestamp_ns=timestamp_ns,
        jpeg=jpeg or encoded,
        rgb=rgb,
    )


def _frame_statuses(observations: tuple) -> list:
    return [
        item
        for item in observations
        if item.key == "perception.image.frame_status"
    ]


def test_sampled_frame_does_not_cover_unsampled_or_missing_intervals() -> None:
    timestamps = np.arange(4, dtype=np.int64) * 100_000_000
    evidence = _scene(timestamps, present_indexes=(0, 2, 3))
    anchors = (
        CameraAnchor(0, (_frame(0, 0),)),
        CameraAnchor(
            200_000_000,
            (_frame(2, 200_000_000, value=120),),
        ),
    )

    observations = label_image_quality(evidence, anchors)
    statuses = _frame_statuses(observations)

    dropped = next(item for item in statuses if item.values == ("dropped_frame",))
    assert (dropped.start_timestamp_ns, dropped.end_timestamp_ns) == (
        100_000_000,
        200_000_000,
    )
    normal_intervals = {
        (item.start_timestamp_ns, item.end_timestamp_ns)
        for item in statuses
        if item.values == ("normal",)
    }
    assert normal_intervals == {
        (0, 100_000_000),
        (200_000_000, 300_000_000),
    }
    assert len(observations) == len(
        {item.observation_uid for item in observations}
    )


def test_sampled_evidence_gap_is_unavailable_not_dropped() -> None:
    timestamps = np.arange(4, dtype=np.int64) * 100_000_000
    evidence = _scene(
        timestamps,
        present_indexes=(0, 2, 3),
        frame_inventory_mode="sampled_evidence",
    )
    anchors = (
        CameraAnchor(0, (_frame(0, 0),)),
        CameraAnchor(
            200_000_000,
            (_frame(2, 200_000_000, value=120),),
        ),
    )

    observations = label_image_quality(evidence, anchors)
    statuses = _frame_statuses(observations)
    gap = next(
        item
        for item in statuses
        if item.start_timestamp_ns == 100_000_000
    )

    assert gap.status == "unavailable"
    assert gap.values == ()
    assert gap.provenance["frame_inventory_mode"] == "sampled_evidence"
    assert (
        gap.provenance["reason"]
        == "camera_frame_inventory_not_authoritative"
    )
    assert not any(item.values == ("dropped_frame",) for item in statuses)
    assert {
        item.status
        for item in observations
        if item.start_timestamp_ns == 100_000_000
        and item.key in DEPENDENT_QC_KEYS
    } == {"unavailable"}


def test_invalid_jpeg_becomes_corrupted_evidence_without_failing_scene() -> None:
    timestamps = np.arange(3, dtype=np.int64) * 100_000_000
    first, _ = _jpeg(100)
    third, _ = _jpeg(120)
    payloads = {
        "frame-0.jpg": first,
        "frame-1.jpg": b"not-a-jpeg",
        "frame-2.jpg": third,
    }
    evidence = _scene(
        timestamps,
        payload_sizes={
            index: len(payloads[f"frame-{index}.jpg"])
            for index in range(3)
        },
    )

    anchors, failures = load_camera_quality_inputs(
        _MemoryS3(payloads),
        evidence,
        interval_s=0.1,
        maximum_anchors=3,
    )
    observations = label_image_quality(evidence, anchors, failures)

    assert len(failures) == 1
    assert failures[0].reason == "invalid_jpeg"
    corrupted = next(
        item
        for item in _frame_statuses(observations)
        if item.values == ("corrupted_frame",)
    )
    assert (corrupted.start_timestamp_ns, corrupted.end_timestamp_ns) == (
        100_000_000,
        200_000_000,
    )
    assert {
        item.key
        for item in observations
        if item.start_timestamp_ns == 100_000_000
        and item.status == "not_observable"
    } == set(DEPENDENT_QC_KEYS)


def test_identical_frame_requires_independent_motion_to_be_frozen() -> None:
    timestamps = np.asarray([0, 100_000_000], dtype=np.int64)
    encoded, _ = _jpeg(100)
    anchors = (
        CameraAnchor(0, (_frame(0, 0, jpeg=encoded),)),
        CameraAnchor(
            100_000_000,
            (_frame(1, 100_000_000, jpeg=encoded),),
        ),
    )
    moving = _scene(
        timestamps,
        distances_m=np.asarray([0.0, 1.0]),
    )
    stationary = _scene(timestamps)

    moving_status = _frame_statuses(label_image_quality(moving, anchors))[-1]
    stationary_status = _frame_statuses(
        label_image_quality(stationary, anchors)
    )[-1]

    assert moving_status.status == "valid"
    assert moving_status.values == ("frozen_frame",)
    assert moving_status.provenance["ego_distance_m"] > 0.9
    assert stationary_status.status == "ambiguous"
    assert stationary_status.values == ()
    assert (
        stationary_status.provenance["reason"]
        == "identical_content_without_independent_motion"
    )


def test_black_frame_abstains_from_all_dependent_quality_labels() -> None:
    timestamps = np.asarray([0, 100_000_000], dtype=np.int64)
    evidence = _scene(timestamps)
    black = CameraAnchor(0, (_frame(0, 0, value=0),))

    observations = label_image_quality(evidence, (black,))

    status = _frame_statuses(observations)[0]
    assert status.values == ("black_frame",)
    dependent = [
        item
        for item in observations
        if item.start_timestamp_ns == 0
        and item.key in DEPENDENT_QC_KEYS
    ]
    assert {item.key for item in dependent} == set(DEPENDENT_QC_KEYS)
    assert all(
        item.status == "not_observable" and item.values == ()
        for item in dependent
    )
