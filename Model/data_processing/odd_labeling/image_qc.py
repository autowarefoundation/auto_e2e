"""Deterministic camera loading and image-quality observations."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
from collections.abc import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from .published_snapshot import (
    CameraObject,
    CanonicalSceneEvidence,
    S3Client,
    object_bytes,
)
from .schema import LabelObservation, make_observation


IMAGE_QC_VERSION = "odd_image_qc_v3"
IMAGE_QC_POLICY_VERSION = "odd_image_qc_policy_v3"
FROZEN_EGO_MOTION_THRESHOLD_M = 0.5
DEPENDENT_QC_KEYS = (
    "perception.image.exposure",
    "perception.visual.contrast",
    "perception.image.blur",
    "perception.visual.lighting",
    "perception.visual.glare",
)


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    frame_index: int
    camera_index: int
    camera_role: str
    timestamp_ns: int
    jpeg: bytes
    rgb: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("camera image must have shape [H,W,3]")
        if not self.jpeg:
            raise ValueError("camera JPEG must not be empty")
        object.__setattr__(self, "rgb", np.ascontiguousarray(rgb))

    def data_url(self) -> str:
        encoded = base64.b64encode(self.jpeg).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


@dataclasses.dataclass(frozen=True)
class CameraAnchor:
    timestamp_ns: int
    frames: tuple[CameraFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("camera anchor must contain frames")
        if any(frame.timestamp_ns != self.timestamp_ns for frame in self.frames):
            raise ValueError("camera anchor timestamps differ")
        indexes = [frame.camera_index for frame in self.frames]
        if indexes != sorted(set(indexes)):
            raise ValueError("camera anchor indexes must be unique and ordered")


@dataclasses.dataclass(frozen=True)
class CameraDecodeFailure:
    frame_index: int
    camera_index: int
    camera_role: str
    timestamp_ns: int
    reason: str

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.camera_index < 0 or self.timestamp_ns < 0:
            raise ValueError("camera decode failure identity must be non-negative")
        if not self.camera_role or self.reason not in {
            "invalid_object_size",
            "invalid_jpeg",
        }:
            raise ValueError("camera decode failure reason is invalid")


def _decode_camera(camera: CameraObject, payload: bytes) -> CameraFrame:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid camera JPEG: {camera.key}") from exc
    return CameraFrame(
        frame_index=camera.frame_index,
        camera_index=camera.camera_index,
        camera_role=camera.camera_role,
        timestamp_ns=camera.timestamp_ns,
        jpeg=payload,
        rgb=rgb,
    )


def load_camera_anchors(
    client: S3Client,
    evidence: CanonicalSceneEvidence,
    *,
    interval_s: float = 2.0,
    maximum_anchors: int = 32,
    trigger_timestamps_ns: Iterable[int] = (),
    trigger_context_s: float = 1.0,
) -> tuple[CameraAnchor, ...]:
    output: list[CameraAnchor] = []
    for objects in evidence.camera_anchors(
        interval_s=interval_s,
        maximum_anchors=maximum_anchors,
        trigger_timestamps_ns=trigger_timestamps_ns,
        trigger_context_s=trigger_context_s,
    ):
        frames = tuple(
            _decode_camera(camera, object_bytes(client, camera))
            for camera in objects
        )
        output.append(
            CameraAnchor(timestamp_ns=frames[0].timestamp_ns, frames=frames)
        )
    return tuple(output)


def load_camera_quality_inputs(
    client: S3Client,
    evidence: CanonicalSceneEvidence,
    *,
    interval_s: float = 2.0,
    maximum_anchors: int = 32,
) -> tuple[tuple[CameraAnchor, ...], tuple[CameraDecodeFailure, ...]]:
    if interval_s <= 0.0 or maximum_anchors <= 0:
        raise ValueError("camera quality sampling must be positive")
    by_frame: dict[int, list[CameraObject]] = {}
    for camera in evidence.camera_objects:
        by_frame.setdefault(camera.frame_index, []).append(camera)
    object_anchors = [
        tuple(sorted(objects, key=lambda item: item.camera_index))
        for _, objects in sorted(by_frame.items())
    ]
    selected: list[tuple[CameraObject, ...]] = []
    next_timestamp_ns = -1
    interval_ns = max(1, int(interval_s * 1_000_000_000))
    for objects in object_anchors:
        timestamp_ns = objects[0].timestamp_ns
        if timestamp_ns < next_timestamp_ns and selected:
            continue
        selected.append(objects)
        next_timestamp_ns = timestamp_ns + interval_ns
    if object_anchors and (
        not selected
        or selected[-1][0].frame_index != object_anchors[-1][0].frame_index
    ):
        selected.append(object_anchors[-1])
    if len(selected) > maximum_anchors:
        indexes = {
            round(position * (len(selected) - 1) / (maximum_anchors - 1))
            for position in range(maximum_anchors)
        } if maximum_anchors > 1 else {len(selected) // 2}
        selected = [selected[index] for index in sorted(indexes)]

    anchors: list[CameraAnchor] = []
    failures: list[CameraDecodeFailure] = []
    for objects in selected:
        frames: list[CameraFrame] = []
        for camera in objects:
            if camera.byte_size <= 0 or camera.byte_size > 16 << 20:
                failures.append(
                    CameraDecodeFailure(
                        frame_index=camera.frame_index,
                        camera_index=camera.camera_index,
                        camera_role=camera.camera_role,
                        timestamp_ns=camera.timestamp_ns,
                        reason="invalid_object_size",
                    )
                )
                continue
            payload = object_bytes(client, camera)
            try:
                frames.append(_decode_camera(camera, payload))
            except ValueError:
                failures.append(
                    CameraDecodeFailure(
                        frame_index=camera.frame_index,
                        camera_index=camera.camera_index,
                        camera_role=camera.camera_role,
                        timestamp_ns=camera.timestamp_ns,
                        reason="invalid_jpeg",
                    )
                )
        if frames:
            anchors.append(
                CameraAnchor(
                    timestamp_ns=objects[0].timestamp_ns,
                    frames=tuple(frames),
                )
            )
    return tuple(anchors), tuple(failures)


def _metrics(rgb: np.ndarray) -> dict[str, float]:
    values = rgb.astype(np.float32) / 255.0
    gray = (
        values[:, :, 0] * 0.2126
        + values[:, :, 1] * 0.7152
        + values[:, :, 2] * 0.0722
    )
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    gradient_x = np.diff(gray, axis=1)
    gradient_y = np.diff(gray, axis=0)
    return {
        "mean_luminance": float(gray.mean()),
        "luminance_stddev": float(gray.std()),
        "dark_fraction": float(np.mean(gray <= 0.03)),
        "bright_fraction": float(np.mean(gray >= 0.97)),
        "laplacian_variance": float(laplacian.var()),
        "edge_density": float(
            0.5
            * (
                np.mean(np.abs(gradient_x) >= 0.08)
                + np.mean(np.abs(gradient_y) >= 0.08)
            )
        ),
    }


def _exposure(metrics: dict[str, float]) -> str:
    dark = metrics["dark_fraction"]
    bright = metrics["bright_fraction"]
    mean = metrics["mean_luminance"]
    if dark >= 0.20 and bright >= 0.08:
        return "mixed"
    if dark >= 0.55 or mean <= 0.18:
        return "underexposed"
    if bright >= 0.25 or mean >= 0.86:
        return "overexposed"
    return "normal"


def _blur(metrics: dict[str, float]) -> tuple[str, str]:
    if (
        metrics["luminance_stddev"] < 0.035
        or metrics["edge_density"] < 0.005
    ):
        return "not_observable", ""
    if metrics["laplacian_variance"] < 0.00035:
        return "valid", "defocus_blur"
    return "valid", "none"


def _intervals(
    evidence: CanonicalSceneEvidence,
    anchors: tuple[CameraAnchor, ...],
) -> Iterable[tuple[CameraAnchor, int, int]]:
    timestamps = evidence.path_latlon_heading_timestamp[:, 3].astype(np.int64)
    for anchor in anchors:
        index = int(np.searchsorted(timestamps, anchor.timestamp_ns))
        if index >= len(timestamps) or timestamps[index] != anchor.timestamp_ns:
            raise ValueError("camera anchor timestamp is outside canonical timeline")
        end = (
            int(timestamps[index + 1])
            if index + 1 < len(timestamps)
            else evidence.end_timestamp_ns
        )
        yield anchor, anchor.timestamp_ns, end


def _dependent_quality_observations(
    evidence: CanonicalSceneEvidence,
    *,
    camera_role: str,
    start_ns: int,
    end_ns: int,
    status: str,
    provenance: dict[str, object],
) -> list[LabelObservation]:
    return [
        make_observation(
            scene_uid=evidence.scene_uid,
            key=key,
            status=status,
            confidence=1.0,
            source="image_qc",
            start_timestamp_ns=start_ns,
            end_timestamp_ns=end_ns,
            provenance=provenance,
            camera_id=camera_role,
        )
        for key in DEPENDENT_QC_KEYS
    ]


def _camera_availability(
    evidence: CanonicalSceneEvidence,
    camera_role: str,
) -> str | None:
    if evidence.capability_manifest is None:
        return None
    return next(
        (
            camera.channel.availability
            for camera in evidence.capability_manifest.cameras
            if camera.canonical_role == camera_role
        ),
        None,
    )


def _camera_frame_inventory_mode(
    evidence: CanonicalSceneEvidence,
    camera_role: str,
) -> str:
    if evidence.capability_manifest is None:
        return "capture_timeline"
    return next(
        (
            camera.frame_inventory_mode
            for camera in evidence.capability_manifest.cameras
            if camera.canonical_role == camera_role
        ),
        "unknown",
    )


def _base_provenance(
    evidence: CanonicalSceneEvidence,
    camera_role: str,
) -> dict[str, object]:
    return {
        "labeler_version": IMAGE_QC_VERSION,
        "image_qc_policy_version": IMAGE_QC_POLICY_VERSION,
        "frozen_ego_motion_threshold_m": FROZEN_EGO_MOTION_THRESHOLD_M,
        "frame_inventory_mode": _camera_frame_inventory_mode(
            evidence,
            camera_role,
        ),
    }


def _expected_camera_roles(
    evidence: CanonicalSceneEvidence,
) -> tuple[str, ...]:
    if (
        evidence.capability_manifest is not None
        and len(evidence.capability_manifest.cameras)
        == evidence.descriptor.camera_count
    ):
        return tuple(
            camera.canonical_role
            for camera in evidence.capability_manifest.cameras
        )
    roles_by_index: dict[int, str] = {}
    for camera in evidence.camera_objects:
        existing = roles_by_index.setdefault(
            camera.camera_index,
            camera.camera_role,
        )
        if existing != camera.camera_role:
            raise ValueError("camera index maps to multiple canonical roles")
    return tuple(
        roles_by_index.get(index, f"camera_{index}")
        for index in range(evidence.descriptor.camera_count)
    )


def _missing_frame_observations(
    evidence: CanonicalSceneEvidence,
) -> list[LabelObservation]:
    timeline_timestamps = evidence.path_latlon_heading_timestamp[
        :, 3
    ].astype(np.int64)
    timeline_ends = np.concatenate(
        (
            timeline_timestamps[1:],
            np.asarray([evidence.end_timestamp_ns]),
        )
    )
    timestamp_ends = dict(
        zip(timeline_timestamps, timeline_ends, strict=True)
    )
    available = {
        (camera.timestamp_ns, camera.camera_role)
        for camera in evidence.camera_objects
    }
    sampled_timestamps = np.asarray(
        sorted({camera.timestamp_ns for camera in evidence.camera_objects}),
        dtype=np.int64,
    )
    observations: list[LabelObservation] = []
    for camera_role in _expected_camera_roles(evidence):
        availability = _camera_availability(evidence, camera_role)
        if availability == "absent":
            provenance = {
                **_base_provenance(evidence, camera_role),
                "reason": "camera_channel_absent",
            }
            observations.append(
                make_observation(
                    scene_uid=evidence.scene_uid,
                    key="perception.image.frame_status",
                    status="unavailable",
                    confidence=1.0,
                    source="image_qc",
                    start_timestamp_ns=evidence.start_timestamp_ns,
                    end_timestamp_ns=evidence.end_timestamp_ns,
                    provenance=provenance,
                    camera_id=camera_role,
                )
            )
            observations.extend(
                _dependent_quality_observations(
                    evidence,
                    camera_role=camera_role,
                    start_ns=evidence.start_timestamp_ns,
                    end_ns=evidence.end_timestamp_ns,
                    status="unavailable",
                    provenance=provenance,
                )
            )
            continue

        timestamps = (
            timeline_timestamps
            if _camera_frame_inventory_mode(evidence, camera_role)
            == "capture_timeline"
            else sampled_timestamps
        )
        run_start: int | None = None
        run_end = 0
        missing_count = 0
        for timestamp in timestamps:
            timestamp_end = timestamp_ends.get(int(timestamp))
            if timestamp_end is None:
                raise ValueError(
                    "sampled camera timestamp is outside canonical timeline"
                )
            if (int(timestamp), camera_role) not in available:
                if run_start is None:
                    run_start = int(timestamp)
                run_end = int(timestamp_end)
                missing_count += 1
                continue
            if run_start is not None:
                observations.extend(
                    _missing_camera_frame_observations(
                        evidence,
                        camera_role=camera_role,
                        start_ns=run_start,
                        end_ns=run_end,
                        missing_count=missing_count,
                    )
                )
                run_start = None
                missing_count = 0
        if run_start is not None:
            observations.extend(
                _missing_camera_frame_observations(
                    evidence,
                    camera_role=camera_role,
                    start_ns=run_start,
                    end_ns=run_end,
                    missing_count=missing_count,
                )
            )
    return observations


def _missing_camera_frame_observations(
    evidence: CanonicalSceneEvidence,
    *,
    camera_role: str,
    start_ns: int,
    end_ns: int,
    missing_count: int,
) -> list[LabelObservation]:
    inventory_mode = _camera_frame_inventory_mode(evidence, camera_role)
    authoritative = inventory_mode == "capture_timeline"
    provenance: dict[str, object] = {
        **_base_provenance(evidence, camera_role),
        "reason": (
            "expected_camera_frame_missing"
            if authoritative
            else "camera_frame_inventory_not_authoritative"
        ),
    }
    observations = [
        make_observation(
            scene_uid=evidence.scene_uid,
            key="perception.image.frame_status",
            status="valid" if authoritative else "unavailable",
            values=("dropped_frame",) if authoritative else (),
            confidence=1.0,
            source="image_qc",
            start_timestamp_ns=start_ns,
            end_timestamp_ns=end_ns,
            measurements={
                (
                    "missing_frame_count"
                    if authoritative
                    else "unobserved_frame_count"
                ): missing_count
            },
            provenance=provenance,
            camera_id=camera_role,
        )
    ]
    observations.extend(
        _dependent_quality_observations(
            evidence,
            camera_role=camera_role,
            start_ns=start_ns,
            end_ns=end_ns,
            status="not_observable" if authoritative else "unavailable",
            provenance=provenance,
        )
    )
    return observations


def _path_distance_m(
    evidence: CanonicalSceneEvidence,
    start_timestamp_ns: int,
    end_timestamp_ns: int,
) -> float:
    path = evidence.path_latlon_heading_timestamp
    timestamps = path[:, 3].astype(np.int64)
    start = max(
        0,
        int(np.searchsorted(timestamps, start_timestamp_ns, side="right")) - 1,
    )
    end = min(
        len(path) - 1,
        int(np.searchsorted(timestamps, end_timestamp_ns, side="left")),
    )
    if end <= start:
        return 0.0
    latlon = np.radians(path[start : end + 1, :2])
    dlat = np.diff(latlon[:, 0])
    dlon = np.diff(latlon[:, 1])
    mean_lat = (latlon[:-1, 0] + latlon[1:, 0]) * 0.5
    return float(
        6_371_008.8 * np.hypot(dlat, dlon * np.cos(mean_lat)).sum()
    )


def _frozen_frame_states(
    evidence: CanonicalSceneEvidence,
    anchors: tuple[CameraAnchor, ...],
) -> dict[tuple[int, str], dict[str, object]]:
    states: dict[tuple[int, str], dict[str, object]] = {}
    ordered = tuple(sorted(anchors, key=lambda item: item.timestamp_ns))
    for previous, current in zip(ordered, ordered[1:]):
        previous_by_role = {
            frame.camera_role: frame for frame in previous.frames
        }
        current_by_role = {
            frame.camera_role: frame for frame in current.frames
        }
        shared_roles = set(previous_by_role) & set(current_by_role)
        changed_roles = {
            role
            for role in shared_roles
            if previous_by_role[role].jpeg != current_by_role[role].jpeg
        }
        ego_distance_m = _path_distance_m(
            evidence,
            previous.timestamp_ns,
            current.timestamp_ns,
        )
        for role in shared_roles:
            previous_frame = previous_by_role[role]
            current_frame = current_by_role[role]
            if previous_frame.jpeg != current_frame.jpeg:
                continue
            motion_evidence = (
                ego_distance_m >= FROZEN_EGO_MOTION_THRESHOLD_M
                or bool(changed_roles - {role})
            )
            states[(current.timestamp_ns, role)] = {
                "status": "valid" if motion_evidence else "ambiguous",
                "value": "frozen_frame" if motion_evidence else None,
                "reason": (
                    "identical_content_with_independent_motion"
                    if motion_evidence
                    else "identical_content_without_independent_motion"
                ),
                "ego_distance_m": ego_distance_m,
                "other_camera_changed": bool(changed_roles - {role}),
                "previous_timestamp_ns": previous.timestamp_ns,
            }
    return states


def _decode_failure_observations(
    evidence: CanonicalSceneEvidence,
    failure: CameraDecodeFailure,
) -> list[LabelObservation]:
    timestamps = evidence.path_latlon_heading_timestamp[:, 3].astype(np.int64)
    index = int(np.searchsorted(timestamps, failure.timestamp_ns))
    if index >= len(timestamps) or timestamps[index] != failure.timestamp_ns:
        raise ValueError("decode failure timestamp is outside canonical timeline")
    end_ns = (
        int(timestamps[index + 1])
        if index + 1 < len(timestamps)
        else evidence.end_timestamp_ns
    )
    provenance: dict[str, object] = {
        **_base_provenance(evidence, failure.camera_role),
        "frame_index": failure.frame_index,
        "reason": failure.reason,
    }
    observations = [
        make_observation(
            scene_uid=evidence.scene_uid,
            key="perception.image.frame_status",
            status="valid",
            values=("corrupted_frame",),
            confidence=1.0,
            source="image_qc",
            start_timestamp_ns=failure.timestamp_ns,
            end_timestamp_ns=end_ns,
            provenance=provenance,
            camera_id=failure.camera_role,
        )
    ]
    observations.extend(
        _dependent_quality_observations(
            evidence,
            camera_role=failure.camera_role,
            start_ns=failure.timestamp_ns,
            end_ns=end_ns,
            status="not_observable",
            provenance=provenance,
        )
    )
    return observations


def label_image_quality(
    evidence: CanonicalSceneEvidence,
    anchors: tuple[CameraAnchor, ...],
    decode_failures: tuple[CameraDecodeFailure, ...] = (),
) -> tuple[LabelObservation, ...]:
    decoded_identities = {
        (frame.timestamp_ns, frame.camera_role)
        for anchor in anchors
        for frame in anchor.frames
    }
    failed_identities = {
        (failure.timestamp_ns, failure.camera_role)
        for failure in decode_failures
    }
    if decoded_identities & failed_identities:
        raise ValueError("camera frame cannot be both decoded and corrupted")
    observations = _missing_frame_observations(evidence)
    for failure in decode_failures:
        observations.extend(_decode_failure_observations(evidence, failure))
    frozen_states = _frozen_frame_states(evidence, anchors)
    for anchor, start_ns, end_ns in _intervals(evidence, anchors):
        for frame in anchor.frames:
            metrics = _metrics(frame.rgb)
            provenance = {
                **_base_provenance(evidence, frame.camera_role),
                "frame_index": frame.frame_index,
                "frame_content_sha256": hashlib.sha256(frame.jpeg).hexdigest(),
            }
            common = {
                "scene_uid": evidence.scene_uid,
                "confidence": 0.98,
                "source": "image_qc",
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
                "measurements": metrics,
                "provenance": provenance,
                "camera_id": frame.camera_role,
            }
            if metrics["dark_fraction"] >= 0.98:
                frame_status_state = "valid"
                frame_status = "black_frame"
            elif (frame.timestamp_ns, frame.camera_role) in frozen_states:
                frozen_state = frozen_states[
                    (frame.timestamp_ns, frame.camera_role)
                ]
                frame_status_state = str(frozen_state["status"])
                frame_status = frozen_state["value"]
                provenance = {**provenance, **frozen_state}
                common["provenance"] = provenance
            else:
                frame_status_state = "valid"
                frame_status = "normal"
            observations.append(
                make_observation(
                    key="perception.image.frame_status",
                    status=frame_status_state,
                    values=(frame_status,) if frame_status else (),
                    **common,
                )
            )
            if frame_status == "black_frame":
                observations.extend(
                    _dependent_quality_observations(
                        evidence,
                        camera_role=frame.camera_role,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        status="not_observable",
                        provenance={
                            **provenance,
                            "reason": "black_frame",
                        },
                    )
                )
                continue

            exposure = _exposure(metrics)
            observations.append(
                make_observation(
                    key="perception.image.exposure",
                    status="valid",
                    values=(exposure,),
                    **common,
                )
            )
            contrast = (
                "low_contrast"
                if metrics["luminance_stddev"] < 0.10
                else "normal"
            )
            observations.append(
                make_observation(
                    key="perception.visual.contrast",
                    status="valid",
                    values=(contrast,),
                    **common,
                )
            )
            blur_status, blur_value = _blur(metrics)
            observations.append(
                make_observation(
                    scene_uid=evidence.scene_uid,
                    key="perception.image.blur",
                    status=blur_status,
                    values=(blur_value,) if blur_value else (),
                    confidence=0.9 if blur_status == "valid" else 0.0,
                    source="image_qc",
                    start_timestamp_ns=start_ns,
                    end_timestamp_ns=end_ns,
                    measurements=metrics,
                    provenance=provenance,
                    camera_id=frame.camera_role,
                )
            )

            if exposure == "mixed":
                lighting = "high_dynamic_range"
            elif exposure == "underexposed":
                lighting = "deep_shadow"
            elif exposure == "normal":
                lighting = "normal"
            else:
                lighting = None
            observations.append(
                make_observation(
                    scene_uid=evidence.scene_uid,
                    key="perception.visual.lighting",
                    status="valid" if lighting else "ambiguous",
                    values=(lighting,) if lighting else (),
                    confidence=0.85 if lighting else 0.0,
                    source="image_qc",
                    start_timestamp_ns=start_ns,
                    end_timestamp_ns=end_ns,
                    measurements=metrics,
                    provenance=provenance,
                    camera_id=frame.camera_role,
                )
            )

            if metrics["bright_fraction"] < 0.01:
                observations.append(
                    make_observation(
                        key="perception.visual.glare",
                        status="valid",
                        values=("none",),
                        confidence=0.85,
                        source="image_qc",
                        scene_uid=evidence.scene_uid,
                        start_timestamp_ns=start_ns,
                        end_timestamp_ns=end_ns,
                        measurements=metrics,
                        provenance=provenance,
                        camera_id=frame.camera_role,
                    )
                )
            else:
                observations.append(
                    make_observation(
                        key="perception.visual.glare",
                        status="ambiguous",
                        confidence=0.0,
                        source="image_qc",
                        scene_uid=evidence.scene_uid,
                        start_timestamp_ns=start_ns,
                        end_timestamp_ns=end_ns,
                        measurements=metrics,
                        provenance=provenance,
                        camera_id=frame.camera_role,
                    )
                )
    return tuple(observations)
