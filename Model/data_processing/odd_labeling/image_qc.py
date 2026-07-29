"""Deterministic camera loading and image-quality observations."""

from __future__ import annotations

import base64
import dataclasses
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


IMAGE_QC_VERSION = "odd_image_qc_v1"


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
    for index, anchor in enumerate(anchors):
        start = max(evidence.start_timestamp_ns, anchor.timestamp_ns)
        if index + 1 < len(anchors):
            end = anchors[index + 1].timestamp_ns
        else:
            end = evidence.end_timestamp_ns
        end = min(evidence.end_timestamp_ns, end)
        if end > start:
            yield anchor, start, end


def label_image_quality(
    evidence: CanonicalSceneEvidence,
    anchors: tuple[CameraAnchor, ...],
) -> tuple[LabelObservation, ...]:
    observations: list[LabelObservation] = []
    for anchor, start_ns, end_ns in _intervals(evidence, anchors):
        for frame in anchor.frames:
            metrics = _metrics(frame.rgb)
            provenance = {
                "labeler_version": IMAGE_QC_VERSION,
                "frame_index": frame.frame_index,
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
                frame_status = "black_frame"
            else:
                frame_status = "normal"
            observations.append(
                make_observation(
                    key="perception.image.frame_status",
                    status="valid",
                    values=(frame_status,),
                    **common,
                )
            )
            if frame_status != "normal":
                observations.append(
                    make_observation(
                        scene_uid=evidence.scene_uid,
                        key="perception.image.exposure",
                        status="not_observable",
                        confidence=0.0,
                        source="image_qc",
                        start_timestamp_ns=start_ns,
                        end_timestamp_ns=end_ns,
                        measurements=metrics,
                        provenance=provenance,
                        camera_id=frame.camera_role,
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
