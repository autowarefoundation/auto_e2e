"""Benchmark population, model input samples and canonical ground truth.

The model-input side and the evaluation side are separate:

* ``InferenceAnchor`` (scene, frame) is all the input builder receives. It
  carries no future length, validity or ground truth.
* ``build_input_sample`` uses only the bounded ``*_until(t0)`` scene accessors,
  which drop every pose, timestamp and camera row after the anchor frame.
* ``validate_evaluation_anchor`` and ``build_ground_truth`` read the full scene.
  The runner calls them only after the prediction has been persisted.

``Anchor`` (with ``available_future_steps``) belongs to the population and to
scoring; it is never passed to the input builder.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from accuracy_benchmark.simforge.contracts import (
    CAMERA_FRAMES,
    EGO_HISTORY_STEPS,
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_HORIZON_COUNTS,
    FUTURE_STEPS,
    POPULATION_ID,
    POPULATION_MIN_FUTURE_STEPS,
    POPULATION_START_FRAME,
    POPULATION_STRIDE_FRAMES,
    SCORING_HORIZONS_S,
    TIME_STEP_S,
    TIMESTAMP_TOLERANCE_NS,
    VIRTUAL_TELE_ASPECT_RATIO,
    VIRTUAL_TELE_HFOV_DEG,
    CameraConfig,
)
from accuracy_benchmark.simforge.kitscenes_data import (
    REFERENCE_TIMESTAMP_FILENAME,
    KitScene,
    PoseTrack,
    crop_to_png,
    parse_calibration,
    virtual_telephoto_crop,
)

BUNDLED_POPULATION_PATH = Path(__file__).with_name("population.json")
MINIMUM_ANCHOR_FRAME = EGO_HISTORY_STEPS - 1


def sample_uid(scene_id: str, frame_index: int) -> str:
    return f"kitscenes-v1-{scene_id}-f{frame_index:06d}"


@dataclass(frozen=True)
class InferenceAnchor:
    """Model-facing anchor identity: no future-derived field of any kind."""

    scene_id: str
    frame_index: int

    @property
    def uid(self) -> str:
        return sample_uid(self.scene_id, self.frame_index)


@dataclass(frozen=True)
class Anchor:
    """Population/evaluation anchor. ``available_future_steps`` is evaluation metadata only."""

    scene_id: str
    frame_index: int
    available_future_steps: int

    @property
    def uid(self) -> str:
        return sample_uid(self.scene_id, self.frame_index)

    def for_inference(self) -> InferenceAnchor:
        return InferenceAnchor(self.scene_id, self.frame_index)

    def horizon_valid(self, horizon_s: float) -> bool:
        return self.available_future_steps >= round(horizon_s / TIME_STEP_S)


def load_population(path: Path = BUNDLED_POPULATION_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("population_id") != POPULATION_ID:
        raise ValueError(f"{path} is not the {POPULATION_ID} population")
    return payload


def derive_anchors(scene_usable_frames: Mapping[str, int]) -> list[Anchor]:
    """Frames ``40 + 90k`` per scene that still have >= 30 genuine future frames.

    ``scene_usable_frames`` is the number of frames indexable on the shared
    10 Hz timeline for poses and every camera; the future available to an
    anchor is ``usable - 1 - frame``.
    """
    anchors: list[Anchor] = []
    for scene_id in sorted(scene_usable_frames):
        usable = int(scene_usable_frames[scene_id])
        frame = POPULATION_START_FRAME
        while usable - 1 - frame >= POPULATION_MIN_FUTURE_STEPS:
            anchors.append(Anchor(scene_id, frame, min(usable - 1 - frame, FUTURE_STEPS)))
            frame += POPULATION_STRIDE_FRAMES
    return anchors


def horizon_counts(anchors: Sequence[Anchor]) -> dict[str, int]:
    return {f"{h:g}s": sum(1 for a in anchors if a.horizon_valid(h)) for h in SCORING_HORIZONS_S}


def verify_population(anchors: Sequence[Anchor], bundled: Mapping[str, Any]) -> dict[str, Any]:
    """Check the derived population against the expected counts and the bundled list."""
    counts = horizon_counts(anchors)
    if len(anchors) != EXPECTED_ANCHOR_COUNT or counts != EXPECTED_HORIZON_COUNTS:
        raise ValueError(f"population has {len(anchors)} anchors with horizon counts {counts}")
    bundled_uids = [entry["sample_uid"] for entry in bundled["anchors"]]
    derived_uids = [anchor.uid for anchor in anchors]
    if derived_uids != bundled_uids:
        raise ValueError("derived anchors differ from the bundled population list")
    return {"anchor_count": len(anchors), "horizon_counts": counts, "matches_bundled_list": True}


def select_anchors(anchors: Sequence[Anchor], *, scene_ids: Sequence[str] = (), limit: int | None = None) -> list[Anchor]:
    selected = [a for a in anchors if not scene_ids or a.scene_id in set(scene_ids)]
    return selected[:limit] if limit is not None else selected


# ---------------------------------------------------------------------------
# Past-only model input
# ---------------------------------------------------------------------------


def relative_to_t0(xyz_world: np.ndarray, rot_world: np.ndarray, t0_index: int) -> tuple[np.ndarray, np.ndarray]:
    """``xyz_local = R_t0^T (p - p_t0)``, ``rot_local = R_t0^T R_t``."""
    rot_t0 = rot_world[t0_index]
    xyz_local = (xyz_world - xyz_world[t0_index]) @ rot_t0
    rot_local = np.einsum("ji,tjk->tik", rot_t0, rot_world)
    return xyz_local, rot_local


@dataclass(frozen=True)
class ImageFrame:
    simforge_slot: int
    kit_camera: str
    frame_index: int
    payload: bytes
    encoding: str  # "jpeg" (raw, untouched) or "png_lossless_crop"
    timestamp_ns: int
    expected_timestamp_ns: int
    timestamp_source: str
    source_sha256: str
    crop: dict[str, Any] | None = None

    @property
    def skew_ms(self) -> float:
        return (self.timestamp_ns - self.expected_timestamp_ns) / 1e6

    def record(self) -> dict[str, Any]:
        return {
            "simforge_slot": self.simforge_slot,
            "kit_camera": self.kit_camera,
            "frame_index": self.frame_index,
            "encoding": self.encoding,
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "bytes": len(self.payload),
            "timestamp_ns": self.timestamp_ns,
            "expected_timestamp_ns": self.expected_timestamp_ns,
            "skew_ms": round(self.skew_ms, 4),
            "timestamp_source": self.timestamp_source,
            "source_jpeg_sha256": self.source_sha256,
            "crop": self.crop,
        }


@dataclass(frozen=True)
class InputSample:
    """Everything the model may see for one anchor. Contains no future data."""

    anchor: InferenceAnchor
    config: CameraConfig
    t0_ns: int
    ego_timestamps_ns: np.ndarray  # (16,)
    ego_xyz: np.ndarray  # (16, 3), t0 ego frame
    ego_rot: np.ndarray  # (16, 3, 3), t0 ego frame
    images: tuple[tuple[ImageFrame, ...], ...]  # [slot][4 frames, oldest first]
    provenance: dict[str, Any] = field(default_factory=dict)

    def slot_frames(self) -> list[list[bytes]]:
        return [[frame.payload for frame in slot] for slot in self.images]


def _expected_ns(t0_ns: int, offsets_s: Sequence[float]) -> np.ndarray:
    return np.asarray([t0_ns + round(offset * 1e9) for offset in offsets_s], dtype=np.int64)


class PastSceneAccess(Protocol):
    """The scene interface the input builder is allowed to use (all bounded to t0)."""

    scene_id: str

    def pose_rows_until(self, frame_index: int) -> PoseTrack: ...

    def reference_timestamps_until(self, frame_index: int) -> np.ndarray: ...

    def camera_timestamps_until(self, camera: str, frame_index: int) -> np.ndarray | None: ...

    def camera_image_until(self, camera: str, frame: int, frame_index: int) -> bytes: ...

    def calibration(self) -> dict[str, Any]: ...


def build_input_sample(
    scene: PastSceneAccess,
    anchor: InferenceAnchor,
    config: CameraConfig,
    *,
    tolerance_ns: int = TIMESTAMP_TOLERANCE_NS,
) -> InputSample:
    """Past-only model input: 16 poses at t0-1.5..t0 and 4 frames per camera at t0-0.3..t0.

    Every array used here ends at the anchor frame; nothing after t0 is visible.
    """
    if not isinstance(anchor, InferenceAnchor):
        raise TypeError("build_input_sample takes an InferenceAnchor (no future metadata)")
    t0_index = anchor.frame_index
    if t0_index < MINIMUM_ANCHOR_FRAME:
        raise ValueError(f"{anchor.uid}: frame lacks a {EGO_HISTORY_STEPS}-pose history")
    track: PoseTrack = scene.pose_rows_until(t0_index)
    reference = scene.reference_timestamps_until(t0_index)

    history = list(range(t0_index - EGO_HISTORY_STEPS + 1, t0_index + 1))
    image_indices = list(range(t0_index - CAMERA_FRAMES + 1, t0_index + 1))
    t0_ns = int(reference[t0_index])

    pose_ns = track.timestamps_ns[history]
    if int(np.abs(pose_ns - reference[history]).max()) >= tolerance_ns:
        raise ValueError(f"{anchor.uid}: poses.txt and the reference timeline disagree")
    expected_history = _expected_ns(t0_ns, [(i - (EGO_HISTORY_STEPS - 1)) * TIME_STEP_S for i in range(EGO_HISTORY_STEPS)])
    if int(np.abs(pose_ns - expected_history).max()) >= tolerance_ns:
        raise ValueError(f"{anchor.uid}: ego history is off the 10 Hz grid")
    rotations = track.rotation_matrices()[history]
    ego_xyz, ego_rot = relative_to_t0(track.xyz[history], rotations, EGO_HISTORY_STEPS - 1)

    expected_images = _expected_ns(t0_ns, [(i - (CAMERA_FRAMES - 1)) * TIME_STEP_S for i in range(CAMERA_FRAMES)])
    calibration_payload = scene.calibration() if any(view.virtual_tele for view in config.views) else None
    slots: list[tuple[ImageFrame, ...]] = []
    per_view: list[dict[str, Any]] = []
    for view in config.views:
        camera_stamps = scene.camera_timestamps_until(view.kit_camera, t0_index)
        if camera_stamps is not None:
            stamps = camera_stamps[image_indices]
            source_name = f"timestamp.{view.kit_camera}.txt"
        else:
            stamps = reference[image_indices]
            source_name = REFERENCE_TIMESTAMP_FILENAME
        deviation = int(np.abs(stamps - expected_images).max())
        if deviation >= tolerance_ns:
            raise ValueError(f"{anchor.uid}: {view.kit_camera} frames deviate {deviation / 1e6:.3f} ms")
        raw = [scene.camera_image_until(view.kit_camera, index, t0_index) for index in image_indices]
        if view.virtual_tele:
            calibration = parse_calibration(calibration_payload or {}, view.kit_camera)
            crop = virtual_telephoto_crop(
                calibration,
                hfov_deg=VIRTUAL_TELE_HFOV_DEG,
                aspect_ratio=VIRTUAL_TELE_ASPECT_RATIO,
            )
            payloads = [crop_to_png(jpeg, crop.box) for jpeg in raw]
            encoding = "png_lossless_crop"
            crop_meta: dict[str, Any] | None = crop.to_metadata()
        else:
            payloads = raw
            encoding = "jpeg"
            crop_meta = None
        frames = tuple(
            ImageFrame(
                simforge_slot=view.slot,
                kit_camera=view.kit_camera,
                frame_index=index,
                payload=payload,
                encoding=encoding,
                timestamp_ns=int(stamps[k]),
                expected_timestamp_ns=int(expected_images[k]),
                timestamp_source=source_name,
                source_sha256=hashlib.sha256(raw[k]).hexdigest(),
                crop=crop_meta,
            )
            for k, (index, payload) in enumerate(zip(image_indices, payloads))
        )
        slots.append(frames)
        per_view.append({"slot": view.slot, "kit_camera": view.kit_camera, "encoding": encoding})
    provenance = {
        "scene_id": anchor.scene_id,
        "frame_index": t0_index,
        "t0_ns": t0_ns,
        "t0_source": REFERENCE_TIMESTAMP_FILENAME,
        "scene_rows_visible": "0..t0 (pose, reference and camera timestamp rows after t0 are dropped)",
        "image_frame_indices": image_indices,
        "ego_history_frame_indices": history,
        "ego_history_timestamps_ns": [int(v) for v in pose_ns],
        "max_pose_reference_skew_ms": round(float(np.abs(pose_ns - reference[history]).max()) / 1e6, 4),
        "views": per_view,
        "config": config.to_metadata(),
    }
    return InputSample(
        anchor=anchor,
        config=config,
        t0_ns=t0_ns,
        ego_timestamps_ns=pose_ns.astype(np.int64),
        ego_xyz=ego_xyz,
        ego_rot=ego_rot,
        images=tuple(slots),
        provenance=provenance,
    )


class PopulationMismatchError(ValueError):
    """The live scene does not match the fixed population metadata."""


def validate_evaluation_anchor(scene: KitScene, anchor: Anchor, config: CameraConfig) -> dict[str, int]:
    """Post-prediction check that the live scene still offers the population's recorded future.

    ``usable`` is the frame count indexable for poses, the reference timeline and
    every configured camera; the anchor's future is ``min(usable - 1 - frame, 64)``.
    """
    usable = min(
        len(scene.pose_track()),
        len(scene.reference_timestamps_ns()),
        *(_contiguous_prefix(scene.camera_frame_indices(camera)) for camera in config.kit_cameras),
    )
    live_future = min(usable - 1 - anchor.frame_index, FUTURE_STEPS)
    if anchor.frame_index >= usable or live_future != anchor.available_future_steps:
        raise PopulationMismatchError(
            f"{anchor.uid}: the scene offers {live_future} future steps but the population recorded "
            f"{anchor.available_future_steps}; the population is fixed and is not changed"
        )
    return {"usable_frame_count": int(usable), "available_future_steps": int(live_future)}


def _contiguous_prefix(indices: Sequence[int]) -> int:
    expected = 0
    for value in sorted(set(indices)):
        if value != expected:
            break
        expected += 1
    return expected


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)  # a writable copy: torch.from_numpy needs it


# ---------------------------------------------------------------------------
# Canonical ground truth (computed after prediction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    anchor: Anchor
    t0_ns: int
    xyz: np.ndarray  # (64, 3) t0 ego frame, NaN where not valid
    rot: np.ndarray  # (64, 3, 3)
    valid: np.ndarray  # (64,) bool, a prefix
    timestamps_ns: np.ndarray  # (64,) int64, -1 where not valid

    @property
    def available_steps(self) -> int:
        return int(self.valid.sum())


def build_ground_truth(scene: KitScene, anchor: Anchor, *, tolerance_ns: int = TIMESTAMP_TOLERANCE_NS) -> GroundTruth:
    """Future poses from ``poses.txt`` in the t0 frame.

    A step is valid when its row exists and lies strictly within the tolerance
    of ``t0 + n * 0.1 s``; validity is the contiguous prefix of such rows.
    """
    track = scene.pose_track()
    reference = scene.reference_timestamps_ns()
    requested = min(anchor.available_future_steps, FUTURE_STEPS)
    if anchor.frame_index + requested >= len(track):
        raise ValueError(f"{anchor.uid}: pose track is shorter than the requested future")
    pose_t0 = int(track.timestamps_ns[anchor.frame_index])
    t0_ns = int(reference[anchor.frame_index])
    if abs(pose_t0 - t0_ns) >= tolerance_ns:
        raise ValueError(f"{anchor.uid}: pose and reference t0 differ")
    indices = list(range(anchor.frame_index, anchor.frame_index + requested + 1))
    xyz_local, rot_local = relative_to_t0(track.xyz[indices], track.rotation_matrices()[indices], 0)
    rows = track.timestamps_ns[indices[1:]]
    expected = np.asarray([pose_t0 + round((k + 1) * 1e8) for k in range(requested)], dtype=np.int64)
    timed = np.abs(rows - expected) < tolerance_ns
    steps = int(np.argmin(timed)) if requested and not timed.all() else requested
    xyz = np.full((FUTURE_STEPS, 3), np.nan)
    rot = np.full((FUTURE_STEPS, 3, 3), np.nan)
    stamps = np.full(FUTURE_STEPS, -1, dtype=np.int64)
    valid = np.zeros(FUTURE_STEPS, dtype=bool)
    xyz[:steps] = xyz_local[1 : steps + 1]
    rot[:steps] = rot_local[1 : steps + 1]
    stamps[:steps] = rows[:steps]
    valid[:steps] = True
    return GroundTruth(anchor=anchor, t0_ns=t0_ns, xyz=xyz, rot=rot, valid=valid, timestamps_ns=stamps)
