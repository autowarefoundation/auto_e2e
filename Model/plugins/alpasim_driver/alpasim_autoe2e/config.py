"""Camera rig profiles for the AutoE2E AlpaSim driver plugin.

A profile describes the cameras of one dataset rig: native intrinsics, image
size and camera-to-reference extrinsics, plus the semantic slot each camera
fills in the model input. The plugin selects the profile whose cameras match
the cameras AlpaSim renders, so the model receives the projections it was
trained with.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).resolve().parent / "configs" / "calibration"

# ImageNet statistics used by data_parsing/pre_extracted.py for every dataset.
IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CameraCalibration:
    """One camera of a rig at its native resolution."""

    logical_id: str
    slot: str
    image_size_wh: tuple[int, int]
    intrinsic: np.ndarray
    """``(3, 3)`` pinhole intrinsics at ``image_size_wh``."""
    camera_to_reference: np.ndarray
    """``(4, 4)`` transform from camera to the rig reference frame."""

    def projection(self, target_hw: tuple[int, int]) -> np.ndarray:
        """``(3, 4)`` matrix mapping reference-frame points to target pixels.

        Matches ``compute_camera_projection_matrices`` in
        ``data_parsing/kit_scenes/camera.py``: the intrinsics are scaled per axis
        to the resized (squashed) image.
        """
        return self.scaled_intrinsic(target_hw) @ self.reference_to_camera()[:3, :]

    def scaled_intrinsic(self, target_hw: tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_hw
        source_w, source_h = self.image_size_wh
        intrinsic = self.intrinsic.astype(np.float64).copy()
        intrinsic[0, :] *= target_w / source_w
        intrinsic[1, :] *= target_h / source_h
        return intrinsic

    def reference_to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.camera_to_reference)


@dataclass(frozen=True)
class EgomotionRule:
    """How a dataset turns yaw rate into curvature.

    The defaults are ``data_parsing/kit_scenes/egomotion.py``; nuPlan zeroes
    and clips it (``_state_signals`` in ``data_parsing/nuplan/packing.py``).
    """

    curvature_speed_floor_mps: float = 0.1
    curvature_zero_below_mps: float | None = None
    curvature_limit: float | None = None

    def curvature(self, yaw_rate: np.ndarray, speed: np.ndarray) -> np.ndarray:
        curvature = yaw_rate / np.maximum(speed, self.curvature_speed_floor_mps)
        if self.curvature_zero_below_mps is not None:
            curvature = np.where(speed > self.curvature_zero_below_mps, curvature, 0.0)
        if self.curvature_limit is not None:
            curvature = np.clip(curvature, -self.curvature_limit, self.curvature_limit)
        return curvature


@dataclass(frozen=True)
class RigProfile:
    """Camera rig of one dataset, as recorded in a calibration JSON file."""

    name: str
    dataset_name: str
    """Dataset name understood by ``training.dataset_policy``."""
    native_front_camera_tile: bool
    """Whether the model also receives the full-resolution front camera."""
    cameras: tuple[CameraCalibration, ...]
    egomotion: EgomotionRule = EgomotionRule()

    @property
    def logical_ids(self) -> list[str]:
        return [camera.logical_id for camera in self.cameras]

    def ordered_by_slots(self, slots: list[str]) -> tuple[CameraCalibration, ...]:
        """Return the cameras in the model's slot order."""
        by_slot = {camera.slot: camera for camera in self.cameras}
        missing = [slot for slot in slots if slot not in by_slot]
        if missing:
            raise ValueError(
                f"Rig profile {self.name!r} has no camera for slots {missing}"
            )
        return tuple(by_slot[slot] for slot in slots)


def load_rig_profile(path: str | Path) -> RigProfile:
    """Load a calibration profile written by ``tools/export_*_calibration.py``."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cameras = tuple(
        CameraCalibration(
            logical_id=entry["logical_id"],
            slot=entry["slot"],
            image_size_wh=(int(entry["image_size_wh"][0]), int(entry["image_size_wh"][1])),
            intrinsic=np.asarray(entry["intrinsic"], dtype=np.float64),
            camera_to_reference=np.asarray(entry["camera_to_reference"], dtype=np.float64),
        )
        for entry in payload["cameras"]
    )
    for camera in cameras:
        if camera.intrinsic.shape != (3, 3) or camera.camera_to_reference.shape != (4, 4):
            raise ValueError(f"{path}: camera {camera.logical_id} has malformed matrices")
    return RigProfile(
        name=path.stem,
        dataset_name=payload["dataset_name"],
        native_front_camera_tile=bool(payload.get("native_front_camera_tile", False)),
        cameras=cameras,
        egomotion=EgomotionRule(**payload.get("egomotion", {})),
    )


def available_rig_profiles() -> list[RigProfile]:
    return [load_rig_profile(path) for path in sorted(CALIBRATION_DIR.glob("*.json"))]


def find_rig_profile(camera_ids: list[str]) -> RigProfile:
    """Return the profile whose cameras are exactly ``camera_ids``.

    ``AUTOE2E_RIG_PROFILE`` may name a profile file to use instead of the
    shipped ones, e.g. for a world recorded with another calibration revision.
    """
    requested = set(camera_ids)
    override = os.environ.get("AUTOE2E_RIG_PROFILE")
    profiles = [load_rig_profile(override)] if override else available_rig_profiles()
    for profile in profiles:
        if set(profile.logical_ids) == requested:
            return profile
    known = {profile.name: profile.logical_ids for profile in profiles}
    raise ValueError(
        f"No calibration profile matches the cameras {sorted(requested)}. "
        f"Known profiles: {known}"
    )
