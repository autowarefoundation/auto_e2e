"""The camera rig AutoE2E sees in a CARLA scenario.

ACS declares every camera it renders at session start: OpenCV pinhole
intrinsics, the resolution, and ``rig_to_camera``, the camera's optical frame
(x right, y down, z along the optical axis) in the rig frame (rear axle,
x forward, y left, z up), as contract revision 2 states it. That is exactly
what the model needs to project, so the profile is built from the declared
cameras rather than from a dataset's calibration file: if the CARLA rig is not
quite the dataset's, the projections still describe the images it rendered.

The dataset profile (``nuplan`` by default) only contributes what the cameras
cannot tell: each camera's model slot (matched by ``logical_id``), the dataset
policy and its egomotion curvature rule.

Cameras declared with an F-theta lens (the Cosmos3 world model's Hyperion
cameras) stand in for the dataset's pinholes through virtual views
(:mod:`.lens`): a dataset camera declared under its own ``logical_id`` but
through F-theta is rectified at its declared pose; one not declared at all is
drawn, at the dataset's mount, from the F-theta cameras that see it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from alpasim_autoe2e.config import (
    CALIBRATION_DIR,
    CameraCalibration,
    RigProfile,
    load_rig_profile,
)
from carla_driver_interface.geometry import Pose

from .lens import FthetaLens, SourceCamera, VirtualView


def dataset_profile(name_or_path: str) -> RigProfile:
    """A shipped profile by name (``nuplan``, ``kit_scenes``) or a profile file."""
    path = CALIBRATION_DIR / f"{name_or_path}.json"
    return load_rig_profile(path if path.is_file() else name_or_path)


def _pose(camera: Any) -> np.ndarray:
    return np.asarray(Pose.from_proto(camera.rig_to_camera).as_matrix(), dtype=np.float64)


def _ftheta_source(camera: Any) -> SourceCamera | None:
    spec = camera.intrinsics
    if spec.WhichOneof("camera_param") != "ftheta_param":
        return None
    lens = FthetaLens.from_proto(spec.ftheta_param, (int(spec.resolution_w), int(spec.resolution_h)))
    return SourceCamera(camera.logical_id, lens, _pose(camera))


def pinhole_from_declaration(camera: Any, slot: str) -> CameraCalibration:
    """A :class:`CameraCalibration` from one ACS ``AvailableCamera`` with a pinhole."""
    spec = camera.intrinsics
    kind = spec.WhichOneof("camera_param")
    if kind != "opencv_pinhole_param":
        raise ValueError(
            f"camera {camera.logical_id}: AutoE2E needs a pinhole or F-theta camera, got {kind}"
        )
    pinhole = spec.opencv_pinhole_param
    if list(pinhole.radial_coeffs) and any(pinhole.radial_coeffs):
        raise ValueError(f"camera {camera.logical_id}: lens distortion is not supported")
    intrinsic = np.array(
        [
            [pinhole.focal_length_x, 0.0, pinhole.principal_point_x],
            [0.0, pinhole.focal_length_y, pinhole.principal_point_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraCalibration(
        logical_id=camera.logical_id,
        slot=slot,
        image_size_wh=(int(spec.resolution_w), int(spec.resolution_h)),
        intrinsic=intrinsic,
        camera_to_reference=_pose(camera),
    )


def profile_from_session(
    cameras: Mapping[str, Any], dataset: RigProfile, tile_hw: tuple[int, int] = (512, 512)
) -> tuple[RigProfile, dict[str, VirtualView]]:
    """The rig of a session, and the virtual views of its cameras that are not
    declared pinholes (by the dataset camera's ``logical_id``).

    For every camera of ``dataset``:

    * declared under its ``logical_id`` as a pinhole: taken as declared;
    * declared under its ``logical_id`` through F-theta: rectified into the
      dataset camera's intrinsics at the declared pose;
    * not declared: drawn at the dataset camera's mount from every declared
      F-theta camera that sees part of it, the one seeing most first.

    Cameras the dataset does not use otherwise are ignored (ACS may render
    extra cameras for recording).
    """
    sources = [s for s in (_ftheta_source(c) for c in cameras.values()) if s is not None]
    rig: list[CameraCalibration] = []
    views: dict[str, VirtualView] = {}
    for camera in dataset.cameras:
        declared = cameras.get(camera.logical_id)
        if declared is not None and _ftheta_source(declared) is None:
            rig.append(pinhole_from_declaration(declared, camera.slot))
            continue
        view_sources: tuple[SourceCamera, ...]
        if declared is not None:
            source = _ftheta_source(declared)
            assert source is not None  # declared pinholes were taken above
            pose, view_sources = _pose(declared), (source,)
        else:
            pose = camera.camera_to_reference.astype(np.float64)
            probe = VirtualView(camera.intrinsic, camera.image_size_wh, pose, tuple(sources))
            coverage = probe.coverage_by_source(tile_hw)
            ranked = sorted(zip(coverage, range(len(sources))), reverse=True)
            view_sources = tuple(sources[i] for c, i in ranked if c > 0.0)
            if not view_sources:
                raise ValueError(
                    f"AutoE2E ({dataset.name} rig) needs the camera {camera.logical_id} "
                    f"({camera.slot}); the scenario declares {sorted(cameras)}, none of "
                    "which is it or an F-theta camera that sees its view. Run it with the "
                    "AutoE2E driver preset."
                )
        virtual = CameraCalibration(
            logical_id=camera.logical_id,
            slot=camera.slot,
            image_size_wh=camera.image_size_wh,
            intrinsic=camera.intrinsic.astype(np.float64),
            camera_to_reference=pose,
        )
        rig.append(virtual)
        views[camera.logical_id] = VirtualView(
            virtual.intrinsic, virtual.image_size_wh, pose, view_sources
        ).drawing(tile_hw)
    profile = RigProfile(
        name=f"{dataset.name}@carla",
        dataset_name=dataset.dataset_name,
        native_front_camera_tile=dataset.native_front_camera_tile,
        cameras=tuple(rig),
        egomotion=dataset.egomotion,
    )
    return profile, views
