"""Camera frame loading for the KIT Scenes Multimodal dataset.

KIT Scenes stores per-frame JPEGs on disk (not videos), already at the 10 Hz
reference timeline, so a single ``frame_idx`` indexes every camera and the ego
poses alike. The ``kitscenes`` SDK's ``SensorDataLoader`` decodes a frame to an
RGB ``np.ndarray``; this module resizes/normalises it for the AutoE2E backbone
and stacks the 6 camera views into the tensor the model expects.

Camera projection matrices are computed from KITScenes calibration files, with
intrinsics scaled to match the backbone's actual resize/crop transform.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from kitscenes.sensors import SensorDataLoader
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision.transforms import Compose

# Shared, dataset-agnostic intrinsic scaling (re-exported for backward compat).
from ..calibration import scale_intrinsic
from ..camera_slots import CANONICAL_SIX_CAMERA_SLOTS

# Camera directories used as visual tiles for the KIT Scenes dataset.
# Order: long-range front, then the 5 remaining surround ring cameras.
#
# camera_ring_front is omitted because it duplicates the long-range front
# camera's approximately 88-degree forward coverage at a lower resolution.
# The narrower stereo front pair remains outside the model input contract.
CAMERA_NAMES: list[str] = [
    "camera_base_front_center",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear_left",
    "camera_ring_rear",
    "camera_ring_rear_right",
]

# Total views fed to the model = 6 cameras.
NUM_VIEWS = len(CAMERA_NAMES)
CAMERA_SLOTS = CANONICAL_SIX_CAMERA_SLOTS
CAMERA_SLOT_BY_NAME = dict(zip(CAMERA_NAMES, CAMERA_SLOTS))


def _target_hw(
    image_size: int | tuple[int, int] | None,
) -> tuple[int, int] | None:
    if isinstance(image_size, int):
        return (image_size, image_size)
    return image_size


def _scaled_intrinsic(
    loader: SensorDataLoader,
    camera_name: str,
    *,
    transform: Compose | None,
    target_hw: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    calib = loader.get_camera_calibration(camera_name)
    source_wh = calib.image_size
    if source_wh is None:
        source_wh = loader.get_camera_image_size(
            camera_name,
            frame_idx=0,
        )
    if target_hw is not None:
        target_h, target_w = target_hw
        source_w, source_h = source_wh
        intrinsic = calib.intrinsic.copy().astype(np.float64)
        intrinsic[0, :] *= target_w / source_w
        intrinsic[1, :] *= target_h / source_h
    else:
        assert transform is not None
        intrinsic = scale_intrinsic(
            calib.intrinsic,
            source_wh,
            transform,
        )
    return intrinsic, np.asarray(calib.extrinsic, dtype=np.float64)


def _pose_matrix(pose: object) -> np.ndarray:
    translation = np.asarray(
        getattr(pose, "translation"),
        dtype=np.float64,
    )
    rotation = np.asarray(
        getattr(pose, "rotation"),
        dtype=np.float64,
    )
    if (
        translation.shape != (3,)
        or rotation.shape != (4,)
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError("KITScenes ego pose is invalid")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(rotation).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def compute_camera_projection_matrices(
    loader: SensorDataLoader,
    transform: Compose | None = None,
    camera_names: list[str] | None = None,
    image_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Compute ``(3, 4)`` projection matrices for each camera view.
 
    ``P = K_scaled @ T_ref_to_cam`` maps 3-D reference-frame points to
    pixel coordinates in the backbone-resized image.
 
    Args:
        loader: ``SensorDataLoader`` for the scene.
        transform: Optional backbone transform used by the standalone parser.
        camera_names: Cameras to compute matrices for, in slot order.
            Defaults to ``CAMERA_NAMES``.
        image_size: Optional packed output size as an int (square) or ``(H, W)``.
            This is the pipeline path and is mutually exclusive with transform.
 
    Returns:
        Float32 tensor of shape ``(len(camera_names), 3, 4)``.
        Does not include a slot for the map tile.
    """
    if camera_names is None:
        camera_names = CAMERA_NAMES
    if (transform is None) == (image_size is None):
        raise ValueError("provide exactly one of transform or image_size")

    target_hw = _target_hw(image_size)
 
    matrices = []
    for cam_name in camera_names:
        intrinsic, camera_to_reference = _scaled_intrinsic(
            loader,
            cam_name,
            transform=transform,
            target_hw=target_hw,
        )
        reference_to_camera = np.linalg.inv(camera_to_reference)
        P = intrinsic @ reference_to_camera[:3, :]
        matrices.append(P)
 
    return torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)  # (V, 3, 4)


def compute_temporal_camera_projection_matrices(
    loader: SensorDataLoader,
    poses: Sequence[object],
    *,
    reference_frame_idx: int,
    history_frame_indices: Sequence[int],
    camera_names: list[str] | None = None,
    image_size: int | tuple[int, int],
) -> torch.Tensor:
    """Map current reference-frame points into historical camera images."""
    if camera_names is None:
        camera_names = CAMERA_NAMES
    if not history_frame_indices:
        raise ValueError("history_frame_indices must not be empty")
    if not 0 <= reference_frame_idx < len(poses):
        raise IndexError("reference_frame_idx leaves the pose sequence")
    typed_history = [int(index) for index in history_frame_indices]
    if any(
        index < 0 or index >= len(poses)
        for index in typed_history
    ):
        raise IndexError("history frame leaves the pose sequence")
    if typed_history != sorted(typed_history) or any(
        left >= right
        for left, right in zip(typed_history, typed_history[1:])
    ):
        raise ValueError("history frames must be strictly increasing")
    if typed_history[-1] >= reference_frame_idx:
        raise ValueError("history frames must precede the reference frame")

    target_hw = _target_hw(image_size)
    current_global_from_reference = _pose_matrix(
        poses[reference_frame_idx]
    )
    camera_contracts = [
        _scaled_intrinsic(
            loader,
            camera_name,
            transform=None,
            target_hw=target_hw,
        )
        for camera_name in camera_names
    ]
    frame_matrices = []
    for history_index in typed_history:
        history_reference_from_current_reference = (
            np.linalg.inv(_pose_matrix(poses[history_index]))
            @ current_global_from_reference
        )
        camera_matrices = []
        for intrinsic, camera_to_reference in camera_contracts:
            camera_from_current_reference = (
                np.linalg.inv(camera_to_reference)
                @ history_reference_from_current_reference
            )
            camera_matrices.append(
                intrinsic @ camera_from_current_reference[:3, :]
            )
        frame_matrices.append(np.stack(camera_matrices))
    matrices = np.stack(frame_matrices).astype(np.float32)
    if not np.isfinite(matrices).all():
        raise ValueError("KITScenes temporal projection is non-finite")
    return torch.from_numpy(matrices)


def load_camera_frame(
    loader: SensorDataLoader,
    frame_idx: int,
    transform: Compose | None = None,
    camera_names: list[str] | None = None,
    image_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load and preprocess the camera views at a single reference frame.

    Args:
        loader: ``SensorDataLoader`` for the scene, supplied by the dataset so
            its per-scene caches are reused across __getitem__ calls.
        frame_idx: Index into the scene's reference timeline.
        transform: Optional backbone preprocessing transform.
        camera_names: Ordered list of camera directory names to load.
            Defaults to ``CAMERA_NAMES``.
        image_size: Optional raw pipeline output size as an int (square) or
            ``(H, W)``. Images are resized but not normalized.

    Returns:
        Float tensor of shape ``(len(camera_names), 3, H, W)``.
    """
    if camera_names is None:
        camera_names = CAMERA_NAMES

    if transform is not None and image_size is not None:
        raise ValueError("transform and image_size are mutually exclusive")
    if isinstance(image_size, int):
        target_wh = (image_size, image_size)
    elif image_size is not None:
        target_wh = (image_size[1], image_size[0])
    else:
        target_wh = None

    camera_tensors = []
    for cam_name in camera_names:
        rgb_frame = loader.get_camera_image(cam_name, frame_idx)  # (H, W, 3) RGB
        image = Image.fromarray(rgb_frame)
        if transform is not None:
            camera_tensors.append(transform(image))
            continue
        if target_wh is not None:
            image = image.resize(target_wh, resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
        camera_tensors.append(torch.from_numpy(array).permute(2, 0, 1))

    return torch.stack(camera_tensors, dim=0)  # (V, 3, H, W)
