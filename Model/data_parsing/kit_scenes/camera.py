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

import numpy as np
import torch
from kitscenes.sensors import SensorDataLoader
from PIL import Image
from torchvision.transforms import Compose

# Shared, dataset-agnostic intrinsic scaling (re-exported for backward compat).
from ..calibration import scale_intrinsic

# Camera directories used as visual tiles for the KIT Scenes dataset.
# Order: long-range front, then the 5 remaining surround ring cameras.
#
# Which camera is which (from the calibration published in the SDK's
# notebooks/04_calibration_and_multimodal.ipynb; HFOV = 2*atan(W / 2*focal)):
#
#   camera_base_front_center       5856x3104   18.2 MPix   88.2 deg  <- long-range
#   camera_ring_*                  3504x2272    8.0 MPix   87.1 deg
#   camera_base_front_*_rect       2272x3488    7.9 MPix   63.3 deg  <- stereo pair
#
# Those three groups match the sensor suite in the dataset paper (arXiv:2606.02956):
# one long-range 88.4 deg camera, six 87.1 deg surround cameras and a tilted 63.3 deg
# stereo pair. So camera_base_front_center is the long-range imager, at 2.3x the
# pixel count of the ring cameras over essentially the same field of view.
#
# camera_ring_front is dropped (#146): it points the same way as the long-range
# camera and covers the same 87 deg, so it is the redundant one of the two. The
# stereo pair stays out — at 63.3 deg it trades field of view for baseline, and it
# is not the long-range imager.
CAMERA_NAMES: list[str] = [
    "camera_base_front_center",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
]

# Total views fed to the model = 6 cameras.
NUM_VIEWS = 6

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

    target_hw: tuple[int, int] | None
    if isinstance(image_size, int):
        target_hw = (image_size, image_size)
    else:
        target_hw = image_size
 
    matrices = []
    for cam_name in camera_names:
        calib = loader.get_camera_calibration(cam_name)
 
        source_wh = calib.image_size
        if source_wh is None:
            source_wh = loader.get_camera_image_size(cam_name, frame_idx=0)
        if target_hw is not None:
            target_h, target_w = target_hw
            source_w, source_h = source_wh
            K_scaled = calib.intrinsic.copy().astype(np.float64)
            K_scaled[0, :] *= target_w / source_w
            K_scaled[1, :] *= target_h / source_h
        else:
            assert transform is not None
            K_scaled = scale_intrinsic(
                calib.intrinsic, source_wh, transform
            )
 
        # invert calib.extrinsic to get T_ref_to_cam.
        T_ref_to_cam = np.linalg.inv(calib.extrinsic)   # (4, 4)
        P = K_scaled @ T_ref_to_cam[:3, :]              # (3, 4)
        matrices.append(P)
 
    return torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)  # (V, 3, 4)


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
