"""Camera frame loading for the KIT Scenes Multimodal dataset.

KIT Scenes stores per-frame JPEGs on disk (not videos), already at the 10 Hz
reference timeline, so a single ``frame_idx`` indexes every camera and the ego
poses alike. The ``kitscenes`` SDK's ``SensorDataLoader`` decodes a frame to an
RGB ``np.ndarray``; this module resizes/normalises it for the AutoE2E backbone
and stacks the 7 views (6 cameras plus a derived tele crop) into the tensor
the model expects.

Camera projection matrices are computed from KITScenes calibration files, with
intrinsics scaled to match the backbone's actual resize/crop transform.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose

if TYPE_CHECKING:  # the SDK is only needed for the loader's type, not at runtime
    from kitscenes.sensors import SensorDataLoader

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

# A seventh view, derived rather than physical: a narrow centre crop of the
# long-range camera, taken at native resolution and then resized like any other
# view (#146).
#
# Why it is needed. What sets how far the model can see is not the sensor's pixel
# count but the pixels per degree that survive preprocessing, and every view is
# resized to `image_size` (256 by default). At 256 px the full 88.2 deg frame of
# the long-range camera carries 2.90 px/deg — the ring cameras carry 2.94 px/deg
# over 87.1 deg. So feeding the long-range camera whole buys no extra reach: the
# resize discards exactly the resolution that would have provided it.
#
# Cropping first keeps it. A centre crop of TELE_VIEW_HFOV_DEG taken before the
# resize yields 256 / 30 = 8.5 px/deg, 2.9x what any full view carries today.
TELE_VIEW_SOURCE = "camera_base_front_center"
TELE_VIEW_NAME = "tele_center"
TELE_VIEW_HFOV_DEG = 30.0

# Slot order of the tensors handed to the model: the six physical cameras above,
# then the derived tele view.
VIEW_NAMES: list[str] = [*CAMERA_NAMES, TELE_VIEW_NAME]

# Total views fed to the model = 6 cameras + 1 derived tele view.
NUM_VIEWS = 7


def tele_crop_box(
    intrinsic: np.ndarray,
    source_wh: tuple[int, int],
    hfov_deg: float = TELE_VIEW_HFOV_DEG,
) -> tuple[int, int, int]:
    """Square centre crop of ``hfov_deg`` around the source camera's optical axis.

    The crop is square so it survives the square resize without anisotropic
    distortion, and it is centred on the principal point rather than on the image
    centre, so the optical axis stays at the centre of the cropped frame.

    Args:
        intrinsic: ``(3, 3)`` pinhole matrix of the source camera.
        source_wh: ``(width, height)`` of the source image in pixels.
        hfov_deg: Field of view the crop should span, in degrees.

    Returns:
        ``(x0, y0, side)`` in source-image pixels.

    Raises:
        ValueError: if the requested field of view does not fit in the frame.
    """
    fx, cx, cy = float(intrinsic[0, 0]), float(intrinsic[0, 2]), float(intrinsic[1, 2])
    side = int(round(2.0 * fx * math.tan(math.radians(hfov_deg) / 2.0)))
    width, height = source_wh
    if side > width or side > height:
        raise ValueError(
            f"tele crop of {hfov_deg} deg needs {side}px but the source frame is "
            f"{width}x{height}"
        )
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    # Keep the crop inside the frame if the principal point sits off-centre.
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    return x0, y0, side


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
        camera_names: Views to compute matrices for, in slot order. May include
            the derived ``TELE_VIEW_NAME``. Defaults to ``VIEW_NAMES``.
        image_size: Optional packed output size as an int (square) or ``(H, W)``.
            This is the pipeline path and is mutually exclusive with transform.
 
    Returns:
        Float32 tensor of shape ``(len(camera_names), 3, 4)``.
        Does not include a slot for the map tile.
    """
    if camera_names is None:
        camera_names = VIEW_NAMES
    if (transform is None) == (image_size is None):
        raise ValueError("provide exactly one of transform or image_size")

    target_hw: tuple[int, int] | None
    if isinstance(image_size, int):
        target_hw = (image_size, image_size)
    else:
        target_hw = image_size

    matrices = []
    for cam_name in camera_names:
        is_tele = cam_name == TELE_VIEW_NAME
        calib = loader.get_camera_calibration(
            TELE_VIEW_SOURCE if is_tele else cam_name
        )

        source_wh = calib.image_size
        if source_wh is None:
            source_wh = loader.get_camera_image_size(
                TELE_VIEW_SOURCE if is_tele else cam_name, frame_idx=0
            )

        intrinsic = calib.intrinsic.astype(np.float64)
        if is_tele:
            # Cropping does not change focal length; it only moves the principal
            # point into the cropped frame. Everything downstream then treats the
            # crop as a narrower camera of its own.
            x0, y0, side = tele_crop_box(intrinsic, source_wh)
            intrinsic = intrinsic.copy()
            intrinsic[0, 2] -= x0
            intrinsic[1, 2] -= y0
            source_wh = (side, side)

        if target_hw is not None:
            target_h, target_w = target_hw
            source_w, source_h = source_wh
            K_scaled = intrinsic.copy()
            K_scaled[0, :] *= target_w / source_w
            K_scaled[1, :] *= target_h / source_h
        else:
            assert transform is not None
            K_scaled = scale_intrinsic(intrinsic, source_wh, transform)

        # invert calib.extrinsic to get T_ref_to_cam. The tele view shares the
        # source camera's pose: it is the same optical centre, seen through a
        # narrower window.
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
        camera_names: Ordered list of view names to load. May include the derived
            ``TELE_VIEW_NAME``. Defaults to ``VIEW_NAMES``.
        image_size: Optional raw pipeline output size as an int (square) or
            ``(H, W)``. Images are resized but not normalized.

    Returns:
        Float tensor of shape ``(len(camera_names), 3, H, W)``.
    """
    if camera_names is None:
        camera_names = VIEW_NAMES

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
        is_tele = cam_name == TELE_VIEW_NAME
        source_name = TELE_VIEW_SOURCE if is_tele else cam_name
        rgb_frame = loader.get_camera_image(source_name, frame_idx)  # (H, W, 3) RGB
        image = Image.fromarray(rgb_frame)
        if is_tele:
            # Crop at native resolution, before any resize: that is the whole
            # point of this view, and cropping afterwards would preserve nothing.
            calib = loader.get_camera_calibration(source_name)
            x0, y0, side = tele_crop_box(
                calib.intrinsic.astype(np.float64), (image.width, image.height)
            )
            image = image.crop((x0, y0, x0 + side, y0 + side))
        if transform is not None:
            camera_tensors.append(transform(image))
            continue
        if target_wh is not None:
            image = image.resize(target_wh, resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
        camera_tensors.append(torch.from_numpy(array).permute(2, 0, 1))

    return torch.stack(camera_tensors, dim=0)  # (V, 3, H, W)
