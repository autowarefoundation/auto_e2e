"""Published L2D camera calibration converted to the model projection ABI.

L2D publishes inter-camera extrinsics, but not the reference camera pose in the
vehicle frame. The conversion therefore assumes coincident origins and an exact
FLU-to-RDF axis permutation (zero reference-camera pitch, roll, and yaw). Any
real mounting rotation shifts the projected horizon and every ground-plane/BEV
sample; consumers must treat this projection as an audited approximation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS

from .camera import CAMERA_NAMES

L2D_EXTRINSICS_URL = (
    "https://huggingface.co/datasets/yaak-ai/L2D/raw/"
    "1e7578b183c1cabd3b0f1061828b4cc76a323bd2/extrinsic_RDF.yaml"
)
L2D_EXTRINSICS_SHA256 = (
    "d4bb7e0a1a771a0f81acc1042d56affa8255d0fa1944a71d9fb8c188eddd5bd5"
)
L2D_HARDWARE_SOURCE_URL = (
    "https://github.com/huggingface/blog/blob/main/"
    "lerobot-goes-to-driving-school.md"
)
L2D_HARDWARE_LAYOUT_URL = (
    "https://slabstatic.com/prod/assets/iripw2qz/post/1w8z0ln0/"
    "preimages/Sa-fklsLOuDJGSVyRQrPMjj8.png"
)
L2D_HARDWARE_LAYOUT_SHA256 = (
    "690e8d2381d3e496e47217e9ddcfe64a034703ba9065fadac82dd8694265f4dd"
)
L2D_SOURCE_IMAGE_SIZE_WH = (1920, 1080)
L2D_VENDOR_SENSOR_ACTIVE_AREA_WH = (2048, 1280)
L2D_ASSUMED_REFERENCE_CAMERA_HEIGHT_M = 1.3
L2D_PINHOLE_FOV_AXIS = "vertical"

_EXTRINSICS_PATH = Path(__file__).with_name("extrinsic_RDF.yaml")
_REFERENCE_CAMERA = "cam_front_left"
_COORDINATE_SYSTEM = {"x": "right", "y": "down", "z": "front"}
_CAMERA_TO_EXTRINSIC_KEY = {
    "observation.images.front_left": "cam_front_left",
    "observation.images.left_forward": "cam_left_forward",
    "observation.images.right_forward": "cam_right_forward",
    "observation.images.left_backward": "cam_left_backward",
    "observation.images.rear": "cam_rear",
    "observation.images.right_backward": "cam_right_backward",
}
_NILECAM21_FOV_DEGREES = (110.65, 61.16)
_STURDECAM21_FOV_DEGREES = (98.22, 50.34)
_NILECAM21_SPEC_URL = (
    "https://www.e-consystems.com/camera-modules/"
    "ar0233-hdr-gmsl-camera-module.asp"
)
_STURDECAM21_SPEC_URL = (
    "https://www.e-consystems.com/camera-modules/"
    "ip67-ar0233-gmsl2-hdr-camera.asp"
)
_CAMERA_LENS = {
    **{
        camera_name: (
            "STURDeCAM21",
            _STURDECAM21_FOV_DEGREES,
            _STURDECAM21_SPEC_URL,
        )
        for camera_name in CAMERA_NAMES
    },
    "observation.images.front_left": (
        "NileCAM21",
        _NILECAM21_FOV_DEGREES,
        _NILECAM21_SPEC_URL,
    ),
}

# Vehicle FLU (x=front, y=left, z=up) to the published reference-camera RDF
# axes. L2D does not publish the reference camera's vehicle-frame orientation or
# translation, so this assumes zero pitch/roll/yaw and coincident origins.
_T_REFERENCE_RDF_FROM_EGO_FLU = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _normalize_image_size(
    image_size: int | tuple[int, int],
) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = image_size
    else:
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or (height, width)")
        height, width = image_size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image dimensions must be positive integers")
    return height, width


def intrinsic_from_fov(
    image_size: int | tuple[int, int],
    *,
    fov_x_degrees: float | None = None,
    fov_y_degrees: float | None = None,
) -> np.ndarray:
    """Build a centered pinhole intrinsic from one or two FOV axes."""
    height, width = _normalize_image_size(image_size)
    if fov_x_degrees is None and fov_y_degrees is None:
        raise ValueError("at least one FOV axis is required")
    for name, value in (
        ("fov_x_degrees", fov_x_degrees),
        ("fov_y_degrees", fov_y_degrees),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value < 180.0
        ):
            raise ValueError(f"{name} must be finite and in (0, 180)")

    fx = (
        width
        / (2.0 * math.tan(math.radians(fov_x_degrees) / 2.0))
        if fov_x_degrees is not None
        else None
    )
    fy = (
        height
        / (2.0 * math.tan(math.radians(fov_y_degrees) / 2.0))
        if fov_y_degrees is not None
        else None
    )
    # A single published FOV axis plus square pixels determines the other axis.
    if fx is None:
        fx = fy
    if fy is None:
        fy = fx
    assert fx is not None and fy is not None
    return np.asarray(
        [
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_extrinsics(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path)
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"L2D extrinsics are unavailable: {resolved}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != L2D_EXTRINSICS_SHA256:
        raise ValueError(
            "L2D extrinsics digest differs from the pinned official file"
        )
    try:
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ValueError("L2D extrinsics YAML is invalid") from exc
    if not isinstance(document, Mapping):
        raise ValueError("L2D extrinsics YAML must contain a mapping")

    expected_keys = {
        *_CAMERA_TO_EXTRINSIC_KEY.values(),
        "coordinate_system",
        "ref_cam",
    }
    if set(document) != expected_keys:
        raise ValueError("L2D extrinsics camera set is invalid")
    if document.get("coordinate_system") != _COORDINATE_SYSTEM:
        raise ValueError("L2D extrinsics must use the published RDF axes")
    if document.get("ref_cam") != _REFERENCE_CAMERA:
        raise ValueError("L2D extrinsics reference camera is invalid")

    for camera_key in _CAMERA_TO_EXTRINSIC_KEY.values():
        entry = document[camera_key]
        if not isinstance(entry, Mapping) or set(entry) != {
            "extrinsic_rotation_ref_cam_from_cam",
            "extrinsic_t_ref_cam_from_cam",
        }:
            raise ValueError(f"L2D extrinsics entry is invalid: {camera_key}")
        rotation = np.asarray(
            entry["extrinsic_rotation_ref_cam_from_cam"],
            dtype=np.float64,
        )
        translation = np.asarray(
            entry["extrinsic_t_ref_cam_from_cam"],
            dtype=np.float64,
        )
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                rtol=0.0,
                atol=1e-6,
            )
            or not math.isclose(
                float(np.linalg.det(rotation)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"L2D extrinsics transform is invalid: {camera_key}"
            )
    return document


def compute_l2d_projection_matrices(
    image_size: int | tuple[int, int],
    *,
    extrinsics_path: str | Path = _EXTRINSICS_PATH,
) -> np.ndarray:
    """Return ``K @ T_camera_from_ego`` in canonical six-camera slot order."""
    transforms = compute_l2d_camera_from_ego_transforms(
        extrinsics_path=extrinsics_path,
    )
    matrices = []
    for camera_name, camera_from_ego in zip(CAMERA_NAMES, transforms):
        _, (_, fov_y), _ = _CAMERA_LENS[camera_name]
        source_width, source_height = L2D_SOURCE_IMAGE_SIZE_WH
        source_intrinsic = intrinsic_from_fov(
            (source_height, source_width),
            fov_y_degrees=fov_y,
        )
        target_height, target_width = _normalize_image_size(image_size)
        resize = np.diag([
            target_width / source_width,
            target_height / source_height,
            1.0,
        ])
        intrinsic = resize @ source_intrinsic
        matrices.append(intrinsic @ camera_from_ego[:3])
    result = np.stack(matrices).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("L2D projection matrices are non-finite")
    return result


def compute_l2d_camera_from_ego_transforms(
    *,
    extrinsics_path: str | Path = _EXTRINSICS_PATH,
) -> np.ndarray:
    """Return camera-from-ego transforms in canonical six-camera slot order."""
    document = _load_extrinsics(extrinsics_path)
    transforms = []
    for camera_name in CAMERA_NAMES:
        entry = document[_CAMERA_TO_EXTRINSIC_KEY[camera_name]]
        reference_from_camera = np.eye(4, dtype=np.float64)
        reference_from_camera[:3, :3] = np.asarray(
            entry["extrinsic_rotation_ref_cam_from_cam"],
            dtype=np.float64,
        )
        reference_from_camera[:3, 3] = np.asarray(
            entry["extrinsic_t_ref_cam_from_cam"],
            dtype=np.float64,
        )
        transforms.append(
            np.linalg.inv(reference_from_camera)
            @ _T_REFERENCE_RDF_FROM_EGO_FLU
        )
    result = np.stack(transforms)
    if result.shape != (len(CAMERA_NAMES), 4, 4):
        raise ValueError("L2D camera transform count is invalid")
    return result


def l2d_projection_spec(
    image_size: int | tuple[int, int],
    *,
    extrinsics_path: str | Path = _EXTRINSICS_PATH,
) -> dict[str, Any]:
    """Serialize the audited L2D pinhole approximation and its limitations."""
    height, width = _normalize_image_size(image_size)
    matrices = compute_l2d_projection_matrices(
        (height, width),
        extrinsics_path=extrinsics_path,
    )
    source_width, source_height = L2D_SOURCE_IMAGE_SIZE_WH
    lens_by_camera = {
        camera_name: {
            "fov_x_degrees": fov[0],
            "fov_y_degrees": fov[1],
            "implied_fov_x_degrees": math.degrees(
                2.0
                * math.atan(
                    source_width
                    * math.tan(math.radians(fov[1]) / 2.0)
                    / source_height
                )
            ),
            "published_axis_focal_ratio_fy_over_fx": (
                (
                    source_height
                    / (2.0 * math.tan(math.radians(fov[1]) / 2.0))
                )
                / (
                    source_width
                    / (2.0 * math.tan(math.radians(fov[0]) / 2.0))
                )
            ),
            "selected_fov_axis": L2D_PINHOLE_FOV_AXIS,
            "model": model,
            "spec_url": spec_url,
        }
        for camera_name, (model, fov, spec_url) in _CAMERA_LENS.items()
    }
    return {
        "type": "pinhole",
        "matrix": matrices.tolist(),
        "reference_frame": (
            "ego_flu_with_reference_camera_origin_approximation"
        ),
        "ground_z_m": -L2D_ASSUMED_REFERENCE_CAMERA_HEIGHT_M,
        "camera_order": list(CAMERA_NAMES),
        "camera_slots": list(CANONICAL_SIX_CAMERA_SLOTS),
        "image_size_hw": [height, width],
        "provenance": {
            "distortion_coefficients": None,
            "distortion_status": "unpublished_not_applied",
            "extrinsics_sha256": L2D_EXTRINSICS_SHA256,
            "extrinsics_url": L2D_EXTRINSICS_URL,
            "ground_plane_status": (
                "assumed_from_unpublished_reference_camera_height"
            ),
            "reference_camera_height_basis": (
                "engineering_prior_for_passenger_vehicle_roof_camera_"
                "not_published_or_measured"
            ),
            "hardware_layout_url": L2D_HARDWARE_LAYOUT_URL,
            "hardware_layout_sha256": L2D_HARDWARE_LAYOUT_SHA256,
            "hardware_source_url": L2D_HARDWARE_SOURCE_URL,
            "image_rectification_status": "unverified",
            "intrinsic_model_status": (
                "vertical_fov_square_pixel_source_pinhole_scaled_to_"
                "packed_image"
            ),
            "intrinsic_axis_selection_rationale": (
                "preserve_published_vertical_angles_used_by_ground_plane_"
                "projection_and_derive_horizontal_angles_for_square_pixels"
            ),
            "lens_by_camera": lens_by_camera,
            "lens_mapping_status": (
                "inferred_front_reference_standard_other_five_rugged_from_"
                "official_count_and_hardware_layout"
            ),
            "published_fov_consistency_status": (
                "horizontal_and_vertical_axes_do_not_form_a_square_pixel_"
                "pinhole_at_source_aspect_ratio_numeric_error_recorded_per_"
                "lens"
            ),
            "reference_camera_translation_m": [0.0, 0.0, 0.0],
            "reference_camera_translation_status": (
                "unpublished_assumed_zero"
            ),
            "reference_camera_orientation_status": (
                "unpublished_assumed_axis_permutation_zero_pitch_roll_yaw"
            ),
            "reference_camera_orientation_projection_impact": (
                "real_mounting_rotation_would_shift_horizon_ground_plane_"
                "and_bev_samples"
            ),
            "source_image_size_wh": list(L2D_SOURCE_IMAGE_SIZE_WH),
            "stream_field_of_view_status": (
                "unpublished_assumed_vendor_full_sensor_fov_applies_to_"
                "1920x1080_stream"
            ),
            "vendor_sensor_active_area_wh": list(
                L2D_VENDOR_SENSOR_ACTIVE_AREA_WH
            ),
        },
    }
