"""Audited L2D FOV/extrinsic projection contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_parsing.l2d.calibration import (
    L2D_ASSUMED_REFERENCE_CAMERA_HEIGHT_M,
    L2D_EXTRINSICS_SHA256,
    L2D_HARDWARE_LAYOUT_SHA256,
    L2D_PINHOLE_FOV_AXIS,
    L2D_SOURCE_IMAGE_SIZE_WH,
    L2D_VENDOR_SENSOR_ACTIVE_AREA_WH,
    compute_l2d_camera_from_ego_transforms,
    compute_l2d_projection_matrices,
    intrinsic_from_fov,
    l2d_projection_spec,
)
from data_parsing.l2d.camera import CAMERA_NAMES


def test_intrinsic_uses_explicit_horizontal_and_vertical_fov():
    intrinsic = intrinsic_from_fov(
        (1080, 1920),
        fov_x_degrees=110.65,
        fov_y_degrees=61.16,
    )

    assert intrinsic[0, 0] == pytest.approx(
        1920 / (2 * math.tan(math.radians(110.65) / 2))
    )
    assert intrinsic[1, 1] == pytest.approx(
        1080 / (2 * math.tan(math.radians(61.16) / 2))
    )
    assert intrinsic[0, 2] == pytest.approx(960.0)
    assert intrinsic[1, 2] == pytest.approx(540.0)


@pytest.mark.parametrize(
    ("fov_x", "fov_y"),
    [(90.0, None), (None, 60.0)],
)
def test_single_fov_axis_preserves_square_pixels(fov_x, fov_y):
    intrinsic = intrinsic_from_fov(
        (720, 1280),
        fov_x_degrees=fov_x,
        fov_y_degrees=fov_y,
    )

    assert intrinsic[0, 0] == pytest.approx(intrinsic[1, 1])


def test_projection_order_and_camera_optical_axes():
    image_size = (512, 768)
    matrices = compute_l2d_projection_matrices(image_size)
    camera_from_ego = compute_l2d_camera_from_ego_transforms()

    assert matrices.shape == (len(CAMERA_NAMES), 3, 4)
    expected_yaws = [0.0, 70.0, -70.0, 137.0, 179.0, -139.0]
    for index, (matrix, transform, expected_yaw) in enumerate(
        zip(matrices, camera_from_ego, expected_yaws)
    ):
        ego_from_camera = np.linalg.inv(transform)
        optical_point_ego = ego_from_camera @ np.asarray(
            [0.0, 0.0, 10.0, 1.0]
        )
        pixel_h = matrix @ optical_point_ego
        assert pixel_h[2] > 0.0
        np.testing.assert_allclose(
            pixel_h[:2] / pixel_h[2],
            [image_size[1] / 2, image_size[0] / 2],
            rtol=0.0,
            atol=1e-4,
        )
        optical_axis_ego = ego_from_camera[:3, :3] @ np.asarray(
            [0.0, 0.0, 1.0]
        )
        yaw = math.degrees(
            math.atan2(optical_axis_ego[1], optical_axis_ego[0])
        )
        assert yaw == pytest.approx(expected_yaw, abs=1.0), index


def test_projection_scales_to_packed_image_size():
    source = compute_l2d_projection_matrices((1080, 1920))
    packed = compute_l2d_projection_matrices((512, 512))

    np.testing.assert_allclose(
        packed[:, 0],
        source[:, 0] * (512 / 1920),
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        packed[:, 1],
        source[:, 1] * (512 / 1080),
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(packed[:, 2], source[:, 2])


def test_projection_uses_one_square_pixel_focal_length_at_source():
    source = compute_l2d_projection_matrices((1080, 1920))
    camera_from_ego = compute_l2d_camera_from_ego_transforms()
    intrinsic = source[0, :, :3] @ camera_from_ego[0, :3, :3].T
    expected_focal = 1080 / (
        2 * math.tan(math.radians(61.16) / 2)
    )

    assert intrinsic[0, 0] == pytest.approx(expected_focal)
    assert intrinsic[1, 1] == pytest.approx(expected_focal)
    assert intrinsic[0, 2] == pytest.approx(960.0)
    assert intrinsic[1, 2] == pytest.approx(540.0)


def test_projection_spec_is_honest_about_unpublished_distortion():
    spec = l2d_projection_spec(512)
    provenance = spec["provenance"]

    assert spec["type"] == "pinhole"
    assert spec["camera_order"] == CAMERA_NAMES
    assert spec["camera_slots"] == list(CANONICAL_SIX_CAMERA_SLOTS)
    assert provenance["extrinsics_sha256"] == L2D_EXTRINSICS_SHA256
    assert provenance["distortion_coefficients"] is None
    assert provenance["image_rectification_status"] == "unverified"
    assert provenance["reference_camera_translation_status"] == (
        "unpublished_assumed_zero"
    )
    assert provenance["reference_camera_orientation_status"] == (
        "unpublished_assumed_axis_permutation_zero_pitch_roll_yaw"
    )
    assert provenance["reference_camera_orientation_projection_impact"] == (
        "real_mounting_rotation_would_shift_horizon_ground_plane_"
        "and_bev_samples"
    )
    assert provenance["intrinsic_model_status"] == (
        "vertical_fov_square_pixel_source_pinhole_scaled_to_packed_image"
    )
    assert provenance["intrinsic_axis_selection_rationale"] == (
        "preserve_published_vertical_angles_used_by_ground_plane_"
        "projection_and_derive_horizontal_angles_for_square_pixels"
    )
    assert provenance["published_fov_consistency_status"] == (
        "horizontal_and_vertical_axes_do_not_form_a_square_pixel_"
        "pinhole_at_source_aspect_ratio_numeric_error_recorded_per_lens"
    )
    lens_by_camera = provenance["lens_by_camera"]
    assert lens_by_camera["observation.images.front_left"]["model"] == (
        "NileCAM21"
    )
    assert {
        lens_by_camera[camera_name]["model"]
        for camera_name in CAMERA_NAMES
        if camera_name != "observation.images.front_left"
    } == {"STURDeCAM21"}
    assert {
        lens_by_camera[camera_name]["selected_fov_axis"]
        for camera_name in CAMERA_NAMES
    } == {L2D_PINHOLE_FOV_AXIS}
    assert lens_by_camera["observation.images.front_left"][
        "published_axis_focal_ratio_fy_over_fx"
    ] == pytest.approx(1.376, rel=1e-3)
    assert lens_by_camera["observation.images.front_left"][
        "implied_fov_x_degrees"
    ] == pytest.approx(92.8, abs=0.1)
    assert provenance["lens_mapping_status"] == (
        "inferred_front_reference_standard_other_five_rugged_from_"
        "official_count_and_hardware_layout"
    )
    assert provenance["hardware_layout_url"].endswith(".png")
    assert (
        provenance["hardware_layout_sha256"]
        == L2D_HARDWARE_LAYOUT_SHA256
    )
    assert spec["ground_z_m"] == pytest.approx(
        -L2D_ASSUMED_REFERENCE_CAMERA_HEIGHT_M
    )
    assert provenance["ground_plane_status"] == (
        "assumed_from_unpublished_reference_camera_height"
    )
    assert provenance["reference_camera_height_basis"] == (
        "engineering_prior_for_passenger_vehicle_roof_camera_"
        "not_published_or_measured"
    )
    assert provenance["source_image_size_wh"] == list(
        L2D_SOURCE_IMAGE_SIZE_WH
    )
    assert provenance["vendor_sensor_active_area_wh"] == list(
        L2D_VENDOR_SENSOR_ACTIVE_AREA_WH
    )
    assert provenance["stream_field_of_view_status"] == (
        "unpublished_assumed_vendor_full_sensor_fov_applies_to_"
        "1920x1080_stream"
    )
    assert "pseudo" not in str(spec).lower()
    assert "rectified_pinhole" not in str(spec).lower()


def test_assumed_ground_plane_projects_below_front_camera_horizon():
    spec = l2d_projection_spec(512)
    matrix = np.asarray(spec["matrix"][0])
    ground_z = spec["ground_z_m"]

    near = matrix @ np.asarray([10.0, 0.0, ground_z, 1.0])
    far = matrix @ np.asarray([40.0, 0.0, ground_z, 1.0])
    near_uv = near[:2] / near[2]
    far_uv = far[:2] / far[2]

    assert near[2] > 0.0
    assert far[2] > 0.0
    assert near_uv[0] == pytest.approx(256.0)
    assert far_uv[0] == pytest.approx(256.0)
    assert near_uv[1] > far_uv[1] > 256.0


def test_missing_or_tampered_extrinsics_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        compute_l2d_projection_matrices(
            512,
            extrinsics_path=tmp_path / "missing.yaml",
        )

    tampered = tmp_path / "extrinsic_RDF.yaml"
    tampered.write_text("ref_cam: cam_front_left\n", encoding="ascii")
    with pytest.raises(ValueError, match="digest"):
        compute_l2d_projection_matrices(
            512,
            extrinsics_path=tampered,
        )
