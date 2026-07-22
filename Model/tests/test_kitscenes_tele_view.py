"""The derived tele view must buy angular resolution, not just exist (#146).

The point of the view is that a centre crop taken *before* the resize keeps
pixels per degree that the resize would otherwise throw away. These tests pin
that property with the real calibration of camera_base_front_center, published
in the KITScenes SDK notebook 04_calibration_and_multimodal.ipynb:

    focal 3022.0 px, principal point (2924.0, 1545.0), frame 5856x3104
"""

import math

import numpy as np
import pytest

from data_parsing.kit_scenes.camera import (
    TELE_VIEW_HFOV_DEG,
    TELE_VIEW_NAME,
    TELE_VIEW_SOURCE,
    VIEW_NAMES,
    tele_crop_box,
)

FOCAL_PX = 3022.0
FRAME_WH = (5856, 3104)
INTRINSIC = np.array(
    [[FOCAL_PX, 0.0, 2924.0], [0.0, FOCAL_PX, 1545.0], [0.0, 0.0, 1.0]]
)
PACKED_PX = 256


def _hfov_deg(width_px: float, focal_px: float) -> float:
    return 2.0 * math.degrees(math.atan(width_px / (2.0 * focal_px)))


class TestTeleCropBox:
    def test_crop_spans_the_requested_field_of_view(self):
        _, _, side = tele_crop_box(INTRINSIC, FRAME_WH)
        assert _hfov_deg(side, FOCAL_PX) == pytest.approx(TELE_VIEW_HFOV_DEG, abs=0.05)

    def test_crop_is_centred_on_the_principal_point(self):
        x0, y0, side = tele_crop_box(INTRINSIC, FRAME_WH)
        # After cropping, the optical axis must land in the middle of the frame,
        # otherwise the projection matrices and the pixels disagree.
        assert INTRINSIC[0, 2] - x0 == pytest.approx(side / 2.0, abs=1.0)
        assert INTRINSIC[1, 2] - y0 == pytest.approx(side / 2.0, abs=1.0)

    def test_crop_fits_inside_the_source_frame(self):
        x0, y0, side = tele_crop_box(INTRINSIC, FRAME_WH)
        assert 0 <= x0 and x0 + side <= FRAME_WH[0]
        assert 0 <= y0 and y0 + side <= FRAME_WH[1]

    def test_rejects_a_field_of_view_that_does_not_fit(self):
        with pytest.raises(ValueError, match="does not fit|needs"):
            tele_crop_box(INTRINSIC, FRAME_WH, hfov_deg=170.0)


class TestTeleViewBuysResolution:
    def test_crop_carries_more_pixels_per_degree_than_the_full_frame(self):
        """The reason this view exists. Without the crop the long-range camera
        arrives at the same angular resolution as a ring camera, because both are
        resized to the same square."""
        _, _, side = tele_crop_box(INTRINSIC, FRAME_WH)
        full_hfov = _hfov_deg(FRAME_WH[0], FOCAL_PX)
        tele_hfov = _hfov_deg(side, FOCAL_PX)

        full_px_per_deg = PACKED_PX / full_hfov
        tele_px_per_deg = PACKED_PX / tele_hfov

        assert tele_px_per_deg > 2.5 * full_px_per_deg

    def test_full_long_range_frame_is_no_sharper_than_a_ring_camera(self):
        """Control for the claim above: at 256 px the long-range camera's own
        frame carries no more angular resolution than a ring camera, so simply
        feeding it whole cannot extend range."""
        long_range = PACKED_PX / _hfov_deg(FRAME_WH[0], FOCAL_PX)
        ring = PACKED_PX / _hfov_deg(3504, 1841.0)  # camera_ring_front calibration
        assert long_range == pytest.approx(ring, abs=0.1)


class TestViewSlots:
    def test_tele_view_is_the_last_slot(self):
        assert VIEW_NAMES[-1] == TELE_VIEW_NAME
        assert len(VIEW_NAMES) == 7

    def test_tele_view_derives_from_a_camera_that_is_also_fed_whole(self):
        assert TELE_VIEW_SOURCE in VIEW_NAMES
