"""Registering the L2D map image and reading its road surface into the BEV grid."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from data_parsing.l2d.map_image import (  # noqa: E402
    MapImageRegistration,
    bev_layers,
    register,
    road_mask,
)

BACKGROUND = (242, 239, 233)
ROAD = (255, 255, 255)
DOT = (110, 129, 224)
EGO = (60, 62, 110)


def _image(pixels_per_meter, rotation_rad, waypoints, road_along_east=False):
    """A Carto-like tile: one white road running north (or east), route dots on it."""
    image = np.full((256, 256, 3), BACKGROUND, dtype=np.uint8)
    truth = MapImageRegistration((127.5, 127.5), pixels_per_meter, rotation_rad, 0.0, len(waypoints))
    # Road: 8 m wide strip along one local axis, drawn in metres.
    along = np.linspace(-120.0, 120.0, 2000)
    for across in np.linspace(-4.0, 4.0, 40):
        points = np.stack([np.full_like(along, across), along], axis=-1)
        if road_along_east:
            points = points[:, ::-1]
        pixels = truth.to_pixels(points)
        column, row = np.rint(pixels).astype(int).T
        keep = (column >= 0) & (column < 256) & (row >= 0) & (row < 256)
        image[row[keep], column[keep]] = ROAD
    for point in waypoints:
        column, row = truth.to_pixels(np.asarray(point, dtype=float))
        image[int(round(row)) - 1:int(round(row)) + 2, int(round(column)) - 1:int(round(column)) + 2] = DOT
    image[126:130, 126:130] = EGO
    return image


@pytest.mark.parametrize("rotation", [0.0, math.radians(35.0)])
def test_route_dots_register_scale_and_rotation(rotation):
    waypoints = np.asarray([[0.0, 8.0 * k] for k in range(1, 11)])
    image = _image(1.25, rotation, waypoints)
    fit = register(image, waypoints, expected_rotation_rad=rotation)
    assert fit is not None and fit.dot_count == 10
    assert fit.pixels_per_meter == pytest.approx(1.25, rel=0.03)
    assert fit.rotation_rad == pytest.approx(rotation, abs=0.03)
    assert fit.residual_px < 1.0


def test_real_tile_failure_modes_still_register():
    """The nearest waypoint hides under the ego, and other blue blobs appear."""
    waypoints = np.asarray([[0.0, 1.3]] + [[0.0, 8.0 * k] for k in range(1, 10)])
    image = _image(1.25, 0.0, waypoints)
    image[40:44, 200:204] = DOT
    image[60:63, 30:33] = DOT
    image[126:131, 126:131] = EGO
    fit = register(image, waypoints)
    assert fit is not None and fit.pixels_per_meter == pytest.approx(1.25, rel=0.03)


def test_unregistrable_images_are_rejected():
    waypoints = np.asarray([[0.0, 8.0 * k] for k in range(1, 11)])
    blank = np.full((256, 256, 3), BACKGROUND, dtype=np.uint8)
    assert register(blank, waypoints) is None
    # A north-up tile that only fits rotated 10 degrees is not this route.
    assert register(_image(1.25, np.radians(10.0), waypoints), waypoints) is None
    # Dots drawn for a different route do not fit these waypoints.
    other = _image(1.25, 0.0, np.asarray([[8.0 * k, 0.0] for k in range(1, 6)] + [[0.0, -9.0 * k] for k in range(1, 6)]))
    assert register(other, waypoints) is None


def test_road_surface_lands_on_the_bev_grid_in_ego_coordinates():
    waypoints = np.asarray([[0.0, 8.0 * k] for k in range(1, 11)])
    image = _image(1.25, 0.0, waypoints)
    assert road_mask(image).sum() > 0
    assert not road_mask(np.full((4, 4, 3), BACKGROUND, dtype=np.uint8)).any()
    fit = register(image, waypoints)
    forward, left = np.meshgrid(np.arange(-20.0, 40.0, 1.0), np.arange(-20.0, 20.0, 1.0), indexing="xy")
    # Ego heading north: forward is +north, left is -east.
    road, footprint = bev_layers(image, fit, yaw_rad=math.pi / 2, forward_m=forward, left_m=left)
    assert footprint.all()
    on_road = road[np.abs(left) <= 3.0]
    off_road = road[np.abs(left) >= 8.0]
    assert on_road.mean() > 0.9 and off_road.mean() < 0.05
