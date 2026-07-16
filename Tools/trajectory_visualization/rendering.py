"""Deterministic camera and BEV rendering for trajectory report frames."""

from __future__ import annotations

import io
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Tools.trajectory_visualization.artifacts import ShardSample


FRAME_SIZE = (1280, 720)
PANEL_SIZE = 560
PREDICTION_COLOR = (52, 211, 153)
TARGET_COLOR = (59, 130, 246)
BACKGROUND_COLOR = (9, 13, 20)
PANEL_COLOR = (15, 23, 42)
GRID_COLOR = (51, 65, 85)
TEXT_COLOR = (226, 232, 240)
MUTED_COLOR = (148, 163, 184)
_DEPTH_EPS = 1e-5


def trajectory_extent(
    trajectories: Iterable[np.ndarray],
    *,
    minimum: float = 20.0,
    maximum: float = 150.0,
) -> float:
    extent = minimum
    for trajectory in trajectories:
        values = np.asarray(trajectory)
        if values.size:
            extent = max(extent, float(np.abs(values).max()) * 1.15)
    return min(extent, maximum)


def _view_value(value: Any, view: int, fallback: float = 0.0) -> float:
    if isinstance(value, list):
        if not value:
            return fallback
        selected = value[view] if view < len(value) else fallback
    else:
        selected = value
    try:
        result = float(selected)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _view_pair(
    value: Any,
    view: int,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if not isinstance(value, list) or not value:
        return fallback
    selected: Any = value
    if isinstance(value[0], list):
        if view >= len(value):
            return fallback
        selected = value[view]
    if not isinstance(selected, list) or len(selected) != 2:
        return fallback
    return (
        _view_value(selected[0], 0, fallback[0]),
        _view_value(selected[1], 0, fallback[1]),
    )


def _polynomial(coefficients: Sequence[float], theta: float) -> float:
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * theta + float(coefficient)
    return result


def _pinhole_point(
    spec: Mapping[str, Any],
    view: int,
    point: np.ndarray,
    image_wh: tuple[int, int],
) -> tuple[float, float] | None:
    matrices = spec.get("matrix")
    if not isinstance(matrices, list) or view >= len(matrices):
        return None
    matrix = np.asarray(matrices[view], dtype=np.float64)
    if matrix.shape != (3, 4):
        return None
    projected = matrix @ np.array([point[0], point[1], 0.0, 1.0])
    depth = float(projected[2])
    if not math.isfinite(depth) or depth <= _DEPTH_EPS:
        return None
    u = float(projected[0] / depth) / image_wh[0]
    v = float(projected[1] / depth) / image_wh[1]
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


def _ftheta_point(
    spec: Mapping[str, Any],
    view: int,
    point: np.ndarray,
) -> tuple[float, float] | None:
    transforms = spec.get("t_camera_ego")
    if not isinstance(transforms, list) or view >= len(transforms):
        return None
    transform = np.asarray(transforms[view], dtype=np.float64)
    if transform.shape != (4, 4):
        return None
    x_value, y_value, z_value, _ = (
        transform @ np.array([point[0], point[1], 0.0, 1.0])
    )
    rho = max(math.hypot(x_value, y_value), _DEPTH_EPS)
    theta = math.atan2(rho, z_value)
    max_theta = _view_value(spec.get("max_theta"), view, math.nan)
    if (math.isfinite(max_theta) and theta > max_theta) or (
        not math.isfinite(max_theta) and z_value <= _DEPTH_EPS
    ):
        return None

    raw_poly = spec.get("fw_poly")
    if not isinstance(raw_poly, list) or not raw_poly:
        return None
    selected = raw_poly[view] if isinstance(raw_poly[0], list) else raw_poly
    if not isinstance(selected, list) or not selected:
        return None
    radius = _polynomial(selected, theta)
    cx = _view_value(spec.get("cx"), view)
    cy = _view_value(spec.get("cy"), view)
    width, height = _view_pair(spec.get("image_wh"), view, (256.0, 256.0))
    u = (cx + radius * (x_value / rho)) / width
    v = (cy + radius * (y_value / rho)) / height
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


def project_trajectory(
    calibration: Mapping[str, Any],
    trajectory: np.ndarray,
    *,
    camera_index: int,
    image_wh: tuple[int, int],
) -> list[list[tuple[float, float]]]:
    spec = calibration.get("projection")
    if not isinstance(spec, dict):
        return []
    geometry_type = str(
        spec.get("type", calibration.get("geometry_type", "pseudo"))
    )
    if geometry_type == "pseudo":
        return []

    paths: list[list[tuple[float, float]]] = [[]]
    for point in trajectory:
        projected = (
            _ftheta_point(spec, camera_index, point)
            if geometry_type == "ftheta"
            else _pinhole_point(
                spec,
                camera_index,
                point,
                image_wh,
            )
        )
        if projected is None:
            if paths[-1]:
                paths.append([])
            continue
        paths[-1].append(projected)
    return [path for path in paths if len(path) >= 2]


def _draw_camera_path(
    draw: ImageDraw.ImageDraw,
    paths: Sequence[Sequence[tuple[float, float]]],
    *,
    image_wh: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    for path in paths:
        points = [
            (
                round(u * (image_wh[0] - 1)),
                round(v * (image_wh[1] - 1)),
            )
            for u, v in path
        ]
        draw.line(points, fill=(0, 0, 0), width=width + 4, joint="curve")
        draw.line(points, fill=color, width=width, joint="curve")


def _draw_bev(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    extent: float,
) -> Image.Image:
    panel = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), PANEL_COLOR)
    draw = ImageDraw.Draw(panel)
    center_x = PANEL_SIZE / 2
    ego_y = PANEL_SIZE * 0.84
    scale = PANEL_SIZE / 2 / extent

    for metres in range(
        -math.ceil(extent / 10) * 10,
        math.ceil(extent / 10) * 10 + 1,
        10,
    ):
        horizontal = ego_y - metres * scale
        vertical = center_x - metres * scale
        if 0 <= horizontal <= PANEL_SIZE:
            draw.line(
                (0, horizontal, PANEL_SIZE, horizontal),
                fill=GRID_COLOR,
                width=1,
            )
        if 0 <= vertical <= PANEL_SIZE:
            draw.line(
                (vertical, 0, vertical, PANEL_SIZE),
                fill=GRID_COLOR,
                width=1,
            )

    def screen_points(values: np.ndarray) -> list[tuple[int, int]]:
        return [
            (
                round(center_x - float(point[1]) * scale),
                round(ego_y - float(point[0]) * scale),
            )
            for point in values
        ]

    target_points = screen_points(target)
    prediction_points = screen_points(prediction)
    if len(target_points) >= 2:
        draw.line(
            target_points,
            fill=TARGET_COLOR,
            width=4,
            joint="curve",
        )
    if len(prediction_points) >= 2:
        draw.line(
            prediction_points,
            fill=(0, 0, 0),
            width=9,
            joint="curve",
        )
        draw.line(
            prediction_points,
            fill=PREDICTION_COLOR,
            width=5,
            joint="curve",
        )
    draw.polygon(
        (
            (center_x, ego_y - 12),
            (center_x - 8, ego_y + 10),
            (center_x + 8, ego_y + 10),
        ),
        fill=TEXT_COLOR,
    )
    return panel


def render_frame(
    sample: ShardSample,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    v0: float,
    base_seed: int,
    extent: float,
    camera_index: int,
) -> Image.Image:
    """Render a fixed-size camera + BEV frame suitable for H.264 encoding."""
    camera = Image.open(io.BytesIO(sample.camera_jpeg)).convert("RGB")
    camera_draw = ImageDraw.Draw(camera)
    image_wh = camera.size
    _draw_camera_path(
        camera_draw,
        project_trajectory(
            sample.calibration,
            target,
            camera_index=camera_index,
            image_wh=image_wh,
        ),
        image_wh=image_wh,
        color=TARGET_COLOR,
        width=3,
    )
    _draw_camera_path(
        camera_draw,
        project_trajectory(
            sample.calibration,
            prediction,
            camera_index=camera_index,
            image_wh=image_wh,
        ),
        image_wh=image_wh,
        color=PREDICTION_COLOR,
        width=4,
    )
    camera = camera.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    bev = _draw_bev(prediction, target, extent=extent)

    frame = Image.new("RGB", FRAME_SIZE, BACKGROUND_COLOR)
    frame.paste(camera, (24, 136))
    frame.paste(bev, (696, 136))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    draw.text(
        (24, 22),
        "AutoE2E trajectory report",
        fill=TEXT_COLOR,
        font=font,
    )
    draw.text(
        (24, 52),
        (
            f"scene {sample.scene_uid}  frame {sample.frame_idx}  "
            f"v0 {v0:.2f} m/s  seed {base_seed}"
        ),
        fill=MUTED_COLOR,
        font=font,
    )
    draw.text(
        (24, 88),
        f"camera {camera_index}",
        fill=MUTED_COLOR,
        font=font,
    )
    draw.text(
        (696, 88),
        f"BEV +/-{extent:.0f} m",
        fill=MUTED_COLOR,
        font=font,
    )
    draw.line((696, 116, 736, 116), fill=PREDICTION_COLOR, width=5)
    draw.text((744, 108), "model prediction", fill=TEXT_COLOR, font=font)
    draw.line((886, 116, 926, 116), fill=TARGET_COLOR, width=4)
    draw.text((934, 108), "recorded future", fill=TEXT_COLOR, font=font)
    draw.text(
        (24, 704),
        sample.sample_uid,
        fill=MUTED_COLOR,
        font=font,
    )
    return frame
