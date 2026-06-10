"""BEV map rasterisation for the KIT Scenes Multimodal dataset.

Provides ``generate_bev_map_tile`` for rendering using OpenCV. `kitscenes` SDK has visualization utilities based on Matplotlib.

Require only base ``lanelet2`` (no ``ml_converter`` wheel).

Rendering mirrors ``kitscenes.visualization.ml_converter_vis_utils``:
- White background.
- Road borders (green), curbstones, fence, guard rail drawn thick.
- Lane dividers: wide gray background stroke + thin coloured stroke on top.
- Centerlines: dashed darkred.
- Stop lines: red.
- Pedestrian crossings: yellow.

Coordinate frame
----------------
Poses are in the map-local frame (metres from map origin, axes aligned with
UTM 32N). All boundary polylines from ``get_lanelets_in_roi`` are in the same
frame. Tiles are ego-centric: forward (+X) → up, left (+Y) → left.

Caching
-------
``_cached_scene_map`` wraps ``load_scene_map`` in ``lru_cache`` so map.osm is
parsed once per scene per process.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour table (RGB) — mirrors ls_type_to_color in ml_converter_vis_utils
# ---------------------------------------------------------------------------
# Keys: (type_attr, subtype_attr). subtype=None matches any subtype.

_RGB_TABLE: dict[tuple[str, str | None], tuple[int, int, int]] = {
    ("road_border",        None):          (0,   200,   0),   # green
    ("curbstone",          "high"):        (0,   100,   0),   # darkgreen
    ("curbstone",          "low"):         (50,  205,  50),   # limegreen
    ("fence",              None):          (165,  42,  42),   # brown
    ("guard_rail",         None):          (139,  69,  19),   # saddlebrown
    ("wall",               None):          (205, 133,  63),   # peru
    ("building",           None):          (244, 164,  96),   # sandybrown
    ("drivable_area",      None):          (173, 216, 230),   # lightblue
    ("line_thin",          "dashed"):      (  0,   0, 255),   # blue
    ("line_thick",         "dashed"):      (  0,   0, 255),   # blue
    ("line_thin",          "solid"):       (  0,   0, 255),   # blue
    ("line_thick",         "solid"):       (  0,   0, 139),   # darkblue
    ("line_thin",          "solid_solid"): (  0,   0, 139),   # darkblue
    ("line_thin",          "solid_dashed"):(  65, 105, 225),  # royalblue
    ("line_thin",          "dashed_solid"):(100, 149, 237),   # cornflowerblue
    ("virtual",            None):          (105, 105, 105),   # dimgrey
    ("divider",            None):          (128, 128, 128),   # gray
    ("line_thin",          "centerline"):  (139,   0,   0),   # darkred
    ("bike_marking",       "dashed"):      (255, 140,   0),   # darkorange
    ("bike_marking",       "solid"):       (255,  69,   0),   # orangered
    ("stop_line",          None):          (255,   0,   0),   # red
    ("pedestrian_marking", None):          (255, 255,   0),   # yellow
    ("zig-zag",            None):          (255, 215,   0),   # gold
}

_FALLBACK_RGB: tuple[int, int, int] = (128, 0, 128)  # purple

_BORDER_TYPES = {
    "road_border", "curbstone", "fence", "guard_rail",
    "wall", "building", "drivable_area",
}
_DIVIDER_TYPES = {"line_thin", "line_thick"}
_DIVIDER_SUBTYPES = {"solid", "dashed", "solid_solid", "solid_dashed", "dashed_solid"}


def _get_rgb(line_type: str, subtype: str) -> tuple[int, int, int]:
    return _RGB_TABLE.get((line_type, subtype)) \
        or _RGB_TABLE.get((line_type, None), _FALLBACK_RGB)


def _attr(obj, key: str) -> str:
    return obj.attributes[key] if key in obj.attributes else ""


@functools.lru_cache(maxsize=None)
def _cached_scene_map(scene_path: Path):
    from kitscenes.map_api import load_scene_map
    return load_scene_map(scene_path)


def _to_px(pts: np.ndarray, ego: np.ndarray, yaw: float, scale: float, rs: int) -> np.ndarray:
    """Map-local XY → canvas pixels.

    Translates to ego-relative, rotates by -yaw so ego heading points up,
    then maps to pixel coords: forward (+X_ego) → up, left (+Y_ego) → left.
    """
    cx = rs / 2.0
    rel = pts[:, :2] - ego
    c, s = np.cos(-yaw), np.sin(-yaw)
    x_rot = c * rel[:, 0] - s * rel[:, 1]
    y_rot = s * rel[:, 0] + c * rel[:, 1]
    col = cx - y_rot * scale
    row = cx - x_rot * scale
    return np.stack([col, row], axis=1).astype(np.int32).reshape(-1, 1, 2)


def _cv_line(canvas, pts, ego, yaw, scale, rs, rgb, thickness):
    if len(pts) < 2:
        return
    bgr = (rgb[2], rgb[1], rgb[0])
    cv2.polylines(canvas, [_to_px(pts, ego, yaw, scale, rs)],
                  isClosed=False, color=bgr, thickness=thickness,
                  lineType=cv2.LINE_AA)


def _cv_divider(canvas, pts, ego, yaw, scale, rs, rgb):
    """Gray background stroke then thin coloured stroke — mirrors _plot_lane_dividers."""
    if len(pts) < 2:
        return
    px = _to_px(pts, ego, yaw, scale, rs)
    cv2.polylines(canvas, [px], isClosed=False,
                  color=(128, 128, 128), thickness=2, lineType=cv2.LINE_AA)
    bgr = (rgb[2], rgb[1], rgb[0])
    cv2.polylines(canvas, [px], isClosed=False,
                  color=bgr, thickness=1, lineType=cv2.LINE_AA)


def _cv_dashed(canvas, pts, ego, yaw, scale, rs, rgb, thickness, dash, gap):
    """Dashed polyline (OpenCV has no native dash support)."""
    if len(pts) < 2:
        return
    px = _to_px(pts, ego, yaw, scale, rs).reshape(-1, 2)
    bgr = (rgb[2], rgb[1], rgb[0])
    accum, drawing = 0.0, True
    for i in range(len(px) - 1):
        p0, p1 = px[i].astype(float), px[i + 1].astype(float)
        seg = float(np.linalg.norm(p1 - p0))
        if seg < 1e-3:
            continue
        d = (p1 - p0) / seg
        pos = 0.0
        while pos < seg:
            budget = (dash if drawing else gap) - accum
            step = min(budget, seg - pos)
            if drawing:
                cv2.line(canvas,
                         tuple((p0 + d * pos).astype(int)),
                         tuple((p0 + d * (pos + step)).astype(int)),
                         bgr, thickness, lineType=cv2.LINE_AA)
            pos += step
            accum += step
            if accum >= (dash if drawing else gap):
                accum, drawing = 0.0, not drawing


def generate_bev_map_tile(
    scene_path: Path,
    ego_x: float,
    ego_y: float,
    ego_yaw: float = 0.0,
    canvas_size: int = 224,
    radius_meters: float = 60.0,
    supersample: int = 4,
) -> np.ndarray | None:
    """Rasterise a semantic BEV map tile centred on the ego vehicle (OpenCV).

    Args:
        scene_path: Scene directory path.
        ego_local_x: Ego X in map-local frame (metres).
        ego_local_y: Ego Y in map-local frame (metres).
        ego_yaw: Ego heading in map frame (radians, Z-up convention). The tile
            is rotated so the ego's heading always points straight up.
        canvas_size: Output side length in pixels.
        radius_meters: Half-width of the observation window in metres.
        supersample: Render at N×canvas_size then downsample with INTER_AREA.

    Returns:
        uint8 RGB (canvas_size, canvas_size, 3). White if map unavailable.
    """
    scene_map = _cached_scene_map(scene_path)
    if scene_map is None:
        # Returning None to distinguish map load failure from an empty map (white).
        return None

    rs = canvas_size * supersample
    canvas = np.full((rs, rs, 3), 255, dtype=np.uint8)
    scale = rs / (radius_meters * 2.0)
    ego = np.array([ego_x, ego_y], dtype=np.float64)
    yaw = float(ego_yaw)
    ss = supersample

    try:
        lanelets = scene_map.get_lanelets_in_roi(center=ego, radius=radius_meters)
    except Exception:
        logger.debug("get_lanelets_in_roi failed for %s", scene_path.name, exc_info=True)
        lanelets = []

    # Pass 1: road borders and lane dividers
    for llt in lanelets:
        for bound in (llt.leftBound, llt.rightBound):
            pts = np.array([[p.x, p.y] for p in bound], dtype=np.float64)
            b_type = _attr(bound, "type")
            b_sub = _attr(bound, "subtype")
            if b_type in _BORDER_TYPES:
                _cv_line(canvas, pts, ego, yaw, scale, rs, _get_rgb(b_type, b_sub), 1)
            elif b_type in _DIVIDER_TYPES and b_sub in _DIVIDER_SUBTYPES:
                _cv_divider(canvas, pts, ego, yaw, scale, rs,
                            _get_rgb(b_type, b_sub))
            elif b_type == "virtual":
                _cv_line(canvas, pts, ego, yaw, scale, rs, _get_rgb("virtual", ""), 1)

    # Pass 2: centerlines — dashed darkred
    for llt in lanelets:
        if _attr(llt, "subtype") == "crosswalk":
            continue
        cl = np.array([[p.x, p.y] for p in llt.centerline], dtype=np.float64)
        if len(cl) >= 2:
            _cv_dashed(canvas, cl, ego, yaw, scale, rs,
                       _get_rgb("line_thin", "centerline"),
                       1, dash=8 * ss, gap=8 * ss)

    # Pass 3: pedestrian crossings — yellow
    for llt in lanelets:
        if _attr(llt, "subtype") != "crosswalk":
            continue
        for bound in (llt.leftBound, llt.rightBound):
            pts = np.array([[p.x, p.y] for p in bound], dtype=np.float64)
            _cv_line(canvas, pts, ego, yaw, scale, rs,
                     _get_rgb("pedestrian_marking", ""), 1)

    # Pass 4: stop lines — red
    try:
        for line in scene_map.get_stop_lines():
            pts = np.array(line, dtype=np.float64)
            _cv_line(canvas, pts, ego, yaw, scale, rs, _get_rgb("stop_line", ""), 1)
    except Exception:
        pass

    if supersample == 1:
        return canvas
    return cv2.resize(canvas, (canvas_size, canvas_size),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Visualization — legend + display
# ---------------------------------------------------------------------------

# Legend entries matching ls_type_to_color in ml_converter_vis_utils
_LEGEND_ENTRIES: list[tuple[str, tuple[float, float, float]]] = [
    ("Road Border",        (0,      200/255, 0)),
    ("Curbstone High",     (0,      100/255, 0)),
    ("Curbstone Low",      (50/255, 205/255, 50/255)),
    ("Fence / Guard Rail", (139/255, 69/255, 19/255)),
    ("Drivable Area",      (173/255, 216/255, 230/255)),
    ("Dashed lane",        (0,      0,       1.0)),
    ("Solid lane",         (0,      0,       139/255)),
    ("Solid-Dashed",       (65/255, 105/255, 225/255)),
    ("Virtual",            (105/255, 105/255, 105/255)),
    ("Centerline",         (139/255, 0,       0)),
    ("Stop Line",          (1.0,    0,       0)),
    ("Ped. Crossing",      (1.0,    1.0,     0)),
]


def visualise_bev_tile(
    bev_rgb: np.ndarray,
    title: str = "BEV map tile",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (9, 7),
) -> None:
    """Display a BEV map tile with a semantic legend.

    Args:
        bev_rgb: (H, W, 3) uint8 RGB array from either generate function.
        title: Figure title.
        save_path: If provided, saves to this path; otherwise calls plt.show().
        figsize: Matplotlib figure size in inches.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, (ax_map, ax_leg) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [4, 1]},
    )

    ax_map.imshow(bev_rgb)
    ax_map.set_title(title, fontsize=11)
    ax_map.axis("off")

    h, w = bev_rgb.shape[:2]
    ax_map.plot(w / 2, h / 2, marker="^", color="black",
                markersize=8, markeredgecolor="white", markeredgewidth=1, zorder=10)
    ax_map.annotate("N ↑ fwd", (w * 0.02, h * 0.04), color="black", fontsize=7)

    patches = [
        mpatches.Patch(facecolor=colour, edgecolor="grey", linewidth=0.5, label=label)
        for label, colour in _LEGEND_ENTRIES
    ]
    ax_leg.legend(handles=patches, loc="center left", fontsize=8,
                  frameon=False, handlelength=1.5, handleheight=1.2)
    ax_leg.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)