"""The one geometry contract of the AutoBEV camera encoder.

Every AutoBEV tensor lives on this grid and camera size, and every checkpoint
and pack manifest that carries AutoBEV members records ``AUTOBEV_CONTRACT``.
``check_autobev_contract`` compares a recorded contract field by field: an
encoder trained with the BEVFormer default pillar range (-5 .. 3 m) has its
anchors 3 to 4 m away from these and must be refused, not loaded.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

from navigation.geometry import NavigationRasterGeometry

IMAGE_W: Final = 1344
IMAGE_H: Final = 768
NUM_VIEWS: Final = 6
HISTORY_FRAMES: Final = 7
FRAME_INTERVAL_US: Final = 500_000
EMBED_DIM: Final = 256
BEV_H: Final = 220
BEV_W: Final = 160
# z spans -0.5 .. 3.5 m above the road, so the four pillar anchors sit at 0,
# 1, 2 and 3 m and one lies on the road, where the depth gate can match the
# predicted depth of road pixels. Every dataset's ego origin is on the road.
PC_RANGE: Final = (-30.0, -40.0, -0.5, 80.0, 40.0, 3.5)
EGO_ORIGIN: Final = "road_surface"

# Also a NavigationRasterGeometry, so the nuPlan target builders can rasterize
# BEV labels directly on this grid.
AUTOBEV_GEOMETRY: Final = NavigationRasterGeometry(
    geometry_id="autobev-220x160-0p5m-v2",
    height_px=BEV_H,
    width_px=BEV_W,
    meters_per_pixel=0.5,
    x_min_m=-30.0,
    x_max_m=80.0,
    y_min_m=-40.0,
    y_max_m=40.0,
    ego_anchor_row=159.5,
    ego_anchor_col=79.5,
    matching_pc_range=PC_RANGE,
    matching_bev_h=BEV_H,
    matching_bev_w=BEV_W,
    route_corridor_width_m=3.5,
    destination_marker_radius_m=2.0,
    route_rear_clip_m=10.0,
)

AUTOBEV_CONTRACT: Final[dict[str, Any]] = {
    "geometry_id": AUTOBEV_GEOMETRY.geometry_id,
    "bev_hw": [BEV_H, BEV_W],
    "meters_per_cell": AUTOBEV_GEOMETRY.meters_per_pixel,
    "pc_range": list(PC_RANGE),
    "ego_origin": EGO_ORIGIN,
    "image_wh": [IMAGE_W, IMAGE_H],
    "num_views": NUM_VIEWS,
    "history_frames": HISTORY_FRAMES,
    "frame_interval_us": FRAME_INTERVAL_US,
}


def check_autobev_contract(found: Mapping[str, Any] | None, *, source: str) -> None:
    """Raise unless ``found`` equals ``AUTOBEV_CONTRACT`` in every field."""
    if found is None:
        raise ValueError(f"{source} records no AutoBEV geometry contract")
    expected = AUTOBEV_CONTRACT
    differing = sorted(
        key for key in set(expected) | set(found)
        if _normalized(found.get(key)) != _normalized(expected.get(key))
    )
    if differing:
        detail = ", ".join(
            f"{key}: {found.get(key)!r} != {expected.get(key)!r}" for key in differing
        )
        raise ValueError(f"{source} has a different AutoBEV geometry ({detail})")


def _normalized(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [float(item) if isinstance(item, (int, float)) else item for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value
