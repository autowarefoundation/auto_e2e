"""Offline GPS-to-map-tile rendering for datasets that lack BEV map images.

See README.md for the full preprocessing workflow. The rendered tiles match
the L2D BEV map format and can be fed through the same timm transform as
camera tiles.
"""

from .gps_to_map import (
    fetch_road_network,
    gps_to_tensor,
    map_match_waypoints,
    render_map_tile,
)

__all__ = [
    "fetch_road_network",
    "gps_to_tensor",
    "map_match_waypoints",
    "render_map_tile",
]
