"""Every constant that decides the bytes of L2D navigation, in one place.

``L2D_NAVIGATION_CONTRACT`` is derived from them and enters the L2D
pack_version, so changing how the OSM region is clipped, how an episode's
lane graph is cut, or how lanes are built gives a new dataset_id instead of
reusing a dataset packed by other rules.
"""

from __future__ import annotations

import hashlib
import json

L2D_NAVIGATION_VERSION = "l2d_osm_navigation_v1"
OSM_REGION_SCHEMA_VERSION = "l2d_osm_region_v1"
OSM_LANE_GRAPH_BUILDER_VERSION = "osm_lane_graph_builder_v2"
MAP_IMAGE_VERSION = "l2d_map_image_v3"
ANCHOR_PERIOD_NS = 500_000_000
# The OSM extract is clipped this far around every pose and route waypoint:
# far wider than the raster, so map matching keeps road connectivity.
OSM_REGION_BUFFER_M = 1000.0
# Lanes within this distance of an episode's trace form its lane graph.
EPISODE_MAP_MARGIN_M = 250.0
# The route matcher compares every trace point with every lane, so the trace
# is thinned to one point per this many metres.
TRACE_STEP_M = 4.0
MANEUVER_LOOKAHEAD_M = 100.0

DRIVABLE_HIGHWAYS = frozenset({
    "living_street",
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "residential",
    "secondary",
    "secondary_link",
    "service",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "unclassified",
})


def _contract() -> str:
    fields = {
        "navigation": L2D_NAVIGATION_VERSION,
        "region": OSM_REGION_SCHEMA_VERSION,
        "lane_graph": OSM_LANE_GRAPH_BUILDER_VERSION,
        "map_image": MAP_IMAGE_VERSION,
        "anchor_period_ns": ANCHOR_PERIOD_NS,
        "region_buffer_m": OSM_REGION_BUFFER_M,
        # Each partition renders from a region cut to its own episodes.
        "region_scope": "partition",
        "episode_margin_m": EPISODE_MAP_MARGIN_M,
        "trace_step_m": TRACE_STEP_M,
        "maneuver_lookahead_m": MANEUVER_LOOKAHEAD_M,
        "drivable_highways": sorted(DRIVABLE_HIGHWAYS),
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return f"{L2D_NAVIGATION_VERSION}-{digest[:12]}"


L2D_NAVIGATION_CONTRACT = _contract()
