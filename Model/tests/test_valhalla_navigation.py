"""Offline OSM adapter and vehicle-local Valhalla provider tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from navigation.osm_adapter import (
    OSM_LANE_GRAPH_SCHEMA_VERSION,
    OSMMapAdapter,
)
from navigation.valhalla import (
    GeoRoutePose,
    LocalLaneSequenceResolver,
    ProviderRoute,
    ValhallaRouteProvider,
    decode_polyline6,
)


def _lane_graph():
    return {
        "schema_version": OSM_LANE_GRAPH_SCHEMA_VERSION,
        "map_version": "karlsruhe-v1",
        "frame": {
            "frame_id": "karlsruhe-local",
            "origin_latitude_deg": 49.0,
            "origin_longitude_deg": 8.0,
            "projection": "EPSG:32632 local ENU",
        },
        "lanes": [
            {
                "way_id": 10,
                "direction": "forward",
                "lane_index": 0,
                "centerline_enu_m": [[0, 0], [20, 0]],
                "left_boundary_enu_m": [[0, 2], [20, 2]],
                "right_boundary_enu_m": [[0, -2], [20, -2]],
                "successors": ["11:forward:0"],
                "maneuver": "straight",
            },
            {
                "way_id": 11,
                "direction": "forward",
                "lane_index": 0,
                "centerline_enu_m": [[20, 0], [40, 0]],
                "left_boundary_enu_m": [[20, 2], [40, 2]],
                "right_boundary_enu_m": [[20, -2], [40, -2]],
                "successors": [],
                "maneuver": "straight",
            },
        ],
        "semantic": {
            "crosswalks": [
                {
                    "id": "cw-1",
                    "points_enu_m": [
                        [18, -3],
                        [22, -3],
                        [22, 3],
                        [18, 3],
                    ],
                }
            ],
            "stop_lines": [
                {
                    "id": "stop-1",
                    "points_enu_m": [[17, -2], [17, 2]],
                }
            ],
            "traffic_signals": [
                {
                    "id": "signal-1",
                    "position_enu_m": [17, 3],
                }
            ],
        },
    }


def _encode_polyline6(points):
    output = []
    previous = [0, 0]
    for point in points:
        scaled = [int(round(float(value) * 1e6)) for value in point]
        for axis in range(2):
            delta = scaled[axis] - previous[axis]
            previous[axis] = scaled[axis]
            value = ~(delta << 1) if delta < 0 else delta << 1
            while value >= 0x20:
                output.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            output.append(chr(value + 63))
    return "".join(output)


def test_osm_adapter_extracts_stable_lane_and_semantic_contract(tmp_path):
    payload = _lane_graph()
    path = tmp_path / "lane-graph.json"
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )

    adapter = OSMMapAdapter.from_file(path)
    navigation_map = adapter.extract()

    assert [lane.lane_id for lane in adapter.lane_segments] == [
        "osm:karlsruhe-v1:10:forward:0",
        "osm:karlsruhe-v1:11:forward:0",
    ]
    assert adapter.lane_segments[0].successor_ids == (
        adapter.lane_segments[1].lane_id,
    )
    assert len(navigation_map.drivable_polygons) == 2
    assert len(navigation_map.crosswalk_polygons) == 1
    assert len(navigation_map.stop_lines) == 1
    assert len(navigation_map.static_traffic_signals) == 1
    assert navigation_map.provenance["source_sha256"]


def test_valhalla_provider_and_resolver_emit_selected_lane_sequence(
    monkeypatch,
):
    shape = np.asarray(
        [
            [49.0, 8.0],
            [49.0, 8.0001],
            [49.0, 8.0002],
            [49.0, 8.0003],
            [49.0, 8.0004],
        ]
    )
    enu_shape = np.column_stack(
        [
            np.arange(0.0, 50.0, 10.0),
            np.zeros(5),
            np.zeros(5),
        ]
    )
    monkeypatch.setattr(
        "navigation.valhalla._wgs84_to_local_enu",
        lambda points, frame: enu_shape,
    )
    captured = {}

    def transport(endpoint, payload, timeout):
        captured.update(
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
        )
        return {
            "id": payload["id"],
            "trip": {
                "legs": [
                    {
                        "shape": _encode_polyline6(shape),
                        "maneuvers": [
                            {
                                "begin_shape_index": 0,
                                "end_shape_index": 4,
                                "instruction": "Continue straight",
                                "type": 1,
                            }
                        ],
                    }
                ]
            },
        }

    provider = ValhallaRouteProvider(transport=transport)
    current = GeoRoutePose(
        float(shape[0, 0]),
        float(shape[0, 1]),
        90.0,
        123,
    )
    destination = GeoRoutePose(
        float(shape[-1, 0]),
        float(shape[-1, 1]),
        90.0,
        123,
    )
    provider_route = provider.plan(
        current,
        destination,
        "karlsruhe-v1",
    )
    adapter = OSMMapAdapter(_lane_graph(), source_sha256="source-sha")
    route = LocalLaneSequenceResolver(adapter).resolve(
        provider_route,
        revision=7,
    )

    assert captured["endpoint"] == "http://127.0.0.1:8002/route"
    assert captured["payload"]["costing"] == "auto"
    assert captured["payload"]["shape_format"] == "polyline6"
    np.testing.assert_allclose(
        decode_polyline6(_encode_polyline6(shape)),
        shape,
        atol=0.5e-6,
    )
    assert route.valid
    assert route.revision == 7
    assert [lane.lane_id for lane in route.lane_sequence] == [
        "osm:karlsruhe-v1:10:forward:0",
        "osm:karlsruhe-v1:11:forward:0",
    ]
    assert route.quality.unresolved_discontinuities == 0
    np.testing.assert_allclose(
        route.destination.position_enu_m[:2],
        [40.0, 0.0],
        atol=0.1,
    )


def test_resolver_marks_unmatched_provider_shape_invalid(monkeypatch):
    adapter = OSMMapAdapter(_lane_graph(), source_sha256="source-sha")
    shape = np.asarray(
        [[49.0, 8.0], [49.0, 8.0002], [49.0, 8.0004]]
    )
    monkeypatch.setattr(
        "navigation.valhalla._wgs84_to_local_enu",
        lambda points, frame: np.asarray(
            [[0, 100, 0], [20, 100, 0], [40, 100, 0]],
            dtype=np.float64,
        ),
    )
    provider_route = ProviderRoute(
        request_id="far-route",
        map_version=adapter.map_version,
        timestamp_ns=1,
        shape_wgs84=shape,
        maneuvers=(),
    )

    route = LocalLaneSequenceResolver(adapter).resolve(
        provider_route,
        revision=1,
    )

    assert not route.valid
    assert not route.lane_sequence
    assert "no_lane_sequence" in route.quality.failure_reasons


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8002",
        "http://valhalla.example.com:8002",
        "unix:///var/run/valhalla.sock",
    ],
)
def test_valhalla_provider_rejects_non_loopback_http(endpoint):
    with pytest.raises(ValueError, match="local HTTP|loopback"):
        ValhallaRouteProvider(endpoint)
