"""The AlpaSim driver's map context and route mask, from a trajdata vector map.

A straight two-lane road along +x (lanes at y = 0 and y = 3.5 m, 3.5 m wide)
with a wait line at x = 100 m. Runs where AlpaSim's packages are installed
(the AlpaSim venv on the GPU host).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "alpasim_driver"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

pytest.importorskip("alpasim_driver")
pytest.importorskip("trajdata")

from alpasim_grpc.v0 import common_pb2, egodriver_pb2  # noqa: E402
from alpasim_autoe2e.parser import pose_trajectory, route_in_local  # noqa: E402
from alpasim_autoe2e.plugin import MOCK_CHECKPOINT, AutoE2EDriver  # noqa: E402
from alpasim_autoe2e.scene_map import SceneMap  # noqa: E402
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY, MapChannel  # noqa: E402
from trajdata.maps.vec_map import VectorMap  # noqa: E402
from trajdata.maps.vec_map_elements import Polyline, RoadLane, WaitLine  # noqa: E402

from tests.test_alpasim_stream import KIT_CAMERAS, T0_US, _pose, _prediction_input  # noqa: E402

GEOMETRY = AUTOE2E_NAVIGATION_GEOMETRY


def _line(y: float, x0: float = -100.0, x1: float = 300.0) -> Polyline:
    x = np.linspace(x0, x1, 81)
    return Polyline(np.stack([x, np.full_like(x, y), np.zeros_like(x)], axis=1))


def _two_lane_map() -> VectorMap:
    vector_map = VectorMap(map_id="test:two_lanes")
    vector_map.add_map_element(
        RoadLane("right", _line(0.0), left_edge=_line(1.75), right_edge=_line(-1.75), adj_lanes_left={"left"})
    )
    vector_map.add_map_element(
        RoadLane("left", _line(3.5), left_edge=_line(5.25), right_edge=_line(1.75), adj_lanes_right={"right"})
    )
    vector_map.add_map_element(
        WaitLine("stop", Polyline(np.array([[100.0, -1.75, 0.0], [100.0, 1.75, 0.0]])), "STOP", False)
    )
    vector_map.__post_init__()  # as Artifact._finalize_map does
    vector_map.compute_search_indices()
    return vector_map


def _pixel(x: float, y: float) -> tuple[int, int]:
    row, col = GEOMETRY.ego_to_pixel(np.array([[x, y]]))[0]
    return int(round(row)), int(round(col))


@pytest.fixture(scope="module")
def scene_map() -> SceneMap:
    return SceneMap.from_vector_map(_two_lane_map(), destination_xy=np.array([150.0, 0.0, 0.0]))


def test_map_context_draws_the_road_in_the_ego_frame(scene_map: SceneMap) -> None:
    context, valid = scene_map.map_context((80.0, 0.0, 0.0))
    assert valid and context.shape == (14, 450, 300)
    ahead, off_road = _pixel(10.0, 0.0), _pixel(10.0, -20.0)
    assert context[MapChannel.DRIVABLE_AREA][ahead] == 1.0
    assert context[MapChannel.DRIVABLE_AREA][off_road] == 0.0
    assert context[MapChannel.LANE_CENTERLINE][ahead] > 0.0
    assert context[MapChannel.LANE_BOUNDARY][_pixel(10.0, 1.75)] > 0.0
    # The wait line at x = 100 m is 20 m ahead of the ego at x = 80 m.
    assert context[MapChannel.STOP_LINE][_pixel(20.0, 0.0)] > 0.0
    # Traffic flows along +x: cos = 1, sin = 0, stored as (v + 1) / 2.
    assert context[MapChannel.TRAFFIC_DIRECTION_COS][ahead] == pytest.approx(1.0)
    assert context[MapChannel.TRAFFIC_DIRECTION_SIN][ahead] == pytest.approx(0.5)
    assert context[MapChannel.KNOWN_MAP_AREA][ahead] == 1.0
    for channel in (MapChannel.INTERSECTION, MapChannel.STATIC_TRAFFIC_SIGNAL, MapChannel.ROAD_LEVEL):
        assert not context[channel].any()


def test_map_context_rotates_with_the_ego_heading(scene_map: SceneMap) -> None:
    # Ego heading +90 deg: the road along +x crosses in front of the ego,
    # running from its left (+y) to its right (-y).
    context, _ = scene_map.map_context((80.0, 0.0, np.pi / 2))
    assert context[MapChannel.DRIVABLE_AREA][_pixel(0.0, -10.0)] == 1.0
    assert context[MapChannel.DRIVABLE_AREA][_pixel(0.0, 10.0)] == 1.0
    assert context[MapChannel.DRIVABLE_AREA][_pixel(10.0, 0.0)] == 0.0


def test_route_corridor_covers_the_roadblock_and_destination(scene_map: SceneMap) -> None:
    route = np.stack([np.arange(80.0, 160.0, 4.0), np.zeros(20), np.zeros(20)], axis=1)
    assert scene_map.roadblock_lanes(route) == {"right", "left"}
    mask, valid = scene_map.route_mask(route, (80.0, 0.0, 0.0))
    assert valid and mask.shape == (2, 450, 300)
    assert mask[0][_pixel(10.0, 0.0)] == 1.0 and mask[0][_pixel(10.0, 3.5)] == 1.0
    assert mask[0][_pixel(10.0, -10.0)] == 0.0
    assert mask[1].max() == pytest.approx(1.0)
    assert np.unravel_index(mask[1].argmax(), mask[1].shape) == _pixel(70.0, 0.0)


def test_destination_outside_the_raster_is_not_drawn() -> None:
    far = SceneMap.from_vector_map(_two_lane_map(), destination_xy=np.array([900.0, 0.0, 0.0]))
    mask, valid = far.route_mask(np.array([[100.0, 0.0, 0.0]]), (80.0, 0.0, 0.0))
    assert valid and mask[0].any() and not mask[1].any()


def test_route_in_local_undoes_the_rig_frame_at_the_route_time() -> None:
    # Heading +90 deg, moving from y = 5 to y = 15 m; at the route's 0.5 s the rig is at y = 10.
    poses = [_pose(0, 10.0, 5.0, np.pi / 2), _pose(1_000_000, 10.0, 15.0, np.pi / 2)]
    route = egodriver_pb2.Route(timestamp_us=500_000)
    for x in (4.0, float("nan")):
        route.waypoints.append(common_pb2.Vec3(x=x, y=0.0, z=0.0))
    local = route_in_local(route, pose_trajectory(poses))
    assert local.shape == (1, 3)
    assert np.allclose(local[0, :2], [10.0, 14.0], atol=1e-4)


class _OneSceneMap:
    def __init__(self, scene_map: SceneMap) -> None:
        self.scene_map = scene_map

    def get(self, scene_id: str) -> SceneMap | None:
        return self.scene_map if scene_id == "clipgt-test" else None


def test_parse_feeds_map_and_route_for_a_known_scene(scene_map: SceneMap) -> None:
    driver = AutoE2EDriver(MOCK_CHECKPOINT, KIT_CAMERAS)
    driver.parser.scene_maps = _OneSceneMap(scene_map)
    prediction_input = _prediction_input()  # straight at 8 m/s, at x = 80 m at t0
    prediction_input.scene_id = "clipgt-test"
    prediction_input.route = egodriver_pb2.Route(timestamp_us=T0_US)
    for x in range(0, 80, 4):
        prediction_input.route.waypoints.append(common_pb2.Vec3(x=float(x), y=0.0, z=0.0))

    inputs = driver.parser.parse(prediction_input).model_inputs
    assert inputs["map_valid"].tolist() == [True] and inputs["route_valid"].tolist() == [True]
    assert inputs["map_context"][0, MapChannel.DRIVABLE_AREA][_pixel(10.0, 0.0)] == 1.0
    assert inputs["route_mask"][0, 0][_pixel(10.0, 0.0)] == 1.0

    prediction_input.scene_id = "clipgt-unknown"
    inputs = driver.parser.parse(prediction_input).model_inputs
    assert not inputs["map_valid"].any() and not inputs["route_mask"].any()


def test_destination_is_drawn_without_a_route(scene_map: SceneMap) -> None:
    driver = AutoE2EDriver(MOCK_CHECKPOINT, KIT_CAMERAS)
    driver.parser.scene_maps = _OneSceneMap(scene_map)
    for route in (None, egodriver_pb2.Route(timestamp_us=T0_US)):  # none, and no waypoints
        prediction_input = _prediction_input()
        prediction_input.scene_id = "clipgt-test"
        prediction_input.route = route
        inputs = driver.parser.parse(prediction_input).model_inputs
        assert inputs["route_valid"].tolist() == [True]
        assert not inputs["route_mask"][0, 0].any()
        assert inputs["route_mask"][0, 1].max() == pytest.approx(1.0)


def test_route_corridor_keeps_one_carriageway() -> None:
    vector_map = VectorMap(map_id="test:two_way")
    vector_map.add_map_element(
        RoadLane("east", _line(0.0), left_edge=_line(1.75), right_edge=_line(-1.75), adj_lanes_left={"west"})
    )
    west = _line(3.5)
    west = Polyline(west.points[::-1].copy())
    vector_map.add_map_element(
        RoadLane("west", west, left_edge=Polyline(_line(1.75).points[::-1].copy()),
                 right_edge=Polyline(_line(5.25).points[::-1].copy()), adj_lanes_right={"east"})
    )
    vector_map.__post_init__()
    vector_map.compute_search_indices()
    two_way = SceneMap.from_vector_map(vector_map, destination_xy=np.array([150.0, 0.0, 0.0]))
    route = np.stack([np.arange(80.0, 160.0, 4.0), np.zeros(20), np.zeros(20)], axis=1)
    assert two_way.roadblock_lanes(route) == {"east"}
    # Route points far off the map add no lane.
    assert two_way.roadblock_lanes(route + [0.0, 50.0, 0.0]) == set()
