"""Driving reward for one sampled plan: progress gated by safety, as in CaRL and EPDMS.

    R = NC * DAC * DDC * V * EP * TTC * LK * C

scored at the plan's 0.1 s steps over its first 4 s, in the scene's local frame:

* NC (no at-fault collision), NAVSIM's rules: a contact is at fault when the
  object stands still, when the ego's front edge hits it, or when it is a
  side contact and the ego is not inside one lane. The ego standing still and
  contacts on the ego's rear edge are not at fault; after a contact that is
  not at fault the object is ignored, as NAVSIM does. An at-fault contact
  gives 0 for a road user (vehicle, person, rider, animal; parked cars
  included) and 0.5 for a static obstacle.
* DAC (drivable area), AlpaSim's offroad rule: the ego box is off the road
  when the lanes do not cover it and it touches a road edge (in a map without
  road edges: when its centre leaves the lanes). Lanes are the
  polygons between their edges, or the centreline widened to 3.7 m where an
  edge is missing, and their union is grown by 5 cm so the slivers between
  neighbouring lanes sampled at different vertices do not count.
* DDC (driving direction): metres driven where every lane under the ego box's
  centre points against the plan (> 120 deg, AlpaSim's wrong-lane rule), per 1 s
  window; <= 2 m keeps 1, <= 6 m gives 0.5, more gives 0 (EPDMS).
* V (renderer validity, #123): the plan's largest distance to the recorded
  path; 1 up to 2.5 m, falling to 0 at 3.0 m. A path distance, so braking or
  falling behind the log is not penalised (RAD's position deviation).
* EP (ego progress): arc length the plan gains along the recorded path from
  the ego's own position, over what the recording gained from the same place
  in 4 s of its moving time (by place, since the ego can trail or lead the
  log; waiting excluded, since the reward has no traffic-light term), clipped
  to [0, 1]; 1 when the recording moves less than 5 m (NAVSIM). Multiplying by
  EP gives a plan that stops 0, where PDMS's weighted average would still give
  it 9/14 (#123).
* TTC: 0.5 when, at any step, the ego box projected at constant velocity 0.3,
  0.6 or 0.9 s ahead would make an at-fault contact; else 1 (CaRL, NAVSIM).
* LK (lane keeping): mean over the steps of 1 within 0.5 m of the nearest
  lane centreline in the plan's direction, falling linearly to 0 at 1.75 m.
* C (comfort): 1 - 0.5 n / 6, with n the nuPlan comfort bounds the plan
  violates (CaRL), from the realised speed (the ego's speed at t0 first) and
  curvature.

The human filter of EPDMS forgives DAC, DDC and LK where the recorded drive
breaks the same rule at the same place: a plan step is compared with the two
recorded poses around its place along the recorded path. The lane polygons under-cover
real roads, and the log is the reference driver.

``gated`` marks a plan that NC, DAC, DDC or V sets to 0, and ``safe_fraction``
is the share of its steps before the first such violation. CaRL ends the
episode at a violation, so a policy earns progress until then; ranking gated
plans by ``safe_fraction`` keeps that order among plans that all fail.

References: CaRL, arXiv:2504.17838; NAVSIM PDMS, arXiv:2406.15349; EPDMS,
arXiv:2506.04218; RAD, arXiv:2502.13144.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

DT_S = 0.1
HORIZON_STEPS = 40
STOPPED_MPS = 0.05
TTC_LOOKAHEAD_S = (0.3, 0.6, 0.9)
LANE_KEEP_FULL_M, LANE_KEEP_ZERO_M = 0.5, 1.75
VALID_FULL_M, VALID_ZERO_M = 2.5, 3.0
ONCOMING_HEADING_RAD = 2.0 * math.pi / 3.0
MIN_RECORDED_PROGRESS_M = 5.0
LANE_WIDTH_WITHOUT_EDGES_M = 3.7
LANE_UNION_TOLERANCE_M = 0.05
# Tracks farther than this from every plan step cannot touch the ego or its TTC projection.
TRACK_REACH_M = 40.0
# NuRec track classes that are obstacles rather than road users (NAVSIM's static objects).
STATIC_OBSTACLE_CLASSES = frozenset({"protruding_object"})
# nuPlan comfort bounds (min, max): lon. accel, lat. accel, jerk, lon. jerk, yaw rate, yaw accel.
COMFORT_BOUNDS = ((-4.05, 2.40), (-4.89, 4.89), (-8.37, 8.37), (-4.13, 4.13), (-0.95, 0.95), (-1.93, 1.93))


@dataclasses.dataclass(frozen=True)
class Box:
    """A rectangle: its centre offset from the pose origin (forward, left), length, width."""

    length: float
    width: float
    centre_offset: float = 0.0
    lateral_offset: float = 0.0

    def corners(self, x: float, y: float, heading: float) -> np.ndarray:
        """Front-left, front-right, rear-right, rear-left."""
        c, s = math.cos(heading), math.sin(heading)
        cx = x + c * self.centre_offset - s * self.lateral_offset
        cy = y + s * self.centre_offset + c * self.lateral_offset
        half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [self.length / 2, self.width / 2]
        return half @ np.array([[c, s], [-s, c]]) + [cx, cy]

    def at(self, x: float, y: float, heading: float) -> Polygon:
        return Polygon(self.corners(x, y, heading))


@dataclasses.dataclass(frozen=True)
class Track:
    """A logged object: its box and its centre pose over time (``[N]`` us, ``[N, 3]`` x, y, yaw)."""

    box: Box
    times_us: np.ndarray
    poses: np.ndarray
    road_user: bool

    def pose(self, time_us: int) -> np.ndarray | None:
        """The pose at ``time_us``, or None outside the track: replayed objects exist only then."""
        if not self.times_us[0] <= time_us <= self.times_us[-1]:
            return None
        return np.array([np.interp(time_us, self.times_us, self.poses[:, 0]),
                         np.interp(time_us, self.times_us, self.poses[:, 1]),
                         np.interp(time_us, self.times_us, np.unwrap(self.poses[:, 2]))])

    def at(self, time_us: int) -> Polygon | None:
        pose = self.pose(time_us)
        return None if pose is None else self.box.at(*pose)

    def speed(self, time_us: int) -> float:
        """Finite-difference speed over 0.1 s, one-sided at the ends of the track."""
        if self.times_us[-1] - self.times_us[0] < 100_000:
            return 0.0
        start = int(np.clip(time_us - 50_000, self.times_us[0], self.times_us[-1] - 100_000))
        before, after = self.pose(start), self.pose(start + 100_000)
        return float(np.linalg.norm(after[:2] - before[:2]) / 0.1)


@dataclasses.dataclass(frozen=True)
class Lane:
    polygon: Polygon
    centreline: np.ndarray  # [N, 2]

    def heading_at(self, xy: np.ndarray) -> float:
        line = self.centreline[np.r_[True, np.linalg.norm(np.diff(self.centreline, axis=0), axis=1) > 1e-6]]
        index = min(int(np.argmin(np.linalg.norm(line - xy, axis=1))), len(line) - 2)
        delta = line[index + 1] - line[index]
        return math.atan2(delta[1], delta[0])

    def distance(self, xy: np.ndarray) -> float:
        return LineString(self.centreline).distance(Point(xy))


def _angle(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class SceneScorer:
    """Scores plans in one scene, given its lanes, road edges, tracks, ego box and recorded drive."""

    def __init__(self, lanes: list[Lane], road_edges: list[np.ndarray], tracks: list[Track], ego: Box,
                 recorded_times_us: np.ndarray, recorded_xy: np.ndarray, recorded_yaw: np.ndarray) -> None:
        if not lanes:
            raise ValueError("the scene map has no lanes to score against")
        self.lanes = lanes
        self.lane_tree = shapely.STRtree([lane.polygon for lane in lanes])
        self.drivable = unary_union([lane.polygon for lane in lanes]).buffer(LANE_UNION_TOLERANCE_M)
        self.road_edges = unary_union([LineString(edge) for edge in road_edges]) if road_edges else None
        self.tracks = tracks
        self.ego = ego
        self.recorded_times_us = recorded_times_us
        self.recorded_xy = recorded_xy
        self.recorded_yaw = np.unwrap(recorded_yaw)
        self.recorded_path = LineString(recorded_xy)
        steps_m = np.linalg.norm(np.diff(recorded_xy, axis=0), axis=1)
        self.recorded_arc = np.r_[0.0, np.cumsum(steps_m)]
        # Time the log spent moving (faster than STOPPED_MPS) up to each sample.
        steps_s = np.diff(recorded_times_us) / 1e6
        moving = steps_m > STOPPED_MPS * np.maximum(steps_s, 1e-6)
        self.recorded_moving_s = np.r_[0.0, np.cumsum(np.where(moving, steps_s, 0.0))]
        # Where the recorded drive itself breaks a rule, for the human filter.
        self.recorded_rules = self._rule_steps(np.column_stack([recorded_xy, self.recorded_yaw]))

    # --- per-step geometry -------------------------------------------------

    def _lanes_at(self, xy: np.ndarray) -> list[Lane]:
        return [self.lanes[i] for i in self.lane_tree.query(Point(xy), predicate="within")]

    def _in_one_lane(self, box: Polygon) -> bool:
        return any(self.lanes[i].polygon.covers(box) for i in self.lane_tree.query(box))

    def _off_road(self, box: Polygon) -> bool:
        if self.drivable.covers(box):
            return False
        if self.road_edges is None:  # no road edges to touch: off when the centre leaves the lanes
            return not self.drivable.covers(box.centroid)
        return self.road_edges.intersects(box)

    def _rule_steps(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per pose of ``[T, 3]``: off the road, oncoming, lane-keeping factor."""
        off_road = np.array([self._off_road(self.ego.at(*pose)) for pose in poses])
        oncoming, keep = np.zeros(len(poses), bool), np.zeros(len(poses))
        for k, pose in enumerate(poses):
            # The box centre, as NAVSIM and AlpaSim's wrong-lane rule use, not the rear axle.
            xy, heading = np.array(self.ego.at(*pose).centroid.coords[0]), float(pose[2])
            lanes = self._lanes_at(xy)
            along = [lane for lane in lanes if _angle(lane.heading_at(xy), heading) <= ONCOMING_HEADING_RAD]
            oncoming[k] = bool(lanes) and not along
            distance = min((lane.distance(xy) for lane in along), default=LANE_KEEP_ZERO_M)
            keep[k] = np.clip((LANE_KEEP_ZERO_M - distance) / (LANE_KEEP_ZERO_M - LANE_KEEP_FULL_M), 0.0, 1.0)
        return off_road, oncoming, keep

    def _recorded_neighbours(self, xy: np.ndarray) -> tuple[int, int]:
        """Indices of the recorded poses on either side of ``xy``'s place along the recorded path."""
        after = int(np.clip(np.searchsorted(self.recorded_arc, self.recorded_path.project(Point(xy))),
                            0, len(self.recorded_arc) - 1))
        return max(after - 1, 0), after

    def _contact(self, ego_pose: np.ndarray, speed: float, track: Track, time_us: int) -> str | None:
        """NAVSIM's collision type of the ego box at ``ego_pose`` with ``track``, or None."""
        other = track.at(time_us)
        ego_corners = self.ego.corners(*ego_pose)
        ego_box = Polygon(ego_corners)
        if other is None or not ego_box.intersects(other):
            return None
        if speed < STOPPED_MPS:
            return "stopped_ego"
        if track.speed(time_us) < STOPPED_MPS:
            return "stopped_track"
        if LineString(ego_corners[[0, 1]]).intersects(other):
            return "front"
        if LineString(ego_corners[[2, 3]]).intersects(other):
            return "rear"
        return "lateral"

    def _at_fault(self, kind: str, ego_pose: np.ndarray) -> bool:
        if kind in ("front", "stopped_track"):
            return True
        if kind == "lateral":
            box = self.ego.at(*ego_pose)
            return not self._in_one_lane(box) or self._off_road(box)
        return False

    def _progress(self, xy: np.ndarray) -> float:
        return float(self.recorded_path.project(Point(xy)))

    # --- the reward -----------------------------------------------------------

    def score(self, plan: np.ndarray, speeds: np.ndarray, controls: np.ndarray, start: np.ndarray,
              start_speed: float, t0_us: int) -> dict[str, float]:
        """Score one plan.

        ``plan [T, 3]`` local x, y, heading at ``t0 + 0.1 s * (1..T)``; ``speeds
        [T]``; ``controls [T, 2]`` commanded acceleration and curvature;
        ``start [3]`` and ``start_speed``: the ego's local pose and speed at ``t0``.
        """
        times = t0_us + np.arange(1, min(HORIZON_STEPS, len(plan)) + 1, dtype=np.int64) * int(DT_S * 1e6)
        times = times[times <= self.recorded_times_us[-1]]
        steps = len(times)
        if steps < 2:
            raise ValueError("fewer than two plan steps lie within the recording")
        plan, speeds, controls = plan[:steps], speeds[:steps], controls[:steps]
        recorded = np.column_stack([np.interp(times, self.recorded_times_us, column) for column in
                                    (*self.recorded_xy.T, self.recorded_yaw)])

        # --- NC and TTC, against the tracks that come near the plan ---
        start_box = self.ego.at(*start)
        near = []
        for track in self.tracks:
            positions = [track.pose(int(t)) for t in times]
            gaps = [np.linalg.norm(p[:2] - q[:2]) for p, q in zip(positions, plan) if p is not None]
            if gaps and min(gaps) < TRACK_REACH_M:
                near.append(track)
        ignored = {id(t) for t in near if (b := t.at(t0_us)) is not None and start_box.intersects(b)}
        nc, ttc = 1.0, 1.0
        first_violation = [steps]  # first step at which a gate becomes 0
        for k in range(steps):
            for track in near:
                if id(track) in ignored:
                    continue
                kind = self._contact(plan[k], float(speeds[k]), track, int(times[k]))
                if kind is None:
                    continue
                if not self._at_fault(kind, plan[k]):
                    ignored.add(id(track))  # NAVSIM: later contacts with it are not counted
                    continue
                value = 0.0 if track.road_user else 0.5
                if value == 0.0 and nc > 0.0:
                    first_violation.append(k)
                nc = min(nc, value)
            heading = float(plan[k, 2])
            for ahead in TTC_LOOKAHEAD_S:
                moved = np.array([*(plan[k, :2] + speeds[k] * ahead * np.array([math.cos(heading), math.sin(heading)])),
                                  heading])
                for track in near:
                    if id(track) in ignored:
                        continue
                    kind = self._contact(moved, float(speeds[k]), track, int(times[k] + ahead * 1e6))
                    if kind is not None and self._at_fault(kind, moved):
                        ttc = 0.5

        # --- DAC, DDC, LK with the human filter by place ---
        off_road, oncoming, keep = self._rule_steps(plan)
        # The recording breaks a rule here if it does at either recorded pose around this place.
        before, after = np.array([self._recorded_neighbours(p) for p in plan[:, :2]]).T
        rec_off_road, rec_oncoming, rec_keep = self.recorded_rules
        off_road_rec = rec_off_road[before] | rec_off_road[after]
        oncoming_rec = rec_oncoming[before] | rec_oncoming[after]
        keep_rec = np.minimum(rec_keep[before], rec_keep[after])
        off_road &= ~off_road_rec
        dac = float(not off_road.any())
        steps_m = np.linalg.norm(np.diff(np.vstack([start[None, :2], plan[:, :2]]), axis=0), axis=1)
        oncoming_m = steps_m * (oncoming & ~oncoming_rec)
        window_m = np.array([oncoming_m[max(0, k - 9):k + 1].sum() for k in range(steps)])
        worst = window_m.max()
        ddc = 1.0 if worst <= 2.0 else 0.5 if worst <= 6.0 else 0.0
        lk = float(np.mean(np.where(keep_rec < 1.0, 1.0, keep)))
        first_violation += [int(np.argmax(off_road))] if off_road.any() else []
        first_violation += [int(np.argmax(window_m > 6.0))] if worst > 6.0 else []

        # --- V and EP ---
        distances = np.array([self.recorded_path.distance(Point(p)) for p in plan[:, :2]])
        valid = float(np.clip((VALID_ZERO_M - distances.max()) / (VALID_ZERO_M - VALID_FULL_M), 0.0, 1.0))
        if valid == 0.0:
            first_violation.append(int(np.argmax(distances >= VALID_ZERO_M)))
        # The log's gain from the ego's place over the plan's horizon of the log's
        # moving time: where the log waited (a light, a queue) is not counted, or a
        # plan that waits there for good would match it.
        ego_arc = self._progress(start[:2])
        moving_s = np.interp(ego_arc, self.recorded_arc, self.recorded_moving_s)
        target_s = moving_s + (times[-1] - t0_us) / 1e6
        recorded_gain = float(np.interp(target_s, self.recorded_moving_s, self.recorded_arc)) - ego_arc
        plan_gain = self._progress(plan[-1, :2]) - ego_arc
        ep = 1.0 if recorded_gain < MIN_RECORDED_PROGRESS_M else float(np.clip(plan_gain / recorded_gain, 0.0, 1.0))

        # --- C, from realised motion including the step out of t0 ---
        speed_path = np.r_[start_speed, speeds]
        curvature = np.r_[controls[0, 1], controls[:, 1]]
        accel = np.gradient(speed_path, DT_S)
        lateral, yaw_rate = speed_path**2 * curvature, speed_path * curvature
        lon_jerk, lat_jerk = np.gradient(accel, DT_S), np.gradient(lateral, DT_S)
        signals = (accel, lateral, np.hypot(lon_jerk, lat_jerk), lon_jerk, yaw_rate, np.gradient(yaw_rate, DT_S))
        violated = sum(bool(((x < lo) | (x > hi)).any()) for x, (lo, hi) in zip(signals, COMFORT_BOUNDS))
        comfort = 1.0 - 0.5 * violated / 6.0

        terms = {"nc": nc, "dac": dac, "ddc": ddc, "valid": valid, "ep": ep, "ttc": ttc, "lk": lk,
                 "comfort": comfort}
        terms["reward"] = float(np.prod(list(terms.values())))
        terms["gated"] = float(min(nc, dac, ddc, valid) == 0.0)
        terms["safe_fraction"] = min(first_violation) / steps
        return terms

    def path_distance(self, xy: np.ndarray) -> float:
        """Distance from ``xy`` to the recorded path, the renderer's validity measure."""
        return float(self.recorded_path.distance(Point(xy)))


MIN_TRAFFIC_DURATION_US = 3_000_000
"""AlpaSim's ``simulation_config.min_traffic_duration_us``: shorter tracks are not simulated."""


def scene_scorer(artifact: Any, recorded: Any, end_us: int) -> SceneScorer:
    """Build the scorer from an AlpaSim ``Artifact`` and the rollout's recorded ego ``Trajectory``.

    The objects are those the rollout simulated: AlpaSim clips the log's tracks
    to the rollout, from the rig trajectory's start to ``end_us``, drops those
    shorter than 3 s and hides them from the renders (``unbound_rollout.py``).
    """
    from trajdata.maps.vec_map_elements import MapElementType

    elements = artifact.map.elements
    lanes = []
    for lane in elements[MapElementType.ROAD_LANE].values():
        if lane.left_edge is not None and lane.right_edge is not None:
            polygon = Polygon(np.concatenate([lane.left_edge.xy, lane.right_edge.xy[::-1]]))
        else:
            polygon = LineString(lane.center.xy).buffer(LANE_WIDTH_WITHOUT_EDGES_M / 2)
        lanes.append(Lane(shapely.make_valid(polygon), lane.center.xy))
    road_edges = [edge.polyline.xy for edge in elements.get(MapElementType.ROAD_EDGE, {}).values()]
    tracks = []
    simulated = artifact.traffic_objects.clip_trajectories(
        artifact.rig.trajectory.time_range_us.start, end_us + 1, exclude_empty=True
    ).filter_short_trajectories(MIN_TRAFFIC_DURATION_US)
    for obj in simulated.values():
        times, xy, yaw = _poses(obj.trajectory)
        tracks.append(Track(Box(obj.aabb.x, obj.aabb.y), times, np.column_stack([xy, yaw]),
                            obj.label_class not in STATIC_OBSTACLE_CLASSES))
    vehicle = artifact.rig.vehicle_config
    if vehicle is None:
        raise ValueError("the scene's rig has no vehicle config, so the ego box is unknown")
    # aabb_x_offset_m is the rig origin to the box's rear face (alpasim_utils.scenario).
    ego = Box(vehicle.aabb_x_m, vehicle.aabb_y_m, vehicle.aabb_x_offset_m + vehicle.aabb_x_m / 2,
              vehicle.aabb_y_offset_m)
    return SceneScorer(lanes, road_edges, tracks, ego, *_poses(recorded))


def _poses(trajectory: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps, XY and yaw of an AlpaSim ``Trajectory``."""
    times = np.asarray(trajectory.timestamps_us, dtype=np.int64)
    yaw = np.array([trajectory.interpolate_pose(int(t)).yaw() for t in times])
    return times, np.asarray(trajectory.positions, dtype=np.float64)[:, :2], yaw


__all__ = ["Box", "Lane", "SceneScorer", "Track", "scene_scorer"]
