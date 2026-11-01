"""Driving reward for one sampled plan: progress gated by safety, as in CaRL and EPDMS.

    R = NC * DAC * DDC * V * EP * TTC * LK * C

scored at the plan's 0.1 s steps over its first 4 s, in the scene's local frame:

* NC (no at-fault collision): the ego box at each step against the log's
  tracked boxes at the same time. 0 for a road user (vehicle, person, rider,
  animal), 0.5 for a static obstacle (NAVSIM decides by object type, not by
  whether it moves: parked cars are road users). Not at fault, as in NAVSIM:
  the ego is stopped (< 0.05 m/s); the object already overlapped the ego at
  t0; or it reaches the ego from behind while the ego stays within one lane.
* DAC (drivable area): every corner of the ego box stays inside the union of
  lane polygons. NuRec maps have no road-area polygons, so lanes are the
  drivable area, as in AlpaSim's offroad metric.
* DDC (driving direction): metres driven where every lane under the ego
  points against the plan (> 120 deg, AlpaSim's wrong-lane rule), per 1 s
  window; <= 2 m keeps 1, <= 6 m gives 0.5, more gives 0 (EPDMS).
* V (renderer validity, #123): the plan's largest distance to the recorded
  path; 1 up to 2.5 m, falling to 0 at 3.0 m. A path distance, so braking or
  falling behind the log is not penalised (RAD's position deviation).
* EP (ego progress): arc length the plan gains along the recorded path from
  the ego's own position, over the recording's gain in the same 4 s, clipped
  to [0, 1]; 1 when the recording moves less than 5 m (NAVSIM). Multiplying by
  EP gives a plan that stops 0, where PDMS's weighted average would still give
  it 9/14 (#123).
* TTC: 0.5 when the ego box, projected at constant velocity 0.3, 0.6 and
  0.9 s ahead of a plan step, would hit an object under the same at-fault
  rules; else 1 (CaRL).
* LK (lane keeping): mean over the steps of 1 within 0.5 m of the nearest
  lane centreline in the plan's direction, falling linearly to 0 at 1.75 m.
* C (comfort): 1 - 0.5 n / 6, with n the nuPlan comfort bounds the plan
  violates (CaRL), from the plan's realised speed and curvature.

DAC, DDC and LK forgive a step where the recorded drive breaks the same rule
at the same step (EPDMS's human filter): the lane polygons under-cover real
roads, and the log is the reference driver.

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
from shapely.geometry import Polygon
from shapely.ops import unary_union

DT_S = 0.1
HORIZON_STEPS = 40
STOPPED_MPS = 0.05
TTC_LOOKAHEAD_S = (0.3, 0.6, 0.9)
LANE_KEEP_FULL_M, LANE_KEEP_ZERO_M = 0.5, 1.75
VALID_FULL_M, VALID_ZERO_M = 2.5, 3.0
ONCOMING_HEADING_RAD = 2.0 * math.pi / 3.0
MIN_RECORDED_PROGRESS_M = 5.0
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

    def at(self, x: float, y: float, heading: float) -> Polygon:
        c, s = math.cos(heading), math.sin(heading)
        cx = x + c * self.centre_offset - s * self.lateral_offset
        cy = y + s * self.centre_offset + c * self.lateral_offset
        half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [self.length / 2, self.width / 2]
        corners = half @ np.array([[c, s], [-s, c]]) + [cx, cy]
        return Polygon(corners)


@dataclasses.dataclass(frozen=True)
class Track:
    """A logged object: its box and its centre pose over time (``[N]`` us, ``[N, 3]`` x, y, yaw)."""

    box: Box
    times_us: np.ndarray
    poses: np.ndarray
    road_user: bool

    def at(self, time_us: int) -> Polygon | None:
        if not self.times_us[0] <= time_us <= self.times_us[-1]:
            return None
        x = np.interp(time_us, self.times_us, self.poses[:, 0])
        y = np.interp(time_us, self.times_us, self.poses[:, 1])
        yaw = np.interp(time_us, self.times_us, np.unwrap(self.poses[:, 2]))
        return self.box.at(float(x), float(y), float(yaw))


@dataclasses.dataclass(frozen=True)
class Lane:
    polygon: Polygon
    centreline: np.ndarray  # [N, 2]

    def heading_at(self, xy: np.ndarray) -> float:
        index = int(np.argmin(np.linalg.norm(self.centreline - xy, axis=1)))
        index = min(index, len(self.centreline) - 2)
        delta = self.centreline[index + 1] - self.centreline[index]
        return math.atan2(delta[1], delta[0])

    def distance(self, xy: np.ndarray) -> float:
        return shapely.LineString(self.centreline).distance(shapely.Point(xy))


def _angle(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class SceneScorer:
    """Scores plans in one scene, given its lanes, logged tracks, ego box and recorded drive."""

    def __init__(self, lanes: list[Lane], tracks: list[Track], ego: Box, recorded_times_us: np.ndarray,
                 recorded_xy: np.ndarray, recorded_yaw: np.ndarray) -> None:
        if not lanes:
            raise ValueError("the scene map has no lanes with edges to score against")
        self.lanes = lanes
        self.lane_tree = shapely.STRtree([lane.polygon for lane in lanes])
        self.drivable = unary_union([lane.polygon for lane in lanes])
        self.tracks = tracks
        self.ego = ego
        self.recorded_times_us = recorded_times_us
        self.recorded_xy = recorded_xy
        self.recorded_yaw = np.unwrap(recorded_yaw)
        self.recorded_path = shapely.LineString(recorded_xy)

    # --- per-step geometry -------------------------------------------------

    def _lanes_at(self, xy: np.ndarray) -> list[Lane]:
        return [self.lanes[i] for i in self.lane_tree.query(shapely.Point(xy), predicate="within")]

    def _in_one_lane(self, box: Polygon) -> bool:
        return any(self.lanes[i].polygon.covers(box) for i in self.lane_tree.query(box))

    def _hit(self, ego_box: Polygon, xy: np.ndarray, heading: float, time_us: int,
             ignored: set[int]) -> float:
        """1 when clear, 0.5 for a static obstacle, 0 for a road user; at-fault hits only.

        Adds objects that hit the ego from behind to ``ignored``.
        """
        worst = 1.0
        forward = np.array([math.cos(heading), math.sin(heading)])
        for index, track in enumerate(self.tracks):
            if index in ignored:
                continue
            other = track.at(time_us)
            if other is None or not ego_box.intersects(other):
                continue
            centre = np.array(other.centroid.coords[0])
            behind = float((centre - xy) @ forward) < self.ego.centre_offset - self.ego.length / 4
            if behind and self._in_one_lane(ego_box):
                # Reached from behind while the ego keeps its lane: not at fault. A
                # log-replayed follower then drives through the ego, so ignore it.
                ignored.add(index)
                continue
            worst = min(worst, 0.0 if track.road_user else 0.5)
        return worst

    def _progress(self, xy: np.ndarray) -> float:
        return float(self.recorded_path.project(shapely.Point(xy)))

    def _rule_steps(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per step of ``[T, 3]`` poses: on the drivable area, oncoming, lane-keeping factor."""
        drivable = np.array([self.drivable.covers(self.ego.at(*pose)) for pose in poses])
        oncoming, keep = np.zeros(len(poses), bool), np.zeros(len(poses))
        for k, pose in enumerate(poses):
            xy, heading = pose[:2], float(pose[2])
            lanes = self._lanes_at(xy)
            along = [lane for lane in lanes if _angle(lane.heading_at(xy), heading) <= ONCOMING_HEADING_RAD]
            oncoming[k] = bool(lanes) and not along
            distance = min((lane.distance(xy) for lane in along), default=LANE_KEEP_ZERO_M)
            keep[k] = np.clip((LANE_KEEP_ZERO_M - distance) / (LANE_KEEP_ZERO_M - LANE_KEEP_FULL_M), 0.0, 1.0)
        return drivable, oncoming, keep

    # --- the reward -----------------------------------------------------------

    def score(self, plan: np.ndarray, speeds: np.ndarray, controls: np.ndarray, start: np.ndarray,
              t0_us: int) -> dict[str, float]:
        """Score one plan.

        ``plan [T, 3]`` local x, y, heading at ``t0 + 0.1 s * (1..T)``; ``speeds
        [T]``; ``controls [T, 2]`` commanded acceleration and curvature;
        ``start [3]`` the ego's local pose at ``t0``.
        """
        times = t0_us + np.arange(1, min(HORIZON_STEPS, len(plan)) + 1, dtype=np.int64) * int(DT_S * 1e6)
        times = times[times <= self.recorded_times_us[-1]]
        steps = len(times)
        if steps < 2:
            raise ValueError("fewer than two plan steps lie within the recording")
        plan, speeds, controls = plan[:steps], speeds[:steps], controls[:steps]
        recorded = np.column_stack([np.interp(times, self.recorded_times_us, column) for column in
                                    (*self.recorded_xy.T, self.recorded_yaw)])

        start_box = self.ego.at(*start)
        ignored = {i for i, track in enumerate(self.tracks)
                   if (other := track.at(t0_us)) is not None and start_box.intersects(other)}
        nc, ttc = 1.0, 1.0
        first_violation = [steps]  # first step at which a gate becomes 0
        for k in range(steps):
            if speeds[k] < STOPPED_MPS:
                continue
            xy, heading = plan[k, :2], float(plan[k, 2])
            hit = self._hit(self.ego.at(*plan[k]), xy, heading, int(times[k]), ignored)
            if hit == 0.0 and nc > 0.0:
                first_violation.append(k)
            nc = min(nc, hit)
            if k % 5:
                continue
            for ahead in TTC_LOOKAHEAD_S:
                moved = xy + speeds[k] * ahead * np.array([math.cos(heading), math.sin(heading)])
                # A copy: a projection that overshoots an object must not excuse it.
                if self._hit(self.ego.at(*moved, heading), moved, heading, int(times[k] + ahead * 1e6), set(ignored)) < 1.0:
                    ttc = 0.5

        # EPDMS's human filter, step by step: what the recorded drive does at a step is allowed.
        drivable, oncoming, keep = self._rule_steps(plan)
        drivable_rec, oncoming_rec, keep_rec = self._rule_steps(recorded)
        off_road = ~drivable & drivable_rec
        dac = float(not off_road.any())
        steps_m = np.linalg.norm(np.diff(np.vstack([start[None, :2], plan[:, :2]]), axis=0), axis=1)
        oncoming_m = steps_m * (oncoming & ~oncoming_rec)
        # Oncoming metres in the 1 s window ending at each step.
        window_m = np.array([oncoming_m[max(0, k - 9):k + 1].sum() for k in range(steps)])
        worst = window_m.max()
        ddc = 1.0 if worst <= 2.0 else 0.5 if worst <= 6.0 else 0.0
        first_violation += [int(np.argmax(off_road))] if off_road.any() else []
        first_violation += [int(np.argmax(window_m > 6.0))] if worst > 6.0 else []
        lk = float(np.mean(np.where(keep_rec < 1.0, np.maximum(keep, keep_rec), keep)))

        distances = np.array([self.recorded_path.distance(shapely.Point(p)) for p in plan[:, :2]])
        valid = float(np.clip((VALID_ZERO_M - distances.max()) / (VALID_ZERO_M - VALID_FULL_M), 0.0, 1.0))
        if valid == 0.0:
            first_violation.append(int(np.argmax(distances >= VALID_ZERO_M)))

        recorded_start = np.array([np.interp(t0_us, self.recorded_times_us, c) for c in self.recorded_xy.T])
        recorded_gain = self._progress(recorded[-1, :2]) - self._progress(recorded_start)
        plan_gain = self._progress(plan[-1, :2]) - self._progress(start[:2])
        ep = 1.0 if recorded_gain < MIN_RECORDED_PROGRESS_M else float(np.clip(plan_gain / recorded_gain, 0.0, 1.0))

        # Realised motion: the integrator clamps speed at 0, so commanded braking of a stopped plan is not felt.
        curvature = controls[:, 1]
        accel = np.gradient(speeds, DT_S)
        lateral, yaw_rate = speeds**2 * curvature, speeds * curvature
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


def scene_scorer(artifact: Any, recorded: Any) -> SceneScorer:
    """Build the scorer from an AlpaSim ``Artifact`` and the rollout's recorded ego ``Trajectory``."""
    from trajdata.maps.vec_map_elements import MapElementType

    lanes = []
    for lane in artifact.map.elements[MapElementType.ROAD_LANE].values():
        if lane.left_edge is None or lane.right_edge is None:
            continue
        polygon = shapely.make_valid(Polygon(np.concatenate([lane.left_edge.xy, lane.right_edge.xy[::-1]])))
        lanes.append(Lane(polygon, lane.center.xy))
    tracks = []
    for obj in artifact.traffic_objects.values():
        times, xy, yaw = _poses(obj.trajectory)
        tracks.append(Track(Box(obj.aabb.x, obj.aabb.y), times, np.column_stack([xy, yaw]),
                            obj.label_class not in STATIC_OBSTACLE_CLASSES))
    vehicle = artifact.rig.vehicle_config
    if vehicle is None:
        raise ValueError("the scene's rig has no vehicle config, so the ego box is unknown")
    # aabb_x_offset_m is the rig origin to the box's rear face (alpasim_utils.scenario).
    ego = Box(vehicle.aabb_x_m, vehicle.aabb_y_m, vehicle.aabb_x_offset_m + vehicle.aabb_x_m / 2,
              vehicle.aabb_y_offset_m)
    return SceneScorer(lanes, tracks, ego, *_poses(recorded))


def _poses(trajectory: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps, XY and yaw of an AlpaSim ``Trajectory``."""
    times = np.asarray(trajectory.timestamps_us, dtype=np.int64)
    yaw = np.array([trajectory.interpolate_pose(int(t)).yaw() for t in times])
    return times, np.asarray(trajectory.positions, dtype=np.float64)[:, :2], yaw


__all__ = ["Box", "Lane", "SceneScorer", "Track", "scene_scorer"]
