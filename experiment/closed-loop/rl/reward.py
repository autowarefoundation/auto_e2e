"""Driving reward for one sampled plan: progress gated by safety, as in CaRL and EPDMS.

    R = NC * DAC * DDC * V * EP * TTC * LK * C

scored over the first 4 s (40 steps) of the plan, in the scene's local frame:

* NC (no at-fault collision): the ego box swept along the plan against the
  log's tracked boxes at the same times. 0 for a moving object, 0.5 for a
  static one. Not at fault, as in NAVSIM: the ego is stopped (< 0.05 m/s),
  the object already overlapped the ego at t0, or it hits the ego from behind.
* DAC (drivable area): every corner of the ego box stays inside the union of
  lane polygons. NuRec maps have no road-area polygons, so lanes are the
  drivable area, as in AlpaSim's offroad metric.
* DDC (driving direction): metres driven where every lane under the ego
  points against the plan (> 120 deg, AlpaSim's wrong-lane rule), per 1 s
  window; <= 2 m keeps 1, <= 6 m gives 0.5, more gives 0 (EPDMS).
* V (renderer validity, #123): the plan's largest distance to the recorded
  path; 1 up to 2.5 m, falling to 0 at 3.0 m. A path distance, so braking or
  falling behind the log is not penalised (RAD's position deviation).
* EP (ego progress): arc length gained along the recorded path over the
  recording's own gain in the same 4 s, clipped to [0, 1]; 1 when the
  recording moves less than 5 m (NAVSIM). Multiplying by EP gives a plan that
  stops 0, where PDMS's weighted average would still give it 9/14 (#123).
* TTC: 0.5 when the ego box, projected at constant velocity 0.3, 0.6 and
  0.9 s ahead of a plan step, would hit an object under the same at-fault
  rules; else 1 (CaRL).
* LK (lane keeping): mean over the steps of 1 within 0.5 m of the nearest
  lane centreline in the plan's direction, falling linearly to 0 at 1.75 m.
* C (comfort): 1 - 0.5 n / 6, with n the nuPlan comfort bounds the plan
  violates (CaRL).

DAC, DDC and LK keep 1 where the recorded drive breaks the same rule over the
same window (EPDMS's human filter): the lane polygons under-cover real roads.

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
# nuPlan comfort bounds (min, max): lon. accel, lat. accel, jerk, lon. jerk, yaw rate, yaw accel.
COMFORT_BOUNDS = ((-4.05, 2.40), (-4.89, 4.89), (-8.37, 8.37), (-4.13, 4.13), (-0.95, 0.95), (-1.93, 1.93))


@dataclasses.dataclass(frozen=True)
class Box:
    """A rectangle: centre offset from the pose origin along its heading, length, width."""

    length: float
    width: float
    centre_offset: float = 0.0

    def at(self, x: float, y: float, heading: float) -> Polygon:
        c, s = math.cos(heading), math.sin(heading)
        cx, cy = x + c * self.centre_offset, y + s * self.centre_offset
        half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [self.length / 2, self.width / 2]
        corners = half @ np.array([[c, s], [-s, c]]) + [cx, cy]
        return Polygon(corners)


@dataclasses.dataclass(frozen=True)
class Track:
    """A logged object: its box and its centre pose over time (``[N]`` us, ``[N, 3]`` x, y, yaw)."""

    box: Box
    times_us: np.ndarray
    poses: np.ndarray
    is_static: bool

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

    def _hit(self, ego_box: Polygon, xy: np.ndarray, heading: float, time_us: int,
             ignored: set[int]) -> float:
        """1 when clear, 0.5 for a static object, 0 for a moving one; at-fault hits only.

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
            if float((centre - xy) @ forward) < self.ego.centre_offset - self.ego.length / 4:
                # It reaches the ego from behind: not at fault. A log-replayed
                # follower then drives through the ego, so ignore it from here on.
                ignored.add(index)
                continue
            worst = min(worst, 0.5 if track.is_static else 0.0)
        return worst

    def _progress(self, xy: np.ndarray) -> float:
        return float(self.recorded_path.project(shapely.Point(xy)))

    def _lane_flags(self, xy: np.ndarray, heading: float) -> tuple[bool, float]:
        """Whether every lane under ``xy`` is oncoming, and the lane-keeping factor."""
        lanes = self._lanes_at(xy)
        along = [lane for lane in lanes if _angle(lane.heading_at(xy), heading) <= ONCOMING_HEADING_RAD]
        distance = min((lane.distance(xy) for lane in along), default=LANE_KEEP_ZERO_M)
        keep = (LANE_KEEP_ZERO_M - distance) / (LANE_KEEP_ZERO_M - LANE_KEEP_FULL_M)
        return bool(lanes) and not along, float(np.clip(keep, 0.0, 1.0))

    def _rule_terms(self, poses: np.ndarray) -> tuple[bool, float, float]:
        """Drivable area kept, driving-direction factor and mean lane keeping of ``[T, 3]`` poses."""
        drivable = all(self.drivable.covers(self.ego.at(*pose)) for pose in poses)
        flags = [self._lane_flags(pose[:2], float(pose[2])) for pose in poses]
        steps_m = np.r_[0.0, np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)]
        oncoming_m = steps_m * np.array([oncoming for oncoming, _ in flags])
        worst = max(oncoming_m[i:i + 10].sum() for i in range(max(1, len(poses) - 9)))
        direction = 1.0 if worst <= 2.0 else 0.5 if worst <= 6.0 else 0.0
        return drivable, direction, float(np.mean([keep for _, keep in flags]))

    # --- the reward -----------------------------------------------------------

    def score(self, plan: np.ndarray, speeds: np.ndarray, controls: np.ndarray, t0_us: int) -> dict[str, float]:
        """``plan [T, 3]`` local x, y, heading; ``speeds [T]``; ``controls [T, 2]`` accel, curvature."""
        times = np.array([t0_us + int((k + 1) * DT_S * 1e6) for k in range(min(HORIZON_STEPS, len(plan)))])
        times = times[times <= self.recorded_times_us[-1]]
        if not len(times):
            raise ValueError("the plan starts after the end of the recording")
        steps = len(times)
        plan, speeds, controls = plan[:steps], speeds[:steps], controls[:steps]
        recorded = np.column_stack([np.interp(times, self.recorded_times_us, column) for column in
                                    (*self.recorded_xy.T, self.recorded_yaw)])
        start_xy = np.array([np.interp(t0_us, self.recorded_times_us, column) for column in self.recorded_xy.T])

        start_box = self.ego.at(*plan[0])
        ignored = {i for i, track in enumerate(self.tracks)
                   if (other := track.at(t0_us)) is not None and start_box.intersects(other)}
        nc, ttc = 1.0, 1.0
        for k in range(steps):
            if speeds[k] < STOPPED_MPS:
                continue
            xy, heading = plan[k, :2], float(plan[k, 2])
            nc = min(nc, self._hit(self.ego.at(*plan[k]), xy, heading, int(times[k]), ignored))
            if k % 5:
                continue
            for ahead in TTC_LOOKAHEAD_S:
                moved = xy + speeds[k] * ahead * np.array([math.cos(heading), math.sin(heading)])
                # A copy: a projection that overshoots an object must not excuse it.
                if self._hit(self.ego.at(*moved, heading), moved, heading, int(times[k] + ahead * 1e6), set(ignored)) < 1.0:
                    ttc = 0.5

        # EPDMS's human filter: a rule the recorded drive breaks in the same window does not count.
        drivable, direction, lane_keep = self._rule_terms(plan)
        drivable_rec, direction_rec, lane_keep_rec = self._rule_terms(recorded)
        dac = 1.0 if drivable or not drivable_rec else 0.0
        ddc = 1.0 if direction_rec < 1.0 else direction
        lk = 1.0 if lane_keep_rec < 1.0 else lane_keep

        deviation = max(self.recorded_path.distance(shapely.Point(p)) for p in plan[:, :2])
        valid = float(np.clip((VALID_ZERO_M - deviation) / (VALID_ZERO_M - VALID_FULL_M), 0.0, 1.0))

        recorded_gain = self._progress(recorded[-1, :2]) - self._progress(start_xy)
        plan_gain = self._progress(plan[-1, :2]) - self._progress(start_xy)
        ep = 1.0 if recorded_gain < MIN_RECORDED_PROGRESS_M else float(np.clip(plan_gain / recorded_gain, 0.0, 1.0))

        accel, curvature = controls[:, 0], controls[:, 1]
        lateral, yaw_rate = speeds**2 * curvature, speeds * curvature
        signals = (accel, lateral, np.gradient(np.hypot(accel, lateral), DT_S), np.gradient(accel, DT_S),
                   yaw_rate, np.gradient(yaw_rate, DT_S))
        violated = sum(bool(((x < lo) | (x > hi)).any()) for x, (lo, hi) in zip(signals, COMFORT_BOUNDS))
        comfort = 1.0 - 0.5 * violated / 6.0

        terms = {"nc": nc, "dac": dac, "ddc": ddc, "valid": valid, "ep": ep, "ttc": ttc, "lk": lk,
                 "comfort": comfort}
        terms["reward"] = float(np.prod(list(terms.values())))
        terms["gated"] = float(min(nc, dac, ddc, valid) == 0.0)
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
        tracks.append(Track(Box(obj.aabb.x, obj.aabb.y), times, np.column_stack([xy, yaw]), bool(obj.is_static)))
    vehicle = artifact.rig.vehicle_config
    # aabb_x_offset_m is the rig origin to the box's rear face.
    ego = Box(vehicle.aabb_x_m, vehicle.aabb_y_m, vehicle.aabb_x_offset_m + vehicle.aabb_x_m / 2)
    return SceneScorer(lanes, tracks, ego, *_poses(recorded))


def _poses(trajectory: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps, XY and yaw of an AlpaSim ``Trajectory``."""
    times = np.asarray(trajectory.timestamps_us, dtype=np.int64)
    yaw = np.array([trajectory.interpolate_pose(int(t)).yaw() for t in times])
    return times, np.asarray(trajectory.positions, dtype=np.float64)[:, :2], yaw


__all__ = ["Box", "Lane", "SceneScorer", "Track", "scene_scorer"]
