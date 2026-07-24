"""Scene-level Lanelet2 trace matching without future-trajectory rasterization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

import numpy as np

from .contracts import (
    Destination,
    Maneuver,
    NavigationMap,
    NavigationRoute,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
    TransitionType,
)
from .lanelet2_adapter import _attr, _level, _points


LANELET2_MATCHER_VERSION = "lanelet2_trace_matcher_v1"


@dataclasses.dataclass(frozen=True)
class Lanelet2MatcherConfig:
    candidate_radius_m: float = 8.0
    max_candidates_per_pose: int = 8
    distance_sigma_m: float = 2.0
    heading_sigma_rad: float = math.radians(20.0)
    same_lane_cost: float = 0.0
    following_cost: float = 0.25
    adjacent_cost: float = 1.0
    disconnected_cost: float = 25.0
    skipped_pose_cost: float = 0.5
    minimum_matched_pose_ratio: float = 0.80
    maximum_p95_distance_m: float = 5.0
    maximum_p95_heading_error_rad: float = math.radians(45.0)

    def __post_init__(self) -> None:
        positive = (
            self.candidate_radius_m,
            self.distance_sigma_m,
            self.heading_sigma_rad,
            self.maximum_p95_distance_m,
            self.maximum_p95_heading_error_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("matcher metric parameters must be positive")
        if self.max_candidates_per_pose <= 0:
            raise ValueError("max_candidates_per_pose must be positive")
        if not 0.0 <= self.minimum_matched_pose_ratio <= 1.0:
            raise ValueError("minimum_matched_pose_ratio must be in [0,1]")

    def sha256(self) -> str:
        payload = json.dumps(
            dataclasses.asdict(self),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class _Candidate:
    lanelet: Any
    distance_m: float
    heading_error_rad: float
    emission_cost: float

    @property
    def lanelet_id(self) -> int:
        return int(self.lanelet.id)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _nearest_polyline(
    point_xy: np.ndarray,
    polyline_xy: np.ndarray,
) -> tuple[float, float]:
    """Return distance and tangent yaw at the closest polyline segment."""
    if len(polyline_xy) < 2:
        return math.inf, 0.0
    best_distance_sq = math.inf
    best_yaw = 0.0
    for start, end in zip(polyline_xy[:-1], polyline_xy[1:]):
        delta = end[:2] - start[:2]
        length_sq = float(np.dot(delta, delta))
        if length_sq <= 1e-12:
            continue
        fraction = float(np.dot(point_xy - start[:2], delta) / length_sq)
        fraction = min(1.0, max(0.0, fraction))
        nearest = start[:2] + fraction * delta
        distance_sq = float(np.dot(point_xy - nearest, point_xy - nearest))
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        if (
            distance_sq < best_distance_sq
            or (
                math.isclose(distance_sq, best_distance_sq, abs_tol=1e-12)
                and yaw < best_yaw
            )
        ):
            best_distance_sq = distance_sq
            best_yaw = yaw
    return math.sqrt(best_distance_sq), best_yaw


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


class Lanelet2TraceMatcher:
    """Match one complete scene pose trace to a deterministic lanelet sequence."""

    def __init__(
        self,
        scene_map: Any,
        navigation_map: NavigationMap,
        *,
        map_sha256: str,
        source_revision: str,
        config: Lanelet2MatcherConfig | None = None,
    ) -> None:
        self.scene_map = scene_map
        self.navigation_map = navigation_map
        self.map_sha256 = map_sha256
        self.source_revision = source_revision
        self.config = config or Lanelet2MatcherConfig()
        self._following_cache: dict[int, tuple[Any, ...]] = {}
        self._previous_cache: dict[int, tuple[Any, ...]] = {}
        self._adjacent_cache: dict[int, tuple[Any | None, Any | None]] = {}

    def _can_pass(self, lanelet: Any) -> bool:
        try:
            return bool(self.scene_map.traffic_rules.canPass(lanelet))
        except Exception:
            return True

    def _following(self, lanelet: Any) -> tuple[Any, ...]:
        lanelet_id = int(lanelet.id)
        if lanelet_id not in self._following_cache:
            try:
                values = tuple(self.scene_map.routing_graph.following(lanelet))
            except Exception:
                values = ()
            self._following_cache[lanelet_id] = values
        return self._following_cache[lanelet_id]

    def _previous(self, lanelet: Any) -> tuple[Any, ...]:
        lanelet_id = int(lanelet.id)
        if lanelet_id not in self._previous_cache:
            try:
                values = tuple(self.scene_map.routing_graph.previous(lanelet))
            except Exception:
                values = ()
            self._previous_cache[lanelet_id] = values
        return self._previous_cache[lanelet_id]

    def _adjacent(self, lanelet: Any) -> tuple[Any | None, Any | None]:
        lanelet_id = int(lanelet.id)
        if lanelet_id not in self._adjacent_cache:
            try:
                left = self.scene_map.routing_graph.left(lanelet)
            except Exception:
                left = None
            try:
                right = self.scene_map.routing_graph.right(lanelet)
            except Exception:
                right = None
            self._adjacent_cache[lanelet_id] = (left, right)
        return self._adjacent_cache[lanelet_id]

    def _candidates(
        self,
        position_xy: np.ndarray,
        yaw_rad: float,
    ) -> list[_Candidate]:
        try:
            lanelets = self.scene_map.get_lanelets_in_roi(
                center=position_xy,
                radius=self.config.candidate_radius_m,
            )
        except Exception:
            lanelets = ()
        candidates: list[_Candidate] = []
        for lanelet in lanelets:
            if not self._can_pass(lanelet):
                continue
            centerline = _points(lanelet.centerline)
            distance, lane_yaw = _nearest_polyline(position_xy, centerline)
            heading_error = abs(_wrap_angle(yaw_rad - lane_yaw))
            if not math.isfinite(distance):
                continue
            emission = 0.5 * (
                distance / self.config.distance_sigma_m
            ) ** 2 + 0.5 * (
                heading_error / self.config.heading_sigma_rad
            ) ** 2
            candidates.append(
                _Candidate(
                    lanelet=lanelet,
                    distance_m=distance,
                    heading_error_rad=heading_error,
                    emission_cost=emission,
                )
            )
        candidates.sort(
            key=lambda candidate: (
                candidate.emission_cost,
                candidate.lanelet_id,
            )
        )
        return candidates[: self.config.max_candidates_per_pose]

    def _transition(
        self,
        previous: Any,
        current: Any,
        skipped_poses: int,
    ) -> float:
        previous_id = int(previous.id)
        current_id = int(current.id)
        gap_cost = skipped_poses * self.config.skipped_pose_cost
        if previous_id == current_id:
            return self.config.same_lane_cost + gap_cost
        if any(int(lane.id) == current_id for lane in self._following(previous)):
            return self.config.following_cost + gap_cost
        left, right = self._adjacent(previous)
        if left is not None and int(left.id) == current_id:
            return self.config.adjacent_cost + gap_cost
        if right is not None and int(right.id) == current_id:
            return self.config.adjacent_cost + gap_cost
        return self.config.disconnected_cost + gap_cost

    def _optimize(
        self,
        candidate_rows: list[tuple[int, list[_Candidate]]],
    ) -> list[tuple[int, _Candidate]]:
        if not candidate_rows:
            return []
        costs: list[dict[int, float]] = []
        parents: list[dict[int, int | None]] = []
        first_candidates = candidate_rows[0][1]
        costs.append(
            {
                candidate.lanelet_id: candidate.emission_cost
                for candidate in first_candidates
            }
        )
        parents.append(
            {candidate.lanelet_id: None for candidate in first_candidates}
        )

        for row_index in range(1, len(candidate_rows)):
            pose_index, row_candidates = candidate_rows[row_index]
            previous_pose_index, previous_candidates = candidate_rows[
                row_index - 1
            ]
            previous_by_id = {
                candidate.lanelet_id: candidate
                for candidate in previous_candidates
            }
            row_costs: dict[int, float] = {}
            row_parents: dict[int, int | None] = {}
            skipped = pose_index - previous_pose_index - 1
            for current in row_candidates:
                options: list[tuple[float, int]] = []
                for previous_id, previous_cost in costs[-1].items():
                    previous = previous_by_id[previous_id]
                    options.append(
                        (
                            previous_cost
                            + self._transition(
                                previous.lanelet,
                                current.lanelet,
                                skipped,
                            )
                            + current.emission_cost,
                            previous_id,
                        )
                    )
                best_cost, best_parent = min(
                    options,
                    key=lambda option: (option[0], option[1]),
                )
                row_costs[current.lanelet_id] = best_cost
                row_parents[current.lanelet_id] = best_parent
            costs.append(row_costs)
            parents.append(row_parents)

        selected_ids = [0] * len(candidate_rows)
        selected_ids[-1] = min(
            costs[-1],
            key=lambda lanelet_id: (costs[-1][lanelet_id], lanelet_id),
        )
        for row_index in range(len(candidate_rows) - 1, 0, -1):
            parent = parents[row_index][selected_ids[row_index]]
            if parent is None:
                raise RuntimeError("matcher backtrace ended before first pose")
            selected_ids[row_index - 1] = parent

        selected: list[tuple[int, _Candidate]] = []
        for (pose_index, candidates), lanelet_id in zip(
            candidate_rows,
            selected_ids,
        ):
            by_id = {
                candidate.lanelet_id: candidate for candidate in candidates
            }
            selected.append((pose_index, by_id[lanelet_id]))
        return selected

    def _relation(
        self,
        previous: Any,
        current: Any,
    ) -> TransitionType | None:
        if int(previous.id) == int(current.id):
            return TransitionType.FOLLOW
        following = self._following(previous)
        if any(int(lane.id) == int(current.id) for lane in following):
            if len(following) > 1:
                return TransitionType.SPLIT
            if len(self._previous(current)) > 1:
                return TransitionType.MERGE
            return TransitionType.FOLLOW
        left, right = self._adjacent(previous)
        if left is not None and int(left.id) == int(current.id):
            return TransitionType.LEFT_ADJACENT
        if right is not None and int(right.id) == int(current.id):
            return TransitionType.RIGHT_ADJACENT
        return None

    def _shortest_path(self, previous: Any, current: Any) -> list[Any]:
        try:
            path = self.scene_map.routing_graph.shortestPath(
                previous,
                current,
            )
        except Exception:
            return []
        return list(path) if path is not None else []

    def _lane_sequence(
        self,
        selected: list[tuple[int, _Candidate]],
    ) -> tuple[list[tuple[Any, TransitionType]], int, float, int, int]:
        deduplicated: list[Any] = []
        for _, candidate in selected:
            if (
                not deduplicated
                or int(deduplicated[-1].id) != candidate.lanelet_id
            ):
                deduplicated.append(candidate.lanelet)
        if not deduplicated:
            return [], 0, 0.0, 0, 0

        output: list[tuple[Any, TransitionType]] = [
            (deduplicated[0], TransitionType.FOLLOW)
        ]
        fill_count = 0
        fill_length_m = 0.0
        adjacent_count = 0
        unresolved = 0
        for current in deduplicated[1:]:
            previous = output[-1][0]
            relation = self._relation(previous, current)
            if relation is not None:
                output.append((current, relation))
                if relation in (
                    TransitionType.LEFT_ADJACENT,
                    TransitionType.RIGHT_ADJACENT,
                ):
                    adjacent_count += 1
                continue

            path = self._shortest_path(previous, current)
            if (
                len(path) >= 2
                and int(path[0].id) == int(previous.id)
                and int(path[-1].id) == int(current.id)
            ):
                fill_count += 1
                for lanelet in path[1:]:
                    if int(lanelet.id) == int(output[-1][0].id):
                        continue
                    output.append((lanelet, TransitionType.FOLLOW))
                    fill_length_m += _polyline_length(_points(lanelet.centerline))
            else:
                unresolved += 1
                output.append((current, TransitionType.FOLLOW))
        return output, fill_count, fill_length_m, adjacent_count, unresolved

    @staticmethod
    def _maneuver(lanelet: Any) -> Maneuver:
        tagged = _attr(lanelet, "turn_direction").lower()
        tagged_map = {
            "straight": Maneuver.STRAIGHT,
            "left": Maneuver.LEFT,
            "right": Maneuver.RIGHT,
            "u_turn": Maneuver.U_TURN,
        }
        if tagged in tagged_map:
            return tagged_map[tagged]
        centerline = _points(lanelet.centerline)
        if len(centerline) < 3:
            return Maneuver.STRAIGHT
        first = centerline[1, :2] - centerline[0, :2]
        last = centerline[-1, :2] - centerline[-2, :2]
        delta = _wrap_angle(
            math.atan2(float(last[1]), float(last[0]))
            - math.atan2(float(first[1]), float(first[0]))
        )
        if delta > math.radians(25.0):
            return Maneuver.LEFT
        if delta < -math.radians(25.0):
            return Maneuver.RIGHT
        return Maneuver.STRAIGHT

    def _route_segment(
        self,
        lanelet: Any,
        transition: TransitionType,
        *,
        destination: bool,
    ) -> RouteLaneSegment:
        lanelet_id = int(lanelet.id)
        return RouteLaneSegment(
            lane_id=f"lanelet2:{lanelet_id}",
            provider_segment_id=str(lanelet_id),
            centerline_enu_m=_points(lanelet.centerline),
            left_boundary_enu_m=_points(lanelet.leftBound),
            right_boundary_enu_m=_points(lanelet.rightBound),
            level=_level(lanelet),
            transition_from_previous=transition,
            maneuver=(
                Maneuver.DESTINATION
                if destination
                else self._maneuver(lanelet)
            ),
            confidence=1.0,
        )

    def match(
        self,
        *,
        scene_id: str,
        positions_enu_m: np.ndarray,
        yaws_rad: np.ndarray,
        timestamps_ns: np.ndarray,
    ) -> NavigationRoute:
        positions = np.asarray(positions_enu_m, dtype=np.float64)
        yaws = np.asarray(yaws_rad, dtype=np.float64)
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("positions_enu_m must have shape [N,2] or [N,3]")
        if len(positions) == 0:
            raise ValueError("trace must not be empty")
        if yaws.shape != (len(positions),):
            raise ValueError("yaws_rad length differs from positions")
        if timestamps.shape != (len(positions),):
            raise ValueError("timestamps_ns length differs from positions")
        if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
            raise ValueError("trace contains non-finite values")
        if np.any(timestamps < 0) or np.any(np.diff(timestamps) < 0):
            raise ValueError("trace timestamps must be non-negative and ordered")

        candidate_rows: list[tuple[int, list[_Candidate]]] = []
        for index, (position, yaw) in enumerate(zip(positions, yaws)):
            candidates = self._candidates(position[:2], float(yaw))
            if candidates:
                candidate_rows.append((index, candidates))
        selected = self._optimize(candidate_rows)
        (
            lane_sequence,
            fill_count,
            fill_length,
            adjacent_count,
            unresolved,
        ) = self._lane_sequence(selected)

        distances = np.asarray(
            [candidate.distance_m for _, candidate in selected],
            dtype=np.float64,
        )
        heading_errors = np.asarray(
            [candidate.heading_error_rad for _, candidate in selected],
            dtype=np.float64,
        )
        matched_ratio = len(selected) / len(positions)
        median_distance = float(np.median(distances)) if len(distances) else 0.0
        p95_distance = (
            float(np.quantile(distances, 0.95)) if len(distances) else 0.0
        )
        median_heading = (
            float(np.median(heading_errors)) if len(heading_errors) else 0.0
        )
        p95_heading = (
            float(np.quantile(heading_errors, 0.95))
            if len(heading_errors)
            else 0.0
        )
        failure_reasons: list[str] = []
        if not lane_sequence:
            failure_reasons.append("no_lane_sequence")
        if matched_ratio < self.config.minimum_matched_pose_ratio:
            failure_reasons.append("matched_pose_ratio_below_threshold")
        if p95_distance > self.config.maximum_p95_distance_m:
            failure_reasons.append("p95_distance_above_threshold")
        if p95_heading > self.config.maximum_p95_heading_error_rad:
            failure_reasons.append("p95_heading_error_above_threshold")
        if unresolved:
            failure_reasons.append("unresolved_discontinuity")
        valid = not failure_reasons

        quality = RouteQuality(
            matched_pose_ratio=matched_ratio,
            median_lateral_distance_m=median_distance,
            p95_lateral_distance_m=p95_distance,
            median_heading_error_rad=median_heading,
            p95_heading_error_rad=p95_heading,
            shortest_path_fill_count=fill_count,
            shortest_path_fill_length_m=fill_length,
            adjacent_transition_count=adjacent_count,
            unresolved_discontinuities=unresolved,
            failure_reasons=tuple(failure_reasons),
        )
        confidence = matched_ratio
        confidence *= math.exp(-p95_distance / 10.0)
        confidence *= math.exp(-p95_heading / math.pi)
        confidence *= math.exp(-float(unresolved))
        confidence = min(1.0, max(0.0, confidence))

        trace_hasher = hashlib.sha256()
        trace_hasher.update(np.ascontiguousarray(positions, dtype="<f8").tobytes())
        trace_hasher.update(np.ascontiguousarray(yaws, dtype="<f8").tobytes())
        trace_hasher.update(
            np.ascontiguousarray(timestamps, dtype="<i8").tobytes()
        )
        route_identity = hashlib.sha256(
            (
                scene_id
                + self.navigation_map.map_version
                + self.config.sha256()
            ).encode("utf-8")
        ).hexdigest()[:24]

        route_segments = tuple(
            self._route_segment(
                lanelet,
                transition,
                destination=index == len(lane_sequence) - 1,
            )
            for index, (lanelet, transition) in enumerate(lane_sequence)
        )
        return NavigationRoute(
            route_id=f"kitscenes:{scene_id}:{route_identity}",
            revision=1,
            provider="kitscenes_lanelet2",
            timestamp_ns=int(timestamps[0]),
            valid_from_ns=int(timestamps[0]),
            map_version=self.navigation_map.map_version,
            frame=self.navigation_map.frame,
            lane_sequence=route_segments,
            destination=Destination(
                position_enu_m=positions[-1],
                source="kitscenes_scene_end",
            ),
            confidence=confidence,
            valid=valid,
            quality=quality,
            estimated_destination=True,
            provenance=RouteProvenance(
                source_revision=self.source_revision,
                matcher_version=LANELET2_MATCHER_VERSION,
                matcher_config_sha256=self.config.sha256(),
                map_sha256=self.map_sha256,
                trace_sha256=trace_hasher.hexdigest(),
            ),
        )
