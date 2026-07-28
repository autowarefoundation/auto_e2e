"""Scene-native ODD statistics with duration and trajectory weighting."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


STATISTICS_SCHEMA_VERSION = "odd_statistics_v2"
MIN_COOCCURRENCE_NS = 100_000_000
BOOTSTRAP_REPLICATES = 256
CONFIDENCE_BOUNDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

Interval = tuple[int, int]


def _union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(
        (int(start), int(end))
        for start, end in intervals
        if int(end) > int(start)
    )
    if not ordered:
        return []
    output: list[Interval] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        output.append((start, end))
        start, end = next_start, next_end
    output.append((start, end))
    return output


def union_duration(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in _union_intervals(intervals))


def _intersection(
    left: Sequence[Interval],
    right: Sequence[Interval],
    *,
    minimum_ns: int = 1,
) -> list[Interval]:
    left_rows = _union_intervals(left)
    right_rows = _union_intervals(right)
    output: list[Interval] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_rows) and right_index < len(right_rows):
        start = max(
            left_rows[left_index][0],
            right_rows[right_index][0],
        )
        end = min(
            left_rows[left_index][1],
            right_rows[right_index][1],
        )
        if end - start >= minimum_ns:
            output.append((start, end))
        if left_rows[left_index][1] <= right_rows[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return output


@dataclasses.dataclass(frozen=True)
class _DistanceSegment:
    start_ns: int
    end_ns: int
    distance_m: float


@dataclasses.dataclass(frozen=True)
class _DistanceProfile:
    segments: tuple[_DistanceSegment, ...]
    method: str

    def distance(self, intervals: Iterable[Interval]) -> float:
        total = 0.0
        for start_ns, end_ns in _union_intervals(intervals):
            for segment in self.segments:
                overlap_ns = min(end_ns, segment.end_ns) - max(
                    start_ns,
                    segment.start_ns,
                )
                if overlap_ns <= 0:
                    continue
                total += segment.distance_m * overlap_ns / (
                    segment.end_ns - segment.start_ns
                )
        return total


def _distance_profile(record: Mapping[str, Any]) -> _DistanceProfile:
    scene_start = int(record["start_timestamp_ns"])
    scene_end = int(record["end_timestamp_ns"])
    scene_duration = scene_end - scene_start
    scene_distance = float(record["distance_m"])
    speed_rows = []
    for observation in record.get("observations", []):
        if (
            observation.get("key") != "odd.ego.speed_bin"
            or observation.get("status") != "valid"
        ):
            continue
        speed = observation.get("measurements", {}).get("ego_speed_mps")
        if not isinstance(speed, (int, float)) or not math.isfinite(speed):
            continue
        speed_rows.append(
            (
                max(scene_start, int(observation["start_timestamp_ns"])),
                min(scene_end, int(observation["end_timestamp_ns"])),
                max(0.0, float(speed)),
                float(observation["confidence"]),
            )
        )
    boundaries = {scene_start, scene_end}
    for start_ns, end_ns, _, _ in speed_rows:
        if end_ns > start_ns:
            boundaries.update((start_ns, end_ns))
    ordered = sorted(boundaries)
    raw_segments: list[tuple[int, int, float]] = []
    used_speed = False
    fallback_speed = (
        scene_distance / (scene_duration / 1e9)
        if scene_duration > 0
        else 0.0
    )
    for start_ns, end_ns in zip(ordered, ordered[1:]):
        active = [
            row
            for row in speed_rows
            if row[0] <= start_ns and row[1] >= end_ns
        ]
        if active:
            selected = max(active, key=lambda row: (row[3], -row[0]))
            speed_mps = selected[2]
            used_speed = True
        else:
            speed_mps = fallback_speed
        raw_segments.append(
            (
                start_ns,
                end_ns,
                speed_mps * (end_ns - start_ns) / 1e9,
            )
        )
    raw_total = sum(row[2] for row in raw_segments)
    if raw_total > 0.0:
        scale = scene_distance / raw_total
    else:
        scale = 0.0
    segments = tuple(
        _DistanceSegment(start_ns, end_ns, distance_m * scale)
        for start_ns, end_ns, distance_m in raw_segments
    )
    if not segments:
        segments = (
            _DistanceSegment(scene_start, scene_end, scene_distance),
        )
    return _DistanceProfile(
        segments=segments,
        method=(
            "speed_integrated_scene_normalized"
            if used_speed
            else "duration_proportional"
        ),
    )


def _wilson_interval(successes: int, trials: int) -> dict[str, Any]:
    if trials <= 0:
        return {
            "lower": 0.0,
            "upper": 0.0,
            "method": "wilson_scene_95",
        }
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "method": "wilson_scene_95",
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _cluster_bootstrap_interval(
    numerators: Mapping[str, float],
    denominators: Mapping[str, float],
    *,
    seed_key: str,
) -> dict[str, Any]:
    scene_ids = sorted(set(numerators) | set(denominators))
    if not scene_ids or sum(denominators.values()) <= 0.0:
        return {
            "lower": 0.0,
            "upper": 0.0,
            "method": "scene_clustered_bootstrap_95",
            "replicates": BOOTSTRAP_REPLICATES,
        }
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8],
        "big",
    )
    generator = random.Random(seed)
    ratios = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = generator.choices(scene_ids, k=len(scene_ids))
        denominator = sum(denominators.get(scene_uid, 0.0) for scene_uid in sampled)
        numerator = sum(numerators.get(scene_uid, 0.0) for scene_uid in sampled)
        ratios.append(numerator / denominator if denominator else 0.0)
    return {
        "lower": _percentile(ratios, 0.025),
        "upper": _percentile(ratios, 0.975),
        "method": "scene_clustered_bootstrap_95",
        "replicates": BOOTSTRAP_REPLICATES,
    }


def _weighted_quantile(
    rows: Sequence[tuple[float, float]],
    probability: float,
) -> float:
    ordered = sorted(rows)
    total_weight = sum(weight for _, weight in ordered)
    if not ordered or total_weight <= 0.0:
        return 0.0
    target = probability * total_weight
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= target:
            return value
    return ordered[-1][0]


def _confidence_summary(
    rows: Sequence[tuple[float, int, float]],
) -> dict[str, Any]:
    if not rows:
        return {
            "observation_count": 0,
            "duration_weighted_mean": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "bins": [],
        }
    duration_total = sum(duration for _, duration, _ in rows)
    duration_weights = [
        (confidence, float(duration))
        for confidence, duration, _ in rows
    ]
    bins = []
    for index, lower in enumerate(CONFIDENCE_BOUNDS[:-1]):
        upper = CONFIDENCE_BOUNDS[index + 1]
        selected = [
            row
            for row in rows
            if lower <= row[0] <= upper
            and (row[0] < upper or upper == 1.0)
        ]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "observation_count": len(selected),
                "duration_ns": sum(row[1] for row in selected),
                "distance_m": sum(row[2] for row in selected),
            }
        )
    return {
        "observation_count": len(rows),
        "duration_weighted_mean": (
            sum(confidence * duration for confidence, duration, _ in rows)
            / duration_total
            if duration_total
            else sum(row[0] for row in rows) / len(rows)
        ),
        "p10": _weighted_quantile(duration_weights, 0.10),
        "p50": _weighted_quantile(duration_weights, 0.50),
        "p90": _weighted_quantile(duration_weights, 0.90),
        "bins": bins,
    }


def _event_value_instances(
    record: Mapping[str, Any],
) -> dict[tuple[str, str], set[str]]:
    observations = {
        str(observation["observation_uid"]): observation
        for observation in record.get("observations", [])
    }
    output: dict[tuple[str, str], set[str]] = defaultdict(set)
    for event in record.get("events", []):
        event_uid = str(event["event_uid"])
        for observation_uid in event.get("observation_uids", []):
            observation = observations.get(str(observation_uid))
            if not observation or observation.get("status") != "valid":
                continue
            for value in observation.get("values", []):
                output[(str(observation["key"]), str(value))].add(event_uid)
    return output


def _value_statistic(
    *,
    labelset_id: str,
    key: str,
    value: str,
    valid_scenes: set[str],
    value_scenes: set[str],
    valid_duration_by_scene: Mapping[str, float],
    value_duration_by_scene: Mapping[str, float],
    valid_distance_by_scene: Mapping[str, float],
    value_distance_by_scene: Mapping[str, float],
    interval_count: int,
    event_instance_count: int,
    confidence_rows: Sequence[tuple[float, int, float]],
) -> dict[str, Any]:
    scene_count = len(value_scenes)
    valid_scene_count = len(valid_scenes)
    duration_ns = int(round(sum(value_duration_by_scene.values())))
    valid_duration_ns = int(round(sum(valid_duration_by_scene.values())))
    distance_m = sum(value_distance_by_scene.values())
    valid_distance_m = sum(valid_distance_by_scene.values())
    return {
        "value": value,
        "scene_count": scene_count,
        "scene_ratio": (
            scene_count / valid_scene_count if valid_scene_count else 0.0
        ),
        "scene_ratio_ci95": _wilson_interval(
            scene_count,
            valid_scene_count,
        ),
        "duration_ns": duration_ns,
        "duration_ratio": (
            duration_ns / valid_duration_ns if valid_duration_ns else 0.0
        ),
        "duration_ratio_ci95": _cluster_bootstrap_interval(
            value_duration_by_scene,
            valid_duration_by_scene,
            seed_key=f"{labelset_id}:{key}:{value}:duration",
        ),
        "distance_m": distance_m,
        "distance_ratio": (
            distance_m / valid_distance_m if valid_distance_m else 0.0
        ),
        "distance_ratio_ci95": _cluster_bootstrap_interval(
            value_distance_by_scene,
            valid_distance_by_scene,
            seed_key=f"{labelset_id}:{key}:{value}:distance",
        ),
        "valid_interval_count": interval_count,
        "event_instance_count": event_instance_count,
        "confidence": _confidence_summary(confidence_rows),
    }


def _key_statistic(
    records: Sequence[Mapping[str, Any]],
    definition: Mapping[str, Any],
    *,
    labelset_id: str,
    profiles: Mapping[str, _DistanceProfile],
) -> dict[str, Any]:
    key = str(definition["key"])
    scene_duration = {
        str(record["scene_uid"]): (
            int(record["end_timestamp_ns"])
            - int(record["start_timestamp_ns"])
        )
        for record in records
    }
    scene_distance = {
        str(record["scene_uid"]): float(record["distance_m"])
        for record in records
    }
    valid_scenes: set[str] = set()
    status_scenes: dict[str, set[str]] = defaultdict(set)
    value_scenes: dict[str, set[str]] = defaultdict(set)
    status_duration_by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    status_distance_by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    value_duration_by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    value_distance_by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    source_scenes: dict[str, set[str]] = defaultdict(set)
    source_duration_by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    source_distance_by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    value_interval_count: dict[str, int] = defaultdict(int)
    confidence_rows: list[tuple[float, int, float]] = []
    value_confidence_rows: dict[
        str, list[tuple[float, int, float]]
    ] = defaultdict(list)
    attempted_count = 0
    successful_count = 0
    conflict_count = 0
    valid_interval_count = 0
    event_instances: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        scene_uid = str(record["scene_uid"])
        profile = profiles[scene_uid]
        observations = [
            observation
            for observation in record.get("observations", [])
            if observation.get("key") == key
        ]
        for evidence in record.get("evidence", []):
            if evidence.get("label_key") != key:
                continue
            attempted_count += 1
            if evidence.get("status") == "valid":
                successful_count += 1
        for identity, ids in _event_value_instances(record).items():
            event_instances[identity].update(ids)
        conflict_count += sum(
            bool(observation.get("conflicting_evidence_uids"))
            for observation in observations
        )
        status_intervals: dict[str, list[Interval]] = defaultdict(list)
        value_intervals: dict[str, list[Interval]] = defaultdict(list)
        source_intervals: dict[str, list[Interval]] = defaultdict(list)
        for observation in observations:
            interval = (
                int(observation["start_timestamp_ns"]),
                int(observation["end_timestamp_ns"]),
            )
            status = str(observation["status"])
            source = str(observation["source"])
            status_scenes[status].add(scene_uid)
            source_scenes[source].add(scene_uid)
            status_intervals[status].append(interval)
            source_intervals[source].append(interval)
            duration_ns = interval[1] - interval[0]
            distance_m = profile.distance([interval])
            confidence_rows.append(
                (
                    float(observation["confidence"]),
                    duration_ns,
                    distance_m,
                )
            )
            if status != "valid":
                continue
            valid_scenes.add(scene_uid)
            for value in observation.get("values", []):
                value = str(value)
                value_scenes[value].add(scene_uid)
                value_intervals[value].append(interval)
                value_confidence_rows[value].append(
                    (
                        float(observation["confidence"]),
                        duration_ns,
                        distance_m,
                    )
                )
        for status, intervals in status_intervals.items():
            merged = _union_intervals(intervals)
            if status == "valid":
                valid_interval_count += len(merged)
            status_duration_by_scene[status][scene_uid] = union_duration(
                merged
            )
            status_distance_by_scene[status][scene_uid] = profile.distance(
                merged
            )
        for source, intervals in source_intervals.items():
            merged = _union_intervals(intervals)
            source_duration_by_scene[source][scene_uid] = union_duration(
                merged
            )
            source_distance_by_scene[source][scene_uid] = profile.distance(
                merged
            )
        for value, intervals in value_intervals.items():
            merged = _union_intervals(intervals)
            value_duration_by_scene[value][scene_uid] = union_duration(
                merged
            )
            value_distance_by_scene[value][scene_uid] = profile.distance(
                merged
            )
            value_interval_count[value] += len(merged)

    valid_duration_by_scene = status_duration_by_scene.get("valid", {})
    valid_distance_by_scene = status_distance_by_scene.get("valid", {})
    valid_duration = int(round(sum(valid_duration_by_scene.values())))
    valid_distance = sum(valid_distance_by_scene.values())
    total_duration = sum(scene_duration.values())
    total_distance = sum(scene_distance.values())
    values = []
    for candidate in definition["values"]:
        value = str(candidate["value"])
        values.append(
            _value_statistic(
                labelset_id=labelset_id,
                key=key,
                value=value,
                valid_scenes=valid_scenes,
                value_scenes=value_scenes[value],
                valid_duration_by_scene=valid_duration_by_scene,
                value_duration_by_scene=value_duration_by_scene[value],
                valid_distance_by_scene=valid_distance_by_scene,
                value_distance_by_scene=value_distance_by_scene[value],
                interval_count=value_interval_count[value],
                event_instance_count=len(event_instances[(key, value)]),
                confidence_rows=value_confidence_rows[value],
            )
        )
    statuses = ("valid", "unavailable", "not_observable", "ambiguous")
    return {
        "key": key,
        "namespace": str(definition["namespace"]),
        "quality_tier": str(definition.get("quality_tier", "experimental")),
        "valid_scene_count": len(valid_scenes),
        "eligible_scene_count": len(records),
        "observable_scene_coverage": (
            len(valid_scenes) / len(records) if records else 0.0
        ),
        "eligible_duration_ns": total_duration,
        "valid_duration_ns": valid_duration,
        "observable_duration_coverage": (
            valid_duration / total_duration if total_duration else 0.0
        ),
        "eligible_distance_m": total_distance,
        "valid_distance_m": valid_distance,
        "observable_distance_coverage": (
            valid_distance / total_distance if total_distance else 0.0
        ),
        "valid_interval_count": valid_interval_count,
        "attempted_count": attempted_count,
        "successful_count": successful_count,
        "conflict_count": conflict_count,
        "status_scene_counts": {
            status: len(status_scenes[status]) for status in statuses
        },
        "status_duration_ns": {
            status: int(
                round(sum(status_duration_by_scene[status].values()))
            )
            for status in statuses
        },
        "status_distance_m": {
            status: sum(status_distance_by_scene[status].values())
            for status in statuses
        },
        "source_scene_counts": {
            source: len(scenes)
            for source, scenes in sorted(source_scenes.items())
        },
        "source_duration_ns": {
            source: int(round(sum(by_scene.values())))
            for source, by_scene in sorted(source_duration_by_scene.items())
        },
        "source_distance_m": {
            source: sum(by_scene.values())
            for source, by_scene in sorted(source_distance_by_scene.items())
        },
        "confidence": _confidence_summary(confidence_rows),
        "values": values,
    }


def _scene_value_intervals(
    record: Mapping[str, Any],
    *,
    namespace: str,
) -> dict[tuple[str, str], list[Interval]]:
    output: dict[tuple[str, str], list[Interval]] = defaultdict(list)
    for observation in record.get("observations", []):
        key = str(observation["key"])
        if (
            not key.startswith(f"{namespace}.")
            or observation.get("status") != "valid"
        ):
            continue
        interval = (
            int(observation["start_timestamp_ns"]),
            int(observation["end_timestamp_ns"]),
        )
        for value in observation.get("values", []):
            output[(key, str(value))].append(interval)
    return {
        identity: _union_intervals(intervals)
        for identity, intervals in output.items()
    }


def _cooccurrences(
    records: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, _DistanceProfile],
) -> dict[str, Any]:
    odd_pairs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    odd_event: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        scene_uid = str(record["scene_uid"])
        profile = profiles[scene_uid]
        odd_values = _scene_value_intervals(record, namespace="odd")
        identities = sorted(odd_values)
        for left_index, left in enumerate(identities):
            for right in identities[left_index + 1 :]:
                if left[0] == right[0]:
                    continue
                overlap = _intersection(
                    odd_values[left],
                    odd_values[right],
                    minimum_ns=MIN_COOCCURRENCE_NS,
                )
                if not overlap:
                    continue
                identity = (left[0], left[1], right[0], right[1])
                row = odd_pairs.setdefault(
                    identity,
                    {
                        "left_key": left[0],
                        "left_value": left[1],
                        "right_key": right[0],
                        "right_value": right[1],
                        "scene_uids": set(),
                        "overlap_duration_ns": 0,
                        "overlap_distance_m": 0.0,
                    },
                )
                row["scene_uids"].add(scene_uid)
                row["overlap_duration_ns"] += union_duration(overlap)
                row["overlap_distance_m"] += profile.distance(overlap)

        for event in record.get("events", []):
            event_interval = [
                (
                    int(event["start_timestamp_ns"]),
                    int(event["end_timestamp_ns"]),
                )
            ]
            event_values = event.get("provenance", {}).get(
                "primary_values",
                [],
            )
            if not event_values:
                event_values = ["present"]
            for odd_identity, intervals in odd_values.items():
                overlap = _intersection(
                    intervals,
                    event_interval,
                    minimum_ns=MIN_COOCCURRENCE_NS,
                )
                if not overlap:
                    continue
                for event_value in event_values:
                    identity = (
                        odd_identity[0],
                        odd_identity[1],
                        str(event["primary_event_key"]),
                        str(event_value),
                    )
                    row = odd_event.setdefault(
                        identity,
                        {
                            "odd_key": odd_identity[0],
                            "odd_value": odd_identity[1],
                            "event_key": str(event["primary_event_key"]),
                            "event_value": str(event_value),
                            "scene_uids": set(),
                            "event_uids": set(),
                            "overlap_duration_ns": 0,
                            "overlap_distance_m": 0.0,
                        },
                    )
                    row["scene_uids"].add(scene_uid)
                    row["event_uids"].add(str(event["event_uid"]))
                    row["overlap_duration_ns"] += union_duration(overlap)
                    row["overlap_distance_m"] += profile.distance(overlap)

    pair_rows = []
    for identity in sorted(odd_pairs):
        row = odd_pairs[identity]
        pair_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "scene_uids"
            }
            | {"scene_count": len(row["scene_uids"])}
        )
    event_rows = []
    for identity in sorted(odd_event):
        row = odd_event[identity]
        event_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"scene_uids", "event_uids"}
            }
            | {
                "scene_count": len(row["scene_uids"]),
                "event_instance_count": len(row["event_uids"]),
            }
        )
    return {
        "minimum_overlap_ns": MIN_COOCCURRENCE_NS,
        "odd_pairs": pair_rows,
        "odd_event": event_rows,
    }


def build_statistics(
    records: Sequence[Mapping[str, Any]],
    ontology: Mapping[str, Any],
    labelset_id: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("ODD statistics require at least one scene")
    ordered = sorted(records, key=lambda record: str(record["scene_uid"]))
    scene_ids = [str(record["scene_uid"]) for record in ordered]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("ODD statistics require unique scene identities")
    profiles = {
        str(record["scene_uid"]): _distance_profile(record)
        for record in ordered
    }
    keys = [
        _key_statistic(
            ordered,
            definition,
            labelset_id=labelset_id,
            profiles=profiles,
        )
        for definition in ontology["labels"]
    ]
    method_counts: dict[str, int] = defaultdict(int)
    for profile in profiles.values():
        method_counts[profile.method] += 1
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "labelset_id": labelset_id,
        "scene_count": len(ordered),
        "scene_duration_ns": sum(
            int(record["end_timestamp_ns"])
            - int(record["start_timestamp_ns"])
            for record in ordered
        ),
        "scene_distance_m": sum(
            float(record["distance_m"]) for record in ordered
        ),
        "distance_weighting": {
            "method_scene_counts": dict(sorted(method_counts.items())),
            "normalization": "per_scene_recorded_distance",
        },
        "keys": keys,
        "cooccurrences": _cooccurrences(ordered, profiles),
    }
