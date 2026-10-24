"""20 m / 2 m commands from a supplied, ordered route centreline.

The geometry uses the recorded KITScenes ego-pose reference. It does not
silently apply NAVSIM vehicle-specific rear-axle offsets. Route reconstruction
is a separate offline operation; no future target points enter this function.
"""

import numpy as np

UNKNOWN = [0.0, 0.0, 0.0, 1.0]


def command_from_route(route, metadata, lookahead_m=20.0, lateral_threshold_m=2.0):
    if lookahead_m <= 0 or lateral_threshold_m < 0:
        raise ValueError("Invalid command geometry thresholds")
    if not route.valid or not metadata["route_valid"] or not metadata["map_valid"]:
        return UNKNOWN.copy(), {"reason": "invalid_route_or_map"}
    pose = metadata["sample_pose_enu"]
    position = np.array([pose["x_enu_m"], pose["y_enu_m"]], dtype=np.float64)
    yaw = float(pose["yaw_rad"])
    if not np.isfinite(position).all() or not np.isfinite(yaw):
        raise ValueError("Non-finite current ego pose")
    candidates = []
    for lane_index, lane in enumerate(route.lane_sequence):
        if lane.lane_id != metadata["current_route_lane_id"]:
            continue
        points = np.asarray(lane.centerline_enu_m, dtype=np.float64)[:, :2]
        if not np.isfinite(points).all():
            raise ValueError("Non-finite route centreline")
        for point_index, (a, b) in enumerate(zip(points[:-1], points[1:])):
            delta = b - a
            length_squared = float(delta @ delta)
            if length_squared == 0:
                continue
            fraction = np.clip(float((position - a) @ delta) / length_squared, 0, 1)
            projection = a + fraction * delta
            candidates.append(
                (
                    float(np.linalg.norm(position - projection)),
                    lane_index,
                    point_index,
                    projection,
                )
            )
    if not candidates:
        return UNKNOWN.copy(), {"reason": "active_lane_missing_or_degenerate"}
    _, lane_index, point_index, start = min(candidates, key=lambda c: c[:3])
    path = [start]
    path.extend(
        np.asarray(route.lane_sequence[lane_index].centerline_enu_m)[
            point_index + 1 :, :2
        ]
    )
    for lane in route.lane_sequence[lane_index + 1 :]:
        if (
            sum(
                float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
                for a, b in zip(path[:-1], path[1:])
            )
            >= lookahead_m
        ):
            break
        points = np.asarray(lane.centerline_enu_m, dtype=np.float64)[:, :2]
        if len(points) == 0:
            continue
        if not np.isfinite(points).all():
            raise ValueError("Non-finite route centreline")
        if np.linalg.norm(np.asarray(path[-1]) - points[0]) > 2.0:
            # Do not invent a centreline across a lane-change or unresolved gap.
            return UNKNOWN.copy(), {"reason": "discontinuous_route"}
        path.extend(points)
    remaining = float(lookahead_m)
    for a, b in zip(path[:-1], path[1:]):
        delta = np.asarray(b) - np.asarray(a)
        length = float(np.linalg.norm(delta))
        if length == 0:
            continue
        if remaining <= length:
            goal = np.asarray(a) + delta * (remaining / length)
            relative = goal - position
            lateral = float(-np.sin(yaw) * relative[0] + np.cos(yaw) * relative[1])
            slot = (
                0
                if lateral > lateral_threshold_m
                else 2
                if lateral < -lateral_threshold_m
                else 1
            )
            command = [0.0] * 4
            command[slot] = 1.0
            return command, {
                "reason": "available",
                "goal_enu_m": goal.tolist(),
                "goal_lateral_m": lateral,
            }
        remaining -= length
    return UNKNOWN.copy(), {"reason": "route_shorter_than_lookahead"}
