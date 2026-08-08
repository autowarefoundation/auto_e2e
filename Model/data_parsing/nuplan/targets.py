"""nuPlan scenario-to-target conversion without a hard devkit dependency."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_MEMBER,
    BEV_SEGMENTATION_CLASSES,
    TRAJECTORY_XY_MEMBER,
    encode_bev_segmentation,
    encode_reactive_navigation,
    encode_trajectory_xy,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    MapChannel,
    NavigationRasterGeometry,
)


@dataclasses.dataclass(frozen=True)
class NuPlanReactiveTargets:
    trajectory_xy_m: np.ndarray
    trajectory_valid: np.ndarray
    initial_speed_mps: float
    map_context: np.ndarray
    map_valid: bool
    bev_segmentation: np.ndarray
    bev_segmentation_valid: np.ndarray
    route_target: np.ndarray
    route_channel_valid: np.ndarray


def _pose_xy_heading(state: Any) -> tuple[float, float, float]:
    pose = state.rear_axle
    return float(pose.x), float(pose.y), float(pose.heading)


def _global_to_ego(
    points_xy: np.ndarray,
    reference_pose: tuple[float, float, float],
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2]")
    x, y, heading = reference_pose
    delta = points - np.asarray([x, y], dtype=np.float64)
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return np.column_stack((
        cosine * delta[:, 0] + sine * delta[:, 1],
        -sine * delta[:, 0] + cosine * delta[:, 1],
    ))


def future_trajectory_xy(
    scenario: Any,
    *,
    iteration: int = 0,
    horizon_s: float = 6.4,
    num_samples: int = 64,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the 10 Hz future XY target from nuPlan ego states."""
    if horizon_s <= 0.0 or num_samples <= 0:
        raise ValueError("trajectory horizon and sample count must be positive")
    current = scenario.get_ego_state_at_iteration(iteration)
    reference_pose = _pose_xy_heading(current)
    future_states = list(
        scenario.get_ego_future_trajectory(
            iteration,
            time_horizon=horizon_s,
            num_samples=num_samples,
        )
    )
    trajectory = np.zeros((num_samples, 2), dtype=np.float32)
    valid = np.zeros(num_samples, dtype=np.bool_)
    usable = min(len(future_states), num_samples)
    if usable:
        global_xy = np.asarray(
            [
                _pose_xy_heading(state)[:2]
                for state in future_states[:usable]
            ],
            dtype=np.float64,
        )
        ego_xy = _global_to_ego(global_xy, reference_pose)
        finite = np.isfinite(ego_xy).all(axis=1)
        trajectory[:usable] = np.where(
            finite[:, None],
            ego_xy,
            0.0,
        ).astype(np.float32)
        valid[:usable] = finite
    velocity = current.dynamic_car_state.rear_axle_velocity_2d
    initial_speed = math.hypot(float(velocity.x), float(velocity.y))
    if not math.isfinite(initial_speed):
        raise ValueError("nuPlan initial speed is non-finite")
    return trajectory, valid, initial_speed


def _polygon_coordinates(polygon: Any) -> np.ndarray:
    if polygon is None or polygon.is_empty:
        return np.empty((0, 2), dtype=np.float64)
    if polygon.geom_type == "MultiPolygon":
        parts = [
            np.asarray(part.exterior.coords[:-1], dtype=np.float64)
            for part in polygon.geoms
            if not part.is_empty
        ]
        return max(parts, key=len) if parts else np.empty((0, 2))
    return np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)


def _object_polygons(objects: Iterable[Any]) -> list[np.ndarray]:
    polygons = []
    for map_object in objects:
        coordinates = _polygon_coordinates(
            getattr(map_object, "polygon", None)
        )
        if len(coordinates) >= 3:
            polygons.append(coordinates)
    return polygons


def _tracked_object_polygons(
    tracked_objects: Iterable[Any],
) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = {
        "vehicle": [],
        "vulnerable_road_user": [],
        "other_obstacle": [],
    }
    for tracked_object in tracked_objects:
        object_type = getattr(
            getattr(tracked_object, "tracked_object_type", None),
            "name",
            "",
        )
        coordinates = _polygon_coordinates(
            getattr(getattr(tracked_object, "box", None), "geometry", None)
        )
        if len(coordinates) < 3:
            continue
        if object_type == "VEHICLE":
            grouped["vehicle"].append(coordinates)
        elif object_type in {"PEDESTRIAN", "BICYCLE"}:
            grouped["vulnerable_road_user"].append(coordinates)
        else:
            grouped["other_obstacle"].append(coordinates)
    return grouped


def _rasterize_polygons(
    polygons_global: Sequence[np.ndarray],
    reference_pose: tuple[float, float, float],
    geometry: NavigationRasterGeometry,
    *,
    supersample: int = 4,
) -> np.ndarray:
    if supersample <= 0:
        raise ValueError("supersample must be positive")
    high = Image.new(
        "L",
        (
            geometry.width_px * supersample,
            geometry.height_px * supersample,
        ),
        color=0,
    )
    draw = ImageDraw.Draw(high)
    for polygon in polygons_global:
        ego_polygon = _global_to_ego(polygon, reference_pose)
        pixels = geometry.ego_to_pixel(ego_polygon)
        high_pixels = np.rint(
            (pixels + 0.5) * supersample - 0.5
        ).astype(np.int32)
        draw.polygon(
            [tuple(value) for value in high_pixels[:, ::-1]],
            fill=255,
        )
    resized = high.resize(
        (geometry.width_px, geometry.height_px),
        resample=Image.Resampling.BOX,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _rasterize_polylines(
    polylines_global: Sequence[np.ndarray],
    reference_pose: tuple[float, float, float],
    geometry: NavigationRasterGeometry,
    *,
    width_m: float,
    supersample: int = 4,
) -> np.ndarray:
    if width_m <= 0.0 or supersample <= 0:
        raise ValueError("polyline width and supersample must be positive")
    high = Image.new(
        "L",
        (
            geometry.width_px * supersample,
            geometry.height_px * supersample,
        ),
        color=0,
    )
    draw = ImageDraw.Draw(high)
    width_px = max(
        1,
        round(width_m / geometry.meters_per_pixel * supersample),
    )
    for polyline in polylines_global:
        values = np.asarray(polyline, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
            continue
        pixels = geometry.ego_to_pixel(
            _global_to_ego(values, reference_pose)
        )
        high_pixels = np.rint(
            (pixels + 0.5) * supersample - 0.5
        ).astype(np.int32)
        draw.line(
            [tuple(value) for value in high_pixels[:, ::-1]],
            fill=255,
            width=width_px,
            joint="curve",
        )
    resized = high.resize(
        (geometry.width_px, geometry.height_px),
        resample=Image.Resampling.BOX,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _map_layer_polygons(
    scenario: Any,
    reference_pose: tuple[float, float, float],
    geometry: NavigationRasterGeometry,
) -> tuple[dict[str, list[np.ndarray]], dict[str, bool]]:
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuplan-devkit is required to extract native map layers"
        ) from exc
    radius = math.hypot(
        max(abs(geometry.x_min_m), abs(geometry.x_max_m)),
        max(abs(geometry.y_min_m), abs(geometry.y_max_m)),
    )
    lane_layers = (
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    )
    layer_map = {
        # nuPlan exposes DRIVABLE_AREA as a vector/raster layer but not as a
        # MapObject. Lane polygons are the stable object-level approximation.
        "drivable_area": lane_layers,
        "lane_area": (
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ),
        "intersection": (SemanticMapLayer.INTERSECTION,),
        "crosswalk": (SemanticMapLayer.CROSSWALK,),
        "stop_line": (SemanticMapLayer.STOP_LINE,),
    }
    available = set(scenario.map_api.get_available_map_objects())
    requested = [
        layer
        for layers in layer_map.values()
        for layer in layers
        if layer in available
    ]
    proximal = scenario.map_api.get_proximal_map_objects(
        Point2D(reference_pose[0], reference_pose[1]),
        radius,
        requested,
    )
    polygons = {
        name: [
            polygon
            for layer in layers
            for polygon in _object_polygons(proximal.get(layer, ()))
        ]
        for name, layers in layer_map.items()
    }
    validity = {
        name: all(layer in available for layer in layers)
        for name, layers in layer_map.items()
    }
    return polygons, validity


def _lane_features(
    scenario: Any,
    reference_pose: tuple[float, float, float],
    geometry: NavigationRasterGeometry,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not hasattr(scenario, "map_api"):
        return [], []
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuplan-devkit is required to extract lane features"
        ) from exc
    radius = math.hypot(
        max(abs(geometry.x_min_m), abs(geometry.x_max_m)),
        max(abs(geometry.y_min_m), abs(geometry.y_max_m)),
    )
    layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]
    available = set(scenario.map_api.get_available_map_objects())
    requested = [layer for layer in layers if layer in available]
    if not requested:
        return [], []
    proximal = scenario.map_api.get_proximal_map_objects(
        Point2D(reference_pose[0], reference_pose[1]),
        radius,
        requested,
    )
    centerlines: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    for layer in requested:
        for lane in proximal.get(layer, ()):
            baseline = getattr(
                getattr(lane, "baseline_path", None),
                "discrete_path",
                (),
            )
            centerline = np.asarray(
                [
                    (float(point.x), float(point.y))
                    for point in baseline
                ],
                dtype=np.float64,
            )
            if centerline.shape[0] >= 2:
                centerlines.append(centerline)
            for boundary_name in ("left_boundary", "right_boundary"):
                boundary_path = getattr(
                    getattr(lane, boundary_name, None),
                    "discrete_path",
                    (),
                )
                boundary = np.asarray(
                    [
                        (float(point.x), float(point.y))
                        for point in boundary_path
                    ],
                    dtype=np.float64,
                )
                if boundary.shape[0] >= 2:
                    boundaries.append(boundary)
    return centerlines, boundaries


def _build_navigation_map_context(
    scenario: Any,
    map_polygons: dict[str, list[np.ndarray]],
    reference_pose: tuple[float, float, float],
    geometry: NavigationRasterGeometry,
) -> tuple[np.ndarray, bool]:
    context = np.zeros(
        (MAP_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    drivable_polygons = (
        map_polygons["drivable_area"]
        + map_polygons["intersection"]
    )
    context[MapChannel.DRIVABLE_AREA] = _rasterize_polygons(
        drivable_polygons,
        reference_pose,
        geometry,
    )
    context[MapChannel.INTERSECTION] = _rasterize_polygons(
        map_polygons["intersection"],
        reference_pose,
        geometry,
    )
    context[MapChannel.CROSSWALK] = _rasterize_polygons(
        map_polygons["crosswalk"],
        reference_pose,
        geometry,
    )
    context[MapChannel.STOP_LINE] = _rasterize_polygons(
        map_polygons["stop_line"],
        reference_pose,
        geometry,
    )
    centerlines, boundaries = _lane_features(
        scenario,
        reference_pose,
        geometry,
    )
    context[MapChannel.LANE_CENTERLINE] = _rasterize_polylines(
        centerlines,
        reference_pose,
        geometry,
        width_m=geometry.meters_per_pixel,
    )
    context[MapChannel.LANE_BOUNDARY] = _rasterize_polylines(
        boundaries,
        reference_pose,
        geometry,
        width_m=geometry.meters_per_pixel,
    )

    direction_sin = np.zeros(context.shape[1:], dtype=np.float32)
    direction_cos = np.zeros_like(direction_sin)
    direction_count = np.zeros_like(direction_sin)
    for centerline in centerlines:
        ego_line = _global_to_ego(centerline, reference_pose)
        delta = ego_line[-1] - ego_line[0]
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-6:
            continue
        mask = _rasterize_polylines(
            [centerline],
            reference_pose,
            geometry,
            width_m=geometry.route_corridor_width_m,
            supersample=1,
        ) > 0.0
        direction_sin[mask] += float(delta[1] / norm)
        direction_cos[mask] += float(delta[0] / norm)
        direction_count[mask] += 1.0
    direction_valid = direction_count > 0.0
    context[MapChannel.TRAFFIC_DIRECTION_VALID] = direction_valid
    context[MapChannel.TRAFFIC_DIRECTION_SIN][direction_valid] = (
        (
            direction_sin[direction_valid]
            / direction_count[direction_valid]
            + 1.0
        )
        * 0.5
    )
    context[MapChannel.TRAFFIC_DIRECTION_COS][direction_valid] = (
        (
            direction_cos[direction_valid]
            / direction_count[direction_valid]
            + 1.0
        )
        * 0.5
    )
    known = np.maximum.reduce(
        [
            context[MapChannel.DRIVABLE_AREA],
            context[MapChannel.INTERSECTION],
            context[MapChannel.CROSSWALK],
            context[MapChannel.STOP_LINE],
            context[MapChannel.LANE_CENTERLINE],
            context[MapChannel.LANE_BOUNDARY],
        ]
    )
    context[MapChannel.KNOWN_MAP_AREA] = known
    return context, bool(known.any())


def _route_polygons(scenario: Any) -> list[np.ndarray]:
    try:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuplan-devkit is required to extract route roadblocks"
        ) from exc
    polygons = []
    for roadblock_id in scenario.get_route_roadblock_ids():
        roadblock = scenario.map_api.get_map_object(
            roadblock_id,
            SemanticMapLayer.ROADBLOCK,
        )
        if roadblock is None:
            roadblock = scenario.map_api.get_map_object(
                roadblock_id,
                SemanticMapLayer.ROADBLOCK_CONNECTOR,
            )
        coordinates = _polygon_coordinates(
            getattr(roadblock, "polygon", None)
        )
        if len(coordinates) >= 3:
            polygons.append(coordinates)
    return polygons


def _destination_heatmap(
    goal_xy_ego: np.ndarray,
    geometry: NavigationRasterGeometry,
) -> tuple[np.ndarray, bool]:
    heatmap = np.zeros(
        (geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    if (
        goal_xy_ego.shape != (2,)
        or not np.isfinite(goal_xy_ego).all()
        or not bool(
            geometry.contains_ego_points(goal_xy_ego[None])[0]
        )
    ):
        return heatmap, False
    center = geometry.ego_to_pixel(goal_xy_ego[None])[0]
    center_row = int(
        np.clip(round(float(center[0])), 0, geometry.height_px - 1)
    )
    center_col = int(
        np.clip(round(float(center[1])), 0, geometry.width_px - 1)
    )
    rows, cols = np.meshgrid(
        np.arange(geometry.height_px, dtype=np.float32),
        np.arange(geometry.width_px, dtype=np.float32),
        indexing="ij",
    )
    sigma_px = geometry.destination_marker_radius_m / (
        2.0 * geometry.meters_per_pixel
    )
    heatmap = np.exp(
        -((rows - center_row) ** 2 + (cols - center_col) ** 2)
        / (2.0 * sigma_px**2)
    ).astype(np.float32)
    return heatmap, True


def build_nuplan_reactive_targets(
    scenario: Any,
    *,
    iteration: int = 0,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    camera_visibility: np.ndarray | None = None,
    lidar_observability: np.ndarray | None = None,
) -> NuPlanReactiveTargets:
    """Build trajectory, BEV semantics, and route targets for one scenario."""
    current = scenario.get_ego_state_at_iteration(iteration)
    reference_pose = _pose_xy_heading(current)
    trajectory, trajectory_valid, initial_speed = future_trajectory_xy(
        scenario,
        iteration=iteration,
    )
    shape = (geometry.height_px, geometry.width_px)
    if camera_visibility is None or lidar_observability is None:
        raise ValueError(
            "nuPlan BEV targets require explicit camera and lidar "
            "observability masks"
        )
    camera_valid = np.asarray(camera_visibility, dtype=np.bool_)
    lidar_valid = np.asarray(lidar_observability, dtype=np.bool_)
    if camera_valid.shape != shape or lidar_valid.shape != shape:
        raise ValueError("observability masks must match the BEV geometry")

    map_polygons, map_available = _map_layer_polygons(
        scenario,
        reference_pose,
        geometry,
    )
    map_context, map_valid = _build_navigation_map_context(
        scenario,
        map_polygons,
        reference_pose,
        geometry,
    )
    detections = scenario.get_tracked_objects_at_iteration(iteration)
    dynamic_polygons = _tracked_object_polygons(
        detections.tracked_objects
    )
    sources = {
        **map_polygons,
        **dynamic_polygons,
    }
    known_map_area = (
        map_context[MapChannel.KNOWN_MAP_AREA] > 0.0
    )
    semantic = np.zeros(
        (len(BEV_SEGMENTATION_CLASSES), *shape),
        dtype=np.float32,
    )
    semantic_valid = np.zeros_like(semantic, dtype=np.bool_)
    for class_index, class_name in enumerate(BEV_SEGMENTATION_CLASSES):
        semantic[class_index] = _rasterize_polygons(
            sources[class_name],
            reference_pose,
            geometry,
        )
        if class_name in map_available:
            semantic_valid[class_index] = (
                camera_valid
                & known_map_area
                & map_available[class_name]
            )
        else:
            # Positive footprints remain supervised even if the conservative
            # lidar coverage mask excludes their cell.
            semantic_valid[class_index] = (
                camera_valid
                & (
                    lidar_valid
                    | (semantic[class_index] > 0.0)
                )
            )

    corridor_polygons = _route_polygons(scenario)
    route_target = np.zeros((2, *shape), dtype=np.float32)
    if corridor_polygons:
        route_target[0] = _rasterize_polygons(
            corridor_polygons,
            reference_pose,
            geometry,
        )
    route_channel_valid = np.asarray(
        [bool(corridor_polygons), False],
        dtype=np.bool_,
    )
    mission_goal = scenario.get_mission_goal()
    if mission_goal is not None:
        goal_ego = _global_to_ego(
            np.asarray(
                [[float(mission_goal.x), float(mission_goal.y)]],
                dtype=np.float64,
            ),
            reference_pose,
        )[0]
        destination, destination_valid = _destination_heatmap(
            goal_ego,
            geometry,
        )
        route_target[1] = destination
        route_channel_valid[1] = destination_valid

    return NuPlanReactiveTargets(
        trajectory_xy_m=trajectory,
        trajectory_valid=trajectory_valid,
        initial_speed_mps=initial_speed,
        map_context=map_context,
        map_valid=map_valid,
        bev_segmentation=semantic,
        bev_segmentation_valid=semantic_valid,
        route_target=route_target,
        route_channel_valid=route_channel_valid,
    )


def nuplan_reactive_target_members(
    targets: NuPlanReactiveTargets,
    *,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    metadata: dict[str, object] | None = None,
) -> dict[str, bytes]:
    """Encode one nuPlan target set into the common packed sample ABI."""
    members = encode_reactive_navigation(
        targets.map_context,
        targets.route_target,
        map_valid=targets.map_valid,
        route_channel_valid=targets.route_channel_valid,
        geometry=geometry,
        metadata={
            "map_source": "nuplan_native",
            "route_source": "nuplan_route_roadblock_ids",
            **(metadata or {}),
        },
    )
    members[TRAJECTORY_XY_MEMBER] = encode_trajectory_xy(
        targets.trajectory_xy_m,
        targets.trajectory_valid,
    )
    members[BEV_SEGMENTATION_MEMBER] = encode_bev_segmentation(
        targets.bev_segmentation,
        targets.bev_segmentation_valid,
    )
    return members
