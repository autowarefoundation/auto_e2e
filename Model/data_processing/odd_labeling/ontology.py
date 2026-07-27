"""Machine-readable ontology for scene-level ODD labeling."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable
from typing import Any


ONTOLOGY_VERSION = "odd_ontology_v1"
LABEL_STATUSES = ("valid", "unavailable", "not_observable", "ambiguous")
LABEL_SOURCES = (
    "map_route",
    "gnss_ins",
    "vlm",
    "image_qc",
    "fusion",
    "can_optional",
)


@dataclasses.dataclass(frozen=True)
class LabelDefinition:
    key: str
    cardinality: str
    values: tuple[str, ...]
    primary_sources: tuple[str, ...]
    backends: tuple[str, ...]
    description: str
    subject: str = "scene"
    temporal_scope: str = "interval"
    quality_tier: str = "experimental"
    none_semantics: str | None = None

    @property
    def namespace(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def display_name(self) -> str:
        return self.key.split(".", 1)[1].replace(".", " ").replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "namespace": self.namespace,
            "display_name": self.display_name,
            "description": self.description,
            "cardinality": self.cardinality,
            "values": [{"value": value} for value in self.values],
            "primary_sources": list(self.primary_sources),
            "backends": list(self.backends),
            "subject": self.subject,
            "temporal_scope": self.temporal_scope,
            "quality_tier": self.quality_tier,
            "none_semantics": self.none_semantics,
        }


def _definition(
    key: str,
    cardinality: str,
    values: str,
    sources: str,
    backends: str,
    description: str,
    *,
    subject: str = "scene",
    temporal_scope: str = "interval",
    quality_tier: str = "experimental",
    none_semantics: str | None = None,
) -> LabelDefinition:
    return LabelDefinition(
        key=key,
        cardinality=cardinality,
        values=tuple(values.split()),
        primary_sources=tuple(sources.split()),
        backends=tuple(backends.split()),
        description=description,
        subject=subject,
        temporal_scope=temporal_scope,
        quality_tier=quality_tier,
        none_semantics=none_semantics,
    )


_DEFINITIONS = (
    _definition("odd.road.context", "single", "urban suburban rural motorway residential industrial parking", "map_route vlm", "deterministic openai_compatible", "Functional character of the road surroundings."),
    _definition("odd.road.type", "single", "motorway trunk primary secondary tertiary residential service ramp parking_aisle shared_space", "map_route", "deterministic bedrock_claude", "Current road hierarchy or use."),
    _definition("odd.road.division", "single", "divided undivided", "map_route", "deterministic", "Whether opposing traffic is physically divided."),
    _definition("odd.road.directionality", "single", "one_way two_way", "map_route", "deterministic", "Permitted travel direction on the current road."),
    _definition("odd.road.horizontal_geometry", "single", "straight curve_left curve_right", "map_route", "deterministic", "Signed horizontal geometry of the ego-connected road."),
    _definition("odd.road.vertical_geometry", "single", "level uphill downhill crest sag", "map_route gnss_ins", "deterministic", "Vertical profile of the driven road."),
    _definition("odd.road.junction_type", "single", "none t_junction y_junction crossroad staggered roundabout merge diverge grade_separated", "map_route", "deterministic bedrock_claude", "Topology of the current or approaching junction."),
    _definition("odd.road.junction_position", "single", "approach inside exit midblock", "map_route", "deterministic", "Ego position relative to a junction."),
    _definition("odd.road.junction_control", "single", "traffic_light stop_sign yield_sign uncontrolled other", "map_route vlm", "deterministic openai_compatible", "Control governing the route-relevant junction."),
    _definition("odd.route.action", "single", "lane_follow straight turn_left turn_right u_turn merge diverge roundabout_enter roundabout_exit", "map_route", "deterministic bedrock_claude", "Action planned by the selected route."),
    _definition("odd.road.lane_count_bin", "single", "one two three four_plus", "map_route", "deterministic", "Lane count in the ego travel direction."),
    _definition("odd.road.lane_type_present", "multi", "general bus bicycle tram emergency turn_only parking shared none", "map_route", "deterministic", "Lane types present in the local road corridor.", none_semantics="Observed corridor contains none of the listed lane types."),
    _definition("odd.road.lane_marking_quality", "single", "clear faded missing temporary occluded", "vlm", "openai_compatible", "Visual condition of lane markings."),
    _definition("odd.road.surface_type", "single", "asphalt concrete paving_stone gravel unpaved", "vlm map_route", "openai_compatible deterministic", "Visible or mapped road-surface material."),
    _definition("odd.road.surface_state", "multi", "dry wet standing_water snow_covered visually_contaminated", "vlm", "openai_compatible", "Visible state of the road surface."),
    _definition("odd.road.edge_type_present", "multi", "curb guardrail solid_barrier temporary_barrier paved_shoulder unpaved_shoulder grass none", "map_route vlm", "deterministic openai_compatible", "Road-edge treatments visible or mapped in the local corridor.", none_semantics="Both relevant road edges were observed and no listed treatment is present."),
    _definition("odd.road.special_structure", "multi", "bridge tunnel railway_crossing pedestrian_crossing access_gate none", "map_route vlm", "deterministic openai_compatible", "Special road structures on the current route.", none_semantics="The route corridor was observed and no listed structure is present."),
    _definition("odd.road.workzone_state", "multi", "roadworks lane_closure detour cones temporary_barrier temporary_signage none", "vlm map_route", "openai_compatible deterministic", "Temporary road-work conditions.", none_semantics="Road and control area were observed with no workzone condition."),
    _definition("odd.traffic_control.present", "multi", "traffic_light stop_sign yield_sign speed_limit_sign temporary_sign traffic_officer none", "map_route vlm", "deterministic openai_compatible", "Traffic controls present in the route-relevant area.", none_semantics="The relevant control area was observed and no listed control is present."),
    _definition("odd.traffic_light.state", "single", "red red_yellow yellow green flashing off not_applicable", "vlm map_route", "openai_compatible deterministic", "State of the route-relevant traffic signal."),
    _definition("odd.environment.day_phase", "single", "day dawn dusk night", "fusion vlm", "deterministic openai_compatible", "Solar or visually inferred phase of day."),
    _definition("odd.environment.sky", "single", "clear partly_cloudy overcast", "vlm", "openai_compatible", "Visible sky condition."),
    _definition("odd.environment.precipitation_visual", "single", "none_visible rain snow mixed", "vlm image_qc", "openai_compatible deterministic", "Precipitation visible in camera imagery."),
    _definition("odd.environment.visibility_degradation", "multi", "fog haze precipitation water_spray smoke_or_dust none", "vlm image_qc", "openai_compatible deterministic", "Atmospheric causes that reduce visibility.", none_semantics="Sufficient distant scene content was visible without a listed degradation."),
    _definition("odd.environment.road_lighting", "single", "daylight street_lit unlit tunnel_lit", "vlm fusion", "openai_compatible deterministic", "Lighting available on the driven road."),
    _definition("odd.environment.glare", "multi", "sun_front sun_side headlight wet_road_reflection none", "vlm image_qc", "openai_compatible deterministic", "Glare source and direction affecting the road view.", none_semantics="Usable camera views contain no supported glare condition."),
    _definition("odd.dynamic.traffic_density", "single", "empty low medium high stop_and_go", "fusion vlm", "openai_compatible deterministic", "Density and flow state of motorized traffic."),
    _definition("odd.dynamic.vru_density", "single", "none low medium high", "fusion vlm", "openai_compatible deterministic", "Density of vulnerable road users."),
    _definition("odd.dynamic.parked_vehicle_density", "single", "none low medium high", "fusion vlm", "openai_compatible deterministic", "Density of parked vehicles along the corridor."),
    _definition("odd.dynamic.oncoming_traffic", "single", "absent present", "fusion vlm", "openai_compatible deterministic", "Presence of traffic moving in the opposing direction."),
    _definition("odd.dynamic.agent_type_present", "multi", "passenger_vehicle light_commercial_vehicle heavy_truck bus motorcycle bicycle e_scooter pedestrian wheelchair_user animal emergency_vehicle construction_vehicle none", "fusion vlm", "openai_compatible deterministic", "Road-agent classes present in the observable scene.", none_semantics="Road and sidewalk regions were observed and no listed agent is present."),
    _definition("odd.ego.speed_bin", "single", "stationary creeping low_speed medium_speed high_speed", "gnss_ins can_optional", "deterministic", "Gap-aware ego-speed category with raw speed retained."),
    _definition("event.ego.motion_state", "single", "stopped starting moving creeping accelerating decelerating reversing", "gnss_ins can_optional", "deterministic", "Observed ego motion state.", temporal_scope="event"),
    _definition("event.ego.maneuver", "single", "lane_follow turn_left turn_right u_turn lane_change_left lane_change_right merge diverge pull_over pull_out overtake stop", "fusion gnss_ins map_route", "deterministic", "Maneuver executed by the driven trajectory.", temporal_scope="event"),
    _definition("event.ego.strong_response", "single", "none hard_brake emergency_stop evasive_steer", "gnss_ins can_optional fusion", "deterministic", "Strong ego response derived from motion signals.", temporal_scope="event"),
    _definition("event.vehicle.interaction", "multi", "cut_in cut_out lead_vehicle_braking vehicle_merging_ahead vehicle_crossing_path oncoming_encroachment parked_vehicle_pull_out door_opening vehicle_yielding vehicle_not_yielding being_overtaken none", "fusion vlm", "openai_compatible deterministic", "Temporal interaction with another vehicle.", temporal_scope="event", none_semantics="The interval was observable and no listed vehicle interaction occurred."),
    _definition("event.vru.interaction", "multi", "pedestrian_crossing pedestrian_entering_road pedestrian_waiting_to_cross pedestrian_walking_along_road cyclist_crossing cyclist_merging vru_sudden_emergence occluded_vru_emergence vru_yielding vru_not_yielding none", "fusion vlm", "openai_compatible deterministic", "Temporal interaction with a vulnerable road user.", temporal_scope="event", none_semantics="The interval was observable and no listed VRU interaction occurred."),
    _definition("event.traffic_control.response", "single", "stop_at_red proceed_on_green stop_at_stop_sign yield_at_yield_sign stop_for_crosswalk stop_at_rail_crossing follow_traffic_officer no_response_required", "fusion map_route vlm gnss_ins", "deterministic openai_compatible", "Actual ego response to an applicable control.", temporal_scope="event"),
    _definition("event.right_of_way", "single", "ego_has_priority other_has_priority ambiguous_priority not_applicable", "fusion map_route vlm", "deterministic bedrock_claude openai_compatible", "Right-of-way state during an interaction.", temporal_scope="event"),
    _definition("event.hazard.type", "multi", "obstacle_on_road debris_on_road blocked_lane wrong_way_vehicle emergency_vehicle_approach collision none", "vlm fusion", "openai_compatible deterministic", "Observed roadway hazard.", temporal_scope="event", none_semantics="The road corridor was observable and no listed hazard occurred."),
    _definition("event.hazard.response", "single", "none slow_down stop obstacle_avoidance lane_change_avoidance yield", "fusion gnss_ins vlm", "deterministic openai_compatible", "Ego response caused by a valid hazard.", temporal_scope="event"),
    _definition("event.traffic_flow", "multi", "congestion_entry congestion_exit queue_entry queue_exit stop_and_go road_closure_encounter workzone_entry workzone_exit none", "fusion gnss_ins vlm map_route", "deterministic openai_compatible", "Transitions in traffic-flow state.", temporal_scope="event", none_semantics="The interval was observable and no listed flow transition occurred."),
    _definition("event.interaction.actor", "multi", "vehicle pedestrian cyclist motorcycle emergency_vehicle animal static_obstacle none", "fusion vlm", "openai_compatible deterministic", "Actor classes participating in an event.", temporal_scope="event", none_semantics="A valid event was observed without a listed participating actor."),
    _definition("event.outcome", "single", "normal_completion interrupted hazard_avoided unresolved collision", "fusion", "deterministic", "Observed resolution of an event.", temporal_scope="event"),
    _definition("event.phase", "single", "onset active resolution", "fusion", "deterministic", "Phase subinterval inside an event instance.", temporal_scope="event_phase"),
    _definition("perception.occlusion.source", "multi", "static_object dynamic_object ego_body weather none", "vlm fusion", "openai_compatible deterministic", "Source of scene or actor occlusion.", subject="camera_or_actor", none_semantics="The target or scene was observable with no supported occlusion source."),
    _definition("perception.occlusion.level", "single", "none partial major full", "vlm fusion", "openai_compatible deterministic", "Severity of scene or actor occlusion.", subject="camera_or_actor"),
    _definition("perception.object.visibility", "single", "fully_visible partially_visible barely_visible not_visible", "vlm fusion", "openai_compatible deterministic", "Visibility of a tracked or visual actor.", subject="actor"),
    _definition("perception.object.scale", "single", "normal small very_small", "fusion vlm", "deterministic openai_compatible", "Image scale of an actor.", subject="actor"),
    _definition("perception.object.range", "single", "near mid far very_far", "fusion vlm", "deterministic openai_compatible", "Range category of an actor.", subject="actor"),
    _definition("perception.fov.state", "single", "centered edge_of_fov truncated entering_fov leaving_fov", "fusion vlm", "deterministic openai_compatible", "Actor position and motion relative to camera field of view.", subject="actor_camera"),
    _definition("perception.scene.clutter", "single", "low medium high", "vlm image_qc", "openai_compatible deterministic", "Semantic and visual clutter of the observable scene.", subject="camera"),
    _definition("perception.object.overlap", "single", "none moderate heavy", "fusion vlm", "deterministic openai_compatible", "Overlap of an actor with nearer scene content.", subject="actor_camera"),
    _definition("perception.visual.contrast", "single", "normal low_contrast silhouette", "image_qc vlm", "deterministic openai_compatible", "Contrast affecting perception.", subject="camera"),
    _definition("perception.visual.lighting", "multi", "normal backlit deep_shadow high_dynamic_range tunnel_transition", "image_qc vlm", "deterministic openai_compatible", "Lighting conditions affecting camera perception.", subject="camera"),
    _definition("perception.visual.glare", "multi", "sun headlight wet_road_reflection none", "image_qc vlm", "deterministic openai_compatible", "Glare affecting a camera view.", subject="camera", none_semantics="The camera view is usable and contains no supported glare source."),
    _definition("perception.image.exposure", "single", "normal overexposed underexposed mixed", "image_qc", "deterministic", "Image exposure condition.", subject="camera"),
    _definition("perception.image.blur", "single", "none motion_blur defocus_blur", "image_qc vlm", "deterministic openai_compatible", "Image blur condition.", subject="camera"),
    _definition("perception.image.weather_artifact", "multi", "rain_streak snow_streak water_spray fog_or_haze none", "image_qc vlm", "deterministic openai_compatible", "Weather artifacts visible in an image.", subject="camera", none_semantics="Usable image content contains no listed weather artifact."),
    _definition("perception.image.lens_contamination", "multi", "water_droplet dirt mud condensation none", "image_qc vlm", "deterministic openai_compatible", "Camera-fixed lens contamination.", subject="camera", none_semantics="The lens was observable over time and no listed contamination is present."),
    _definition("perception.image.frame_status", "single", "normal partial_obstruction full_obstruction black_frame frozen_frame dropped_frame corrupted_frame", "image_qc", "deterministic", "Decode, timing, and obstruction state of a camera frame.", subject="camera"),
    _definition("perception.object.appearance", "multi", "normal unusual_object unusual_pose temporary_object ambiguous_class deceptive_appearance", "vlm", "openai_compatible", "Appearance conditions that make an actor difficult to recognize.", subject="actor"),
    _definition("perception.map_element_condition", "single", "clear occluded faded temporary_conflict visually_missing", "fusion map_route vlm", "deterministic openai_compatible", "Visual condition of an expected mapped element.", subject="map_element"),
    _definition("perception.scene.complexity", "single", "simple moderate complex extreme", "vlm fusion", "openai_compatible deterministic", "Combined topology, actor, control, and visibility complexity."),
    _definition("perception.mixed_traffic", "single", "absent present", "vlm fusion", "openai_compatible deterministic", "Co-presence of heterogeneous motorized and vulnerable traffic."),
    _definition("perception.temporary_traffic_control", "single", "absent present", "vlm map_route fusion", "openai_compatible deterministic", "Visible temporary traffic-control elements."),
)


ONTOLOGY = {definition.key: definition for definition in _DEFINITIONS}


def _validate_registry(definitions: Iterable[LabelDefinition]) -> None:
    seen: set[str] = set()
    for definition in definitions:
        if definition.key in seen:
            raise ValueError(f"duplicate ontology key: {definition.key}")
        seen.add(definition.key)
        if definition.namespace not in {"odd", "event", "perception"}:
            raise ValueError(f"invalid ontology namespace: {definition.key}")
        if definition.cardinality not in {"single", "multi"}:
            raise ValueError(f"invalid cardinality: {definition.key}")
        if not definition.values or len(set(definition.values)) != len(definition.values):
            raise ValueError(f"invalid values: {definition.key}")
        if not set(definition.primary_sources).issubset(LABEL_SOURCES):
            raise ValueError(f"invalid sources: {definition.key}")
        if "none" in definition.values and definition.cardinality == "multi":
            if definition.none_semantics is None:
                raise ValueError(f"multi-select none needs semantics: {definition.key}")


_validate_registry(_DEFINITIONS)


def ontology_document() -> dict[str, Any]:
    labels = [definition.to_dict() for definition in _DEFINITIONS]
    body = {
        "schema_version": "odd_ontology_registry_v1",
        "ontology_version": ONTOLOGY_VERSION,
        "statuses": list(LABEL_STATUSES),
        "sources": list(LABEL_SOURCES),
        "labels": labels,
    }
    body["ontology_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def ontology_sha256() -> str:
    return str(ontology_document()["ontology_sha256"])


def validate_values(key: str, values: Iterable[str]) -> tuple[str, ...]:
    definition = ONTOLOGY.get(key)
    if definition is None:
        raise ValueError(f"unknown ontology key: {key}")
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError(f"valid observation requires a value: {key}")
    unknown = set(normalized) - set(definition.values)
    if unknown:
        raise ValueError(f"unknown values for {key}: {sorted(unknown)}")
    if definition.cardinality == "single" and len(normalized) != 1:
        raise ValueError(f"single-select label has {len(normalized)} values: {key}")
    if "none" in normalized and len(normalized) != 1:
        raise ValueError(f"none cannot coexist with another value: {key}")
    if "normal" in normalized and len(normalized) != 1:
        raise ValueError(f"normal cannot coexist with an abnormal value: {key}")
    return normalized
