from __future__ import annotations

import json
from typing import Any

import numpy as np

from data_processing.odd_labeling.image_qc import CameraAnchor, CameraFrame
from data_processing.odd_labeling.ontology import ONTOLOGY
from data_processing.odd_labeling.openai_compatible import (
    OpenAICompatibleRoadObserver,
    ROAD_VLM_TASK_BUNDLES,
    RoadVLMConfig,
    derive_visual_trigger_timestamps,
    label_visual_scene,
    road_vlm_decoding_bundle_sha256,
    road_vlm_prompt_bundle_sha256,
)
from data_processing.odd_labeling.schema import make_observation


def _frame() -> CameraFrame:
    return CameraFrame(
        frame_index=4,
        camera_index=0,
        camera_role="front_center",
        timestamp_ns=1_000,
        jpeg=b"test-jpeg",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _frame_at(timestamp_ns: int, frame_index: int) -> CameraFrame:
    return CameraFrame(
        frame_index=frame_index,
        camera_index=0,
        camera_role="front_center",
        timestamp_ns=timestamp_ns,
        jpeg=f"jpeg-{frame_index}".encode(),
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _frame_role_at(
    timestamp_ns: int,
    frame_index: int,
    camera_index: int,
    camera_role: str,
) -> CameraFrame:
    return CameraFrame(
        frame_index=frame_index,
        camera_index=camera_index,
        camera_role=camera_role,
        timestamp_ns=timestamp_ns,
        jpeg=f"jpeg-{frame_index}-{camera_role}".encode(),
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _completion(observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"observations": observations})
                }
            }
        ]
    }


def test_road_vlm_semantic_hashes_are_stable() -> None:
    assert road_vlm_prompt_bundle_sha256() == (
        "1dc8231f9b9698e3412b725d6d7013a35ce94f6638f56cbc9a447726140b85ad"
    )
    assert road_vlm_decoding_bundle_sha256(max_tokens=4096) == (
        "0627d339b8fdc715ea1a9115d133723b4b52940f9885debf0daf6bfd38f2d500"
    )
    assert road_vlm_decoding_bundle_sha256(
        max_tokens=2048
    ) != road_vlm_decoding_bundle_sha256(max_tokens=4096)


def test_observer_uses_openai_contract_and_validates_output() -> None:
    requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        requests.append((url, payload, headers))
        return _completion(
            {
                "odd.environment.sky": {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "confidence": 0.91,
                    "camera_id": None,
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "Visible blue sky.",
                },
                "event.hazard.type": {
                    "key": "event.hazard.type",
                    "status": "valid",
                    "values": ["none"],
                    "confidence": 0.82,
                    "camera_id": None,
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "The road corridor is visible.",
                },
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1/",
            model="road-observer",
            api_key="secret",
            retry_count=0,
        ),
        transport=transport,
    )
    observations = observer.observe(
        scene_uid="scene-1",
        task_bundle="contract_test",
        keys=("odd.environment.sky", "event.hazard.type"),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        sampling_parameters={
            "regular_interval_s": 1.0,
            "trigger_context_s": 1.0,
        },
    )

    assert [item.values for item in observations] == [("clear",), ("none",)]
    assert [item.source for item in observations] == ["vlm", "vlm"]
    assert all(
        len(item.provenance["prompt_sha256"]) == 64
        and len(item.provenance["decoding_config_sha256"]) == 64
        and len(item.provenance["request_sha256"]) == 64
        and len(item.provenance["response_sha256"]) == 64
        and item.provenance["input_start_timestamp_ns"] == 1_000
        and item.provenance["input_end_timestamp_ns"] == 1_000
        and item.provenance["lookback_ns"] == 0
        and item.provenance["lookahead_ns"] == 0
        and item.provenance["subject_scope"] == "scene"
        and item.provenance["inference_pass"] == "primary"
        and item.provenance["supporting_timestamps_ns"] == [1_000]
        and item.provenance["sampling_parameters"] == {
            "regular_interval_s": 1.0,
            "trigger_context_s": 1.0,
        }
        for item in observations
    )
    assert requests[0][0] == "https://road-vlm.example/v1/chat/completions"
    assert requests[0][2]["Authorization"] == "Bearer secret"
    request = requests[0][1]
    assert request["model"] == "road-observer"
    assert request["temperature"] == 0.0
    assert request["response_format"]["type"] == "json_schema"
    observation_schema = request["response_format"]["json_schema"]["schema"][
        "properties"
    ]["observations"]
    assert observation_schema["required"] == [
        "odd.environment.sky",
        "event.hazard.type",
    ]
    assert request["messages"][1]["content"][0]["text"].find(
        '"subject_scope":"scene"'
    ) > 0
    assert '"regular_interval_s":1.0' in (
        request["messages"][1]["content"][0]["text"]
    )
    item_variants = observation_schema["properties"]["odd.environment.sky"][
        "oneOf"
    ]
    sky_valid = next(
        variant
        for variant in item_variants
        if variant["properties"]["key"].get("const") == "odd.environment.sky"
        and variant["properties"]["status"].get("const") == "valid"
    )
    sky_missing = next(
        variant
        for variant in item_variants
        if variant["properties"]["key"].get("const") == "odd.environment.sky"
        and "enum" in variant["properties"]["status"]
    )
    assert sky_valid["properties"]["values"] == {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {
            "type": "string",
            "enum": ["clear", "partly_cloudy", "overcast"],
        },
    }
    assert sky_missing["properties"]["values"]["maxItems"] == 0
    assert "camera_id" in sky_valid["required"]
    assert sky_valid["properties"]["camera_id"] == {"type": "null"}
    assert "supporting_timestamps_ns" in sky_valid["required"]
    assert (
        request["messages"][1]["content"][1]["image_url"]["url"]
        == "data:image/jpeg;base64,dGVzdC1qcGVn"
    )
    system_prompt = request["messages"][0]["content"]
    assert "near collision" in system_prompt
    assert "single image" in system_prompt
    assert "reason names or describes an allowed candidate" in system_prompt
    assert "status MUST be valid" in system_prompt
    exchanges = observer.provider_exchanges
    assert len(exchanges) == 1
    assert exchanges[0].backend == "ORV"
    assert exchanges[0].status == "succeeded"
    assert exchanges[0].raw_response == _completion(
        {
            "odd.environment.sky": {
                "key": "odd.environment.sky",
                "status": "valid",
                "values": ["clear"],
                "confidence": 0.91,
                "camera_id": None,
                "supporting_cameras": ["front_center"],
                "supporting_timestamps_ns": [1_000],
                "reason": "Visible blue sky.",
            },
            "event.hazard.type": {
                "key": "event.hazard.type",
                "status": "valid",
                "values": ["none"],
                "confidence": 0.82,
                "camera_id": None,
                "supporting_cameras": ["front_center"],
                "supporting_timestamps_ns": [1_000],
                "reason": "The road corridor is visible.",
            },
        }
    )
    request_metadata = json.dumps(
        exchanges[0].request_metadata,
        sort_keys=True,
    )
    assert "data:image" not in request_metadata
    assert "road-vlm.example" not in request_metadata
    assert "secret" not in request_metadata
    assert exchanges[0].request_metadata["frames"] == [
        {
            "camera_role": "front_center",
            "jpeg_sha256": (
                "95addac620cbcf40dbfcbf5b32d76c58"
                "c4f6c57cfe6d830590141c16db533830"
            ),
            "timestamp_ns": 1_000,
        }
    ]


def test_scene_protocol_repairs_are_bounded_and_audited() -> None:
    raw_response = _completion(
        {
            "event.hazard.type": {
                "key": "event.hazard.type",
                "status": "valid",
                "values": ["obstacle_on_road", "none"],
                "confidence": 0.88,
                "camera_id": "front_center",
                "supporting_cameras": ["front_center"],
                "supporting_timestamps_ns": [1_000],
                "reason": "An obstacle is visible in the road.",
            }
        }
    )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=lambda _url, _payload, _headers: raw_response,
    )
    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="interaction",
        keys=("event.hazard.type",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )[0]

    expected_repairs = [
        {
            "kind": "scene_camera_id_to_null",
            "key": "event.hazard.type",
            "before": "front_center",
            "after": None,
        },
        {
            "kind": "exclusive_neutral_removed",
            "key": "event.hazard.type",
            "before": ["obstacle_on_road", "none"],
            "after": ["obstacle_on_road"],
        },
    ]
    assert observation.status == "valid"
    assert observation.values == ("obstacle_on_road",)
    assert observation.camera_id is None
    assert observation.provenance["protocol_repairs"] == expected_repairs
    exchange = observer.provider_exchanges[0]
    assert exchange.status == "succeeded"
    assert list(exchange.protocol_repairs) == expected_repairs
    assert exchange.raw_response == raw_response
    assert exchange.schema_version == "odd_provider_exchange_v2"


def test_normal_is_removed_only_when_an_abnormal_value_is_valid() -> None:
    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=lambda _url, _payload, _headers: _completion(
            {
                "perception.visual.lighting": {
                    "key": "perception.visual.lighting",
                    "status": "valid",
                    "values": ["normal", "backlit"],
                    "confidence": 0.9,
                    "camera_id": "front_center",
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "The subject is backlit.",
                }
            }
        ),
    )

    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="perception_condition",
        keys=("perception.visual.lighting",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        target_camera_id="front_center",
    )[0]

    assert observation.values == ("backlit",)
    assert observation.provenance["protocol_repairs"] == [
        {
            "kind": "exclusive_neutral_removed",
            "key": "perception.visual.lighting",
            "before": ["normal", "backlit"],
            "after": ["backlit"],
        }
    ]


def test_scene_camera_identity_is_not_repaired_without_matching_evidence() -> None:
    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=lambda _url, _payload, _headers: _completion(
            {
                "odd.environment.sky": {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "confidence": 0.9,
                    "camera_id": "rear",
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "Clear sky is visible.",
                }
            }
        ),
    )

    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="environment",
        keys=("odd.environment.sky",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )[0]

    assert observation.status == "unavailable"
    assert observation.provenance["protocol_repairs"] == []
    assert observer.provider_exchanges[0].status == "invalid_response"
    assert observer.provider_exchanges[0].protocol_repairs == ()


def test_invalid_responses_exhaust_retries_and_abstain() -> None:
    calls = 0

    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _completion(
            {
                "odd.environment.sky": {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["tornado"],
                    "confidence": 1.0,
                    "camera_id": None,
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "Invalid candidate.",
                }
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=1,
        ),
        transport=transport,
    )
    observations = observer.observe(
        scene_uid="scene-1",
        task_bundle="contract_test",
        keys=("odd.environment.sky",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )

    assert calls == 2
    assert len(observations) == 1
    assert observations[0].status == "unavailable"
    assert observations[0].values == ()
    assert observations[0].confidence == 0.0
    assert observations[0].provenance["error_type"] == "ValueError"
    assert observations[0].provenance["attempt"] == 2
    assert len(observations[0].provenance["prompt_sha256"]) == 64
    assert len(observations[0].provenance["decoding_config_sha256"]) == 64
    assert len(observations[0].provenance["request_sha256"]) == 64
    assert [exchange.status for exchange in observer.provider_exchanges] == [
        "invalid_response",
        "invalid_response",
    ]
    assert all(
        exchange.raw_response is not None
        and exchange.error_type == "ValueError"
        for exchange in observer.provider_exchanges
    )


def test_visual_frame_status_cannot_claim_decoder_or_timing_failures() -> None:
    requests: list[dict[str, Any]] = []

    def transport(
        _url: str,
        payload: dict[str, Any],
        _headers: dict[str, str],
    ) -> dict[str, Any]:
        requests.append(payload)
        return _completion(
            {
                "perception.image.frame_status": {
                    "key": "perception.image.frame_status",
                    "status": "valid",
                    "values": ["frozen_frame"],
                    "confidence": 0.99,
                    "camera_id": "front_center",
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "The visible frame appears unchanged.",
                }
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=transport,
    )
    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="perception_condition",
        keys=("perception.image.frame_status",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        target_camera_id="front_center",
    )[0]

    assert observation.status == "unavailable"
    assert observation.values == ()
    assert observer.provider_exchanges[0].status == "invalid_response"
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    variants = schema["properties"]["observations"]["properties"][
        "perception.image.frame_status"
    ]["oneOf"]
    valid = next(
        variant
        for variant in variants
        if variant["properties"]["status"].get("const") == "valid"
    )
    assert valid["properties"]["values"]["items"]["enum"] == [
        "normal",
        "partial_obstruction",
        "full_obstruction",
    ]


def test_task_bundles_are_small_and_scope_camera_conditions() -> None:
    assert [bundle.name for bundle in ROAD_VLM_TASK_BUNDLES] == [
        "road_environment",
        "traffic_dynamic",
        "temporal_event",
        "forward_perception",
    ]
    requested_keys = [
        key
        for bundle in ROAD_VLM_TASK_BUNDLES
        for key in bundle.scene_keys + bundle.camera_keys
    ]
    assert len(requested_keys) == len(set(requested_keys))
    perception = ROAD_VLM_TASK_BUNDLES[-1]
    assert "perception.image.blur" in perception.camera_keys
    assert "perception.image.frame_status" in perception.camera_keys
    assert "perception.image.lens_contamination" in perception.camera_keys
    assert not any(
        key.startswith("perception.object.")
        for key in requested_keys
    )
    by_name = {bundle.name: bundle for bundle in ROAD_VLM_TASK_BUNDLES}
    assert by_name["temporal_event"].scene_camera_roles == ("front_center",)
    assert by_name["temporal_event"].temporal_mode == "event"
    assert by_name["temporal_event"].trigger_only
    assert by_name["traffic_dynamic"].scene_camera_roles == ("front_center",)
    assert by_name["forward_perception"].camera_roles == ("front_center",)


def test_camera_scoped_observation_has_camera_identity() -> None:
    requests: list[dict[str, Any]] = []

    def transport(
        _url: str,
        payload: dict[str, Any],
        _headers: dict[str, str],
    ) -> dict[str, Any]:
        requests.append(payload)
        return _completion(
            {
                "perception.image.blur": {
                    "key": "perception.image.blur",
                    "status": "valid",
                    "values": ["none"],
                    "confidence": 0.93,
                    "camera_id": "front_center",
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "Lane edges remain sharp.",
                }
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=transport,
    )
    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="perception_condition",
        keys=("perception.image.blur",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        target_camera_id="front_center",
    )[0]

    assert observation.camera_id == "front_center"
    assert observation.provenance["subject_scope"] == "camera"
    prompt = requests[0]["messages"][1]["content"][0]["text"]
    assert '"subject_camera_id":"front_center"' in prompt
    assert '"subject_scope":"camera"' in prompt
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    variants = schema["properties"]["observations"]["properties"][
        "perception.image.blur"
    ]["oneOf"]
    assert all(
        variant["properties"]["camera_id"] == {"type": "string"}
        for variant in variants
    )


def test_unknown_frame_citation_is_rejected() -> None:
    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
    ) -> dict[str, Any]:
        return _completion(
            {
                "odd.environment.sky": {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "confidence": 0.9,
                    "camera_id": None,
                    "supporting_cameras": ["rear"],
                    "supporting_timestamps_ns": [999_999],
                    "reason": "Unsupported citation.",
                }
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=transport,
    )
    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="environment",
        keys=("odd.environment.sky",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )[0]

    assert observation.status == "unavailable"
    assert observation.provenance["error_type"] == "ValueError"


def test_interval_end_citation_is_removed_when_input_frame_is_also_cited() -> None:
    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
    ) -> dict[str, Any]:
        return _completion(
            {
                "odd.environment.sky": {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "confidence": 0.9,
                    "camera_id": None,
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000, 2_000],
                    "reason": "The input frame shows clear sky.",
                }
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=transport,
    )
    observation = observer.observe(
        scene_uid="scene-1",
        task_bundle="environment",
        keys=("odd.environment.sky",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )[0]

    assert observation.status == "valid"
    assert observation.provenance["supporting_timestamps_ns"] == [1_000]
    assert observation.provenance["protocol_repairs"] == [
        {
            "kind": "unsupported_timestamps_removed",
            "key": "odd.environment.sky",
            "before": [1_000, 2_000],
            "after": [1_000],
        }
    ]


def test_refinement_pass_has_distinct_evidence_identity() -> None:
    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
    ) -> dict[str, Any]:
        return _completion(
            {
                "odd.environment.sky": {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["overcast"],
                    "confidence": 0.6,
                    "camera_id": None,
                    "supporting_cameras": ["front_center"],
                    "supporting_timestamps_ns": [1_000],
                    "reason": "Cloud cover is visible.",
                }
            }
        )

    observer = OpenAICompatibleRoadObserver(
        RoadVLMConfig(
            base_url="https://road-vlm.example/v1",
            model="road-observer",
            retry_count=0,
        ),
        transport=transport,
    )
    primary = observer.observe(
        scene_uid="scene-1",
        task_bundle="environment",
        keys=("odd.environment.sky",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )[0]
    refinement = observer.observe(
        scene_uid="scene-1",
        task_bundle="environment",
        keys=("odd.environment.sky",),
        frames=(_frame(),),
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        inference_pass="refinement",
        refinement_reasons={"odd.environment.sky": "low_confidence"},
    )[0]

    assert primary.values == refinement.values
    assert primary.observation_uid != refinement.observation_uid
    assert refinement.provenance["inference_pass"] == "refinement"
    assert refinement.provenance["refinement_reasons"] == {
        "odd.environment.sky": "low_confidence"
    }


def test_visual_trigger_timestamps_cover_transitions_and_events() -> None:
    route_before = make_observation(
        scene_uid="scene-1",
        key="odd.route.action",
        status="valid",
        values=("lane_follow",),
        confidence=1.0,
        source="map_route",
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
    )
    route_turn = make_observation(
        scene_uid="scene-1",
        key="odd.route.action",
        status="valid",
        values=("turn_left",),
        confidence=1.0,
        source="map_route",
        start_timestamp_ns=2_000,
        end_timestamp_ns=3_000,
    )
    route_turn_continues = make_observation(
        scene_uid="scene-1",
        key="odd.route.action",
        status="valid",
        values=("turn_left",),
        confidence=1.0,
        source="map_route",
        start_timestamp_ns=3_000,
        end_timestamp_ns=4_000,
    )
    hard_brake = make_observation(
        scene_uid="scene-1",
        key="event.ego.strong_response",
        status="valid",
        values=("hard_brake",),
        confidence=0.9,
        source="gnss_ins",
        start_timestamp_ns=4_000,
        end_timestamp_ns=5_000,
    )
    normal_frame = make_observation(
        scene_uid="scene-1",
        key="perception.image.frame_status",
        status="valid",
        values=("normal",),
        confidence=0.98,
        source="image_qc",
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        camera_id="front_center",
    )

    assert derive_visual_trigger_timestamps(
        (route_before, route_turn, route_turn_continues),
        (hard_brake,),
        (normal_frame,),
    ) == (2_000, 4_000, 4_999)


def test_unavailable_sampled_camera_gap_is_not_a_visual_trigger() -> None:
    observations = (
        make_observation(
            scene_uid="scene-1",
            key="perception.image.frame_status",
            status="valid",
            values=("normal",),
            confidence=0.98,
            source="image_qc",
            start_timestamp_ns=1_000,
            end_timestamp_ns=2_000,
            camera_id="front_center",
        ),
        make_observation(
            scene_uid="scene-1",
            key="perception.image.frame_status",
            status="unavailable",
            confidence=1.0,
            source="image_qc",
            start_timestamp_ns=2_000,
            end_timestamp_ns=3_000,
            camera_id="front_center",
        ),
        make_observation(
            scene_uid="scene-1",
            key="perception.image.frame_status",
            status="valid",
            values=("normal",),
            confidence=0.98,
            source="image_qc",
            start_timestamp_ns=3_000,
            end_timestamp_ns=4_000,
            camera_id="front_center",
        ),
    )

    assert derive_visual_trigger_timestamps(observations) == ()


def test_visual_scene_uses_focused_alternate_frames() -> None:
    calls: list[dict[str, Any]] = []

    class RecordingObserver:
        def observe(self, **kwargs: Any) -> tuple[Any, ...]:
            calls.append(kwargs)
            output = []
            for key in kwargs["keys"]:
                forced_ambiguity = (
                    key == "odd.environment.sky"
                    and kwargs["start_timestamp_ns"] == 2_000
                    and kwargs.get("inference_pass", "primary") == "primary"
                )
                values = ()
                if not forced_ambiguity:
                    candidates = ONTOLOGY[key].values
                    values = (
                        next(
                            (
                                value
                                for value in (
                                    "none",
                                    "none_visible",
                                    "absent",
                                    "not_applicable",
                                    "no_response_required",
                                    "normal",
                                )
                                if value in candidates
                            ),
                            candidates[0],
                        ),
                    )
                output.append(
                    make_observation(
                        scene_uid=kwargs["scene_uid"],
                        key=key,
                        status="ambiguous" if forced_ambiguity else "valid",
                        values=values,
                        confidence=0.9,
                        source="vlm",
                        start_timestamp_ns=kwargs["start_timestamp_ns"],
                        end_timestamp_ns=kwargs["end_timestamp_ns"],
                        camera_id=kwargs.get("target_camera_id"),
                    )
                )
            return tuple(output)

    anchors = tuple(
        CameraAnchor(
            timestamp_ns=timestamp_ns,
            frames=(_frame_at(timestamp_ns, index),),
        )
        for index, timestamp_ns in enumerate((1_000, 2_000, 3_000))
    )
    observations = label_visual_scene(
        RecordingObserver(),  # type: ignore[arg-type]
        scene_uid="scene-1",
        scene_end_timestamp_ns=4_000,
        anchors=anchors,
    )

    primary_bundles = {
        call["task_bundle"]
        for call in calls
        if call.get("inference_pass", "primary") == "primary"
    }
    assert primary_bundles == {
        "road_environment",
        "traffic_dynamic",
        "forward_perception",
    }
    refinement = [
        call
        for call in calls
        if call.get("inference_pass") == "refinement"
    ]
    assert len(refinement) == 1
    assert refinement[0]["keys"] == ("odd.environment.sky",)
    assert [
        frame.timestamp_ns for frame in refinement[0]["frames"]
    ] == [1_000, 2_000, 3_000]
    assert refinement[0]["refinement_reasons"] == {
        "odd.environment.sky": "ambiguous"
    }
    camera_observations = [
        observation
        for observation in observations
        if observation.key in ROAD_VLM_TASK_BUNDLES[-1].camera_keys
    ]
    assert camera_observations
    assert {
        observation.camera_id for observation in camera_observations
    } == {"front_center"}
    assert len(calls) == 10


def test_visual_scene_uses_front_view_and_triggered_temporal_event() -> None:
    calls: list[dict[str, Any]] = []

    class RecordingObserver:
        def observe(self, **kwargs: Any) -> tuple[Any, ...]:
            calls.append(kwargs)
            output = []
            for key in kwargs["keys"]:
                candidates = ONTOLOGY[key].values
                value = next(
                    (
                        neutral
                        for neutral in (
                            "none",
                            "none_visible",
                            "absent",
                            "not_applicable",
                            "no_response_required",
                            "normal",
                        )
                        if neutral in candidates
                    ),
                    candidates[0],
                )
                output.append(
                    make_observation(
                        scene_uid=kwargs["scene_uid"],
                        key=key,
                        status="valid",
                        values=(value,),
                        confidence=1.0,
                        source="vlm",
                        start_timestamp_ns=kwargs["start_timestamp_ns"],
                        end_timestamp_ns=kwargs["end_timestamp_ns"],
                        camera_id=kwargs.get("target_camera_id"),
                    )
                )
            return tuple(output)

    roles = (
        "front_center",
        "front_left",
        "front_right",
        "rear",
        "rear_left",
        "rear_right",
    )
    anchors = tuple(
        CameraAnchor(
            timestamp_ns=timestamp_ns,
            frames=tuple(
                _frame_role_at(
                    timestamp_ns,
                    frame_index,
                    camera_index,
                    role,
                )
                for camera_index, role in enumerate(roles)
            ),
        )
        for frame_index, timestamp_ns in enumerate((1_000, 2_000, 3_000))
    )

    label_visual_scene(
        RecordingObserver(),  # type: ignore[arg-type]
        scene_uid="scene-1",
        scene_end_timestamp_ns=4_000,
        anchors=anchors,
        event_trigger_timestamps_ns=(2_000,),
    )

    middle_scene_calls = {
        call["task_bundle"]: call
        for call in calls
        if call["start_timestamp_ns"] == 2_000
        and call.get("target_camera_id") is None
        and call.get("inference_pass", "primary") == "primary"
    }
    interaction_frames = middle_scene_calls["temporal_event"]["frames"]
    assert {frame.camera_role for frame in interaction_frames} == {
        "front_center"
    }
    assert [frame.timestamp_ns for frame in interaction_frames] == [
        1_000,
        2_000,
        3_000,
    ]
    dynamic_frames = middle_scene_calls["traffic_dynamic"]["frames"]
    assert {frame.camera_role for frame in dynamic_frames} == {
        "front_center"
    }
    assert {frame.timestamp_ns for frame in dynamic_frames} == {2_000}
    assert {
        frame.camera_role
        for frame in middle_scene_calls["road_environment"]["frames"]
    } == {"front_center"}
    camera_calls = [
        call
        for call in calls
        if call.get("target_camera_id") is not None
        and call.get("inference_pass", "primary") == "primary"
    ]
    assert {call["target_camera_id"] for call in camera_calls} == {
        "front_center"
    }
    assert len(calls) == 10
