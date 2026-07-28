from __future__ import annotations

import json
from typing import Any

import numpy as np

from data_processing.odd_labeling.image_qc import CameraFrame
from data_processing.odd_labeling.openai_compatible import (
    OpenAICompatibleRoadObserver,
    RoadVLMConfig,
)


def _frame() -> CameraFrame:
    return CameraFrame(
        frame_index=4,
        camera_index=0,
        camera_role="front_center",
        timestamp_ns=1_000,
        jpeg=b"test-jpeg",
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
                    "supporting_cameras": ["front_center"],
                    "reason": "Visible blue sky.",
                },
                "event.hazard.type": {
                    "key": "event.hazard.type",
                    "status": "valid",
                    "values": ["none"],
                    "confidence": 0.82,
                    "supporting_cameras": ["front_center"],
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
    )

    assert [item.values for item in observations] == [("clear",), ("none",)]
    assert [item.source for item in observations] == ["vlm", "vlm"]
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
    assert (
        request["messages"][1]["content"][1]["image_url"]["url"]
        == "data:image/jpeg;base64,dGVzdC1qcGVn"
    )
    system_prompt = request["messages"][0]["content"]
    assert "near collision" in system_prompt
    assert "single image" in system_prompt
    assert "reason names or describes an allowed candidate" in system_prompt
    assert "status MUST be valid" in system_prompt


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
                    "supporting_cameras": ["front_center"],
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
