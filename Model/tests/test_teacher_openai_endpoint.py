"""Tests for the OpenAI-compatible endpoint teacher backend (issue #98).

Covers @riita10069's model-agnostic teacher endpoint: request construction,
per-frame and clip-horizons modes, the strict-vs-abstain failure policy, and
registry reachability.  No network / GPU — the transport is a stub.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from model_components.reasoning.scenario_taxonomy import DEFAULT_TAXONOMY
from model_components.reasoning.teachers.openai_endpoint import OpenAIEndpointTeacher

B = 2


def _frames(n: int) -> List[torch.Tensor]:
    return [torch.zeros(B, 3, 8, 8) for _ in range(n)]


def _openai_response(content: str) -> Dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class _RecordingTransport:
    """Stub transport: records each call, returns a fixed JSON content."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self, url: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        return _openai_response(self.content)


class TestPerFrameMode:
    def test_label_shapes_and_active_labels(self):
        content = (
            '{"maneuver": ["turn_left"], '
            '"edge_case": ["close_to_vru"], "weather_env": []}'
        )
        teacher = OpenAIEndpointTeacher(transport=_RecordingTransport(content))
        out = teacher.label(_frames(5), num_future_horizons=4)

        for g in DEFAULT_TAXONOMY.groups:
            assert g.name in out
            assert len(out[g.name]) == 5  # current + 4 horizons
            for tensor in out[g.name]:
                assert tensor.shape == (B, len(g))

        man = DEFAULT_TAXONOMY["maneuver"]
        assert torch.allclose(
            out["maneuver"][0][:, man.index("turn_left")], torch.ones(B)
        )
        edge = DEFAULT_TAXONOMY["edge_case"]
        assert torch.allclose(
            out["edge_case"][2][:, edge.index("close_to_vru")], torch.ones(B)
        )
        assert out["maneuver"][0][:, man.index("turn_right")].sum().item() == 0.0

    def test_request_is_openai_compatible(self):
        transport = _RecordingTransport(
            '{"maneuver": [], "edge_case": [], "weather_env": []}'
        )
        teacher = OpenAIEndpointTeacher(
            base_url="http://host:9000/v1/",
            model="cosmos3-nano",
            api_key="secret",
            transport=transport,
        )
        teacher.label(_frames(1), num_future_horizons=0)

        call = transport.calls[0]
        assert call["url"] == "http://host:9000/v1/chat/completions"
        assert call["payload"]["model"] == "cosmos3-nano"
        assert call["headers"]["Authorization"] == "Bearer secret"
        content = call["payload"]["messages"][0]["content"]
        types = [part["type"] for part in content]
        assert "text" in types and "image_url" in types
        image_part = next(p for p in content if p["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


class TestFailurePolicy:
    def test_strict_raises_on_transport_error(self):
        def boom(url: str, payload: Dict[str, Any], headers: Dict[str, str]):
            raise RuntimeError("endpoint down")

        teacher = OpenAIEndpointTeacher(transport=boom)  # strict=True default
        with pytest.raises(RuntimeError, match="teacher endpoint call failed"):
            teacher.label(_frames(1), num_future_horizons=0)

    def test_strict_raises_on_empty_response(self):
        # A malformed response (no choices) must not silently become 0-labels.
        teacher = OpenAIEndpointTeacher(
            transport=lambda url, payload, headers: {"choices": []}
        )
        with pytest.raises(RuntimeError, match="empty/malformed"):
            teacher.label(_frames(1), num_future_horizons=0)

    def test_abstain_when_not_strict(self):
        def boom(url: str, payload: Dict[str, Any], headers: Dict[str, str]):
            raise RuntimeError("endpoint down")

        teacher = OpenAIEndpointTeacher(transport=boom, strict=False)
        out = teacher.label(_frames(1), num_future_horizons=0)
        for g in DEFAULT_TAXONOMY.groups:
            assert out[g.name][0].sum().item() == 0.0


class TestClipHorizonsMode:
    def test_clip_returns_all_horizons_from_one_request(self):
        content = (
            '{"horizons": ['
            '{"maneuver": ["turn_left"], "edge_case": [], "weather_env": []}, '
            '{"maneuver": [], "edge_case": ["give_way"], "weather_env": []}'
            "]}"
        )
        transport = _RecordingTransport(content)
        teacher = OpenAIEndpointTeacher(mode="clip_horizons", transport=transport)
        out = teacher.label(_frames(2), num_future_horizons=1)  # 2 horizons

        man = DEFAULT_TAXONOMY["maneuver"]
        edge = DEFAULT_TAXONOMY["edge_case"]
        assert torch.allclose(
            out["maneuver"][0][:, man.index("turn_left")], torch.ones(B)
        )
        assert torch.allclose(
            out["edge_case"][1][:, edge.index("give_way")], torch.ones(B)
        )
        # clip mode sends ONE request per sample, each carrying every frame.
        assert len(transport.calls) == B
        parts = transport.calls[0]["payload"]["messages"][0]["content"]
        n_images = sum(1 for p in parts if p["type"] == "image_url")
        assert n_images == 2  # both horizon frames in a single request

    def test_clip_prompt_carries_extra_context(self):
        content = '{"horizons": [{"maneuver": [], "edge_case": [], "weather_env": []}]}'
        transport = _RecordingTransport(content)
        teacher = OpenAIEndpointTeacher(
            mode="clip_horizons",
            transport=transport,
            extra_context="ego: 30 km/h; route: turn right at intersection",
        )
        teacher.label(_frames(1), num_future_horizons=0)
        text = transport.calls[0]["payload"]["messages"][0]["content"][0]["text"]
        assert "route: turn right" in text

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            OpenAIEndpointTeacher(mode="nope")


class TestRegistryAndContract:
    def test_registered_in_registry(self):
        from model_components.reasoning.teachers import _TEACHER_REGISTRY

        assert "openai_endpoint" in _TEACHER_REGISTRY

    def test_requires_enough_frames(self):
        teacher = OpenAIEndpointTeacher(transport=_RecordingTransport("{}"))
        with pytest.raises(ValueError, match="frame batches"):
            teacher.label(_frames(2), num_future_horizons=4)
