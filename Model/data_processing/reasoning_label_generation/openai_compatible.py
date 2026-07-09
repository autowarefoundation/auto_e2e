"""OpenAI-compatible teacher backend (issue #98, R2/R9).

Speaks the OpenAI chat-completions API, so the backend behind the URL can be
Cosmos3-Nano on vLLM / vLLM-Omni, a Qwen server, an external API, or a local
stub — with no code change here. AutoE2E depends only on
``base_url / model / prompt_version / request schema / response schema``.

Testability: the network boundary is a single injectable ``transport`` callable
``(url, payload, headers) -> parsed-JSON-dict``, so unit tests run with a stub
(no network, no GPU). ``strict=True`` (default) raises on endpoint/parse failure;
``strict=False`` returns an abstained record so a bad response is marked, never
silently turned into all-zero labels (R9).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Callable, Dict, List, Optional

from model_components.reasoning.reasoning_taxonomy import ReasoningTaxonomy

from .prompt_builder import build_clip_prompt, parse_clip_response, system_prompt
from .schema import ReasoningLabelRecord
from .teacher_client import TeacherClient, TeacherRequest, register_teacher

# transport(url, payload, headers) -> parsed JSON response dict (OpenAI schema).
Transport = Callable[[str, Dict[str, Any], Dict[str, str]], Dict[str, Any]]


def _tensor_to_data_url(img: Any) -> str:
    """Encode a ``[3, H, W]`` image tensor as a base64 PNG ``data:`` URL."""
    from torchvision.transforms.functional import to_pil_image

    pil = to_pil_image(img.detach().cpu())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _urllib_transport(timeout: float) -> Transport:
    """Default transport: POST JSON to an OpenAI-compatible endpoint via urllib."""

    def _post(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    return _post


def _extract_content(response: Dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


class OpenAICompatibleTeacher(TeacherClient):
    """Offline autolabeller backed by any OpenAI-compatible endpoint.

    Args:
        base_url: e.g. ``"http://localhost:8000/v1"`` (the Cosmos3-Nano vLLM PoC).
        model: model name (e.g. ``"cosmos3-nano"``).
        api_key: optional bearer token.
        timeout / max_tokens: default-transport request params.
        transport: injectable ``(url, payload, headers) -> response`` (stub in tests).
        endpoint_type: recorded in provenance (e.g. ``"vllm"``).
        extra_context: optional ego/route/map text folded into the prompt.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str = "cosmos3-nano",
        prompt_version: str = "action_relevant_reasoning_v2",
        request_mode: str = "clip_horizons",
        taxonomy: Optional[ReasoningTaxonomy] = None,
        strict: bool = True,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        transport: Optional[Transport] = None,
        endpoint_type: str = "vllm",
    ) -> None:
        # Only clip_horizons is implemented here — _payload always builds the
        # 5-horizon clip prompt. Reject per_frame rather than silently stamping
        # request_mode="per_frame" into provenance while doing clip labelling.
        if request_mode != "clip_horizons":
            raise ValueError(
                f"OpenAICompatibleTeacher only supports request_mode='clip_horizons'; "
                f"got {request_mode!r}."
            )
        super().__init__(
            provider="openai_compatible", model=model, prompt_version=prompt_version,
            request_mode=request_mode, taxonomy=taxonomy, strict=strict,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.endpoint_type = endpoint_type
        self._transport: Transport = transport or _urllib_transport(timeout)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: TeacherRequest) -> Dict[str, Any]:
        prompt = build_clip_prompt(self.taxonomy, request.extra_context)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in request.frames:
            content.append(
                {"type": "image_url", "image_url": {"url": _tensor_to_data_url(img)}}
            )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

    def label(self, request: TeacherRequest) -> ReasoningLabelRecord:
        try:
            response = self._transport(self.endpoint, self._payload(request), self._headers())
            text = _extract_content(response)
        except Exception as exc:  # noqa: BLE001 — outage / auth / transport error
            if self.strict:
                raise RuntimeError(
                    f"teacher endpoint call failed ({self.endpoint}): {exc}"
                ) from exc
            return self._abstain(request, f"transport error: {exc}")

        if not text:
            if self.strict:
                raise RuntimeError(
                    f"teacher endpoint returned an empty response ({self.endpoint})."
                )
            return self._abstain(request, "empty response")

        record = parse_clip_response(
            text, self.taxonomy,
            sample_id=request.sample_id, dataset_name=request.dataset_name,
            provider=self.provider, model=self.model,
            prompt_version=self.prompt_version, request_mode=self.request_mode,
            timestamp=request.timestamp, provenance="teacher_gt",
        )
        if record is None:
            if self.strict:
                raise RuntimeError(
                    f"teacher response could not be parsed into 5 horizons ({self.endpoint})."
                )
            return self._abstain(request, "unparseable / incomplete response")
        record.teacher_endpoint_type = self.endpoint_type
        record.dataset_version = request.dataset_version
        return record


register_teacher("openai_compatible", OpenAICompatibleTeacher)
