"""OpenAI-compatible endpoint teacher backend for reasoning-band pseudo-labels (issue #98).

Implements the *model-agnostic teacher endpoint* from @riita10069's Enhancement
Proposal in #98.  AutoE2E depends only on ``(base_url, model, prompt_version,
request schema, response schema)`` and speaks the OpenAI chat-completions API,
so the backend behind the URL can be Cosmos3-Nano on vLLM / vLLM-Omni, a Qwen
server, an external API, or a local mock — with no code change here.

TRAIN-ONLY / OFFLINE: like every teacher, this runs during offline label
pre-extraction and is NEVER part of the vehicle inference loop.  No teacher
weights are shipped; only the endpoint URL is referenced, so the runtime
artifact stays lightweight and licence-clean (the Cosmos weights never enter
the repo — only its outputs, as offline labels).

Testability: the network boundary is a single injectable ``transport`` callable,
so unit tests run with a stub (no network, no GPU).  Prompt construction and
response parsing are reused verbatim from :mod:`.qwen2vl` (the same closed JSON
schema over the taxonomy), so every teacher backend stays label-space-consistent
and migrates for free when the taxonomy grows into the compositional ontology.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy
from .base import ReasoningTargets, VLMTeacher
from .qwen2vl import build_scenario_prompt, labels_to_targets, parse_scenario_response

# transport(url, payload, headers) -> parsed JSON response dict (OpenAI schema).
Transport = Callable[[str, Dict[str, Any], Dict[str, str]], Dict[str, Any]]


def _tensor_to_data_url(img: torch.Tensor) -> str:
    """Encode a ``[3, H, W]`` image tensor as a base64 PNG ``data:`` URL."""
    from torchvision.transforms.functional import to_pil_image

    pil = to_pil_image(img.detach().cpu())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _urllib_transport(timeout: float) -> Transport:
    """Default transport: POST JSON to an OpenAI-compatible endpoint via urllib.

    Uses the standard library only (no new runtime dependency); network calls
    happen offline during label pre-extraction, never at inference.
    """

    def _post(
        url: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            decoded: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return decoded

    return _post


def _extract_content(response: Dict[str, Any]) -> str:
    """Pull the assistant message text from an OpenAI chat-completion response."""
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


class OpenAIEndpointTeacher(VLMTeacher):
    """Offline scenario autolabeller backed by any OpenAI-compatible endpoint.

    The intended default backend is @riita10069's Cosmos3-Nano Reasoner served
    behind a vLLM / vLLM-Omni OpenAI-compatible endpoint, but nothing here is
    Cosmos-specific: point ``base_url`` at any compatible server (or inject a
    ``transport`` for tests / a local mock).

    Future horizons are labelled from the **future frames themselves**
    (``frames[1:]``) — privileged offline information, following the same
    leakage-prevention split as :class:`~.qwen2vl.Qwen2VLTeacher`.

    Args:
        taxonomy: label registry.  Defaults to :data:`DEFAULT_TAXONOMY`.
        base_url: OpenAI-compatible base URL (e.g. ``"http://host:8000/v1"``).
        model: model name to request (e.g. ``"cosmos3-nano"``).
        prompt_version: recorded on the teacher for artifact provenance
            (riita's ``prompt_version`` field); does not change the request.
        api_key: optional bearer token for the endpoint.
        timeout: per-request timeout in seconds (default backend only).
        max_tokens: generation budget for the JSON answer.
        transport: injectable ``(url, payload, headers) -> response`` callable.
            Defaults to a stdlib urllib POST.  Inject a stub in tests.
    """

    def __init__(
        self,
        taxonomy: Optional[ScenarioTaxonomy] = None,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str = "cosmos3-nano",
        prompt_version: str = "reasoning_label_v1",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 256,
        transport: Optional[Transport] = None,
    ) -> None:
        super().__init__(taxonomy)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_version = prompt_version
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._transport: Transport = (
            transport if transport is not None else _urllib_transport(timeout)
        )

    @property
    def endpoint(self) -> str:
        """Full chat-completions URL derived from ``base_url``."""
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, img: torch.Tensor, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _tensor_to_data_url(img)},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

    def _label_one_frame_batch(
        self, frame: torch.Tensor
    ) -> List[Dict[str, List[str]]]:
        """Label one ``[B, 3, H, W]`` frame batch → per-sample label dicts."""
        prompt = build_scenario_prompt(self.taxonomy)
        per_sample: List[Dict[str, List[str]]] = []
        for img in frame:
            try:
                payload = self._build_payload(img, prompt)
                response = self._transport(self.endpoint, payload, self._headers())
                text = _extract_content(response)
            except Exception:  # noqa: BLE001
                # Abstain (empty labels), never crash a labelling batch: at
                # scale a dropped frame is far cheaper than a crashed run.
                per_sample.append({g.name: [] for g in self.taxonomy.groups})
                continue
            per_sample.append(parse_scenario_response(text, self.taxonomy))
        return per_sample

    def label(
        self,
        frames: Sequence[torch.Tensor],
        num_future_horizons: int = 4,
    ) -> ReasoningTargets:
        """Generate multi-label scenario targets from front-camera frames.

        Args:
            frames: sequence of ``1 + num_future_horizons`` frame batches, each
                ``[B, 3, H, W]``.  ``frames[0]`` is the current frame;
                ``frames[h]`` is the +h s frame (labelled directly, offline).
            num_future_horizons: number of future horizons.

        Returns:
            :data:`ReasoningTargets` with hard ``{0, 1}`` values.  Soft
            confidence comes from cross-teacher agreement
            (:class:`~.multi_teacher.MultiTeacher`), not from a single
            teacher's self-report.

        Raises:
            ValueError: if fewer than ``1 + num_future_horizons`` frame
                batches are supplied.
        """
        total_horizons = 1 + num_future_horizons
        if len(frames) < total_horizons:
            raise ValueError(
                f"need {total_horizons} frame batches (current + "
                f"{num_future_horizons} future), got {len(frames)}."
            )

        per_horizon: List[Dict[str, torch.Tensor]] = []
        for h in range(total_horizons):
            per_sample = self._label_one_frame_batch(frames[h])
            per_horizon.append(
                labels_to_targets(per_sample, self.taxonomy, device=frames[h].device)
            )

        out: ReasoningTargets = {}
        for group in self.taxonomy.groups:
            out[group.name] = [
                per_horizon[h][group.name] for h in range(total_horizons)
            ]
        return out
