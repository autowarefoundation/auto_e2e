"""OpenAI-compatible endpoint teacher backend for reasoning-band pseudo-labels (issue #98).

Implements the *model-agnostic teacher endpoint* from @riita10069's Enhancement
Proposal in #98.  AutoE2E depends only on ``(base_url, model, prompt_version,
request schema, response schema)`` and speaks the OpenAI chat-completions API,
so the backend behind the URL can be Cosmos3-Nano on vLLM / vLLM-Omni, a Qwen
server, an external API, or a local mock — with no code change here.

Two request modes:

* ``mode="per_frame"`` — one request per frame, each horizon labelled
  independently (simple; good for image-only backends).
* ``mode="clip_horizons"`` — one request per sample with the whole clip
  (current + future frames, plus optional ego/route/map ``extra_context``),
  returning all horizons in a single JSON object.  This preserves temporal
  context and is what video-native backends (Cosmos) want.

Failure policy: ``strict=True`` (default) **raises** on endpoint/parse failure,
so a transport outage, auth error, or malformed response cannot silently poison
the dataset with all-zero labels.  ``strict=False`` abstains (empty labels) for
large best-effort jobs.

TRAIN-ONLY / OFFLINE: like every teacher, this runs during offline label
pre-extraction and is NEVER part of the vehicle inference loop.  No teacher
weights are shipped; only the endpoint URL is referenced.

Testability: the network boundary is a single injectable ``transport`` callable,
so unit tests run with a stub (no network, no GPU).  Prompt construction and
per-frame parsing are reused from :mod:`.qwen2vl` (the same closed JSON schema
over the taxonomy), so every teacher backend stays label-space-consistent.
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
    """Default transport: POST JSON to an OpenAI-compatible endpoint via urllib."""

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


def build_clip_prompt(
    taxonomy: ScenarioTaxonomy,
    num_horizons: int,
    extra_context: Optional[str] = None,
) -> str:
    """Prompt for labelling a whole clip: one JSON with a per-horizon array.

    Unlike :func:`~.qwen2vl.build_scenario_prompt` (one frame), this asks the
    backend to reason over the ordered clip and return every horizon at once,
    optionally conditioned on ego/route/map ``extra_context``.
    """
    lines = [
        f"You are labelling a short driving clip of {num_horizons} front-camera "
        "frames, ordered from the current frame (horizon 0) to the furthest "
        "future frame.",
        "For EACH horizon, list every label that applies from the categories below.",
        "Use only the exact label strings given.  If none applies, use [].",
        "",
    ]
    for group in taxonomy.groups:
        lines.append(f"- {group.name}: {', '.join(group.labels)}")
    if extra_context:
        lines += ["", "Scene context (ego state / route / map):", extra_context]
    per_group = '{"' + '": [...], "'.join(taxonomy.group_names) + '": [...]}'
    lines += [
        "",
        "Answer with ONLY a JSON object, no other text, in the form:",
        '{"horizons": [' + per_group + ", ...]}",
        f"with exactly {num_horizons} entries, in horizon order.",
    ]
    return "\n".join(lines)


def parse_clip_response(
    text: str, taxonomy: ScenarioTaxonomy, num_horizons: int
) -> List[Dict[str, List[str]]]:
    """Parse a clip response into ``num_horizons`` per-group active-label dicts.

    Tolerant like :func:`~.qwen2vl.parse_scenario_response`: chatter around the
    JSON, unknown labels, and missing horizons all degrade to empty (abstain)
    rather than raising — the endpoint's ``strict`` flag decides what to do
    with an empty/malformed answer.
    """
    empty: List[Dict[str, List[str]]] = [
        {g.name: [] for g in taxonomy.groups} for _ in range(num_horizons)
    ]
    start = text.find("{")
    if start == -1:
        return empty
    try:
        raw, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return empty
    if not isinstance(raw, dict):
        return empty
    horizons = raw.get("horizons")
    if not isinstance(horizons, list):
        return empty

    out: List[Dict[str, List[str]]] = []
    for h in range(num_horizons):
        entry = horizons[h] if h < len(horizons) and isinstance(horizons[h], dict) else {}
        parsed: Dict[str, List[str]] = {}
        for group in taxonomy.groups:
            values = entry.get(group.name, [])
            if not isinstance(values, list):
                values = []
            parsed[group.name] = [
                v for v in values if isinstance(v, str) and v in group.labels
            ]
        out.append(parsed)
    return out


class OpenAIEndpointTeacher(VLMTeacher):
    """Offline scenario autolabeller backed by any OpenAI-compatible endpoint.

    Args:
        taxonomy: label registry.  Defaults to :data:`DEFAULT_TAXONOMY`.
        base_url: OpenAI-compatible base URL (e.g. ``"http://host:8000/v1"``).
        model: model name to request (e.g. ``"cosmos3-nano"``).
        prompt_version: recorded for artifact provenance; does not change the
            request wire format.
        api_key: optional bearer token for the endpoint.
        timeout: per-request timeout in seconds (default backend only).
        max_tokens: generation budget for the JSON answer.
        transport: injectable ``(url, payload, headers) -> response`` callable.
            Defaults to a stdlib urllib POST.  Inject a stub in tests.
        mode: ``"per_frame"`` (default) or ``"clip_horizons"`` (see module docstring).
        strict: if ``True`` (default) raise on any endpoint/parse failure so the
            dataset is never silently poisoned; if ``False`` abstain (empty
            labels) for best-effort bulk jobs.  A downstream artifact layer
            should record the abstain/error provenance for filtering (follow-up).
        extra_context: optional text (ego/route/map) injected into the
            ``clip_horizons`` prompt; ignored in ``per_frame`` mode.
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
        mode: str = "per_frame",
        strict: bool = True,
        extra_context: Optional[str] = None,
    ) -> None:
        super().__init__(taxonomy)
        if mode not in ("per_frame", "clip_horizons"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose 'per_frame' or 'clip_horizons'."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_version = prompt_version
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.mode = mode
        self.strict = strict
        self.extra_context = extra_context
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

    def _payload(self, images: Sequence[torch.Tensor], prompt: str) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {"type": "image_url", "image_url": {"url": _tensor_to_data_url(img)}}
            )
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

    def _post(self, payload: Dict[str, Any]) -> Optional[str]:
        """Call the endpoint; on failure raise (strict) or return None (abstain).

        Returns the assistant text, or ``None`` only when ``strict=False`` and
        the call/response failed.  An empty/malformed response is treated as a
        failure under ``strict`` (it is exactly the silent-poisoning case).
        """
        try:
            response = self._transport(self.endpoint, payload, self._headers())
            text = _extract_content(response)
        except Exception as exc:  # noqa: BLE001 — outage / auth / transport error
            if self.strict:
                raise RuntimeError(
                    f"teacher endpoint call failed ({self.endpoint}): {exc}"
                ) from exc
            return None
        if not text and self.strict:
            raise RuntimeError(
                f"teacher endpoint returned an empty/malformed response "
                f"({self.endpoint})."
            )
        return text

    def _label_per_frame(
        self, frames: Sequence[torch.Tensor], total_horizons: int
    ) -> ReasoningTargets:
        prompt = build_scenario_prompt(self.taxonomy)
        per_horizon: List[Dict[str, torch.Tensor]] = []
        for h in range(total_horizons):
            frame = frames[h]
            per_sample: List[Dict[str, List[str]]] = []
            for img in frame:
                text = self._post(self._payload([img], prompt))
                if text is None:  # abstained (strict=False)
                    per_sample.append({g.name: [] for g in self.taxonomy.groups})
                else:
                    per_sample.append(parse_scenario_response(text, self.taxonomy))
            per_horizon.append(
                labels_to_targets(per_sample, self.taxonomy, device=frame.device)
            )
        return self._assemble(per_horizon, total_horizons)

    def _label_clip(
        self, frames: Sequence[torch.Tensor], total_horizons: int
    ) -> ReasoningTargets:
        prompt = build_clip_prompt(self.taxonomy, total_horizons, self.extra_context)
        batch = frames[0].shape[0]
        # per_horizon_samples[h] = list of per-sample label dicts at horizon h.
        per_horizon_samples: List[List[Dict[str, List[str]]]] = [
            [] for _ in range(total_horizons)
        ]
        for b in range(batch):
            images = [frames[h][b] for h in range(total_horizons)]
            text = self._post(self._payload(images, prompt))
            horizon_dicts: List[Dict[str, List[str]]]
            if text is None:  # abstained (strict=False)
                horizon_dicts = [
                    {g.name: [] for g in self.taxonomy.groups}
                    for _ in range(total_horizons)
                ]
            else:
                horizon_dicts = parse_clip_response(
                    text, self.taxonomy, total_horizons
                )
            for h in range(total_horizons):
                per_horizon_samples[h].append(horizon_dicts[h])

        per_horizon = [
            labels_to_targets(
                per_horizon_samples[h], self.taxonomy, device=frames[h].device
            )
            for h in range(total_horizons)
        ]
        return self._assemble(per_horizon, total_horizons)

    def _assemble(
        self, per_horizon: List[Dict[str, torch.Tensor]], total_horizons: int
    ) -> ReasoningTargets:
        out: ReasoningTargets = {}
        for group in self.taxonomy.groups:
            out[group.name] = [
                per_horizon[h][group.name] for h in range(total_horizons)
            ]
        return out

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
            (:class:`~.multi_teacher.MultiTeacher`), not a single teacher's
            self-report.

        Raises:
            ValueError: if fewer than ``1 + num_future_horizons`` frame
                batches are supplied.
            RuntimeError: under ``strict=True``, on any endpoint/parse failure.
        """
        total_horizons = 1 + num_future_horizons
        if len(frames) < total_horizons:
            raise ValueError(
                f"need {total_horizons} frame batches (current + "
                f"{num_future_horizons} future), got {len(frames)}."
            )
        if self.mode == "clip_horizons":
            return self._label_clip(frames, total_horizons)
        return self._label_per_frame(frames, total_horizons)
