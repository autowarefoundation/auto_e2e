"""Qwen2-VL teacher backend for reasoning-band pseudo-label generation (issue #98).

LICENSE NOTE — IMPORTANT:
    Qwen2-VL (``Qwen/Qwen2-VL-*`` on HuggingFace) is released under the
    Apache 2.0 license.  **This teacher is a TRAIN-ONLY offline autolabeller
    and is never deployed in the vehicle inference loop.**  Users must verify
    the licence of the specific model checkpoint they use (including any
    fine-tuned variants) before using it in a project context.  The HuggingFace
    model card may impose additional terms — always check.

    Per the WG decision of 1 July 2026: the VLM teacher generates pseudo-labels
    offline (once, cached to disk); only the labelled dataset — not the VLM
    itself — is used during training.  The teacher is therefore never loaded
    on the inference device / vehicle PC.

Lazy import strategy:
    The ``transformers`` package is imported inside the label pipeline (not at
    module level), so importing the ``teachers`` package — or the full
    ``reasoning`` package — never requires heavy dependencies and CI pipelines
    without a GPU still work.  The prompt construction and response parsing
    are pure module-level functions, unit-testable without any model.

Extension point — Alpamayo CoC (v2):
    See :class:`model_components.reasoning.teachers.base.VLMTeacher` for the
    documented hook to add an Alpamayo Chain-of-Causation autolabeller in v2.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy
from .base import VLMTeacher, ReasoningTargets


def build_scenario_prompt(taxonomy: ScenarioTaxonomy) -> str:
    """Build the structured labelling prompt for one frame.

    Asks for a single JSON object whose keys are the taxonomy groups and whose
    values are lists of active labels — a closed, machine-parseable output
    space (structured labels beat free-form text for downstream decision
    accuracy and parsing cost; arXiv:2506.05442).
    """
    lines = [
        "You are labelling one front-camera frame from a driving log.",
        "For each category below, list every label that applies to the scene.",
        "Use only the exact label strings given.  If none applies, use [].",
        "",
    ]
    for group in taxonomy.groups:
        lines.append(f"- {group.name}: {', '.join(group.labels)}")
    lines += [
        "",
        "Answer with ONLY a JSON object, no other text, in the form:",
        '{"' + '": [...], "'.join(taxonomy.group_names) + '": [...]}',
    ]
    return "\n".join(lines)


def parse_scenario_response(
    text: str, taxonomy: ScenarioTaxonomy
) -> Dict[str, List[str]]:
    """Parse a model response into per-group active-label lists.

    Tolerant to chatter around the JSON (extracts the first ``{...}`` block)
    and to unknown labels (silently dropped — the closed label set is the
    contract).  An unparseable response yields empty lists for every group
    (abstain), never an exception: at labelling scale a crashed batch is worse
    than an abstained frame.
    """
    empty: Dict[str, List[str]] = {g.name: [] for g in taxonomy.groups}
    start = text.find("{")
    if start == -1:
        return empty
    try:
        # raw_decode parses the FIRST valid JSON object and ignores trailing
        # text — robust to chatter after the object (a greedy regex would
        # swallow any later '}' and fail to parse, silently dropping labels).
        raw, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return empty
    if not isinstance(raw, dict):
        return empty

    out: Dict[str, List[str]] = {}
    for group in taxonomy.groups:
        values = raw.get(group.name, [])
        if not isinstance(values, list):
            values = []
        out[group.name] = [
            v for v in values if isinstance(v, str) and v in group.labels
        ]
    return out


def labels_to_targets(
    per_sample_labels: Sequence[Dict[str, List[str]]],
    taxonomy: ScenarioTaxonomy,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Turn per-sample active-label dicts into ``[B, num_classes]`` tensors."""
    out: Dict[str, torch.Tensor] = {}
    for group in taxonomy.groups:
        target = torch.zeros(len(per_sample_labels), len(group), device=device)
        for i, labels in enumerate(per_sample_labels):
            for label in labels.get(group.name, []):
                target[i, group.index(label)] = 1.0
        out[group.name] = target
    return out


class Qwen2VLTeacher(VLMTeacher):
    """Offline scenario autolabeller backed by Qwen2-VL.

    Generates multi-label scenario targets from front-camera frame(s) using a
    Qwen2-VL model loaded from HuggingFace.  Designed for offline
    pre-extraction of pseudo-labels (piggyback on #100): run once on the full
    training split and cache the outputs.  Do NOT instantiate this class on
    the vehicle PC.

    Future horizons are labelled from the **future frames themselves**
    (``frames[1:]``) — privileged information that exists offline in the log.
    Following Alpamayo-R1's leakage-prevention split (arXiv:2511.00088), the
    future frames only decide *which label is true* at each horizon; they are
    never fed to the student.

    Args:
        taxonomy: label registry.  Defaults to :data:`DEFAULT_TAXONOMY`.
        model_name: HuggingFace model identifier (default
            ``"Qwen/Qwen2-VL-7B-Instruct"``).
        device: torch device string for the VLM (default ``"cuda"``).
        max_new_tokens: generation budget for the JSON answer (default 256).

    Raises:
        ImportError: if ``transformers`` is not installed (raised lazily on
            the first :meth:`label` call, not at import time).
    """

    def __init__(
        self,
        taxonomy: Optional[ScenarioTaxonomy] = None,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(taxonomy)
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        # Model and processor are loaded lazily on the first label() call.
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        """Lazily load the Qwen2-VL model and processor.

        Raises:
            ImportError: if ``transformers`` is not available.
        """
        if self._model is not None:
            return
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Qwen2VLTeacher requires the 'transformers' package. "
                "Install it with:  pip install transformers\n"
                "This dependency is intentionally not listed as a core "
                "requirement because the teacher is a TRAIN-ONLY offline "
                "tool that is never used at inference time."
            ) from exc

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto"
        ).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _label_one_frame_batch(self, frame: torch.Tensor) -> List[Dict[str, List[str]]]:
        """Label one ``[B, 3, H, W]`` frame batch → per-sample label dicts."""
        from torchvision.transforms.functional import to_pil_image

        assert self._model is not None and self._processor is not None
        prompt = build_scenario_prompt(self.taxonomy)

        per_sample: List[Dict[str, List[str]]] = []
        for img in frame:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[chat], images=[to_pil_image(img.cpu())], return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            text = self._processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
            per_sample.append(parse_scenario_response(text, self.taxonomy))
        return per_sample

    def label(
        self,
        frames: Sequence[torch.Tensor],
        num_future_horizons: int = 4,
    ) -> ReasoningTargets:
        """Generate multi-label scenario targets from front-camera frames.

        Args:
            frames: sequence of ``1 + num_future_horizons`` frame batches,
                each ``[B, 3, H, W]``.  ``frames[0]`` is the current frame;
                ``frames[h]`` is the +h s frame, labelled directly (privileged
                offline information).
            num_future_horizons: number of future horizons.

        Returns:
            :data:`ReasoningTargets` with hard {0, 1} values.  Soft targets /
            confidence come from cross-teacher agreement
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
        self._ensure_loaded()

        per_horizon: List[Dict[str, torch.Tensor]] = []
        for h in range(total_horizons):
            per_sample = self._label_one_frame_batch(frames[h])
            per_horizon.append(
                labels_to_targets(per_sample, self.taxonomy, device=frames[h].device)
            )

        out: ReasoningTargets = {}
        for group in self.taxonomy.groups:
            out[group.name] = [per_horizon[h][group.name] for h in range(total_horizons)]
        return out
