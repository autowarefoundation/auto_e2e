"""VideoLLaMA3 teacher backend (issue #98) — second opinion for agreement.

VideoLLaMA3 (Apache-2.0, ``DAMO-NLP-SG/VideoLLaMA3-7B``; arXiv:2501.13106) is
the video-native second teacher in the two-teacher agreement pipeline: it
ingests the 1 Hz frame window natively (fps=1 sampling), giving an opinion
that is independent from Qwen2-VL's.  Labels are fused with
:class:`~.multi_teacher.MultiTeacher`, whose agreement fraction doubles as the
confidence signal (#103).

Like every teacher here it is TRAIN-ONLY and never runs in the vehicle.

Status: the prompt/parse layer is shared with the Qwen backend
(:func:`~.qwen2vl.build_scenario_prompt` / ``parse_scenario_response``); the
checkpoint-specific generation call is wired when the labelling pipeline runs
on GPU hardware (it follows the model card's ``transformers`` usage).  Until
then :meth:`label` raises ``NotImplementedError`` with a clear message rather
than risking a silently wrong integration.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy
from .base import VLMTeacher, ReasoningTargets


class VideoLlama3Teacher(VLMTeacher):
    """Offline scenario autolabeller backed by VideoLLaMA3 (video-native).

    Args:
        taxonomy: label registry.  Defaults to :data:`DEFAULT_TAXONOMY`.
        model_name: HuggingFace model identifier (default
            ``"DAMO-NLP-SG/VideoLLaMA3-7B"``).
        device: torch device string for the VLM (default ``"cuda"``).
    """

    def __init__(
        self,
        taxonomy: Optional[ScenarioTaxonomy] = None,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        device: str = "cuda",
    ) -> None:
        super().__init__(taxonomy)
        self.model_name = model_name
        self.device = device
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def label(
        self,
        frames: Sequence[torch.Tensor],
        num_future_horizons: int = 4,
    ) -> ReasoningTargets:
        """Generate targets from the 1 Hz frame window (video-native).

        Raises:
            NotImplementedError: the checkpoint-specific generation call is
                wired together with the GPU labelling pipeline; use
                :class:`~.deterministic.DeterministicTeacher` in CI and
                :class:`~.qwen2vl.Qwen2VLTeacher` as the working backend.
        """
        raise NotImplementedError(
            "VideoLlama3Teacher's generation call is wired with the GPU "
            "labelling pipeline (the prompt/parse layer is shared with the "
            "Qwen2-VL backend). Use DeterministicTeacher for CI."
        )
