"""Offline teacher-embedding production for the alignment loss (#98 §6).

``HorizonReasoningLoss`` already implements the teacher-embedding alignment term
(a cosine loss against ``teacher_embedding_targets [B, 5, D]``), but the term is
inert today: ``lambda_alignment`` defaults to 0 and **nothing in the label
pipeline produces those embeddings**.  The teacher emits structured labels plus a
free-text ``evidence`` rationale per horizon; this module turns that rationale
into the dense target the alignment loss expects.

Why it matters: the structured heads capture the action-relevant enums, but the
teacher's rationale carries softer semantics that do not fit a fixed taxonomy
("the pedestrian is slowing at the kerb and may not enter").  Aligning the
student's horizon tokens to that embedding gives the head dense supervision on
top of the discrete labels.  It is also what makes the ``student_reasoning_embedding``
tap meaningful at inference — an unaligned head emits an untrained projection, so
comparing it against cached teacher prototypes would measure noise.

Same principles as the teacher endpoint: **offline only** (never called in the
training loop), **backend-agnostic** (any encoder satisfying :class:`TextEncoder`),
and with a deterministic dependency-free encoder so CI needs no model download.

Horizons whose ``evidence`` is missing are reported in a validity mask rather than
embedded as zeros — a zero vector is not a valid cosine target, so the caller must
drop those horizons' source weight (``HorizonReasoningLoss`` already ignores
zero-weight horizons in ``_weighted_mean``).
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

from .schema import NUM_HORIZONS, ReasoningLabelRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class TextEncoder(Protocol):
    """Any sentence encoder: ``encode(texts) -> [N, dim]``, L2-normalized."""

    dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class HashingTextEncoder:
    """Deterministic, dependency-free encoder (CI / offline default).

    Hashes each token into a fixed-width bag and L2-normalizes.  It carries no
    real semantics, but it is stable across runs and machines, so tests and the
    pipeline's smoke path never need a model download — the same reason the label
    pipeline ships a ``MockTeacher``.

    NOTE: this is a *stand-in*.  Training a real alignment on hashed bags would
    align the student to lexical overlap, not meaning.  Use a real encoder for
    any run whose numbers are reported.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = hashlib.sha256(tok.encode()).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] % 2 else -1.0
                out[i, idx] += sign
        return torch.nn.functional.normalize(out, dim=-1)


class SentenceTransformerEncoder:
    """Real encoder backed by sentence-transformers (lazy import).

    Args:
        model: any sentence-transformers model id (default a small, permissive one).
        device: torch device string.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: offline dep

        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        vecs = self._model.encode(
            list(texts), convert_to_tensor=True, normalize_embeddings=True
        )
        return vecs.detach().cpu().float()


def embed_record(
    record: ReasoningLabelRecord,
    encoder: TextEncoder,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed one sample's per-horizon ``evidence``.

    Args:
        record: a teacher label record (its ``horizons`` carry the rationales).
        encoder: any :class:`TextEncoder`.

    Returns:
        ``(embeddings [5, D], valid [5] bool)``.  Horizons with no ``evidence``
        get a zero row and ``valid=False`` — the caller must zero their source
        weight so the alignment loss skips them rather than regressing the student
        toward a zero vector.  An **abstained** record (teacher/parse failure) is
        entirely invalid, per the schema's rule that its horizons are masked out
        of the loss rather than turned into all-zero targets.
    """
    texts: List[str] = [""] * NUM_HORIZONS
    valid = torch.zeros(NUM_HORIZONS, dtype=torch.bool)

    if not record.abstained:
        for h_idx, horizon in enumerate(record.horizons[:NUM_HORIZONS]):
            evidence = (horizon.evidence or "").strip()
            if evidence:
                texts[h_idx] = evidence
                valid[h_idx] = True

    embeddings = encoder.encode(texts)          # [5, D]
    embeddings[~valid] = 0.0                    # never align against empty text
    return embeddings, valid


def embed_records(
    records: Sequence[ReasoningLabelRecord],
    encoder: TextEncoder,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch form of :func:`embed_record`.

    Returns:
        ``(embeddings [B, 5, D], valid [B, 5] bool)`` — the exact shape
        ``HorizonReasoningLoss(teacher_embedding_targets=...)`` expects.
    """
    if not records:
        return torch.zeros(0, NUM_HORIZONS, encoder.dim), torch.zeros(
            0, NUM_HORIZONS, dtype=torch.bool
        )
    pairs = [embed_record(r, encoder) for r in records]
    return torch.stack([e for e, _ in pairs]), torch.stack([v for _, v in pairs])
