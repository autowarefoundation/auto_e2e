"""Tests for offline teacher-embedding production (#98 §6).

`HorizonReasoningLoss` implements the alignment term but nothing feeds it: the
label pipeline emits structured labels plus a free-text `evidence` rationale, and
no `teacher_embedding_targets` are produced anywhere. These tests pin the module
that closes that gap, and in particular the contract that makes the loss safe:
a horizon with no evidence must be reported invalid rather than embedded as a
zero vector (a zero row is not a valid cosine target).
"""

from __future__ import annotations

import pytest
import torch

from data_processing.reasoning_label_generation.schema import (
    NUM_HORIZONS,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)
from data_processing.reasoning_label_generation.teacher_embedding import (
    HashingTextEncoder,
    embed_record,
    embed_records,
)


def _record(
    evidences: list[str | None],
    sample_id: str = "s0",
    abstained: bool = False,
) -> ReasoningLabelRecord:
    return ReasoningLabelRecord(
        schema_version="reasoning_label_v1",
        sample_id=sample_id,
        timestamp=0.0,
        dataset_name="test",
        teacher_provider="mock",
        teacher_model="mock",
        prompt_version="v1",
        request_mode="forecast",
        horizons=[
            ReasoningHorizonLabel(horizon_sec=float(i), evidence=e)
            for i, e in enumerate(evidences)
        ],
        abstained=abstained,
    )


class TestHashingEncoder:
    def test_shape_and_normalisation(self):
        enc = HashingTextEncoder(dim=64)
        out = enc.encode(["a pedestrian is crossing", "the road is clear"])
        assert out.shape == (2, 64)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_deterministic_across_calls(self):
        # Same reason as MockTeacher: CI must not depend on a model download.
        a = HashingTextEncoder(dim=64).encode(["pedestrian at the kerb"])
        b = HashingTextEncoder(dim=64).encode(["pedestrian at the kerb"])
        assert torch.equal(a, b)

    def test_different_text_different_embedding(self):
        enc = HashingTextEncoder(dim=128)
        out = enc.encode(["a pedestrian is crossing", "the road is clear"])
        assert not torch.allclose(out[0], out[1])

    def test_empty_text_is_zero(self):
        assert torch.equal(HashingTextEncoder(dim=32).encode([""]), torch.zeros(1, 32))


class TestEmbedRecord:
    def test_shape_matches_the_loss_contract(self):
        enc = HashingTextEncoder(dim=64)
        rec = _record(["lead vehicle ahead"] * NUM_HORIZONS)
        emb, valid = embed_record(rec, enc)
        assert emb.shape == (NUM_HORIZONS, 64)   # what the loss expects per sample
        assert valid.shape == (NUM_HORIZONS,)
        assert valid.all()

    def test_missing_evidence_is_invalid_not_zero_target(self):
        """The contract that keeps the alignment loss honest: a horizon without a
        rationale must be flagged invalid so the caller can zero its source weight.
        Aligning the student toward a zero vector would be a wrong target, not a
        neutral one."""
        enc = HashingTextEncoder(dim=64)
        rec = _record(["pedestrian crossing", None, "", "clear road", None])
        emb, valid = embed_record(rec, enc)
        assert valid.tolist() == [True, False, False, True, False]
        # Invalid horizons are zeroed AND reported, so they can be masked out.
        assert torch.equal(emb[1], torch.zeros(64))
        assert torch.equal(emb[2], torch.zeros(64))
        assert emb[0].norm() > 0 and emb[3].norm() > 0

    def test_fewer_horizons_than_five_does_not_crash(self):
        enc = HashingTextEncoder(dim=32)
        emb, valid = embed_record(_record(["only now"]), enc)
        assert emb.shape == (NUM_HORIZONS, 32)
        assert valid.tolist() == [True, False, False, False, False]

    def test_abstained_record_is_entirely_invalid(self):
        """Schema rule R9: an abstained record's horizons are masked out of the
        loss, not turned into all-zero targets. Its evidence (if any survived the
        failure) must not be aligned against."""
        enc = HashingTextEncoder(dim=32)
        rec = _record(["stale text"] * NUM_HORIZONS, abstained=True)
        emb, valid = embed_record(rec, enc)
        assert not valid.any()
        assert torch.equal(emb, torch.zeros(NUM_HORIZONS, 32))


class TestEmbedRecords:
    def test_batch_shape_is_what_the_loss_takes(self):
        enc = HashingTextEncoder(dim=64)
        recs = [_record(["a"] * NUM_HORIZONS, "s0"), _record(["b"] * NUM_HORIZONS, "s1")]
        emb, valid = embed_records(recs, enc)
        # HorizonReasoningLoss(teacher_embedding_targets=[B, 5, D])
        assert emb.shape == (2, NUM_HORIZONS, 64)
        assert valid.shape == (2, NUM_HORIZONS)

    def test_empty_batch(self):
        emb, valid = embed_records([], HashingTextEncoder(dim=16))
        assert emb.shape == (0, NUM_HORIZONS, 16)
        assert valid.shape == (0, NUM_HORIZONS)


# --- El cableado: lo que hace que los embeddings LLEGUEN a la loss -----------
# Sin esto el módulo es un productor que nadie llama: HorizonReasoningLoss ya acepta
# teacher_embedding_targets [B, 5, D], pero NADA en el pipeline los producía, así que
# el término de alineación estaba inerte. `record_to_target_tensors` es el ÚLTIMO
# sitio que todavía tiene el texto de `evidence` — el loader lo aplana a índices de
# clase justo después y el texto se pierde.

def test_tensorizer_is_byte_identical_without_an_encoder():
    """Opt-in: sin encoder, la salida no cambia en absoluto."""
    from data_processing.reasoning_label_generation.targets import (
        record_to_target_tensors,
    )

    rec = _record(["a car ahead", "it slows", "brake", "stopped", "clear"])
    out = record_to_target_tensors(rec)
    assert "teacher_embedding" not in out
    assert "teacher_embedding_valid" not in out


def test_tensorizer_emits_the_embedding_the_loss_expects():
    from data_processing.reasoning_label_generation.targets import (
        collate_reasoning_targets,
        record_to_target_tensors,
    )
    from data_processing.reasoning_label_generation.teacher_embedding import (
        HashingTextEncoder,
    )

    enc = HashingTextEncoder(dim=32)
    rec = _record(["a car ahead", "it slows", "brake", "stopped", "clear"])
    out = record_to_target_tensors(rec, teacher_encoder=enc)
    assert out["teacher_embedding"].shape == (5, 32)
    assert out["teacher_embedding_valid"].shape == (5,)

    batch = collate_reasoning_targets([out, out])
    # Esto es lo que HorizonReasoningLoss consume: [B, 5, D].
    assert batch.teacher_embedding_targets is not None
    assert batch.teacher_embedding_targets.shape == (2, 5, 32)


def test_a_mixed_batch_fails_loudly_instead_of_dropping_the_term():
    """Media tanda con embeddings y media sin ellos no se puede apilar.

    Tirar el término en silencio para TODO el batch sería peor que decirlo: el
    entrenamiento seguiría, la loss de alineación no haría nada, y nadie se enteraría.
    """
    from data_processing.reasoning_label_generation.targets import (
        collate_reasoning_targets,
        record_to_target_tensors,
    )
    from data_processing.reasoning_label_generation.teacher_embedding import (
        HashingTextEncoder,
    )

    rec = _record(["a car ahead", "it slows", "brake", "stopped", "clear"])
    with_emb = record_to_target_tensors(rec, teacher_encoder=HashingTextEncoder(dim=8))
    without = record_to_target_tensors(rec)
    with pytest.raises(ValueError, match="mixed batch"):
        collate_reasoning_targets([with_emb, without])
