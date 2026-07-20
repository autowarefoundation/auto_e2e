"""Label → training-target tensorization (issue #98).

Bridges the string-valued :class:`ReasoningLabelRecord` (offline, torch-free)
to the tensors :class:`HorizonReasoningLoss` consumes. Kept in ``data_processing``
(a label concern) and separate from ``schema.py`` (which stays torch-free).

Per group, per horizon:
    * multi-label  → ``[5, C]`` float in {0, 1} (a class's index set to 1);
    * single-label → ``[5]`` long class index, ``IGNORE_INDEX`` (-100) when the
      teacher abstained on that group/horizon.
Plus ``confidence_targets [5]`` and ``source_weights [5]`` = provenance weight ×
label confidence (0 for an abstained record, so it is fully masked).

``collate_reasoning_targets`` stacks per-sample tensors into a
:class:`ReasoningTargetBatch` (``[B, 5, ...]``) for the loss.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Sequence

import torch

from model_components.reasoning.reasoning_taxonomy import (
    DEFAULT_TAXONOMY,
    LabelMode,
    ReasoningTaxonomy,
)

from .schema import (
    NUM_HORIZONS,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
    ReasoningTargetBatch,
)

IGNORE_INDEX = -100

# Provenance → source weight (issue #98 Loss §3). Abstained horizons get 0.
_SOURCE_WEIGHT: Dict[str, float] = {
    "audited_gt": 1.0,
    "direct_gt": 0.9,
    "derived_gt": 0.7,
    "counterfactual_gt": 0.8,
    "teacher_gt": 0.5,
    "weak_gt": 0.3,
    "teacher_error": 0.0,
}

# The action-relevant core groups this tensorizer covers, matching the head.
_CORE_GROUPS = (
    "relation_to_ego", "hazard_event", "cause",
    "longitudinal_response", "lateral_response", "tactical_response", "rule_response",
)


def _group_attr(horizon: ReasoningHorizonLabel, group: str) -> Any:
    return getattr(horizon, group)


def record_to_target_tensors(
    record: ReasoningLabelRecord, taxonomy: ReasoningTaxonomy = DEFAULT_TAXONOMY
) -> Dict[str, torch.Tensor]:
    """Tensorize one record into per-sample targets (no batch dim).

    Returns a flat dict:
        ``target__<group>``: ``[5, C]`` float (multi) or ``[5]`` long (single),
        ``confidence``:      ``[5]`` float,
        ``source_weight``:   ``[5]`` float.

    An abstained record yields all-ignore single-label targets, all-zero
    multi-label targets, and zero source weights (fully masked, R9).
    """
    H = NUM_HORIZONS
    out: Dict[str, torch.Tensor] = {}
    abstained = record.abstained or len(record.horizons) != H

    for group in _CORE_GROUPS:
        C = taxonomy.num_classes(group)
        if taxonomy.mode(group) is LabelMode.MULTI:
            t = torch.zeros(H, C, dtype=torch.float32)
        else:
            t = torch.full((H,), IGNORE_INDEX, dtype=torch.long)
        if not abstained:
            for h_idx, horizon in enumerate(record.horizons):
                value = _group_attr(horizon, group)
                if taxonomy.mode(group) is LabelMode.MULTI:
                    for lbl in (value or []):
                        if lbl in taxonomy.labels(group):
                            t[h_idx, taxonomy.index(group, lbl)] = 1.0
                elif isinstance(value, str) and value in taxonomy.labels(group):
                    t[h_idx] = taxonomy.index(group, value)
        out[f"target__{group}"] = t

    confidence = torch.zeros(H, dtype=torch.float32)
    weights = torch.zeros(H, dtype=torch.float32)
    if not abstained:
        for h_idx, horizon in enumerate(record.horizons):
            confidence[h_idx] = float(horizon.confidence)
            base = _SOURCE_WEIGHT.get(horizon.provenance, 0.0)
            weights[h_idx] = base * float(horizon.confidence)
    out["confidence"] = confidence
    out["source_weight"] = weights
    return out


def collate_reasoning_targets(
    per_sample: Sequence[Dict[str, torch.Tensor]],
    taxonomy: ReasoningTaxonomy = DEFAULT_TAXONOMY,
) -> ReasoningTargetBatch:
    """Stack per-sample target dicts into a batched :class:`ReasoningTargetBatch`."""
    targets: Dict[str, torch.Tensor] = {}
    for group in _CORE_GROUPS:
        targets[group] = torch.stack([s[f"target__{group}"] for s in per_sample], dim=0)
    confidence = torch.stack([s["confidence"] for s in per_sample], dim=0)
    weights = torch.stack([s["source_weight"] for s in per_sample], dim=0)
    return ReasoningTargetBatch(
        targets=targets,
        confidence_targets=confidence,
        source_weights=weights,
    )


def target_batch_from_loader(
    batch: Dict[str, torch.Tensor],
    taxonomy: ReasoningTaxonomy = DEFAULT_TAXONOMY,
) -> Optional[ReasoningTargetBatch]:
    """Assemble a ReasoningTargetBatch from a pre-extracted loader batch.

    The loader flattens per-sample targets to top-level ``reasoning__<key>``
    tensors (already stacked to ``[B, ...]`` by WebDataset's default collation).
    Returns None when the batch carries no reasoning labels (shards packed
    without a teacher), so training can skip the reasoning loss.
    """
    if f"reasoning__target__{_CORE_GROUPS[0]}" not in batch:
        return None
    targets = {g: batch[f"reasoning__target__{g}"] for g in _CORE_GROUPS}
    return ReasoningTargetBatch(
        targets=targets,
        confidence_targets=batch["reasoning__confidence"],
        source_weights=batch["reasoning__source_weight"],
    )


# ---------------------------------------------------------------------------
# JSON (de)serialization for the per-sample shard member (reasoning.json).
# ---------------------------------------------------------------------------

def record_to_json(record: ReasoningLabelRecord) -> Dict[str, Any]:
    """Serialize a record to a JSON-able dict (one shard member per sample)."""
    return dataclasses.asdict(record)


def record_from_json(data: Dict[str, Any]) -> ReasoningLabelRecord:
    """Reconstruct a record from a JSON dict written by :func:`record_to_json`."""
    horizon_fields = {f.name for f in dataclasses.fields(ReasoningHorizonLabel)}
    horizons = [
        ReasoningHorizonLabel(**{k: v for k, v in h.items() if k in horizon_fields})
        for h in data.get("horizons", [])
    ]
    record_fields = {f.name for f in dataclasses.fields(ReasoningLabelRecord)}
    kwargs = {k: v for k, v in data.items() if k in record_fields and k != "horizons"}
    return ReasoningLabelRecord(horizons=horizons, **kwargs)


# ---------------------------------------------------------------------------
# Whole-record JSONL (one full record per line) — the JOIN interchange format.
#
# The parquet/jsonl in parquet_writer.py are FLATTENED to per-(sample,horizon)
# rows for analytics/querying; reconstructing whole records from them is lossy
# and awkward. For the label→shard JOIN we instead write one complete record per
# line (record_to_json), which data_processing reads back into a
# {sample_id: record} map to embed as the per-sample reasoning.json member.
# ---------------------------------------------------------------------------

def write_records_jsonl(records, path: str) -> str:
    """Write one full ReasoningLabelRecord per line (JOIN interchange). Returns path."""
    import json
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(record_to_json(r)) + "\n")
    return path


def load_records_by_sample_id(path: str) -> Dict[str, ReasoningLabelRecord]:
    """Read a whole-record JSONL (written by :func:`write_records_jsonl`) into a
    ``{sample_id: ReasoningLabelRecord}`` map for the data_processing JOIN.

    A later line for the same sample_id wins (idempotent re-writes are safe).
    """
    import json
    out: Dict[str, ReasoningLabelRecord] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = record_from_json(json.loads(line))
            out[rec.sample_id] = rec
    return out
