"""Artifact I/O for reasoning labels (issue #98, R4).

Writes versioned label artifacts: JSONL for debugging (always available, stdlib)
and Parquet for training (when ``pyarrow`` is present). Records are flattened to
one row PER (sample, horizon) with full provenance so training can filter and
source-weight. Abstained records write a single provenance row (no horizons) so
the failure is visible in the artifact rather than dropped.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .schema import ReasoningLabelRecord


def _record_to_rows(record: ReasoningLabelRecord) -> List[Dict[str, Any]]:
    """Flatten a record into per-horizon rows (one provenance row if abstained)."""
    base = {
        "schema_version": record.schema_version,
        "sample_id": record.sample_id,
        "timestamp": record.timestamp,
        "dataset_name": record.dataset_name,
        "dataset_version": record.dataset_version,
        "teacher_provider": record.teacher_provider,
        "teacher_model": record.teacher_model,
        "teacher_endpoint_type": record.teacher_endpoint_type,
        "prompt_version": record.prompt_version,
        "request_mode": record.request_mode,
        "labeler_version": record.labeler_version,
        "provenance": record.provenance,
        "created_at": record.created_at,
        "abstained": record.abstained,
        "teacher_error": record.teacher_error,
    }
    if record.abstained or not record.horizons:
        return [{**base, "horizon_sec": None}]
    rows = []
    for h in record.horizons:
        row = dict(base)
        row.update(dataclasses.asdict(h))
        rows.append(row)
    return rows


def records_to_rows(records: Sequence[ReasoningLabelRecord]) -> List[Dict[str, Any]]:
    """Flatten many records to per-horizon rows."""
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.extend(_record_to_rows(r))
    return rows


def write_jsonl(records: Sequence[ReasoningLabelRecord], path: str) -> str:
    """Write one JSON object per (sample, horizon) row. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for row in records_to_rows(records):
            f.write(json.dumps(row, default=_json_default) + "\n")
    return str(p)


def write_parquet(records: Sequence[ReasoningLabelRecord], path: str) -> str:
    """Write a Parquet table of per-horizon rows (requires pyarrow).

    Raises ImportError with a clear message if pyarrow is unavailable so a
    contributor knows to use JSONL (or install the extra) rather than hitting
    an opaque failure.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "write_parquet requires pyarrow; install it or use write_jsonl for "
            "a stdlib-only artifact."
        ) from e

    rows = records_to_rows(records)
    # Multi-label list columns must serialize as JSON strings for a flat schema
    # (Parquet lists-of-strings work too, but JSON keeps the column types simple
    # and round-trips through pandas without dtype surprises).
    normalized = [{k: _cell(v) for k, v in row.items()} for row in rows]
    table = pa.Table.from_pylist(normalized)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(p))
    return str(p)


def _cell(value: Any) -> Any:
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
