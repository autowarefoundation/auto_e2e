"""Cached teacher: re-serve labels from a prior artifact (issue #98, R3).

Lets a contributor (or CI) run the whole pipeline against labels generated once
by a real teacher, with no model, network, or GPU. Reads a JSONL artifact
written by :mod:`.parquet_writer` and reconstructs one
:class:`ReasoningLabelRecord` per sample_id.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_components.reasoning.reasoning_taxonomy import ReasoningTaxonomy

from .schema import (
    SCHEMA_VERSION,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)
from .teacher_client import TeacherClient, TeacherRequest, register_teacher

_HORIZON_FIELDS = {f.name for f in ReasoningHorizonLabel.__dataclass_fields__.values()}


class CachedTeacher(TeacherClient):
    """Serve pre-generated labels from a JSONL artifact, keyed by sample_id."""

    def __init__(
        self,
        *,
        label_artifact: str,
        model: str = "cached",
        prompt_version: str = "action_relevant_reasoning_v2",
        request_mode: str = "clip_horizons",
        taxonomy: Optional[ReasoningTaxonomy] = None,
        strict: bool = True,
    ) -> None:
        super().__init__(
            provider="cached", model=model, prompt_version=prompt_version,
            request_mode=request_mode, taxonomy=taxonomy, strict=strict,
        )
        self._by_sample = self._load(label_artifact)

    def _load(self, artifact: str) -> Dict[str, ReasoningLabelRecord]:
        path = Path(artifact)
        if not path.exists():
            raise FileNotFoundError(f"cached label artifact not found: {artifact}")
        by_sample_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_sample_rows[row["sample_id"]].append(row)
        return {sid: self._rows_to_record(rows) for sid, rows in by_sample_rows.items()}

    def _rows_to_record(self, rows: List[Dict[str, Any]]) -> ReasoningLabelRecord:
        first = rows[0]
        if first.get("abstained"):
            return ReasoningLabelRecord.abstain(
                sample_id=first["sample_id"], dataset_name=first["dataset_name"],
                teacher_provider=first["teacher_provider"], teacher_model=first["teacher_model"],
                prompt_version=first["prompt_version"], request_mode=first["request_mode"],
                teacher_error=first.get("teacher_error") or "cached abstain",
                timestamp=first.get("timestamp", 0.0),
            )
        horizons = [
            ReasoningHorizonLabel(**{k: v for k, v in row.items() if k in _HORIZON_FIELDS})
            for row in sorted(rows, key=lambda r: r.get("horizon_sec") or 0.0)
        ]
        return ReasoningLabelRecord(
            schema_version=first.get("schema_version", SCHEMA_VERSION),
            sample_id=first["sample_id"], timestamp=first.get("timestamp", 0.0),
            dataset_name=first["dataset_name"],
            teacher_provider=first["teacher_provider"], teacher_model=first["teacher_model"],
            prompt_version=first["prompt_version"], request_mode=first["request_mode"],
            horizons=horizons, dataset_version=first.get("dataset_version"),
            teacher_endpoint_type=first.get("teacher_endpoint_type"),
            provenance=first.get("provenance", "teacher_gt"),
        )

    def label(self, request: TeacherRequest) -> ReasoningLabelRecord:
        record = self._by_sample.get(request.sample_id)
        if record is None:
            if self.strict:
                raise KeyError(
                    f"no cached label for sample_id {request.sample_id!r}."
                )
            return self._abstain(request, "missing from cache")
        return record


register_teacher("cached", CachedTeacher)
