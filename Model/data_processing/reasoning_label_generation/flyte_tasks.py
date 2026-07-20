"""Flyte Processing task for reasoning-label generation (issue #98, R3).

Generates labels for a set of samples with a configurable teacher backend
(mock / cached / openai_compatible), validates them, and writes versioned
JSONL/Parquet artifacts.

Runs LOCALLY WITHOUT KUBERNETES: ``flytekit`` is optional. If it is installed,
``label_samples_task`` is a registered Flyte task; if not, it degrades to a
plain Python function with the identical signature, so a contributor can call
``run_labeling(...)`` on a laptop with the mock/cached backend and no cluster.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .parquet_writer import write_jsonl, write_parquet
from .schema import ReasoningLabelRecord
from .teacher_client import TeacherClient, TeacherRequest, build_teacher
from .validators import validate_record

# Optional Flyte: degrade to a no-op decorator when flytekit is unavailable so
# the pipeline stays runnable on a contributor machine without Kubernetes.
try:  # pragma: no cover - environment dependent
    from flytekit import task as _flyte_task

    _HAS_FLYTE = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_FLYTE = False

    def _flyte_task(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        # Support both @task and @task(...) usage.
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


def run_labeling(
    requests: Sequence[TeacherRequest],
    *,
    provider: str = "mock",
    teacher_kwargs: Optional[Dict[str, Any]] = None,
    jsonl_path: Optional[str] = None,
    parquet_path: Optional[str] = None,
    validate: bool = True,
) -> List[ReasoningLabelRecord]:
    """Label ``requests`` with ``provider``, validate, and optionally write.

    This is the plain-Python core (no Flyte/K8s needed). ``label_samples_task``
    wraps it as a Flyte task when flytekit is present.

    Args:
        requests: samples to label.
        provider: teacher backend key (``mock`` / ``cached`` / ``openai_compatible``).
        teacher_kwargs: forwarded to the backend factory (e.g. ``base_url``).
        jsonl_path / parquet_path: optional artifact outputs.
        validate: run schema/taxonomy validation on successful records (R4/R5).

    Returns:
        the produced records (successful and abstained).
    """
    teacher: TeacherClient = build_teacher(provider, **(teacher_kwargs or {}))
    records = teacher.label_batch(requests)

    if validate:
        for record in records:
            validate_record(record, teacher.taxonomy)

    if jsonl_path is not None:
        write_jsonl(records, jsonl_path)
    if parquet_path is not None:
        write_parquet(records, parquet_path)

    return records


@_flyte_task
def label_samples_task(
    sample_ids: List[str],
    dataset_name: str,
    provider: str = "mock",
    jsonl_path: Optional[str] = None,
) -> int:
    """Flyte-compatible entry point: label samples by id, return the count.

    Kept to Flyte-friendly primitive types (lists/strings/ints) so it registers
    as a task; the heavy lifting is in :func:`run_labeling`. Frames are not
    passed here (a real pipeline resolves them from the dataset inside the task);
    with the mock/cached backend they are not needed, which is what makes the
    contributor path cluster-free.
    """
    requests = [
        TeacherRequest(sample_id=sid, dataset_name=dataset_name) for sid in sample_ids
    ]
    records = run_labeling(requests, provider=provider, jsonl_path=jsonl_path)
    return len(records)
