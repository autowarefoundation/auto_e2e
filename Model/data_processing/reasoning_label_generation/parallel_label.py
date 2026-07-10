"""Process-parallel offline reasoning labeling (#98).

The offline labeler is decode-bound: building each sample's temporal front clip
decodes the 1 Hz World-Model window, and lerobot's reader is NOT thread-safe, so
a ThreadPool had to serialize decode under a lock — which left the (many) vLLM
replicas idle. Processes each own an independent dataset + reader, so decode runs
truly in parallel across CPU cores and the teacher calls overlap, finally using
the scaled-out Cosmos endpoint.

Design:
  * ``init_worker`` builds ONE per-process dataset + teacher + label cache
    (constructed once, reused for every sample that process handles).
  * ``label_sample`` (module-level, picklable) does: cache.get → on miss decode
    the front clip + call the teacher + cache.put (only successful records).
Only the small sample index is sent across the process boundary; frames never
cross it. Spawn context is used (torch is imported), so workers re-import this
module cleanly without dragging in the Flyte task module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Per-process globals, populated by init_worker in each child process.
_DS = None
_CLIENT = None
_CACHE = None
_DATASET_NAME = None
_NUM_HORIZONS = 5


def init_worker(
    repo_id: str,
    episodes: Optional[List[int]],
    dataset_name: str,
    teacher: str,
    teacher_kwargs: Dict[str, Any],
    cache_bucket: Optional[str],
    prompt_version: str,
    raw_path: Optional[str] = None,
) -> None:
    """Build this process's dataset, teacher, and cache once (reused per sample).

    The dataset is built in a light-weight FRONT-CLIP mode: it exposes
    ``get_front_clip(idx) -> [5 x [3,H,W]]`` (front camera at the reasoning
    horizons) so the worker never decodes the full multi-view WM window. L2D uses
    lerobot delta_timestamps; NVIDIA uses a sparse front-camera PyAV decode. Both
    keep the SAME sample enumeration as data_processing so sample_ids JOIN.
    """
    global _DS, _CLIENT, _CACHE, _DATASET_NAME
    from .teacher_client import build_teacher
    from .label_cache import LabelCache
    from .schema import NUM_HORIZONS

    global _NUM_HORIZONS
    _NUM_HORIZONS = NUM_HORIZONS
    _DATASET_NAME = dataset_name
    if dataset_name == "nvidia/PhysicalAI-Autonomous-Vehicles":
        from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
        _DS = NvidiaAVDataset(data_root=raw_path, reasoning_clip_only=True)
    else:
        from data_parsing.l2d import L2DDataset
        _DS = L2DDataset(repo_id=repo_id, episodes=episodes,
                         reasoning_clip_only=True)
    _CLIENT = build_teacher(teacher, **teacher_kwargs)
    _CACHE = LabelCache(cache_bucket or None, dataset_name, teacher, prompt_version)


def label_sample(si: int) -> Tuple[int, Dict[str, Any], str]:
    """Label sample ``si``: cache hit → reuse; miss → decode front clip + teacher.

    Returns ``(si, record_json, status)`` where status is 'hit' | 'computed' |
    'abstained'. The record is returned as a plain dict (JSON-able via
    record_to_json) so it pickles cleanly back to the parent.
    """
    from .teacher_client import TeacherRequest
    from .targets import record_to_json

    sample_key = f"s{si:08d}"
    cached = _CACHE.get(sample_key)
    if cached is not None:
        return si, record_to_json(cached), "hit"

    clip = _DS.get_front_clip(si)  # 5 front frames (0/1/2/3/4 s)
    rec = _CLIENT.label(TeacherRequest(
        sample_id=sample_key, dataset_name=_DATASET_NAME, frames=clip))
    # Only cache SUCCESSFUL labels so a re-run retries abstentions.
    if not rec.abstained:
        _CACHE.put(sample_key, rec)
    return si, record_to_json(rec), ("abstained" if rec.abstained else "computed")
