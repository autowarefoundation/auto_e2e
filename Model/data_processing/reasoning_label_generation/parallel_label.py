"""Process-parallel offline reasoning labeling (#98).

The offline labeler is decode-bound: building each sample's temporal front clip
decodes the 1 Hz World-Model window, and lerobot's reader is NOT thread-safe, so
a ThreadPool had to serialize decode under a lock — which left the (many) vLLM
replicas idle. Processes each own an independent dataset + reader, so decode runs
truly in parallel across CPU cores and the teacher calls overlap, finally using
the scaled-out Cosmos endpoint.

Design:
  * ``init_worker`` builds ONE per-process dataset + teacher (constructed once,
    reused for every sample that process handles).
  * ``label_sample`` (module-level, picklable) decodes the front clip + calls the
    teacher and returns the record. There is NO per-sample S3 cache (#121 §3.4):
    the parent aggregates all records into ONE per-partition ``records.jsonl``, so
    the whole partition is a single Flyte-cached artifact instead of ~10M tiny S3
    objects. Re-run protection is the deterministic partition + Flyte task cache.
Only the small sample index is sent across the process boundary; frames never
cross it. Spawn context is used (torch is imported), so workers re-import this
module cleanly without dragging in the Flyte task module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Per-process globals, populated by init_worker in each child process. Typed Any:
# deliberately dynamic per-worker state (dataset class / teacher client differ
# per run) set in init_worker, so a concrete static type would misrepresent them
# and trip mypy on every use.
_DS: Any = None
_CLIENT: Any = None
_DATASET_NAME: Any = None
_NUM_HORIZONS = 5


def init_worker(
    repo_id: str,
    episodes: Optional[List[int | str]],
    dataset_name: str,
    teacher: str,
    teacher_kwargs: Dict[str, Any],
    prompt_version: str,
    raw_path: Optional[str] = None,
) -> None:
    """Build this process's dataset + teacher once (reused per sample).

    The dataset is built in a light-weight FRONT-CLIP mode: it exposes
    ``get_front_clip(idx) -> [5 x [3,H,W]]`` (front camera at the reasoning
    horizons) so the worker never decodes the full multi-view WM window. L2D uses
    lerobot delta_timestamps; NVIDIA uses a sparse front-camera PyAV decode. Both
    keep the SAME sample enumeration as data_processing so sample_ids JOIN.

    ``prompt_version`` is threaded to the teacher via teacher_kwargs upstream; it
    is accepted here for call-site symmetry (and would key any future cache).
    """
    global _DS, _CLIENT, _DATASET_NAME
    from .teacher_client import build_teacher
    from .schema import NUM_HORIZONS

    global _NUM_HORIZONS
    _NUM_HORIZONS = NUM_HORIZONS
    _DATASET_NAME = dataset_name
    if dataset_name == "nvidia/PhysicalAI-Autonomous-Vehicles":
        from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
        # NVIDIA reads from a local raw path (no HF repo); raw_path is required.
        if raw_path is None:
            raise ValueError(
                "raw_path is required for the NVIDIA dataset (no HF repo to pull).")
        _DS = NvidiaAVDataset(data_root=raw_path, reasoning_clip_only=True)
    elif dataset_name == "KIT-MRT/KITScenes-Multimodal":
        from data_parsing.kit_scenes import KitScenesDataset
        if raw_path is None:
            raise ValueError(
                "raw_path is required for KITScenes (scene is already ingested)."
            )
        scene_ids = [str(scene_id) for scene_id in episodes] if episodes else None
        _DS = KitScenesDataset(
            data_root=raw_path,
            split="train",
            scene_ids=scene_ids,
            reasoning_clip_only=True,
        )
    else:
        from data_parsing.l2d import L2DDataset
        # root=raw_path (when provided) makes lerobot read the partition's
        # already-materialized raw dir instead of re-downloading to the shared HF
        # cache in this pod (#121 option B). None → legacy HF-cache path.
        l2d_episodes = (
            [int(episode) for episode in episodes]
            if episodes is not None
            else None
        )
        _DS = L2DDataset(repo_id=repo_id, episodes=l2d_episodes,
                         reasoning_clip_only=True, root=raw_path)
    _CLIENT = build_teacher(teacher, **teacher_kwargs)


def label_sample(si: int) -> Tuple[int, Dict[str, Any], str]:
    """Label sample ``si``: decode the front clip + call the teacher.

    Returns ``(si, record_json, status)`` where status is 'computed' | 'abstained'
    (there is no 'hit' — the per-sample cache is gone, §3.4). The record is a plain
    dict (JSON-able via record_to_json) so it pickles cleanly back to the parent,
    which writes all records to one per-partition records.jsonl.
    """
    from .teacher_client import TeacherRequest
    from .targets import record_to_json

    # Global, partition-independent uid (#121 §3.1): built from the sample's
    # (episode/clip, frame) identity, NOT the positional index — so labels JOIN to
    # packed shards regardless of the episode-range partition boundary.
    sample_key = _DS.sample_uid(si)
    clip = _DS.get_front_clip(si)  # 5 front frames (0/1/2/3/4 s)
    rec = _CLIENT.label(TeacherRequest(
        sample_id=sample_key, dataset_name=_DATASET_NAME, frames=clip))
    return si, record_to_json(rec), ("abstained" if rec.abstained else "computed")
