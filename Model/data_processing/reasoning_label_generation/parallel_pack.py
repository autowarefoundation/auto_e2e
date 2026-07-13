"""Process-parallel shard packing for data_processing (#30/#13).

Packing is decode-bound: each sample decodes its camera views and — for the
World-Model branch — a full 1 Hz past/future window (history N + future N rows x
V cams). Done serially in one process it is ~1 core for an hour on a 1000-sample
WM dataset. This module moves the DECODE + JPEG-encode into worker processes
(each with its own dataset/reader), returning per-sample JPEG/npy BYTES; the
parent just appends those bytes to the shard tar (fast, single-threaded, so the
tar stays valid). Only the small byte blobs cross the process boundary.

Per-sample members: cam_i.jpg (reactive current frame), map.jpg, ego.npy,
meta.json, calib.json, and — for WM shards — window_index.json (§3.4d). The WM
window PIXELS are NOT per-sample anymore: the worker returns them in a separate
``pool`` dict keyed by a GLOBAL frame_id, and the parent writes each frame_id to a
shared ``pool/`` dir exactly ONCE, deduping the ~8x cross-sample frame overlap
(10Hz samples × 1Hz stride-10 window). reasoning.json is JOINed in the parent
(labels_by_id is not shipped to workers). The loader rebuilds identical
history_frames/future_frames from window_index.json + the pool.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

# Per-process globals (set by init_pack_worker in each child). Typed Any: these
# are deliberately dynamic per-worker state (the dataset class differs by dataset,
# the transforms are torchvision objects) filled in init_pack_worker, so a
# concrete static type would misrepresent them and trip mypy on every use.
_DS: Any = None
_RESIZE: Any = None
_TO_PIL: Any = None
_DATASET_VALUE: Any = None
_CALIB_BYTES: Any = None


def init_pack_worker(
    dataset_value: str,
    episodes: Optional[List[int]],
    raw_path: str,
    image_size: int,
    world_model: bool,
    calib_bytes: bytes,
) -> None:
    """Build this process's raw dataset + resize transform once (reused per sample)."""
    global _DS, _RESIZE, _TO_PIL, _DATASET_VALUE, _CALIB_BYTES
    from torchvision import transforms

    _DATASET_VALUE = dataset_value
    _CALIB_BYTES = calib_bytes
    _TO_PIL = transforms.ToPILImage()
    _RESIZE = transforms.Resize((image_size, image_size))
    if dataset_value == "nvidia/PhysicalAI-Autonomous-Vehicles":
        from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
        _DS = NvidiaAVDataset(data_root=raw_path)
    else:
        from data_parsing.l2d import L2DDataset
        # root=raw_path makes lerobot read the partition's materialized raw dir
        # (populated by that partition's ingest) instead of re-downloading to the
        # shared HF cache in this pod (#121 option B). raw_path is always provided
        # to the packer (it's a required arg); None-safe via L2DDataset(root=None).
        _DS = L2DDataset(repo_id=dataset_value, episodes=episodes,
                         include_world_model_windows=world_model, root=raw_path)


def _jpeg(frame_tensor) -> bytes:
    """Resize a RAW (3,H,W) frame to a JPEG byte string (the single pack resize)."""
    t = frame_tensor.cpu()
    if t.dtype.is_floating_point:
        t = t.clamp(0, 1)
    f = _RESIZE(_TO_PIL(t))
    b = io.BytesIO()
    f.save(b, format="JPEG", quality=90)
    return b.getvalue()


def init_row_worker(
    dataset_value: str,
    episodes: Optional[List[int]],
    raw_path: str,
    image_size: int,
) -> None:
    """Build this process's PLAIN-mode dataset for row-level decode (#121 decode-dedup).

    Plain mode (no WM delta_timestamps): ``lerobot_dataset[row]`` decodes ONE row's
    videos (6 cams + map ≈ 7 frames) instead of a whole 8-step window (~49). The
    single-row read is bit-identical to the window read at the same physical row
    (verified in dataset.py / test_preextract_world_model), so pool bytes match the
    per-sample window path exactly.

    INVARIANT (load-bearing, pinned by test_reasoning_alignment_invariant.py):
    the sample enumeration is IDENTICAL between WM-on and WM-off L2DDataset
    instances, because the egomotion margins (64/64) strictly dominate the WM
    margins (30/40) — so ``_build_sample_index`` produces the same ``_samples``
    regardless of ``include_world_model_windows``, and this worker's plain-mode
    L2DDataset._episode_ranges + local_row resolution agree exactly with the
    parent's WM-mode dataset used to enumerate window rows in Pass A. If this
    invariant is ever broken (e.g. wm_num_frames * stride > 64), Pass B assembly
    would silently misalign uid ↔ row. The test guards against that.
    """
    global _DS, _RESIZE, _TO_PIL, _DATASET_VALUE
    from torchvision import transforms
    from data_parsing.l2d import L2DDataset

    _DATASET_VALUE = dataset_value
    _TO_PIL = transforms.ToPILImage()
    _RESIZE = transforms.Resize((image_size, image_size))
    _DS = L2DDataset(repo_id=dataset_value, episodes=episodes,
                     include_world_model_windows=False, root=raw_path)


def decode_row(task: Tuple[int, int]) -> Tuple[Tuple[int, int], Dict[str, bytes], Optional[bytes]]:
    """Decode ONE physical row's cameras (+ map) → JPEG bytes (#121 decode-dedup).

    ``task`` = (ep_idx, frame_index) — GLOBAL identity, so the worker resolves the
    local row from its OWN _episode_ranges (robust across processes/partitions).
    Returns ``((ep_idx, frame_index), {frame_id: jpeg per cam}, map_jpeg_or_None)``.

    Each unique row is decoded exactly ONCE per partition — the parent maps this
    over the UNION of all samples' window rows, killing the ~8x per-sample
    re-decode. Map is skipped when all-zero (same rule as pack_sample). The
    frame_id embeds episode + episode-local frame_index, so it can never
    reference another episode/scene.
    """
    from data_parsing.l2d.dataset import CAMERA_NAMES, MAP_VIEW_NAME
    from data_processing.contract_versions import UID_SCHEMA_VERSION

    ep_idx, frame_index = task
    ep_start, ep_end = _DS._episode_ranges[ep_idx]
    local_row = ep_start + frame_index
    if local_row >= ep_end:
        raise IndexError(f"row {frame_index} outside episode {ep_idx}")
    item = _DS.lerobot_dataset[local_row]
    cams: Dict[str, bytes] = {}
    for v, cam_name in enumerate(CAMERA_NAMES):
        fid = f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame_index:06d}-c{v}"
        cams[fid] = _jpeg(item[cam_name])
    map_tile = item[MAP_VIEW_NAME]
    mt = map_tile[0] if map_tile.ndim == 4 else map_tile
    map_jpeg = _jpeg(mt) if float(mt.abs().max()) > 0 else None
    return (ep_idx, frame_index), cams, map_jpeg


def pack_sample(si: int) -> Tuple[str, int, Dict[str, bytes], Dict[str, bytes]]:
    """Decode + encode sample ``si`` into per-sample members + a frame-pool
    contribution (#121 §3.1, §3.4d dedup).

    Returns ``(sample_uid, num_views, members, pool)``:
      * ``members`` = the sample's OWN members (prefixed ``{uid}.`` by the parent):
        ``cam_i.jpg`` (reactive current frame), ``map.jpg``, ``ego.npy``,
        ``meta.json``, ``calib.json``, and — for WM shards — ``window_index.json``
        (the (step,view)→frame_id map, NOT the pixels).
      * ``pool`` = ``{frame_id: jpeg_bytes}`` for THIS sample's WM window frames.
        The parent writes each frame_id to the shared ``pool/`` dir ONCE (dedup
        across the ~8 overlapping neighbour windows), so the same physical frame is
        never re-encoded/stored. Empty for imitation-only (non-WM) samples.

    The reasoning.json JOIN and split bucketing are unchanged (still keyed by uid /
    meta.json). Boundary safety: frame_ids come from window_frame_ids, which is
    clamped to the sample's own episode.
    """
    import numpy as np
    import torch

    sample = _DS[si]
    uid = _DS.sample_uid(si)
    split_group = _DS.split_group_uid(si)
    members: Dict[str, bytes] = {}
    pool: Dict[str, bytes] = {}

    visual = sample["visual_tiles"]            # (V, 3, H, W)
    for cam_i in range(visual.shape[0]):
        members[f"cam_{cam_i}.jpg"] = _jpeg(visual[cam_i])

    # Only write map.jpg for a REAL nav-map. NVIDIA has no map and hands a
    # zeros_like placeholder; packing it would JPEG+ImageNet-normalize a black
    # tile into a nonzero per-channel CONSTANT at load time, contaminating the
    # shared map encoder (and wrongly flagging has_map=True). Skip all-zero tiles
    # so the loader's zero-fallback fires and has_map stays False.
    map_tile = sample.get("map_tile")
    if map_tile is not None and float(map_tile.abs().max()) > 0:
        members["map.jpg"] = _jpeg(map_tile)

    # World-Model window (#13/#3.4d): store the frames in the shared pool keyed by
    # a GLOBAL frame_id, and record only the (step,view)→frame_id index per sample.
    # Because consecutive 10Hz samples' 1Hz windows overlap 7/8, the pool collapses
    # that duplication (parent dedups across samples). Same frame set as before, so
    # the loader rebuilds identical history_frames/future_frames tensors.
    history_win = sample.get("history_frames")   # (T, V, 3, H, W)
    future_win = sample.get("future_frames")     # (F, V, 3, H, W)
    if history_win is not None and future_win is not None:
        ids = _DS.window_frame_ids(si)           # {"history": [[id/view]/step], "future": [...]}
        for t in range(history_win.shape[0]):
            for v in range(history_win.shape[1]):
                pool[ids["history"][t][v]] = _jpeg(history_win[t, v])
        for fh in range(future_win.shape[0]):
            for v in range(future_win.shape[1]):
                pool[ids["future"][fh][v]] = _jpeg(future_win[fh, v])
        members["window_index.json"] = json.dumps(ids).encode()

    ego_hist = sample["egomotion_history"]
    traj = sample["trajectory_target"]
    ego_data = np.concatenate([
        ego_hist.numpy() if torch.is_tensor(ego_hist) else np.asarray(ego_hist),
        traj.numpy() if torch.is_tensor(traj) else np.asarray(traj),
    ]).astype(np.float32)
    members["ego.npy"] = ego_data.tobytes()
    members["meta.json"] = json.dumps({
        "idx": si, "dataset": _DATASET_VALUE,
        "sample_uid": uid, "split_group_uid": split_group,
    }).encode()
    members["calib.json"] = _CALIB_BYTES

    return uid, int(visual.shape[0]), members, pool
