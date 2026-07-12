"""Process-parallel shard packing for data_processing (#30/#13).

Packing is decode-bound: each sample decodes its camera views and — for the
World-Model branch — a full 1 Hz past/future window (history N + future N rows x
V cams). Done serially in one process it is ~1 core for an hour on a 1000-sample
WM dataset. This module moves the DECODE + JPEG-encode into worker processes
(each with its own dataset/reader), returning per-sample JPEG/npy BYTES; the
parent just appends those bytes to the shard tar (fast, single-threaded, so the
tar stays valid). Only the small byte blobs cross the process boundary.

The worker returns exactly the members the serial packer wrote, so the shard
layout is byte-for-byte identical (cam_i.jpg, map.jpg, hist/fut_*.jpg, ego.npy,
meta.json, calib.json, reasoning.json) — reasoning.json is JOINed in the parent
(labels_by_id is not shipped to workers).
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
        _DS = L2DDataset(repo_id=dataset_value, episodes=episodes,
                         include_world_model_windows=world_model)


def _jpeg(frame_tensor) -> bytes:
    """Resize a RAW (3,H,W) frame to a JPEG byte string (the single pack resize)."""
    t = frame_tensor.cpu()
    if t.dtype.is_floating_point:
        t = t.clamp(0, 1)
    f = _RESIZE(_TO_PIL(t))
    b = io.BytesIO()
    f.save(b, format="JPEG", quality=90)
    return b.getvalue()


def pack_sample(si: int) -> Tuple[str, int, Dict[str, bytes]]:
    """Decode + encode sample ``si`` into ``{member_suffix: bytes}``.

    Returns ``(sample_uid, num_views, members)`` (#121 §3.1): the parent prefixes
    each member with ``{sample_uid}.`` and JOINs reasoning.json by the SAME uid the
    labeler used — so shard keys are partition-independent. ``num_views`` lets the
    parent fill the manifest. ``meta.json`` carries the raw episode/clip identity +
    ``split_group_uid`` (the train/val split unit) for downstream use.
    """
    import numpy as np
    import torch

    sample = _DS[si]
    uid = _DS.sample_uid(si)
    split_group = _DS.split_group_uid(si)
    members: Dict[str, bytes] = {}

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

    history_win = sample.get("history_frames")   # (T, V, 3, H, W)
    future_win = sample.get("future_frames")     # (F, V, 3, H, W)
    if history_win is not None and future_win is not None:
        for t in range(history_win.shape[0]):
            for v in range(history_win.shape[1]):
                members[f"hist_{t}_cam_{v}.jpg"] = _jpeg(history_win[t, v])
        for fh in range(future_win.shape[0]):
            for v in range(future_win.shape[1]):
                members[f"fut_{fh}_cam_{v}.jpg"] = _jpeg(future_win[fh, v])

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

    return uid, int(visual.shape[0]), members
