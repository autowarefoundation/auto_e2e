"""Process-parallel shard packing for data_processing.

Packing is decode-bound: each sample decodes its camera views and, for the
World-Model branch, a full 1 Hz past/future window (history N + future N rows x
V cams). Worker processes (each with its own dataset/reader) do the decode and
JPEG encode and return per-sample JPEG/npy bytes; the parent appends those bytes
to the shard tar single-threaded so the tar stays valid. Only the small byte
blobs cross the process boundary.

Per-sample members: cam_i.jpg (reactive current frame), map.jpg, ego.npy,
optional pose.npy/gps.npy, meta.json, calib.json, and, for WM shards,
window_index.json. WM window pixels are not stored per sample: the worker
returns them in a separate ``pool`` dict keyed by a global frame_id, and the
parent writes each frame_id to a shared ``pool/`` dir exactly once, deduping the
~8x cross-sample frame overlap (10Hz samples × 1Hz stride-10 window).
reasoning.json is joined in the parent (labels_by_id is not shipped to workers).
The loader rebuilds identical history_frames/future_frames from
window_index.json + the pool.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

# Per-process globals (set by init_pack_worker in each child). Typed Any because
# the dataset class differs by dataset and the transforms are torchvision objects.
_DS: Any = None
_RESIZE: Any = None
_TO_PIL: Any = None
_DATASET_VALUE: Any = None
_CALIB_BYTES: Any = None


def init_pack_worker(
    dataset_value: str,
    episodes: Optional[List[int | str]],
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
    elif dataset_value == "KIT-MRT/KITScenes-Multimodal":
        from data_parsing.kit_scenes import KitScenesDataset
        scene_ids = [str(scene_id) for scene_id in episodes] if episodes else None
        _DS = KitScenesDataset(
            data_root=raw_path,
            split="train",
            scene_ids=scene_ids,
            image_size=image_size,
            include_world_model_windows=world_model,
            include_navigation=True,
        )
    else:
        from data_parsing.l2d import L2DDataset
        # root=raw_path makes lerobot read the partition's materialized raw dir
        # (populated by that partition's ingest) instead of re-downloading to the
        # shared HF cache in this pod. raw_path is always provided
        # to the packer (it's a required arg); None-safe via L2DDataset(root=None).
        l2d_episodes = (
            [int(episode) for episode in episodes]
            if episodes is not None
            else None
        )
        _DS = L2DDataset(repo_id=dataset_value, episodes=l2d_episodes,
                         include_world_model_windows=world_model, root=raw_path)


def _jpeg(frame_tensor, output_size: int | None = None) -> bytes:
    """Resize a raw (3,H,W) frame to a JPEG byte string (the single pack resize)."""
    from torchvision import transforms

    t = frame_tensor.cpu()
    if t.dtype.is_floating_point:
        t = t.clamp(0, 1)
    resize = (
        _RESIZE
        if output_size is None
        else transforms.Resize((output_size, output_size))
    )
    f = resize(_TO_PIL(t))
    b = io.BytesIO()
    f.save(b, format="JPEG", quality=90)
    return b.getvalue()


def _native_png(frame_tensor) -> bytes:
    """Encode a raw (3,H,W) frame losslessly at its native size."""
    import torch

    t = frame_tensor.cpu()
    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255.0).round().to(dtype=torch.uint8)
    b = io.BytesIO()
    _TO_PIL(t).save(b, format="PNG")
    return b.getvalue()


def init_row_worker(
    dataset_value: str,
    episodes: Optional[List[int | str]],
    raw_path: str,
    image_size: int,
    source_split: str,
    source_revision: str,
    benchmark_protocol: bool = False,
    allow_mapless: bool = False,
) -> None:
    """Build this process's plain-mode dataset for row-level decode.

    Plain mode (no WM delta_timestamps): ``lerobot_dataset[row]`` decodes one row's
    videos (6 cams + map ≈ 7 frames) instead of a whole 8-step window (~49). The
    single-row read is bit-identical to the window read at the same physical row,
    so pool bytes match the per-sample window path exactly.

    Invariant (pinned by test_reasoning_alignment_invariant.py): the sample
    enumeration is identical between WM-on and WM-off L2DDataset instances,
    because the egomotion margins (64/64) strictly dominate the WM margins
    (30/40), so ``_build_sample_index`` produces the same ``_samples``
    regardless of ``include_world_model_windows``, and this worker's plain-mode
    L2DDataset._episode_ranges + local_row resolution agree exactly with the
    parent's WM-mode dataset used to enumerate window rows in Pass A. If this
    invariant breaks (e.g. wm_num_frames * stride > 64), Pass B assembly would
    silently misalign uid ↔ row.
    """
    global _DS, _RESIZE, _TO_PIL, _DATASET_VALUE
    from torchvision import transforms
    _DATASET_VALUE = dataset_value
    _TO_PIL = transforms.ToPILImage()
    _RESIZE = transforms.Resize((image_size, image_size))
    if dataset_value == "KIT-MRT/KITScenes-Multimodal":
        from data_parsing.kit_scenes import KitScenesDataset
        scene_ids = [str(scene_id) for scene_id in episodes] if episodes else None
        _DS = KitScenesDataset(
            data_root=raw_path,
            split=source_split,
            scene_ids=scene_ids,
            image_size=image_size,
            include_world_model_windows=False,
            include_navigation=False,
            source_revision=source_revision,
            benchmark_protocol=benchmark_protocol,
            allow_mapless=allow_mapless,
        )
    else:
        from data_parsing.l2d import L2DDataset
        l2d_episodes = (
            [int(episode) for episode in episodes]
            if episodes is not None
            else None
        )
        _DS = L2DDataset(
            repo_id=dataset_value,
            revision=source_revision,
            episodes=l2d_episodes,
            include_world_model_windows=False,
            root=raw_path,
        )


def decode_row(
    task: tuple[Any, ...],
) -> tuple[
    tuple[Any, int],
    Dict[str, bytes],
    Optional[Dict[str, Optional[bytes]]],
]:
    """Decode one physical row's cameras and navigation context.

    ``task`` = (ep_idx, frame_index) is a global identity, so the worker resolves
    the local row from its own _episode_ranges (robust across processes and
    partitions). KITScenes returns lossless schema-v8 navigation members for
    current rows; L2D returns the map JPEG and the lossless native map tile.

    Each unique row is decoded exactly once per partition: the parent maps this
    over the union of all samples' window rows. Map is skipped when all-zero (same rule as pack_sample). The
    frame_id embeds episode + episode-local frame_index, so it can never
    reference another episode/scene.
    """
    from data_processing.contract_versions import UID_SCHEMA_VERSION

    if len(task) == 2:
        group_id, frame_index = task
        include_map = True
    elif len(task) == 3:
        group_id, frame_index, include_map = task
    else:
        raise ValueError(f"decode_row expects 2 or 3 values, got {task!r}")

    if _DATASET_VALUE == "KIT-MRT/KITScenes-Multimodal":
        from reactive_training_contracts import (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            REACTIVE_FRONT_CAMERA_INDEX,
        )

        scene_id = str(group_id)
        visual = _DS._load_multiview_frame(scene_id, frame_index)
        kitscenes_cams = {
            (
                f"kitscenes-{UID_SCHEMA_VERSION}-{scene_id}-"
                f"r{frame_index:06d}-c{view}"
            ): _jpeg(visual[view])
            for view in range(visual.shape[0])
        }
        extras: Optional[Dict[str, Optional[bytes]]] = None
        if include_map:
            front_frame_id = (
                f"kitscenes-{UID_SCHEMA_VERSION}-{scene_id}-"
                f"r{frame_index:06d}-c{REACTIVE_FRONT_CAMERA_INDEX}"
            )
            native_front = _DS.front_camera_for_row(
                scene_id,
                frame_index,
                image_size=REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            )
            extras = {
                "front_camera_native_jpeg": _jpeg(
                    native_front,
                    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
                ),
                "front_camera_fpn_jpeg": kitscenes_cams[front_frame_id],
            }
        return (scene_id, frame_index), kitscenes_cams, extras

    from data_parsing.l2d.dataset import CAMERA_NAMES, MAP_VIEW_NAME

    ep_idx = int(group_id)
    ep_start, ep_end = _DS._episode_ranges[ep_idx]
    local_row = ep_start + frame_index
    if local_row >= ep_end:
        raise IndexError(f"row {frame_index} outside episode {ep_idx}")
    item = _DS.lerobot_dataset[local_row]
    l2d_cams: Dict[str, bytes] = {}
    for v, cam_name in enumerate(CAMERA_NAMES):
        fid = f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame_index:06d}-c{v}"
        l2d_cams[fid] = _jpeg(item[cam_name])
    extras = None
    if include_map:
        from reactive_training_contracts import (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            REACTIVE_FRONT_CAMERA_INDEX,
        )

        # Current rows also carry the native front view and its FPN tile,
        # the T8 front contract the nuPlan parent was trained with.
        front_id = f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame_index:06d}-c{REACTIVE_FRONT_CAMERA_INDEX}"
        extras = {
            "map_jpeg": None,
            "map_png": None,
            "front_camera_native_jpeg": _jpeg(item[CAMERA_NAMES[REACTIVE_FRONT_CAMERA_INDEX]], REACTIVE_FRONT_CAMERA_IMAGE_SIZE),
            "front_camera_fpn_jpeg": l2d_cams[front_id],
        }
        map_tile = item[MAP_VIEW_NAME]
        mt = map_tile[0] if map_tile.ndim == 4 else map_tile
        if float(mt.abs().max()) > 0:
            # map.jpg is the resized tile the map encoder reads; the lossless
            # native tile is what navigation registers against, since JPEG
            # and resizing both shift the road colours it classifies.
            extras.update(map_jpeg=_jpeg(mt), map_png=_native_png(mt))
    return (ep_idx, frame_index), l2d_cams, extras


def _l2d_episode_video_frames(ep_idx: int, video_key: str, frame_indices: set, encode: Any) -> Dict[int, Any]:
    """Decode the wanted frames of one episode's video in one forward pass.

    The row path asks LeRobot for one timestamp at a time, which seeks back
    to a keyframe and decodes forward for every row and camera. Reading the
    episode once from its first frame decodes each frame once. Frames are
    chosen exactly as LeRobot chooses them: the frame nearest to
    ``from_timestamp + index / fps``. ``encode(index, frame)`` turns each
    chosen frame into bytes at once, so no decoded 1920x1080 frame outlives
    its encoding.
    """
    import av
    import numpy as np
    import torch

    meta = _DS.lerobot_dataset.meta
    episode = meta.episodes[ep_idx]
    if int(episode["episode_index"]) != ep_idx:
        raise ValueError(f"L2D meta row {ep_idx} is episode {episode['episode_index']}")
    path = _DS.lerobot_dataset.root / meta.get_video_file_path(ep_idx, video_key)
    start = float(episode[f"videos/{video_key}/from_timestamp"])
    fps = float(meta.fps)
    wanted = sorted(frame_indices)
    out: Dict[int, Any] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            raise ValueError(f"{path} has no video time base")
        container.seek(int(start / time_base), stream=stream, backward=True, any_frame=False)
        best: Dict[int, tuple] = {}
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * time_base) - start
            index = int(round(seconds * fps))
            if index < wanted[0] - 1:
                continue
            if index > wanted[-1] + 1:
                break
            if index in frame_indices:
                error = abs(seconds - index / fps)
                if index not in best or error < best[index][0]:
                    rgb = frame.to_ndarray(format="rgb24")
                    # Same float CHW in [0, 1] that LeRobot returns, so the
                    # JPEG bytes are identical to the row path.
                    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
                    best[index] = (error, encode(index, tensor))
    missing = [index for index in wanted if index not in best]
    if missing:
        raise ValueError(f"L2D episode {ep_idx} {video_key} lacks frames {missing[:5]}")
    for index, (_, encoded) in best.items():
        out[index] = encoded
    return out


def decode_l2d_episode(task: tuple[int, list, list]) -> list:
    """Decode every wanted row of one L2D episode, one pass per video.

    ``task`` is ``(episode, frame indices, current frame indices)``. Returns
    ``decode_row``-shaped results for each frame, in frame order.
    """
    from data_processing.contract_versions import UID_SCHEMA_VERSION
    from reactive_training_contracts import (
        REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
        REACTIVE_FRONT_CAMERA_INDEX,
    )

    from data_parsing.l2d.dataset import CAMERA_NAMES, MAP_VIEW_NAME

    ep_idx, frame_list, current_list = task
    frames, current = set(frame_list), set(current_list)
    front = CAMERA_NAMES[REACTIVE_FRONT_CAMERA_INDEX]

    def camera_encoder(name: str):
        def encode(index: int, tensor) -> tuple:
            native = (
                _jpeg(tensor, REACTIVE_FRONT_CAMERA_IMAGE_SIZE)
                if name == front and index in current
                else None
            )
            return _jpeg(tensor), native
        return encode

    def map_encoder(_index: int, tile) -> tuple:
        if float(tile.abs().max()) > 0:
            return _jpeg(tile), _native_png(tile)
        return None, None

    by_camera = {name: _l2d_episode_video_frames(ep_idx, name, frames, camera_encoder(name)) for name in CAMERA_NAMES}
    maps = _l2d_episode_video_frames(ep_idx, MAP_VIEW_NAME, current, map_encoder) if current else {}
    results = []
    for frame_index in sorted(frames):
        cams = {
            f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame_index:06d}-c{view}": by_camera[name][frame_index][0]
            for view, name in enumerate(CAMERA_NAMES)
        }
        extras = None
        if frame_index in current:
            map_jpeg, map_png = maps[frame_index]
            extras = {
                "map_jpeg": map_jpeg,
                "map_png": map_png,
                "front_camera_native_jpeg": by_camera[front][frame_index][1],
                "front_camera_fpn_jpeg": cams[f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame_index:06d}-c{REACTIVE_FRONT_CAMERA_INDEX}"],
            }
        results.append(((ep_idx, frame_index), cams, extras))
    return results


def pack_sample(si: int) -> Tuple[str, int, Dict[str, bytes], Dict[str, bytes]]:
    """Decode + encode sample ``si`` into per-sample members + a frame-pool
    contribution.

    Returns ``(sample_uid, num_views, members, pool)``:
      * ``members`` = the sample's own members (prefixed ``{uid}.`` by the parent):
        ``cam_i.jpg`` (reactive current frame), ``map.jpg``, ``ego.npy``,
        ``meta.json``, ``calib.json``, and — for WM shards — ``window_index.json``
        (the (step,view)→frame_id map, not the pixels).
      * ``pool`` = ``{frame_id: jpeg_bytes}`` for this sample's WM window frames.
        The parent writes each frame_id to the shared ``pool/`` dir once (dedup
        across the ~8 overlapping neighbour windows), so the same physical frame is
        never re-encoded/stored. Empty for imitation-only (non-WM) samples.

    The reasoning.json join and split bucketing are keyed by uid / meta.json.
    Boundary safety: frame_ids come from window_frame_ids, which is
    clamped to the sample's own episode.
    """
    import numpy as np
    import torch

    sample = _DS[si]
    uid = _DS.sample_uid(si)
    split_group = _DS.split_group_uid(si)
    from data_processing.dataset_snapshot import split_bucket
    members: Dict[str, bytes] = {}
    pool: Dict[str, bytes] = {}

    visual = sample["visual_tiles"]            # (V, 3, H, W)
    for cam_i in range(visual.shape[0]):
        members[f"cam_{cam_i}.jpg"] = _jpeg(visual[cam_i])

    # Only write map.jpg for a real nav-map. NVIDIA has no map and hands a
    # zeros_like placeholder; packing it would JPEG+ImageNet-normalize a black
    # tile into a nonzero per-channel constant at load time, contaminating the
    # shared map encoder (and wrongly flagging has_map=True). Skip all-zero tiles
    # so the loader's zero-fallback fires and has_map stays False.
    navigation_members = sample.get("navigation_members")
    if navigation_members is not None:
        members.update(navigation_members)
    else:
        map_tile = sample.get("map_tile")
        if map_tile is not None and float(map_tile.abs().max()) > 0:
            members["map.jpg"] = _jpeg(map_tile)

    # World-Model window: store the frames in the shared pool keyed by a global
    # frame_id, and record only the (step,view)→frame_id index per sample.
    # Consecutive 10Hz samples' 1Hz windows overlap 7/8, so the pool collapses
    # that duplication (parent dedups across samples).
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
    from data_processing.geospatial import geospatial_members
    members.update(geospatial_members(sample))
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_MEMBER,
        TRAJECTORY_XY_MEMBER,
        encode_bev_segmentation,
        encode_trajectory_xy,
    )

    trajectory_xy = sample.get("trajectory_xy_m")
    trajectory_valid = sample.get("trajectory_valid")
    if trajectory_xy is not None and trajectory_valid is not None:
        members[TRAJECTORY_XY_MEMBER] = encode_trajectory_xy(
            trajectory_xy,
            trajectory_valid,
        )

    bev_target = sample.get("bev_segmentation_target")
    bev_valid = sample.get("bev_segmentation_valid")
    if bev_target is not None and bev_valid is not None:
        members[BEV_SEGMENTATION_MEMBER] = encode_bev_segmentation(
            bev_target,
            bev_valid,
        )
    members["meta.json"] = json.dumps({
        "idx": si, "dataset": _DATASET_VALUE,
        "sample_uid": uid, "split_group_uid": split_group,
        "split_bucket": split_bucket(split_group),
        "frame_idx": _DS.frame_index(si),
    }).encode()
    members["calib.json"] = _CALIB_BYTES

    return uid, int(visual.shape[0]), members, pool
