"""PyTorch Dataset for the yaak-ai/L2D LeRobot dataset.

Usage
-----
    from data_parsing.l2d import L2DDataset

    dataset = L2DDataset(repo_id="yaak-ai/L2D")
    sample = dataset[0]
    # sample["visual_tiles"]       (6, 3, H, W)  6 raw cameras (native, 10 Hz frame)
    # sample["map_tile"]           (3, H, W)     BEV nav-map (raw; separate branch)
    # sample["egomotion_history"]  (256,)
    # sample["visual_history"]     (896,)
    # sample["trajectory_target"]  (128,)
    # sample["episode_index"]      int
    # sample["frame_index"]        int

    # This is a pre-extraction source: frames are RAW (no resize/normalize). The
    # shard packer resizes once; the pre-extracted loader normalizes once (#77).

    # World Model training (#16, enables the JEPA loss #13): also emit the 1 Hz
    # multi-view past/future windows.
    dataset = L2DDataset(repo_id="yaak-ai/L2D", include_world_model_windows=True)
    sample = dataset[0]
    # sample["history_frames"]     (N, 6, 3, H, W)  past  @1 Hz, oldest->newest (raw)
    # sample["future_frames"]      (N, 6, 3, H, W)  future @1 Hz (JEPA targets, raw)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, TypedDict

import torch
from torch.utils.data import Dataset

import numpy as np

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # Python 3.10 (local dev venv); CI runs 3.12
    from typing_extensions import NotRequired

from data_processing.contract_versions import UID_SCHEMA_VERSION

from .camera import CAMERA_NAMES, MAP_VIEW_NAME
from .egomotion import (
    MIN_FRAMES,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    extract_egomotion,
)
from .world_model_windows import required_margins, stride_for_hz

logger = logging.getLogger(__name__)

_VISUAL_HISTORY_DIM = 896


class L2DSample(TypedDict):
    visual_tiles: torch.Tensor       # (6, 3, H, W) — 6 real cameras (10 Hz frame)
    map_tile: torch.Tensor           # (3, H, W) — BEV nav-map (map branch)
    egomotion_history: torch.Tensor  # (256,)
    visual_history: torch.Tensor     # (896,)
    trajectory_target: torch.Tensor  # (128,)
    episode_index: int
    frame_index: int
    # Present only when include_world_model_windows=True (#16, enables JEPA #13):
    # the 1 Hz multi-view past/future windows, each (N, 6, 3, H, W), oldest->newest.
    history_frames: NotRequired[torch.Tensor]
    future_frames: NotRequired[torch.Tensor]


class L2DDataset(Dataset):
    """Dataset wrapping the yaak-ai/L2D LeRobotDataset.

    Each item is one valid frame from an episode, where sufficient past and
    future context exists for egomotion extraction.

    Frames are returned RAW (lerobot's decoded CHW float in [0, 1]) — no resize
    or normalize. This is a pre-extraction source; the shard packer owns the
    single geometry-aware resize and the loader owns the single normalize (#77).

    Args:
        repo_id: HuggingFace repo ID for the dataset.
        episodes: Optional list of episode indices to load. If None, all
            episodes are used.
        local_files_only: Accepted for backward compatibility; lerobot 0.5.x
            removed this option (it now reads from cache by default), so the
            flag is currently a no-op.
        root: Optional local directory holding the already-materialized L2D data
            (lerobot layout). When set, lerobot reads from it instead of the
            shared HF cache — this is how the episode-range fan-out (#121 option
            B) makes a partition's label/pack read the raw a partition's ingest
            already downloaded, WITHOUT re-hitting HuggingFace in every pod.
    """

    def __init__(
        self,
        repo_id: str = "yaak-ai/L2D",
        episodes: list[int] | None = None,
        local_files_only: bool = False,
        include_world_model_windows: bool = False,
        wm_num_frames: int = 4,
        wm_hz: float = 1.0,
        source_hz: float = 10.0,
        reasoning_clip_only: bool = False,
        root: str | None = None,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:
            # lerobot-dataset (relaxed-deps standalone) exposes the same class
            # under the `ledataset` namespace.
            from ledataset.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self._episodes = episodes

        # World Model (#16): optionally emit the 1 Hz multi-view past/future
        # windows that the JEPA loss (#13) needs. stride converts the source rate
        # (L2D = 10 Hz) to the World Model rate (1 Hz) -> stride 10.
        self._wm_enabled = include_world_model_windows
        self._wm_num_frames = wm_num_frames
        self._wm_stride = stride_for_hz(source_hz, wm_hz)

        # Reasoning-clip mode (#98): the offline teacher only needs the FRONT
        # camera at the reasoning horizons (0/1/2/3/4 s). Instead of decoding the
        # whole WM window (8 rows x 6 cams x 1080p ~ 28s/sample), ask lerobot for
        # exactly those 5 front frames via delta_timestamps — verified bit-identical
        # to the WM-window front clip, ~2.3x faster, and far less memory (5 frames
        # vs ~48), which lets many more worker processes run without OOM. Sample
        # enumeration is UNCHANGED (the egomotion margins 64/64 dominate the WM
        # margins 30/40, so len/order match the WM path -> sample_id JOIN holds).
        self._reasoning_clip_only = reasoning_clip_only
        self._front_cam = CAMERA_NAMES[0]  # observation.images.front_left
        # World-Model window offsets in SECONDS (1 Hz): history [-(N-1)..0] + future
        # [1..N]. At source 10 Hz these are the stride-10 rows the serial WM path
        # decoded one-at-a-time. We list them oldest->newest so a single
        # delta_timestamps read returns them in window order.
        self._wm_hist_secs = [-(self._wm_num_frames - 1 - i) * (self._wm_stride / source_hz)
                              for i in range(self._wm_num_frames)]
        self._wm_fut_secs = [(i + 1) * (self._wm_stride / source_hz)
                             for i in range(self._wm_num_frames)]

        delta_timestamps = None
        if reasoning_clip_only:
            from data_processing.reasoning_label_generation.schema import HORIZON_SECONDS
            delta_timestamps = {self._front_cam: list(HORIZON_SECONDS)}
        elif include_world_model_windows:
            # Decode the ENTIRE WM window (all cameras, all history+future offsets)
            # in ONE indexed read per sample instead of re-decoding all cameras
            # once per window row (the old path did N_hist+N_fut+1 ~= 9 full
            # multi-cam decodes/sample). lerobot's timestamp-accurate seek returns
            # a [T,3,H,W] stack per camera; verified bit-identical to the per-row
            # decode. The map view only needs the current frame (offset 0.0).
            self._wm_all_secs = self._wm_hist_secs + self._wm_fut_secs  # oldest->newest
            delta_timestamps = {c: list(self._wm_all_secs) for c in CAMERA_NAMES}
            delta_timestamps[MAP_VIEW_NAME] = [0.0]

        # lerobot 0.5.x removed `local_files_only`; it now syncs from cache by
        # default and only re-fetches when `force_cache_sync=True`. We map the
        # legacy flag onto that: local_files_only=True means "don't force a
        # remote sync", which is already the default, so it is simply not passed.
        _kwargs: dict[str, Any] = {"repo_id": repo_id, "episodes": episodes}
        if root is not None:
            # Point lerobot at the partition's materialized raw dir so it loads
            # from there instead of re-downloading to the shared HF cache (#121
            # option B). lerobot reads <root>/data + <root>/meta; the FlyteDirectory
            # a partition's ingest produced has exactly this layout.
            _kwargs["root"] = root
        if delta_timestamps is not None:
            _kwargs["delta_timestamps"] = delta_timestamps
        self.lerobot_dataset = LeRobotDataset(**_kwargs)

        # This is a pre-extraction source: __getitem__ returns RAW frames (lerobot
        # yields CHW float in [0, 1]) — no resize/normalize, no timm/backbone
        # dependency. The shard packer owns the single geometry-aware resize and
        # the pre-extracted loader owns the single normalize, so the projection
        # ABI targets a known frame and there is no double-normalize (#77).

        self._episode_ranges = self._episode_local_ranges()
        self._samples = self._build_sample_index()

        if not self._samples:
            raise ValueError("No valid samples found in the dataset.")

        logger.info("L2DDataset: %d samples", len(self._samples))

    def _episode_local_ranges(self) -> dict[int, tuple[int, int]]:
        """Map each episode to its [start, end) row range in ``hf_dataset``.

        Everything downstream indexes ``hf_dataset`` / ``lerobot_dataset``,
        which are local (0-based) to the loaded subset — when ``episodes`` is a
        subset, row 0 is the first frame of the first requested episode, not a
        global frame. We derive ranges from the ``episode_index`` column rather
        than ``meta.episodes`` (whose ``dataset_from_index`` stays global, so it
        would be off by the subset offset). Local rows are what every accessor
        below actually uses.
        """
        hf = self.lerobot_dataset.hf_dataset
        ep_col = np.asarray(hf["episode_index"])

        ranges: dict[int, tuple[int, int]] = {}
        for ep_idx in np.unique(ep_col):
            rows = np.nonzero(ep_col == ep_idx)[0]
            ranges[int(ep_idx)] = (int(rows[0]), int(rows[-1]) + 1)
        return ranges

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """Enumerate all valid (episode_index, local_frame_idx) pairs.

        A frame is valid when there are _HISTORY_TIMESTEPS frames before it
        and _FUTURE_TIMESTEPS frames after it within the same episode. Indices
        are local rows into ``hf_dataset`` / ``lerobot_dataset``.
        """
        samples = []

        # A frame needs enough past/future for BOTH egomotion (64/64) and, when
        # enabled, the World Model 1 Hz window. Take the max of the two margins.
        past_margin = _HISTORY_TIMESTEPS
        future_margin = _FUTURE_TIMESTEPS
        if self._wm_enabled:
            wm_past, wm_future = required_margins(self._wm_num_frames, self._wm_stride)
            past_margin = max(past_margin, wm_past)
            future_margin = max(future_margin, wm_future)
        min_len = max(MIN_FRAMES, past_margin + future_margin + 1)

        for ep_idx, (ep_start, ep_end) in sorted(self._episode_ranges.items()):
            ep_len = ep_end - ep_start

            if ep_len < min_len:
                continue

            min_frame = past_margin
            max_frame = ep_len - future_margin - 1

            for frame_idx in range(min_frame, max_frame + 1):
                samples.append((ep_idx, ep_start + frame_idx))

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def sample_uid(self, idx: int) -> str:
        """Global, partition-independent sample id (#121 §3.1).

        Built from (episode_index, frame_index) — identity the sample already
        carries — so the SAME physical frame gets the SAME uid regardless of which
        episode subset a given pod loaded. This replaces the positional
        ``f"s{si:08d}"`` that broke the label<->pack JOIN and the S3 label cache
        under episode-range sharding. No `.`/`/` (safe as a WebDataset ``__key__``).
        """
        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        frame_index = row - ep_start
        return f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-f{frame_index:06d}"

    def split_group_uid(self, idx: int) -> str:
        """The train/val SPLIT unit (#121 §3.1): the whole EPISODE, not the frame.

        L2D frames within an episode are strongly correlated, so a per-frame split
        leaks correlated frames across train/val. Splitting on this episode-level
        id keeps train and val disjoint at the episode level.
        """
        ep_idx, _ = self._samples[idx]
        return f"l2d-e{ep_idx:06d}"

    def frame_index(self, idx: int) -> int:
        """Episode-local frame index (0-based within the episode) for sample ``idx``.

        Used to select the 1 Hz reasoning-label subset (label iff
        ``frame_index % stride == 0``) — a STABLE function of the sample's identity,
        so the labeled subset is partition-independent (#121 §3.4d).
        """
        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        return row - ep_start

    def window_frame_ids(self, idx: int) -> dict:
        """Per-(step,view) content-addressed frame ids for the WM window (#121 §3.4d).

        For dedup packing: instead of storing each window frame as its own JPEG, a
        sample stores this index and the packer keeps ONE JPEG per distinct
        ``frame_id`` in a shared pool. Each id is ``l2d-{ver}-e{ep}-r{frame_index}-c{v}``
        — GLOBAL (episode + episode-local frame_index + camera), so the SAME physical
        frame gets the SAME id across samples AND across partitions, which is what
        makes the pool dedup coherent.

        The window rows are the current row + the stride-10 offsets used by the WM
        window (history [-(N-1)*s .. 0], future [+s .. +N*s]); they are guaranteed
        to stay inside the sample's own episode because ``_build_sample_index``
        excludes edge frames (margins 64/64 ≥ WM reach), so NO id ever references a
        neighbouring episode/scene (boundary-safety requirement). Raises if a
        computed row would leave the episode (defence-in-depth; enumeration should
        already prevent it).

        Returns ``{"history": [[id per view] per step], "future": [[...] ...]}``,
        oldest→newest, matching the ``history_frames``/``future_frames`` stack order.
        """
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]
        n = self._wm_num_frames
        s = self._wm_stride
        # Same offsets the window uses: history oldest→newest ending at 0, future +s..+N*s.
        hist_offsets = [-(n - 1 - t) * s for t in range(n)]
        fut_offsets = [(t + 1) * s for t in range(n)]

        def _ids_for(offsets: list) -> list:
            steps = []
            for off in offsets:
                r = row + off
                if r < ep_start or r >= ep_end:
                    raise IndexError(
                        f"WM window row {r} (sample row {row}, offset {off}) leaves "
                        f"episode {ep_idx} [{ep_start},{ep_end}) — enumeration should "
                        f"have excluded this edge sample.")
                frame_index = r - ep_start
                steps.append([
                    f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame_index:06d}-c{v}"
                    for v in range(len(CAMERA_NAMES))
                ])
            return steps

        return {"history": _ids_for(hist_offsets), "future": _ids_for(fut_offsets)}

    def window_rows(self, idx: int) -> list[tuple[int, int]]:
        """Every (ep_idx, episode-local frame_index) row this sample's WM window
        references — WITHOUT decoding anything (#121 decode-dedup pass A).

        The parent packer unions these across the partition to get the UNIQUE row
        set, decodes each row exactly ONCE (row-level workers), and assembles
        samples from the pool afterwards. Same offsets/clamping as
        ``window_frame_ids`` (episode-bounded; raises on an edge sample), so a row
        never references a neighbouring episode/scene.
        """
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]
        n, s = self._wm_num_frames, self._wm_stride
        offsets = [-(n - 1 - t) * s for t in range(n)] + [(t + 1) * s for t in range(n)]
        out = []
        for off in offsets:
            r = row + off
            if r < ep_start or r >= ep_end:
                raise IndexError(
                    f"WM window row {r} (sample row {row}, offset {off}) leaves "
                    f"episode {ep_idx} [{ep_start},{ep_end})")
            out.append((ep_idx, r - ep_start))
        return out

    def egomotion_for(self, idx: int):
        """(egomotion_history, trajectory_target) for sample ``idx`` with ZERO
        video decode (#121 decode-dedup pass B).

        Reads only the numeric vehicle-state table — exactly the computation
        ``__getitem__`` performs before touching video — so the pack parent can
        assemble ego.npy without triggering the expensive multi-cam decode.
        """
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]
        vehicle_states = self._get_vehicle_states_window(ep_start, ep_end)
        return extract_egomotion(vehicle_states, sample_idx=row - ep_start)

    def _get_vehicle_states_window(self, ep_start: int, ep_end: int) -> np.ndarray:
        """Load vehicle state vectors for one episode (local row range).

        Reads directly from the underlying ``hf_dataset`` numeric table instead
        of indexing ``lerobot_dataset[i]``. The latter decodes all 7 camera
        videos per frame, which made this ~35s per sample; the vehicle state we
        need here is just an 8-dim vector, so we skip video decoding entirely.
        """
        hf = self.lerobot_dataset.hf_dataset
        col = hf.select_columns(["observation.state.vehicle"])
        states = np.asarray(
            col[ep_start:ep_end]["observation.state.vehicle"], dtype=np.float32
        )
        return states

    def _load_multiview_frame(self, row: int) -> torch.Tensor:
        """Decode the 6 RAW camera views for one local row -> (6, 3, H, W).

        Returns lerobot's decoded CHW float frames UNMODIFIED (no resize/
        normalize) — the shard packer owns the single geometry-aware resize.
        Decodes video, so it is the expensive path; reused for the current frame
        and (when enabled) every frame of the World Model 1 Hz windows.
        """
        item = self.lerobot_dataset[row]
        tensors = [item[cam_name] for cam_name in CAMERA_NAMES]
        return torch.stack(tensors, dim=0)

    def get_front_clip(self, idx: int) -> list[torch.Tensor]:
        """Front-camera clip at the reasoning horizons (0/1/2/3/4 s) for sample idx.

        Only valid when built with ``reasoning_clip_only=True``. Returns a list of
        ``NUM_HORIZONS`` ``[3, H, W]`` RAW front frames (current -> +4 s), decoded
        by lerobot's timestamp-accurate seek (delta_timestamps) — bit-identical to
        the WM-window front clip but far cheaper. ``idx`` uses the SAME sample
        index as ``__getitem__``/generate, so the sample_id JOIN is preserved.
        """
        if not self._reasoning_clip_only:
            raise RuntimeError(
                "get_front_clip requires L2DDataset(reasoning_clip_only=True).")
        _ep_idx, row = self._samples[idx]
        frames = self.lerobot_dataset[row][self._front_cam]  # [NUM_HORIZONS, 3, H, W]
        return [frames[h] for h in range(frames.shape[0])]

    def __getitem__(self, idx: int) -> L2DSample:
        # row is the local index into hf_dataset / lerobot_dataset.
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]

        # Offset of the current frame within its own episode.
        sample_idx_in_episode = row - ep_start

        # Load vehicle states for egomotion (episode window, no video decode)
        vehicle_states = self._get_vehicle_states_window(ep_start, ep_end)
        egomotion_history, trajectory_target = extract_egomotion(
            vehicle_states, sample_idx=sample_idx_in_episode
        )

        item = self.lerobot_dataset[row]
        visual_history = torch.zeros(_VISUAL_HISTORY_DIM, dtype=torch.float32)

        if self._wm_enabled:
            # WM mode: each camera item is a [T,3,H,W] stack over the window
            # offsets (history oldest->newest then future), decoded in ONE read.
            # Current frame = last history index; map is a [1,3,H,W] stack at 0.0.
            n_hist = self._wm_num_frames
            cur_idx = n_hist - 1  # offset 0.0 within the history part
            per_cam = [item[c] for c in CAMERA_NAMES]           # each [T,3,H,W]
            visual_tiles = torch.stack([c[cur_idx] for c in per_cam], dim=0)  # (V,3,H,W)
            map_stack = item[MAP_VIEW_NAME]
            map_tile = map_stack[0] if map_stack.ndim == 4 else map_stack
            # Build [T_hist, V, 3, H, W] and [T_fut, V, 3, H, W] (oldest->newest).
            history_frames = torch.stack(
                [torch.stack([c[t] for c in per_cam], dim=0) for t in range(n_hist)], dim=0)
            future_frames = torch.stack(
                [torch.stack([c[n_hist + t] for c in per_cam], dim=0)
                 for t in range(self._wm_num_frames)], dim=0)
        else:
            # Non-WM: single-row multi-view decode (cameras + map from one item).
            visual_tiles = torch.stack([item[cam_name] for cam_name in CAMERA_NAMES], dim=0)
            map_tile = item[MAP_VIEW_NAME]

        sample = L2DSample(
            visual_tiles=visual_tiles,
            map_tile=map_tile,
            egomotion_history=egomotion_history,
            visual_history=visual_history,
            trajectory_target=trajectory_target,
            episode_index=ep_idx,
            frame_index=sample_idx_in_episode,
        )
        if self._wm_enabled:
            sample["history_frames"] = history_frames
            sample["future_frames"] = future_frames

        return sample
