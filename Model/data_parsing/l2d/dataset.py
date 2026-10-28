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

    # Frames are raw (no resize/normalize): the shard packer resizes once and
    # the pre-extracted loader normalizes once.

    # World Model training also emits the 1 Hz multi-view past/future windows.
    dataset = L2DDataset(repo_id="yaak-ai/L2D", include_world_model_windows=True)
    sample = dataset[0]
    # sample["history_frames"]     (N, 6, 3, H, W)  past  @1 Hz, oldest->newest (raw)
    # sample["future_frames"]      (N, 6, 3, H, W)  future @1 Hz (JEPA targets, raw)
"""

from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path
from typing import Any, Iterator, TypedDict

import torch
from torch.utils.data import Dataset

import numpy as np

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # Python 3.10 (local dev venv); CI runs 3.12
    from typing_extensions import NotRequired

from data_processing.contract_versions import UID_SCHEMA_VERSION
from data_processing.source_revisions import L2D_DATA_REVISION

from .calibration import l2d_projection_spec
from .camera import CAMERA_NAMES, CAMERA_SLOTS, MAP_VIEW_NAME
from .egomotion import (
    MIN_FRAMES,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    extract_egomotion,
)
from .world_model_windows import required_margins, stride_for_hz
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
)

logger = logging.getLogger(__name__)

# observation.state.vehicle columns read as pose; the only place they are
# interpreted (egomotion.py documents the full vector).
_HEADING, _LATITUDE, _LONGITUDE = 1, 3, 4

_VISUAL_HISTORY_DIM = 896


class L2DSample(TypedDict):
    visual_tiles: torch.Tensor       # (6, 3, H, W), 6 real cameras (10 Hz frame)
    map_tile: torch.Tensor           # (3, H, W), BEV nav-map (map branch)
    egomotion_history: torch.Tensor  # (256,)
    visual_history: torch.Tensor     # (896,)
    trajectory_target: torch.Tensor  # (128,)
    episode_index: int
    frame_index: int
    pose_current: dict[str, float | int]
    gps_future: np.ndarray           # (65, 2) float64: current + 64 future
    route_waypoints_lon_lat: np.ndarray  # (10, 2), route intent
    # Present only when include_world_model_windows=True: the 1 Hz multi-view
    # past/future windows, each (N, 6, 3, H, W), oldest->newest.
    history_frames: NotRequired[torch.Tensor]
    future_frames: NotRequired[torch.Tensor]


class L2DDataset(Dataset):
    """Dataset wrapping the yaak-ai/L2D LeRobotDataset.

    Each item is one valid frame from an episode, where sufficient past and
    future context exists for egomotion extraction.

    Frames are returned raw (lerobot's decoded CHW float in [0, 1]); the shard
    packer owns the resize and the pre-extracted loader owns the normalize.

    Args:
        repo_id: HuggingFace repo ID for the dataset.
        revision: Immutable HuggingFace dataset commit SHA.
        episodes: Optional list of episode indices to load. If None, all
            episodes are used.
        local_files_only: Ignored; lerobot 0.5.x reads from cache by default.
        root: Optional local directory holding already-materialized L2D data
            in lerobot layout. When set, lerobot reads from it instead of the
            shared HF cache.
    """

    def __init__(
        self,
        repo_id: str = "yaak-ai/L2D",
        revision: str = L2D_DATA_REVISION,
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
        self.camera_names = list(CAMERA_NAMES)
        self.camera_slots = list(CAMERA_SLOTS)

        # stride converts the source rate (L2D 10 Hz) to the World Model rate
        # (1 Hz), giving stride 10.
        self._wm_enabled = include_world_model_windows
        self._wm_num_frames = wm_num_frames
        self._wm_stride = stride_for_hz(source_hz, wm_hz)
        self._source_hz = source_hz

        # Reasoning-clip mode: the offline teacher only needs the front camera at
        # the reasoning horizons, so lerobot decodes just those frames via
        # delta_timestamps. Sample enumeration matches the WM path because the
        # egomotion margins (64/64) dominate the WM margins.
        self._reasoning_clip_only = reasoning_clip_only
        self._front_cam = CAMERA_NAMES[0]  # observation.images.front_left
        # World-Model window offsets in seconds: history [-(N-1)..0] + future
        # [1..N], oldest->newest so one delta_timestamps read returns them in
        # window order.
        self._wm_hist_secs = [-(self._wm_num_frames - 1 - i) * (self._wm_stride / source_hz)
                              for i in range(self._wm_num_frames)]
        self._wm_fut_secs = [(i + 1) * (self._wm_stride / source_hz)
                             for i in range(self._wm_num_frames)]

        delta_timestamps = None
        if reasoning_clip_only:
            from data_processing.reasoning_label_generation.schema import HORIZON_SECONDS
            delta_timestamps = {self._front_cam: list(HORIZON_SECONDS)}
        elif include_world_model_windows:
            # Decode the whole WM window in one indexed read per sample; lerobot
            # returns a [T,3,H,W] stack per camera. The map view only needs the
            # current frame (offset 0.0).
            self._wm_all_secs = self._wm_hist_secs + self._wm_fut_secs  # oldest->newest
            delta_timestamps = {c: list(self._wm_all_secs) for c in CAMERA_NAMES}
            delta_timestamps[MAP_VIEW_NAME] = [0.0]

        # `local_files_only` is not passed: lerobot 0.5.x already reads from
        # cache unless `force_cache_sync=True`.
        #
        # Pin an immutable commit rather than the CODEBASE_VERSION tag: the
        # `v3.0` tag on yaak-ai/L2D points to a snapshot whose meta.tasks and
        # episodes.parquet are shorter than the data, which makes lerobot raise
        # IndexError/KeyError.
        _kwargs: dict[str, Any] = {"repo_id": repo_id, "episodes": episodes,
                                    "revision": revision}
        if root is not None:
            # lerobot reads <root>/data and <root>/meta.
            _kwargs["root"] = root
        if delta_timestamps is not None:
            _kwargs["delta_timestamps"] = delta_timestamps
        self.lerobot_dataset = LeRobotDataset(**_kwargs)

        self._episode_ranges = self._episode_local_ranges()
        self._samples = self._build_sample_index()

        if not self._samples:
            raise ValueError("No valid samples found in the dataset.")

        logger.info("L2DDataset: %d samples", len(self._samples))

    def projection_spec(
        self,
        image_size: int | tuple[int, int] = 256,
    ) -> dict[str, Any]:
        """Return the FOV-derived L2D pinhole projection contract."""
        return l2d_projection_spec(image_size)

    def _episode_local_ranges(self) -> dict[int, tuple[int, int]]:
        """Map each episode to its [start, end) row range in ``hf_dataset``.

        Rows are local (0-based) to the loaded episode subset. Ranges come from
        the ``episode_index`` column because ``meta.episodes`` keeps global
        indices, which would be off by the subset offset.
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

        # A frame needs enough past/future for egomotion (64/64) and, when
        # enabled, the World Model 1 Hz window.
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
        """Global, partition-independent sample id.

        Built from (episode_index, frame_index) so a physical frame gets the same
        uid regardless of the loaded episode subset. Contains no `.` or `/`, so
        it is safe as a WebDataset ``__key__``.
        """
        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        frame_index = row - ep_start
        return f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-f{frame_index:06d}"

    def split_group_uid(self, idx: int) -> str:
        """The train/val split unit: the whole episode, not the frame.

        Frames within an episode are strongly correlated, so a per-frame split
        would leak them across train/val.
        """
        ep_idx, _ = self._samples[idx]
        return f"l2d-e{ep_idx:06d}"

    def frame_index(self, idx: int) -> int:
        """Episode-local frame index (0-based within the episode) for sample ``idx``.

        Selects the 1 Hz reasoning-label subset (``frame_index % stride == 0``)
        independently of partitioning.
        """
        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        return row - ep_start

    def window_frame_ids(self, idx: int) -> dict:
        """Per-(step, view) content-addressed frame ids for the WM window.

        The packer keeps one JPEG per distinct id in a shared pool. Each id is
        ``l2d-{ver}-e{ep}-r{frame_index}-c{v}``, so a physical frame gets the
        same id across samples and partitions. Raises IndexError if a window
        row would leave the sample's episode.

        Returns ``{"history": [[id per view] per step], "future": [[...] ...]}``,
        oldest to newest, matching the ``history_frames``/``future_frames`` order.
        """
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]
        n = self._wm_num_frames
        s = self._wm_stride
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
        """Every (ep_idx, episode-local frame_index) row the WM window references.

        Decodes nothing, so the packer can decode each unique row once. Uses the
        same offsets as ``window_frame_ids`` and raises IndexError on an edge
        sample.
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

    def bevformer_history_rows(self, idx: int) -> list[tuple[int, int]]:
        """Seven 0.5 s history rows, oldest first, as (episode, frame index).

        L2D keeps 64 frames (6.4 s) of past before every sample, so the 3.5 s
        T8 history the nuPlan parent's BEVFormer was trained on always fits.
        """
        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        frame = row - ep_start
        stride = round(REACTIVE_BEVFORMER_FRAME_INTERVAL_US / 1_000_000 * self._source_hz)
        if stride <= 0 or not np.isclose(stride / self._source_hz, REACTIVE_BEVFORMER_FRAME_INTERVAL_US / 1_000_000):
            raise ValueError("L2D source frequency cannot represent the T8 interval")
        rows = [frame + offset * stride for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]]
        if len(rows) != REACTIVE_BEVFORMER_HISTORY_FRAMES or rows[0] < 0:
            raise IndexError("L2D T8 history leaves the episode")
        # Row arithmetic assumes no dropped frames; check the clock as
        # KITScenes does, 50 ms around each 0.5 s step.
        _, timestamps = self._episode_numeric(ep_idx)
        for offset, history_row in zip(REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1], rows):
            expected_ns = int(timestamps[frame]) + offset * REACTIVE_BEVFORMER_FRAME_INTERVAL_US * 1000
            if abs(int(timestamps[history_row]) - expected_ns) > 50_000_000:
                raise ValueError(f"L2D episode {ep_idx} T8 history frame {history_row} is off the 0.5 s grid")
        return [(ep_idx, history_row) for history_row in rows]

    def _episode_numeric(self, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Vehicle states and timestamps of one episode, kept for the last episode.

        Samples arrive in episode order, so one cached episode saves reading
        the whole episode from Arrow again for every sample.
        """
        cached = getattr(self, "_numeric_cache", None)
        if cached is None or cached[0] != ep_idx:
            ep_start, ep_end = self._episode_ranges[ep_idx]
            cached = (ep_idx, self._get_vehicle_states_window(ep_start, ep_end), self._get_timestamps_window(ep_start, ep_end))
            self._numeric_cache = cached
        return cached[1], cached[2]

    def bevformer_history_frame_ids(self, idx: int) -> list[list[str]]:
        """Shared-pool frame IDs of the seven T8 history rows."""
        return [
            [f"l2d-{UID_SCHEMA_VERSION}-e{ep_idx:06d}-r{frame:06d}-c{view}" for view in range(len(CAMERA_NAMES))]
            for ep_idx, frame in self.bevformer_history_rows(idx)
        ]

    def reactive_calibration_for(self, idx: int, *, image_size: int) -> dict[str, object]:
        """Current, native-front, and current-ego-aligned T8 projections.

        The same contract KITScenes writes. History cameras see the current
        ego frame through the dead-reckoned planar motion between the poses.
        """
        from reactive_training_contracts import REACTIVE_FRONT_CAMERA_IMAGE_SIZE, REACTIVE_FRONT_CAMERA_INDEX

        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        states, timestamps = self._episode_numeric(ep_idx)
        frame = row - ep_start
        history_frames = [history_frame for _, history_frame in self.bevformer_history_rows(idx)]
        # K @ [R|t] of the current frame; a history camera sees a current-ego
        # point after moving it into that frame's ego: P @ history_from_current.
        current = _projection_matrices(image_size)
        poses = _dead_reckoned_poses(states, timestamps, history_frames[0], frame)
        history_matrix = np.stack([
            current @ np.linalg.inv(poses[history_frame - history_frames[0]])
            for history_frame in history_frames
        ]).astype(np.float32)
        front = _projection_matrices(REACTIVE_FRONT_CAMERA_IMAGE_SIZE)[REACTIVE_FRONT_CAMERA_INDEX:REACTIVE_FRONT_CAMERA_INDEX + 1]
        if not (np.isfinite(history_matrix).all() and np.isfinite(front).all()):
            raise ValueError("L2D T8 projections are non-finite")
        return {
            "dataset": self.repo_id,
            "geometry_type": "pinhole",
            "image_size": image_size,
            "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
            "front_camera_image_size": REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            "front_camera_fpn_image_size": image_size,
            "front_projection": {"matrix": front.tolist(), "type": "pinhole"},
            "projection": {"matrix": current.astype(np.float32).tolist(), "type": "pinhole"},
            "history_projection": {"matrix": history_matrix.tolist(), "reference_frame": "current_ego", "type": "pinhole"},
            "temporal_frame_interval_us": REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
            "temporal_frame_offsets": list(REACTIVE_BEVFORMER_FRAME_OFFSETS),
        }

    def egomotion_for(self, idx: int):
        """(egomotion_history, trajectory_target) for sample ``idx`` without video decode."""
        ego_history, trajectory_target, _, _ = self.numeric_for(idx)
        return ego_history, trajectory_target

    def geospatial_for(
        self, idx: int,
    ) -> tuple[dict[str, float | int], np.ndarray]:
        """Return absolute current pose + current/future GPS without video decode."""
        _, _, pose_current, gps_future = self.numeric_for(idx)
        return pose_current, gps_future

    def numeric_for(self, idx: int):
        """Return every numeric sample field with one read and no video decode."""
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]
        vehicle_states = self._get_vehicle_states_window(ep_start, ep_end)
        timestamps_ns = self._get_timestamps_window(ep_start, ep_end)
        sample_idx = row - ep_start
        ego_history, trajectory_target = extract_egomotion(
            vehicle_states, sample_idx=sample_idx
        )
        pose_current, gps_future = self._extract_geospatial(
            vehicle_states, timestamps_ns, sample_idx
        )
        return ego_history, trajectory_target, pose_current, gps_future

    def route_waypoints_for(self, idx: int) -> np.ndarray:
        """Return the current row's OSM-snapped [longitude, latitude] route."""
        _ep_idx, row = self._samples[idx]
        hf = self.lerobot_dataset.hf_dataset
        column = hf.select_columns(["observation.state.waypoints"])
        waypoints = np.asarray(
            column[row]["observation.state.waypoints"],
            dtype=np.float64,
        )
        if waypoints.shape != (10, 2):
            raise ValueError(
                "L2D route waypoints must have shape [10,2]"
            )
        if not np.isfinite(waypoints).all():
            raise ValueError("L2D route waypoints contain non-finite values")
        return waypoints

    def _get_vehicle_states_window(self, ep_start: int, ep_end: int) -> np.ndarray:
        """Load vehicle state vectors for one episode (local row range).

        Reads the ``hf_dataset`` numeric table directly, because indexing
        ``lerobot_dataset[i]`` would decode every camera video per frame.
        """
        hf = self.lerobot_dataset.hf_dataset
        col = hf.select_columns(["observation.state.vehicle"])
        states = np.asarray(
            col[ep_start:ep_end]["observation.state.vehicle"], dtype=np.float32
        )
        return states

    def _get_timestamps_window(self, ep_start: int, ep_end: int) -> np.ndarray:
        """Load raw nanosecond Unix timestamps for one episode."""
        hf = self.lerobot_dataset.hf_dataset
        col = hf.select_columns(["observation.state.timestamp"])
        values = np.asarray(
            col[ep_start:ep_end]["observation.state.timestamp"], dtype=np.int64
        )
        return values.reshape(-1)

    @staticmethod
    def _extract_geospatial(
        vehicle_states: np.ndarray,
        timestamps_ns: np.ndarray,
        sample_idx: int,
    ) -> tuple[dict[str, float | int], np.ndarray]:
        """Extract map pose fields aligned to the egomotion sample index."""
        end = sample_idx + _FUTURE_TIMESTEPS + 1
        if end > len(vehicle_states) or sample_idx >= len(timestamps_ns):
            raise IndexError(
                f"geospatial window [{sample_idx}:{end}] exceeds episode rows "
                f"{len(vehicle_states)}"
            )
        current = vehicle_states[sample_idx]
        # The L2D source vector is float32. Casting here prevents further loss in
        # storage but cannot recover precision absent from the source.
        gps_future = np.asarray(
            vehicle_states[sample_idx:end, 3:5], dtype=np.float64
        )
        pose_current: dict[str, float | int] = {
            "latitude_deg": float(current[3]),
            "longitude_deg": float(current[4]),
            "heading_deg_cw_from_north": float(current[1]),
            "timestamp_ns": int(timestamps_ns[sample_idx]),
            # L2D exposes no GPS-accuracy channel.
            "gps_accuracy_m": float("nan"),
        }
        return pose_current, gps_future

    def episode_indices(self) -> list[int]:
        """Return loaded source episode ids in stable order."""
        return sorted(self._episode_ranges)

    def episode_path(self, episode_index: int) -> np.ndarray:
        """Return ``[lat, lon, heading, timestamp_ns]`` for the full episode.

        The route artifact stores all columns as float64. Per-sample pose members
        retain the nanosecond timestamp as exact int64.
        """
        ep_start, ep_end = self._episode_ranges[episode_index]
        states = self._get_vehicle_states_window(ep_start, ep_end)
        timestamps = self._get_timestamps_window(ep_start, ep_end)
        path = np.empty((len(states), 4), dtype=np.float64)
        path[:, 0] = states[:, 3]
        path[:, 1] = states[:, 4]
        path[:, 2] = states[:, 1]
        path[:, 3] = timestamps
        return path

    def episode_track(
        self, episode_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(longitude, latitude, heading, timestamp_ns)`` per frame.

        Unlike :meth:`episode_path`, timestamps stay exact int64 nanoseconds,
        which navigation needs for its 500 ms anchor grid.
        """
        ep_start, ep_end = self._episode_ranges[episode_index]
        states = self._get_vehicle_states_window(ep_start, ep_end)
        timestamps = self._get_timestamps_window(ep_start, ep_end)
        return (
            states[:, _LONGITUDE].astype(np.float64),
            states[:, _LATITUDE].astype(np.float64),
            states[:, _HEADING].astype(np.float64),
            timestamps,
        )

    def sample_pose_records(self) -> Iterator[dict[str, Any]]:
        """Yield exact per-sample pose rows for the partition geo parquet."""
        by_episode: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for idx, (ep_idx, row) in enumerate(self._samples):
            if ep_idx not in by_episode:
                ep_start, ep_end = self._episode_ranges[ep_idx]
                by_episode[ep_idx] = (
                    self._get_vehicle_states_window(ep_start, ep_end),
                    self._get_timestamps_window(ep_start, ep_end),
                )
            states, timestamps = by_episode[ep_idx]
            ep_start, _ = self._episode_ranges[ep_idx]
            frame_idx = row - ep_start
            state = states[frame_idx]
            yield {
                "sample_uid": self.sample_uid(idx),
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "latitude_deg": float(state[3]),
                "longitude_deg": float(state[4]),
                "heading_deg_cw_from_north": float(state[1]),
                "timestamp_ns": int(timestamps[frame_idx]),
                "gps_accuracy_m": None,
            }

    def _load_multiview_frame(self, row: int) -> torch.Tensor:
        """Decode the 6 raw camera views for one local row -> (6, 3, H, W)."""
        item = self.lerobot_dataset[row]
        tensors = [item[cam_name] for cam_name in CAMERA_NAMES]
        return torch.stack(tensors, dim=0)

    def get_front_clip(self, idx: int) -> list[torch.Tensor]:
        """Front-camera clip at the reasoning horizons for sample ``idx``.

        Only valid with ``reasoning_clip_only=True``. Returns ``NUM_HORIZONS``
        raw ``[3, H, W]`` front frames (current -> +4 s); ``idx`` matches
        ``__getitem__`` indexing.
        """
        if not self._reasoning_clip_only:
            raise RuntimeError(
                "get_front_clip requires L2DDataset(reasoning_clip_only=True).")
        _ep_idx, row = self._samples[idx]
        frames = self.lerobot_dataset[row][self._front_cam]  # [NUM_HORIZONS, 3, H, W]
        return [frames[h] for h in range(frames.shape[0])]

    def __getitem__(self, idx: int) -> L2DSample:
        ep_idx, row = self._samples[idx]
        ep_start, _ = self._episode_ranges[ep_idx]
        sample_idx_in_episode = row - ep_start

        (
            egomotion_history,
            trajectory_target,
            pose_current,
            gps_future,
        ) = self.numeric_for(idx)

        item = self.lerobot_dataset[row]
        visual_history = torch.zeros(_VISUAL_HISTORY_DIM, dtype=torch.float32)

        if self._wm_enabled:
            # Each camera item is a [T,3,H,W] stack over the window offsets
            # (history oldest->newest, then future). The current frame is the
            # last history index; the map is a [1,3,H,W] stack at 0.0.
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
            pose_current=pose_current,
            gps_future=gps_future,
            route_waypoints_lon_lat=self.route_waypoints_for(idx),
        )
        if self._wm_enabled:
            sample["history_frames"] = history_frames
            sample["future_frames"] = future_frames

        return sample


def track_points(data_root: str | Path, episodes: list[int], *, decimals: int = 4) -> np.ndarray:
    """Distinct ``[longitude, latitude]`` of the episodes' poses and route waypoints.

    Reads the LeRobot parquet files directly, so no video is opened. Points
    are rounded to ``decimals`` degrees (about 10 m) and deduplicated, which
    keeps the result small however many frames the episodes have.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = set(int(episode) for episode in episodes)
    seen: set[int] = set()
    points = np.empty((0, 2))

    def flat(column) -> np.ndarray:
        # Nested lists flatten in Arrow without building Python objects.
        values = column.combine_chunks()
        # Real L2D waypoints are large_list<large_list<float>>, vehicle a
        # fixed_size_list<float>[8].
        while pa.types.is_list(values.type) or pa.types.is_large_list(values.type) or pa.types.is_fixed_size_list(values.type):
            values = values.flatten()
        return values.to_numpy(zero_copy_only=False).astype(np.float64)

    for path in sorted(Path(data_root, "data").rglob("*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "observation.state.vehicle", "observation.state.waypoints"],
        )
        table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(sorted(wanted))))
        if not table.num_rows:
            continue
        seen.update(int(e) for e in pc.unique(table["episode_index"]).to_pylist())
        vehicle = flat(table["observation.state.vehicle"]).reshape(table.num_rows, -1)
        routes = flat(table["observation.state.waypoints"]).reshape(-1, 2)
        batch = np.round(np.concatenate([vehicle[:, [_LONGITUDE, _LATITUDE]], routes]), decimals)
        # Deduplicate per file, so memory follows distinct places, not frames.
        points = np.unique(np.concatenate([points, batch[np.isfinite(batch).all(axis=1)]]), axis=0)
    if seen != wanted:
        raise ValueError(f"L2D data files lack episodes {sorted(wanted - seen)[:5]}")
    return points



@functools.lru_cache(maxsize=4)
def _projection_matrices(image_size: int) -> np.ndarray:
    """The rig projection at one size; it reads and hashes the extrinsics YAML."""
    from .calibration import compute_l2d_projection_matrices

    matrices = compute_l2d_projection_matrices(image_size).astype(np.float64)
    matrices.setflags(write=False)
    return matrices


def _dead_reckoned_poses(states: np.ndarray, timestamps_ns: np.ndarray, first: int, current: int) -> np.ndarray:
    """Current-ego-from-ego poses of frames ``first..current`` by dead reckoning.

    L2D stores latitude and longitude as float32, about 0.4 m steps in
    Germany, which would shift a 10 m distant point by tens of pixels between
    T8 frames. Speed and heading integrated at 10 Hz give the 3.5 s relative
    motion smoothly; GPS is not needed over so short a span.
    """
    span = states[first:current + 1].astype(np.float64)
    seconds = np.diff(timestamps_ns[first:current + 1].astype(np.int64)) / 1e9
    speed = span[:, 0] / 3.6  # km/h to m/s
    heading = np.radians(span[:, _HEADING])
    # Trapezoidal steps in ENU, true-north heading clockwise.
    step_speed = 0.5 * (speed[:-1] + speed[1:])
    step_heading = heading[:-1] + 0.5 * np.angle(np.exp(1j * (heading[1:] - heading[:-1])))
    east = np.concatenate([[0.0], np.cumsum(step_speed * seconds * np.sin(step_heading))])
    north = np.concatenate([[0.0], np.cumsum(step_speed * seconds * np.cos(step_heading))])
    yaw = np.pi / 2 - heading
    world_from_ego = np.tile(np.eye(4), (len(span), 1, 1))
    world_from_ego[:, 0, 0], world_from_ego[:, 0, 1] = np.cos(yaw), -np.sin(yaw)
    world_from_ego[:, 1, 0], world_from_ego[:, 1, 1] = np.sin(yaw), np.cos(yaw)
    world_from_ego[:, 0, 3], world_from_ego[:, 1, 3] = east, north
    return np.linalg.inv(world_from_ego[-1]) @ world_from_ego
