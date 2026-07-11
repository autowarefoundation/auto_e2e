"""Offline pre-extraction of the World Model 1 Hz multi-view windows (#16 / #30).

``L2DDataset(include_world_model_windows=True)`` builds the history/future
windows by decoding video per frame (~tens of seconds per sample) — fine for
correctness, far too slow for training. This module pre-extracts those windows
**once, offline**, so training reads them at disk bandwidth (see
``data_parsing/pre_extracted.py``). Frames are raw (the dataset applies no
transform); normalization happens once in the pre-extracted loader.

Design — *content addressed* (avoids the ~2*N*V x blow-up):
  Windows overlap heavily under 10 Hz -> 1 Hz subsampling, so we do NOT store
  ``2*N*V`` images per sample. Instead, for each episode we store every needed
  1 Hz multi-view frame **once** (keyed by its in-episode frame index), and each
  training sample keeps only the *list of frame indices* for its history/future
  window.

This file keeps the planning/assembly logic **pure and dataset-agnostic** (unit
tested without ``lerobot``): the actual video decode and JPEG I/O are injected as
callables, exactly like ``world_model_windows.build_windows``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from .l2d.world_model_windows import required_margins, window_offsets

# A multi-view frame is [V, 3, H, W]; a frame loader maps an in-episode index to one.
FrameLoader = Callable[[int], torch.Tensor]


def plan_episode_windows(
    episode_len: int,
    *,
    num_frames: int = 4,
    stride: int = 10,
) -> tuple[list[int], dict[int, tuple[list[int], list[int]]], list[int]]:
    """Plan which 1 Hz frames to extract for one episode.

    Args:
        episode_len: number of frames in the episode (local indices ``0..len-1``).
        num_frames: N past = N future window length (default 4, per WG 24/06).
        stride: frames between 1 Hz samples (``stride_for_hz(10, 1) == 10``).

    Returns:
        ``(valid_samples, per_sample, unique_frames)`` where
        - ``valid_samples``: local indices with a full past+future window,
        - ``per_sample``: ``{sample_idx: (history_idxs, future_idxs)}`` (local
          frame indices, oldest→newest, current frame last in history),
        - ``unique_frames``: sorted list of every frame index to decode **once**.

    Uses the same offsets/margins as the online dataset (``world_model_windows``),
    so the pre-extracted windows are identical to the on-the-fly ones.
    """
    if episode_len < 0:
        raise ValueError(f"episode_len must be >= 0, got {episode_len}")
    back, fwd = required_margins(num_frames, stride)
    hist_off, fut_off = window_offsets(num_frames, stride)

    valid_samples = list(range(back, episode_len - fwd))
    per_sample: dict[int, tuple[list[int], list[int]]] = {}
    unique: set[int] = set()
    for s in valid_samples:
        history = [s + o for o in hist_off]
        future = [s + o for o in fut_off]
        per_sample[s] = (history, future)
        unique.update(history)
        unique.update(future)
    return valid_samples, per_sample, sorted(unique)


def extract_episode(
    load_frame: FrameLoader,
    episode_len: int,
    save_frame: Callable[[int, torch.Tensor], None],
    *,
    num_frames: int = 4,
    stride: int = 10,
) -> tuple[list[int], dict[int, tuple[list[int], list[int]]]]:
    """Decode each needed 1 Hz frame **once** and hand it to ``save_frame``.

    Args:
        load_frame: ``frame_idx -> [V, 3, H, W]`` (decode + preprocess one frame).
        episode_len: frames in the episode.
        save_frame: ``(frame_idx, [V,3,H,W]) -> None`` sink (e.g. write JPEGs).
        num_frames, stride: window config (must match the training-time loader).

    Returns:
        ``(valid_samples, per_sample)`` — the per-sample window index lists to
        persist alongside the samples (e.g. as ``{key}.wm_windows.json``).
    """
    valid_samples, per_sample, unique_frames = plan_episode_windows(
        episode_len, num_frames=num_frames, stride=stride)
    for idx in unique_frames:
        save_frame(idx, load_frame(idx))
    return valid_samples, per_sample


def assemble_window(
    frame_store: Mapping[int, torch.Tensor] | Callable[[int], torch.Tensor],
    history_idxs: Sequence[int],
    future_idxs: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild a window from a frame store (training time, no video decode).

    Args:
        frame_store: either a mapping ``idx -> [V,3,H,W]`` or a callable
            ``idx -> [V,3,H,W]`` (e.g. lazily reading a cached JPEG).
        history_idxs, future_idxs: the index lists from ``per_sample``.

    Returns:
        ``(history_frames, future_frames)``, each ``[N, V, 3, H, W]`` (oldest→
        newest), matching ``world_model_windows.build_windows``' contract.
    """
    get = frame_store.__getitem__ if isinstance(frame_store, Mapping) else frame_store
    history = torch.stack([get(i) for i in history_idxs], dim=0)
    future = torch.stack([get(i) for i in future_idxs], dim=0)
    return history, future
