"""Build the teacher's temporal front-camera clip from a World-Model window (#98).

The reasoning teacher is asked to label a short driving *clip* — one front-camera
frame per horizon (0 s current + 1/2/3/4 s future) — so it can reason about how
the scene evolves over time (cut-ins, stops, yields) rather than guessing from a
single instant. This is exactly the 1 Hz window the World Model already builds:

    history_frames [N, V, 3, H, W]  (oldest→newest, current = last)
    future_frames  [N, V, 3, H, W]  (+1s, +2s, …, +N*1s)

so the horizon clip is ``[current, future[0], future[1], future[2], future[3]]``
taking the FRONT camera only (view index 0 for both L2D and NVIDIA). Front-only,
downscaled frames match NVIDIA's CoC autolabeler practice and keep the vision
token count small. Only 1 Hz World-Model samples get labelled — the reactive
10 Hz head takes no reasoning loss, so per-frame labels there would be wasted.
"""

from __future__ import annotations

from typing import List

from .schema import NUM_HORIZONS

FRONT_VIEW_INDEX = 0  # L2D: front_left, NVIDIA: camera_front_wide_120fov


def build_temporal_front_clip(history_frames, future_frames) -> List:
    """Return the ``NUM_HORIZONS`` front-camera frames for the horizon clip.

    Args:
        history_frames: ``[N, V, 3, H, W]`` past window, current = last row.
        future_frames:  ``[N, V, 3, H, W]`` future window (+1s … +Ns), oldest first.

    Returns:
        A list of ``NUM_HORIZONS`` ``[3, H, W]`` front-camera frames, one per
        horizon (0 s, 1 s, …), ordered current → furthest future.

    Raises:
        ValueError: if the windows cannot supply ``NUM_HORIZONS`` horizons
            (need the current frame + ``NUM_HORIZONS - 1`` future frames).
    """
    if history_frames is None or future_frames is None:
        raise ValueError("temporal clip needs both history_frames and future_frames")
    if history_frames.ndim != 5 or future_frames.ndim != 5:
        raise ValueError(
            f"expected [N, V, 3, H, W] windows, got {tuple(history_frames.shape)} / "
            f"{tuple(future_frames.shape)}"
        )
    needed_future = NUM_HORIZONS - 1
    if future_frames.shape[0] < needed_future:
        raise ValueError(
            f"need {needed_future} future frames for {NUM_HORIZONS} horizons, "
            f"got {future_frames.shape[0]}"
        )
    current = history_frames[-1, FRONT_VIEW_INDEX]           # 0 s
    clip = [current]
    for h in range(needed_future):                          # +1s … +4s
        clip.append(future_frames[h, FRONT_VIEW_INDEX])
    return clip
