"""1 Hz sequential multi-view windows for the World Model (#16, enables #13).

The reactive model runs at 10 Hz on the *current* frame; the World Model runs at
~1 Hz over a window of past frames and predicts *future* frame features (the JEPA
feature-reconstruction loss, #13). This module turns a higher-rate frame source
(L2D is 10 Hz) into the World Model's **past** and **future** windows by striding
(``stride = round(source_hz / world_model_hz)`` — e.g. 10 at 10 Hz → 1 Hz).

It is **dataset-agnostic**: it takes a ``load_frame(row) -> [V, 3, H, W]``
callable, so the windowing logic is unit-testable without the real (lerobot)
L2D dataset. ``L2DDataset`` wires its own frame loader into ``build_windows``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def stride_for_hz(source_hz: float, world_model_hz: float) -> int:
    """Frames to skip between 1 Hz samples (>= 1)."""
    if source_hz <= 0 or world_model_hz <= 0:
        raise ValueError("source_hz and world_model_hz must be > 0")
    return max(1, round(source_hz / world_model_hz))


def window_offsets(num_frames: int, stride: int) -> tuple[list[int], list[int]]:
    """Row offsets (relative to the current row) for the past/future windows.

    Returns ``(history_offsets, future_offsets)``, both oldest → newest:
    - history: ``num_frames`` frames *ending at* the current row (current last):
      ``[-(N-1)*stride, …, -stride, 0]``
    - future: the next ``num_frames`` frames: ``[+stride, +2*stride, …, +N*stride]``
    """
    if num_frames < 1 or stride < 1:
        raise ValueError("num_frames and stride must be >= 1")
    history = [-(num_frames - 1 - i) * stride for i in range(num_frames)]
    future = [(i + 1) * stride for i in range(num_frames)]
    return history, future


def required_margins(num_frames: int, stride: int) -> tuple[int, int]:
    """Frames needed (before, after) the current row for a full window.

    before = ``(N-1)*stride`` (history reaches back to the oldest past frame),
    after  = ``N*stride`` (future reaches the furthest target).
    """
    if num_frames < 1 or stride < 1:
        raise ValueError("num_frames and stride must be >= 1")
    return (num_frames - 1) * stride, num_frames * stride


def build_windows(
    load_frame: Callable[[int], torch.Tensor],
    row: int,
    ep_start: int,
    ep_end: int,
    num_frames: int = 4,
    stride: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 1 Hz past/future multi-view windows for ``row``.

    Args:
        load_frame: ``global_row -> [V, 3, H, W]`` multi-view frame loader.
        row: current frame's (local) row index.
        ep_start, ep_end: ``[start, end)`` local row range of ``row``'s episode;
            the whole window must stay inside it (no cross-episode leakage).
        num_frames: frames per window (N_past = N_future).
        stride: rows between 1 Hz samples.

    Returns:
        ``(history_frames, future_frames)``, each ``[num_frames, V, 3, H, W]``,
        ordered oldest → newest.

    Raises:
        IndexError: if the window does not fit within the episode (the caller's
            valid-index enumeration must guarantee the margins; see
            :func:`required_margins`).
    """
    hist_off, fut_off = window_offsets(num_frames, stride)
    if row + hist_off[0] < ep_start or row + fut_off[-1] >= ep_end:
        raise IndexError(
            f"World-model window for row {row} exceeds episode "
            f"[{ep_start}, {ep_end}) (need {required_margins(num_frames, stride)} "
            f"frames before/after)."
        )
    history = torch.stack([load_frame(row + o) for o in hist_off], dim=0)
    future = torch.stack([load_frame(row + o) for o in fut_off], dim=0)
    return history, future
