"""Egomotion loading for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

The parquet contains 100Hz egomotion with columns:
    timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz, ax, ay, az, curvature

Downsamples to 10Hz and produces:

    egomotion_history  (256,) — 64 timesteps before the sample point x 4 signals
                                [speed, acceleration, yaw_rate, curvature]

    trajectory_target  (128,) — 64 timesteps after the sample point x 2 signals
                                [acceleration, curvature]
                                Matches the planner's per-timestep output.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from physical_ai_av.egomotion import EgomotionState

# Must match the planner's dimensions. Placing here for package export.
_HISTORY_TIMESTEPS = 64         # 6.4 s of past context at 10 Hz
_FUTURE_TIMESTEPS = 64          # 6.4 s of future prediction at 10 Hz
_NUM_HISTORY_SIGNALS = 4        # speed, acceleration, yaw_rate, curvature
_NUM_TARGET_SIGNALS = 2         # acceleration, curvature

EGOMOTION_DIM = _HISTORY_TIMESTEPS * _NUM_HISTORY_SIGNALS   # 256
TRAJECTORY_DIM = _FUTURE_TIMESTEPS * _NUM_TARGET_SIGNALS    # 128

# Minimum rows in the downsampled sequence for a clip to be usable.
MIN_ROWS = _HISTORY_TIMESTEPS + _FUTURE_TIMESTEPS + 1  # 129

# The parquet is ~100Hz; downsample to the model's 10Hz.
# TODO: check sampling integrity (e.g. no missing rows).
_SOURCE_HZ = 100.0
_TARGET_HZ = 10.0
_DOWNSAMPLE_STEP = int(_SOURCE_HZ / _TARGET_HZ)  # keep every 10th row

# To read in just the required columns
_EGOMOTION_COLUMNS = ["timestamp", "qx", "qy", "qz", "qw", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "curvature"]

def _to_history_signals(state: EgomotionState) -> np.ndarray:
    """Extract the 4 history signals from an EgomotionState.

    Returns:
        Float32 array of shape (T, 4): [speed, acceleration, yaw_rate, curvature].

    Channel 2 is YAW_RATE (rad/s), matching the L2D parser. The SDK gives an
    absolute yaw ANGLE (from the pose quaternion); emitting it here would put a
    different physical quantity in the same input slot as L2D's yaw_rate, and the
    merged multi-dataset loader feeds both into one shared ego encoder — so a
    single weight column would see absolute angle for NVIDIA and angular rate for
    L2D, corrupting the ego conditioning. Difference the (unwrapped) yaw angle
    over dt so both datasets agree.
    """
    vx = state.velocity[:, 0]
    vy = state.velocity[:, 1]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    acceleration = state.acceleration[:, 0]
    yaw_angle = np.unwrap(state.pose.rotation.as_euler("ZYX")[:, 0])
    yaw_rate = np.zeros_like(yaw_angle)
    yaw_rate[1:] = np.diff(yaw_angle) * _TARGET_HZ  # d(yaw)/dt at 10 Hz (dt=0.1s)
    curvature = state.curvature[:, 0]
    return np.stack([speed, acceleration, yaw_rate, curvature], axis=1).astype(np.float32)


def _to_target_signals(state: EgomotionState) -> np.ndarray:
    """Extract the 2 trajectory target signals from an EgomotionState.

    Returns:
        Float32 array of shape (T, 2): [acceleration, curvature].
    """
    acceleration = state.acceleration[:, 0]
    curvature = state.curvature[:, 0]
    return np.stack([acceleration, curvature], axis=1).astype(np.float32)


def _load_downsampled_df(data_root: Path, clip_uuid: str) -> pd.DataFrame:
    """Load and downsample the egomotion parquet to 10Hz."""

    parquet_path = data_root / "labels" / "egomotion" / f"{clip_uuid}.egomotion.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Egomotion parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=_EGOMOTION_COLUMNS)
    if df.empty:
        raise ValueError(f"Egomotion parquet is empty: {parquet_path}")

    return df.iloc[::_DOWNSAMPLE_STEP].reset_index(drop=True)


def load_egomotion(
    data_root: Path | str,
    clip_uuid: str,
    sample_idx: int | None = None,
    df: pd.DataFrame | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load egomotion history and trajectory target in a single parquet read.

    Selects a sample point within the clip, takes the 64 timesteps before it
    as the egomotion history, and the 64 timesteps after it as the trajectory
    target. Both windows are always fully populated — no padding.

    Args:
        data_root: Root directory of the dataset.
        clip_uuid: UUID of the clip to load.
        sample_idx: Index into the downsampled (10Hz) sequence to treat as
            the current moment. Must satisfy:
                _HISTORY_TIMESTEPS <= sample_idx <= len(df) - _FUTURE_TIMESTEPS - 1
            Defaults to the midpoint of the valid range. During training,
            pass a random value in the valid range to augment over time
            within the clip.

    Returns:
        egomotion_history: Float tensor of shape ``(256,)``.
        trajectory_target: Float tensor of shape ``(128,)``.
    """
    if df is None:
        df = _load_downsampled_df(Path(data_root), clip_uuid)

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Clip {clip_uuid} has only {len(df)} rows after downsampling "
            f"(need at least {MIN_ROWS}). Clip may be too short."
        )

    min_idx = _HISTORY_TIMESTEPS                   # 64
    max_idx = len(df) - _FUTURE_TIMESTEPS - 1         # e.g. 135 for a 200-row clip

    if sample_idx is None:
        sample_idx = (min_idx + max_idx) // 2
    elif not (min_idx <= sample_idx <= max_idx):
        raise ValueError(
            f"sample_idx {sample_idx} out of valid range [{min_idx}, {max_idx}] "
            f"for clip {clip_uuid} ({len(df)} rows)."
        )

    history_df = df.iloc[sample_idx - _HISTORY_TIMESTEPS:sample_idx].reset_index(drop=True)
    future_df = df.iloc[sample_idx + 1:sample_idx + 1 + _FUTURE_TIMESTEPS].reset_index(drop=True)

    history_signals = _to_history_signals(EgomotionState.from_egomotion_df(history_df))
    future_signals = _to_target_signals(EgomotionState.from_egomotion_df(future_df))

    return (
        torch.from_numpy(history_signals.flatten()),   # (256,)
        torch.from_numpy(future_signals.flatten()),    # (128,)
    )
