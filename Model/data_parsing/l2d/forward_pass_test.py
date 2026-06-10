"""
Forward pass test for AutoE2E using the yaak-ai/L2D LeRobot dataset.

Tests two modes:
1. Synthetic: Creates fake tensors matching L2D shapes to verify the model
   accepts num_views=7 and all dimensions align. Always runs.
2. Live: Loads actual L2D data via LeRobotDataset. Skipped if lerobot is
   not installed or the dataset is not cached locally.

Usage:
    cd Model/data_parsing/l2d
    python forward_pass_test.py

    # With real data (requires lerobot + cached dataset):
    python forward_pass_test.py --live --episodes 0
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_MODEL_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.l2d.camera import NUM_VIEWS  # noqa: E402
from data_parsing.l2d.egomotion import (  # noqa: E402
    EGOMOTION_DIM,
    MIN_FRAMES,
    TRAJECTORY_DIM,
    _HISTORY_TIMESTEPS,
    _NUM_HISTORY_SIGNALS,
    extract_egomotion,
)
from model_components.auto_e2e import AutoE2E  # noqa: E402


def test_synthetic_forward_pass(pretrained_backbone: bool = False) -> None:
    """Run a forward pass with synthetic tensors matching L2D shapes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[synthetic] Device: {device}")

    batch_size = 2
    H, W = 256, 256

    visual_tiles = torch.randn(batch_size, NUM_VIEWS, 3, H, W, device=device)
    visual_history = torch.zeros(batch_size, 896, device=device)
    egomotion_history = torch.randn(batch_size, EGOMOTION_DIM, device=device)

    model = AutoE2E(
        num_views=NUM_VIEWS,
        is_pretrained=pretrained_backbone,
    ).to(device)

    t0 = time.time()
    trajectory, compressed, future = model(
        visual_tiles, visual_history, egomotion_history
    )
    t_fwd = time.time() - t0

    assert trajectory.shape == (batch_size, TRAJECTORY_DIM), (
        f"Expected ({batch_size}, {TRAJECTORY_DIM}), got {tuple(trajectory.shape)}"
    )
    print(f"[synthetic] trajectory: {tuple(trajectory.shape)}")
    print(f"[synthetic] compressed: {tuple(compressed.shape)}")
    print(f"[synthetic] forward pass: {t_fwd:.2f}s")
    print("[synthetic] PASSED")


def test_egomotion_extraction() -> None:
    """Test egomotion signal extraction with synthetic vehicle states."""
    T = MIN_FRAMES + 20
    vehicle_states = np.random.randn(T, 8).astype(np.float32)
    vehicle_states[:, 0] = np.abs(vehicle_states[:, 0]) + 0.1  # positive speed

    ego_hist, traj_target = extract_egomotion(vehicle_states)

    assert ego_hist.shape == (EGOMOTION_DIM,), (
        f"Expected ({EGOMOTION_DIM},), got {tuple(ego_hist.shape)}"
    )
    assert traj_target.shape == (TRAJECTORY_DIM,), (
        f"Expected ({TRAJECTORY_DIM},), got {tuple(traj_target.shape)}"
    )
    assert ego_hist.dtype == torch.float32
    assert traj_target.dtype == torch.float32
    print("[egomotion] shapes and dtypes correct")
    print("[egomotion] PASSED")


def test_live_dataset(episodes: list[int], batch_size: int, pretrained_backbone: bool) -> None:
    """Run forward pass with real L2D data."""
    try:
        from data_parsing.l2d import L2DDataset
    except ImportError as e:
        print(f"[live] SKIPPED: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live] Device: {device}")

    try:
        dataset = L2DDataset(
            repo_id="yaak-ai/L2D",
            episodes=episodes,
            local_files_only=True,
        )
    except Exception as e:
        print(f"[live] SKIPPED: cannot load dataset: {e}")
        return

    print(f"[live] Valid samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    visual_tiles = batch["visual_tiles"].to(device)
    visual_history = batch["visual_history"].to(device)
    egomotion_history = batch["egomotion_history"].to(device)
    trajectory_target = batch["trajectory_target"].to(device)

    print(f"[live] visual_tiles: {tuple(visual_tiles.shape)}")
    print(f"[live] egomotion_history: {tuple(egomotion_history.shape)}")
    print(f"[live] trajectory_target: {tuple(trajectory_target.shape)}")

    model = AutoE2E(
        num_views=NUM_VIEWS,
        is_pretrained=pretrained_backbone,
    ).to(device)

    trajectory, compressed, future = model(
        visual_tiles, visual_history, egomotion_history
    )
    print(f"[live] trajectory output: {tuple(trajectory.shape)}")
    print("[live] PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="L2D forward pass test")
    parser.add_argument("--live", action="store_true", help="Run live dataset test")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0],
                        help="Episode indices for live test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    pretrained = not args.no_pretrained

    print("=" * 60)
    print("L2D Forward Pass Test")
    print("=" * 60)

    test_egomotion_extraction()
    print()
    test_synthetic_forward_pass(pretrained_backbone=pretrained)

    if args.live:
        print()
        test_live_dataset(args.episodes, args.batch_size, pretrained)

    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
