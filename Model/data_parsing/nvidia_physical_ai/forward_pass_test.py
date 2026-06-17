"""
Forward pass test for AutoE2E using the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

Loads a single clip, runs one batch through the data pipeline, and optionally
through the model. No backprop — this is a data pipeline and model integration
check only.

Usage:
    cd Model/data_parsing/nvidia_physical_ai
    python forward_pass_test.py \
        --dataset_root data \
        --clip_uuid fd1d1b6b-59bf-4292-8295-5028aa6aa5e3

    # Offline / CI (no pretrained weights):
    cd Model/data_parsing/nvidia_physical_ai
    python forward_pass_test.py \
        --dataset_root data \
        --clip_uuid fd1d1b6b-59bf-4292-8295-5028aa6aa5e3 \
        --no-pretrained
"""

import argparse
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

_MODEL_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.nvidia_physical_ai import NvidiaAVDataset  # noqa: E402
from model_components.auto_e2e import AutoE2E  # noqa: E402


def main(dataset_root: str, clip_uuid: str, batch_size: int = 4, pretrained_backbone: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if clip_uuid is not None:
        clip_uuids = [clip_uuid]
    else:
        parquet_dir = pathlib.Path(dataset_root) / "labels" / "egomotion"
        clip_uuids = [p.stem.split(".")[0] for p in sorted(parquet_dir.glob("*.egomotion.parquet"))]
        print(f"Discovered {len(clip_uuids)} clips")

    t0 = time.time()
    dataset = NvidiaAVDataset(
        data_root=dataset_root,
        backbone_name="swinv2_tiny_window8_256",
        clip_uuids=clip_uuids,
    )
    print(f"Valid samples in clip: {len(dataset)}")

    egomotion_mb = sum(df.memory_usage(deep=True).sum() for df in dataset._egomotion_dfs.values()) / (1024 ** 2)
    timestamps_mb = sum(arr.nbytes for arr in dataset._camera_timestamps.values()) / (1024 ** 2)
    print(f"Egomotion cache: {egomotion_mb:.1f} MB")
    print(f"Timestamp cache: {timestamps_mb:.1f} MB")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    visual_tiles = batch["visual_tiles"].to(device)           # (B, 8, 3, 256, 256)
    visual_history = batch["visual_history"].to(device)       # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device) # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device) # (B, 128)
    t_dataset = time.time() - t0

    print(f"Dataset creation for {len(clip_uuids)} clips: {t_dataset:.2f}s")

    print(f"visual_tiles: {tuple(visual_tiles.shape)}")
    print(f"visual_history: {tuple(visual_history.shape)}")
    print(f"egomotion_history: {tuple(egomotion_history.shape)}")
    print(f"trajectory_target: {tuple(trajectory_target.shape)}")

    # --------------------
    # forward pass
    model = AutoE2E(is_pretrained=pretrained_backbone).to(device)

    t0 = time.time()
    trajectory_, ego_hidden_, future_ = model(visual_tiles, visual_history, egomotion_history)
    t_forward = time.time() - t0
    print(f"Forward pass: {t_forward:.2f}s")

    print(f"trajectory output: {tuple(trajectory_.shape)}")
    print(f"ego_hidden output: {tuple(ego_hidden_.shape)}")
    print(f"future visual features: {[tuple(f.shape) for f in future_]}")
    # --------------------

    # TODO (training): wire in loss and backprop
    # loss = F.mse_loss(trajectory, trajectory_target)
    # loss.backward()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--clip_uuid", type=str, default=None,
                    help="Single clip UUID to test. Defaults to all clips under dataset_root.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no-pretrained",  action="store_true",
                    help="Skip downloading pretrained weights for the backbone and initialize randomly.")
    args = parser.parse_args()

    main(args.dataset_root, args.clip_uuid, args.batch_size, pretrained_backbone = not args.no_pretrained)
