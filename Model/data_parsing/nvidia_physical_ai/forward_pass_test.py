"""
Forward pass test for AutoE2E using the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

Loads a single clip, runs one batch through the data pipeline, and optionally
through the model. No backprop — this is a data pipeline and model integration
check only.

Usage:
    python Model/data_parsing/nvidia_physical_ai/forward_pass_test.py \
        --dataset_root /path/to/nvidia_av_camera_subset \
        --clip_uuid fd1d1b6b-59bf-4292-8295-5028aa6aa5e3
"""

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

_MODEL_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.nvidia_physical_ai import NvidiaAVDataset


def main(dataset_root: str, clip_uuid: str, batch_size: int = 4) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if clip_uuid is not None:
        clip_uuids = [clip_uuid]
    else:
        parquet_dir = pathlib.Path(dataset_root) / "labels" / "egomotion"
        clip_uuids = [p.stem.split(".")[0] for p in sorted(parquet_dir.glob("*.egomotion.parquet"))]
        print(f"Discovered {len(clip_uuids)} clips")

    dataset = NvidiaAVDataset(
        data_root=dataset_root,
        backbone_name="swin_tiny_patch4_window7_224.ms_in22k",
        clip_uuids=clip_uuids,
    )
    print(f"Valid samples in clip: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    visual_tiles = batch["visual_tiles"].to(device)           # (B, 8, 3, 224, 224)
    visual_history = batch["visual_history"].to(device)       # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device) # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device) # (B, 128)

    print(f"visual_tiles: {tuple(visual_tiles.shape)}")
    print(f"egomotion_history: {tuple(egomotion_history.shape)}")
    print(f"trajectory_target: {tuple(trajectory_target.shape)}")
    print(f"sample_idx: {batch['sample_idx'].tolist()}")

    # --- forward pass ---
    from model_components.auto_e2e import AutoE2E
    model = AutoE2E().to(device)
    trajectory_, compressed_, future_ = model(visual_tiles, visual_history, egomotion_history)
    print(f"trajectory output: {tuple(trajectory_.shape)}") 
    print(f"compressed visual feature output: {tuple(compressed_.shape)}") 
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
    args = parser.parse_args()

    main(args.dataset_root, args.clip_uuid, args.batch_size)