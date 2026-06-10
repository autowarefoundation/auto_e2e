"""
Forward pass test for AutoE2E using the KIT Scenes Multimodal dataset.

Loads a single scene (or split), runs one batch through the data pipeline, and
optionally through the model. 

Usage:
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --scene_id <scene-uuid>

    # Whole split:
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --split test_e2e

    # Offline / CI (no pretrained weights):
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --scene_id <scene-uuid> \
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

from data_parsing.kit_scenes import KitScenesDataset  # noqa: E402
from model_components.auto_e2e import AutoE2E # noqa: E402


def main(
    dataset_root: str,
    scene_id: str | None,
    split: str | None,
    batch_size: int = 4,
    pretrained_backbone: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scene_ids = [scene_id] if scene_id is not None else None

    t0 = time.time()
    dataset = KitScenesDataset(
        data_root=dataset_root,
        backbone_name="swinv2_tiny_window8_256",
        split=split,
        scene_ids=scene_ids,
    )
    print(f"Valid samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    visual_tiles = batch["visual_tiles"].to(device)           # (B, 8, 3, H, W)
    visual_history = batch["visual_history"].to(device)       # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device) # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device) # (B, 128)
    camera_params = batch["camera_params"].to(device)
    t_dataset = time.time() - t0

    print(f"Dataset creation: {t_dataset:.2f}s")

    print(f"visual_tiles: {tuple(visual_tiles.shape)}")
    print(f"egomotion_history: {tuple(egomotion_history.shape)}")
    print(f"trajectory_target: {tuple(trajectory_target.shape)}")
    print(f"camera_params: {tuple(camera_params.shape)}")

    # --------------------
    # forward pass
    model = AutoE2E(is_pretrained=pretrained_backbone).to(device)

    t0 = time.time()
    trajectory_, compressed_, future_ = model(visual_tiles, visual_history, egomotion_history)
    t_forward = time.time() - t0
    print(f"Forward pass: {t_forward:.2f}s")

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
    parser.add_argument("--scene_id", type=str, default=None,
                    help="Single scene ID to test. Defaults to all scenes in the split.")
    parser.add_argument("--split", type=str, default=None,
                    help="SDK split to use (train, val, test, test_e2e, overlap_train_val).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true",
                    help="Skip downloading pretrained weights for the backbone and initialize randomly.")
    args = parser.parse_args()

    main(
        args.dataset_root,
        args.scene_id,
        args.split,
        args.batch_size,
        pretrained_backbone=not args.no_pretrained,
    )