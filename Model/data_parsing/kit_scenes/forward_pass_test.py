"""
Forward pass test for AutoE2E using the KIT Scenes Multimodal dataset.

Organizes tests into separate functions for loading the dataset, testing
inference mode, and testing training mode.

Usage:
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --scene_id <scene-uuid>

    # Whole split:
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --split train

    # Offline / CI (no pretrained weights):
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --scene_id <scene-uuid> \
        --no-pretrained

    # Benchmark with zero tensor maps (no runtime rasterization):
    cd Model/data_parsing/kit_scenes
    python forward_pass_test.py \
        --dataset_root data \
        --scene_id <scene-uuid> \
        --no-rasterize-maps
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
from model_components.auto_e2e import AutoE2E  # noqa: E402


def load_dataset(
    dataset_root: str,
    scene_id: str | None,
    split: str | None,
    rasterize_map_at_runtime: bool = True,
) -> tuple[KitScenesDataset, dict]:
    """Load KITScenes dataset and return first batch."""
    print("[dataset] Loading KITScenes dataset...")

    scene_ids = [scene_id] if scene_id is not None else None

    t0 = time.time()
    dataset = KitScenesDataset(
        data_root=dataset_root,
        backbone_name="swinv2_tiny_window8_256",
        split=split,
        scene_ids=scene_ids,
        rasterize_map_at_runtime=rasterize_map_at_runtime,
    )
    print(f"[dataset] Valid samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    t_dataset = time.time() - t0
    print(f"[dataset] Creation time: {t_dataset:.2f}s")

    # Move batch to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_device = {
        "visual_tiles": batch["visual_tiles"].to(device),
        "map_tile": batch["map_tile"].to(device),
        "visual_history": batch["visual_history"].to(device),
        "egomotion_history": batch["egomotion_history"].to(device),
        "trajectory_target": batch["trajectory_target"].to(device),
        "camera_params": batch["camera_params"].to(device),
    }

    # Print shapes
    print(f"[dataset] visual_tiles: {tuple(batch_device['visual_tiles'].shape)}")
    print(f"[dataset] map_tile: {tuple(batch_device['map_tile'].shape)}")
    print(f"[dataset] egomotion_history: {tuple(batch_device['egomotion_history'].shape)}")
    print(f"[dataset] camera_params: {tuple(batch_device['camera_params'].shape)}")
    print(f"[dataset] trajectory_target: {tuple(batch_device['trajectory_target'].shape)}")

    return dataset, batch_device


def test_bev_infer_mode(
    batch: dict,
    device: torch.device,
    pretrained_backbone: bool,
) -> None:
    """Test forward pass with BEV fusion in inference mode."""
    print("[bev_infer] Testing forward pass with BEV fusion in infer mode")

    torch.cuda.empty_cache()
    model = AutoE2E(
        is_pretrained=pretrained_backbone,
        fusion_mode="bev",
    ).to(device)

    t0 = time.time()
    trajectory, ego_hidden, _ = model(
        batch["visual_tiles"],
        batch["map_tile"],
        batch["visual_history"],
        batch["egomotion_history"],
        camera_params=batch["camera_params"],
        mode="infer",
    )
    t_forward = time.time() - t0

    print(f"[bev_infer] Forward pass time: {t_forward:.2f}s")
    print(f"[bev_infer] Trajectory shape: {trajectory.shape}")
    print(f"[bev_infer] Ego Hidden shape: {tuple(ego_hidden.shape)}")
    print("[bev_infer] PASSED")


def test_bev_train_mode(
    batch: dict,
    device: torch.device,
    pretrained_backbone: bool,
) -> None:
    """Test forward pass with BEV fusion in training mode."""
    print("[bev_train] Testing forward pass with BEV fusion in train mode")

    torch.cuda.empty_cache()
    model = AutoE2E(
        is_pretrained=pretrained_backbone,
        fusion_mode="bev",
    ).to(device)

    t0 = time.time()
    planner_loss, ego_hidden, future_visual_features = model(
        batch["visual_tiles"],
        batch["map_tile"],
        batch["visual_history"],
        batch["egomotion_history"],
        camera_params=batch["camera_params"],
        trajectory_target=batch["trajectory_target"],
        mode="train",
    )
    t_forward = time.time() - t0

    print(f"[bev_train] Forward pass time: {t_forward:.2f}s")
    print(f"[bev_train] Planner loss: {planner_loss.item():.4f}")
    print(f"[bev_train] Ego Hidden shape: {tuple(ego_hidden.shape)}")
    print(f"[bev_train] Future Visual Features shapes: {[tuple(f.shape) for f in future_visual_features]}")
    print("[bev_train] PASSED")


def main(
    dataset_root: str,
    scene_id: str | None,
    split: str | None,
    pretrained_backbone: bool = True,
    rasterize_map_at_runtime: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load dataset and get first batch
    dataset, batch = load_dataset(dataset_root, scene_id, split, rasterize_map_at_runtime=rasterize_map_at_runtime)
    print()

    # Test inference mode
    test_bev_infer_mode(batch, device, pretrained_backbone)
    print()

    # Test training mode
    test_bev_train_mode(batch, device, pretrained_backbone)
    print()

    print("All tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, default=None,
                    help="Single scene ID to test. Defaults to all scenes in the split.")
    parser.add_argument("--split", type=str, default=None,
                    help="SDK split to use (train, val, test, test_e2e, overlap_train_val).")
    parser.add_argument("--no-pretrained", action="store_true",
                    help="Skip downloading pretrained weights for the backbone and initialize randomly.")
    parser.add_argument("--no-rasterize-maps", action="store_true",
                    help="Disable runtime map rasterization; use zero tensor maps instead (for benchmarking).")
    args = parser.parse_args()

    main(
        args.dataset_root,
        args.scene_id,
        args.split,
        pretrained_backbone=not args.no_pretrained,
        rasterize_map_at_runtime=not args.no_rasterize_maps,
    )
