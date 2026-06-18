"""PreExtractedDataset: WebDataset-backed DataLoader for training.

Reads from local EBS shard cache (init container syncs from S3).
No video decode, no lerobot dependency. Sequential tar reads at full
disk bandwidth.

Usage:
    from data_parsing.pre_extracted import make_pre_extracted_loader

    loader = make_pre_extracted_loader("/data/shards", batch_size=8)
    for batch in loader:
        # batch["visual_tiles"]       (B, 7, 3, 256, 256)
        # batch["egomotion_history"]  (B, 256)
        # batch["visual_history"]     (B, 896)
        # batch["trajectory_target"]  (B, 128)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms

_HISTORY_STEPS = 64
_FUTURE_STEPS = 64
_HISTORY_SIGNALS = 4
_TARGET_SIGNALS = 2
_VISUAL_HISTORY_DIM = 896

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _decode_sample(sample: dict) -> dict:
    """Decode a single WebDataset sample into training tensors."""
    # WebDataset keys: "cam_0.jpg", "cam_1.jpg", ..., "ego.npy", "meta.json", "__key__"
    cam_keys = sorted(k for k in sample if k.endswith(".jpg"))
    frames = []
    for key in cam_keys:
        data = sample[key] if isinstance(sample[key], bytes) else sample[key]
        img = Image.open(io.BytesIO(data))
        frames.append(_TRANSFORM(img))

    # Ego: raw bytes → numpy → split into history and future
    ego_bytes = sample.get("ego.npy", b"")
    if isinstance(ego_bytes, bytes) and len(ego_bytes) > 0:
        ego = np.frombuffer(ego_bytes, dtype=np.float32).copy()
    else:
        ego = np.zeros(384, dtype=np.float32)

    # History: (64, 4) flattened = 256; Future: (64, 2) flattened = 128
    history_size = _HISTORY_STEPS * _HISTORY_SIGNALS
    ego_history = torch.from_numpy(ego[:history_size])
    ego_future = torch.from_numpy(ego[history_size:])

    return {
        "visual_tiles": torch.stack(frames),
        "egomotion_history": ego_history,
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "trajectory_target": ego_future,
    }


def make_pre_extracted_loader(
    shard_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "train",
    shuffle: int = 1000,
) -> wds.WebLoader:
    """Create a WebDataset DataLoader reading from local EBS shard cache.

    Args:
        shard_dir: Path to directory containing .tar shard files.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        split: Unused currently (all tars in shard_dir are loaded).
        shuffle: Shuffle buffer size (0 to disable).
    """
    shard_pattern = str(Path(shard_dir) / "*.tar")
    tarfiles = sorted(Path(shard_dir).glob("*.tar"))
    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    urls = [str(p) for p in tarfiles]

    dataset = wds.WebDataset(urls, shardshuffle=False, empty_check=False, nodesplitter=wds.split_by_worker)
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.map(_decode_sample)

    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=min(num_workers, len(tarfiles)))
    return loader
