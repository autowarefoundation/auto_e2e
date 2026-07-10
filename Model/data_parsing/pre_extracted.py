"""PreExtractedDataset: WebDataset-backed DataLoader for training.

Reads from local EBS shard cache (init container syncs from S3).
No video decode, no lerobot dependency. Sequential tar reads at full
disk bandwidth.

Usage:
    from data_parsing.pre_extracted import make_pre_extracted_loader

    loader = make_pre_extracted_loader("/data/shards", batch_size=8)
    for batch in loader:
        # batch["visual_tiles"]       (B, V, 3, 256, 256)  V real cameras
        # batch["map_input"]          (B, 3, 256, 256)     nav-map (map branch)
        # batch["egomotion_history"]  (B, 256)
        # batch["visual_history"]     (B, 896)
        # batch["trajectory_target"]  (B, 128)
        # batch["camera_params"]      (B, V, 3, 4)         if the manifest has calib
"""

from __future__ import annotations

import io
import json
import re
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

# Camera frames are keyed "cam_<i>.jpg"; the nav-map is "map.jpg". The map MUST
# NOT be picked up as a camera view — matching cam_ explicitly (not any ".jpg")
# keeps V correct and stops the map being double-counted in the BEV projection.
_CAM_KEY_RE = re.compile(r"^cam_\d+\.jpg$")


def _decode_image(data) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data
    return _TRANSFORM(img)


def _decode_sample(sample: dict) -> dict:
    """Decode a WebDataset sample into training tensors (geometry-free).

    Calibration is a per-dataset rig constant, not per-sample, so it is NOT
    decoded here — it is reconstructed once by ``make_pre_extracted_loader`` and
    exposed on the loader as ``.projection`` / ``.geometry_type``.
    """
    # Keys: "cam_0.jpg" ... "cam_{V-1}.jpg", optional "map.jpg",
    # "ego.npy", "meta.json", "__key__".
    cam_keys = sorted(
        (k for k in sample if _CAM_KEY_RE.match(k)),
        key=lambda k: int(k[len("cam_"):-len(".jpg")]),
    )
    frames = [_decode_image(sample[k]) for k in cam_keys]

    # Map view -> map branch. Absent (legacy shards / NVIDIA zeros) -> zeros.
    if "map.jpg" in sample:
        map_input = _decode_image(sample["map.jpg"])
    else:
        ref = frames[0] if frames else torch.zeros(3, 256, 256)
        map_input = torch.zeros_like(ref)

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
        "map_input": map_input,
        "egomotion_history": ego_history,
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "trajectory_target": ego_future,
    }


def load_projection_from_manifest(shard_dir: str):
    """Reconstruct the per-dataset projection operator from manifest.json.

    Returns ``(projection, geometry_type)``. A dataset with real calibration
    stores an operator spec under ``projection`` in its manifest:

        {"geometry_type": "pinhole",
         "projection": {"type": "pinhole", "matrix": [[...]]}}   # [V,3,4]
        {"geometry_type": "ftheta",
         "projection": {"type": "ftheta", "t_camera_ego": [...],  # [V,4,4]
                        "fw_poly": [...], "cx": [...], "cy": [...],
                        "image_wh": [...], "max_theta": ...}}  # native (W,H), FOV

    A dataset without calibration (pseudo geometry, e.g. L2D) returns
    ``(None, "pseudo")`` and the caller runs the explicit pseudo path. This is
    the single geometry-reconstruction point, keeping the pinhole/f-theta split
    out of the training loop.
    """
    from model_components.view_fusion.projection import (
        FThetaProjection,
        PinholeProjection,
    )

    mpath = Path(shard_dir) / "manifest.json"
    # Missing manifest -> pseudo (a legacy shard has no geometry). But a manifest
    # that EXISTS and cannot be read must RAISE: silently degrading a calibrated
    # run to pseudo geometry would corrupt experiments. Corrupt/unreadable is a
    # hard error, not a fallback.
    if not mpath.exists():
        return None, "pseudo"
    try:
        manifest = json.loads(mpath.read_text())
    except (ValueError, OSError) as e:
        raise ValueError(
            f"manifest.json at {mpath} exists but could not be parsed ({e}); "
            f"refusing to silently fall back to pseudo geometry."
        ) from e

    spec = manifest.get("projection")
    if spec is None:
        return None, manifest.get("geometry_type", "pseudo")

    kind = spec.get("type")
    if kind in ("pinhole", "rectified_pinhole"):
        matrix = torch.tensor(spec["matrix"], dtype=torch.float32).unsqueeze(0)  # [1,V,3,4]
        return PinholeProjection(matrix, geometry_type=kind), kind
    if kind == "ftheta":
        def _t(key):
            return torch.tensor(spec[key], dtype=torch.float32).unsqueeze(0)
        # fw_poly may be serialized as a shared [K] (flat list) or per-view [V,K]
        # (nested list) — to_spec keeps a shared vector whole. Reconstruct the
        # matching shape so to_spec/load round-trip is exact: shared -> [K],
        # per-view -> [1,V,K].
        fw = spec["fw_poly"]
        if fw and isinstance(fw[0], (list, tuple)):
            fw_poly = torch.tensor(fw, dtype=torch.float32).unsqueeze(0)  # [1,V,K]
        else:
            fw_poly = torch.tensor(fw, dtype=torch.float32)               # [K] shared
        max_theta = spec.get("max_theta")
        if isinstance(max_theta, (list, tuple)):
            max_theta = torch.tensor(max_theta, dtype=torch.float32)      # per-view
        return (
            FThetaProjection(
                t_camera_ego=_t("t_camera_ego"),   # [1,V,4,4]
                fw_poly=fw_poly,
                cx=_t("cx"), cy=_t("cy"),          # [1,V]
                image_wh=_t("image_wh"),           # [1,V,2] native (W,H)
                max_theta=max_theta,
            ),
            "ftheta",
        )
    raise ValueError(f"Unknown projection type in manifest: {kind!r}")


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

    The returned loader carries two extra attributes describing the dataset's
    geometry (a rig constant, so it lives on the loader, not per batch):
      - ``.projection``: a CameraProjectionModel operator, or None (pseudo).
      - ``.geometry_type``: "pinhole" / "rectified_pinhole" / "ftheta" / "pseudo".
    Pass these to the model's forward alongside each batch.
    """
    tarfiles = sorted(Path(shard_dir).glob("*.tar"))
    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    urls = [str(p) for p in tarfiles]

    dataset = wds.WebDataset(urls, shardshuffle=False, empty_check=False, nodesplitter=wds.split_by_worker)
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.map(_decode_sample)

    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=min(num_workers, len(tarfiles)))

    # Per-dataset geometry, reconstructed once from the manifest.
    projection, geometry_type = load_projection_from_manifest(shard_dir)
    loader.projection = projection
    loader.geometry_type = geometry_type
    return loader
