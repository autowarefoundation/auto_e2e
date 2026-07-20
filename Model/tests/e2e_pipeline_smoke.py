"""End-to-end pipeline smoke test on real NVIDIA data (run manually / on EC2).

Exercises the FULL post-#77 data path through the projection-operator ABI:

    NvidiaAVDataset (7 real cams + separate map_tile)
      -> WebDataset shard packing (cam_i.jpg + distinct map.jpg + manifest)
      -> make_pre_extracted_loader (reconstruct projection operator, map_input)
      -> AutoE2E forward (projection/geometry_type) -> imitation loss -> backward

Not a pytest target (needs the real dataset + heavy decode); invoked directly:
    python tests/e2e_pipeline_smoke.py --data-root /home/ubuntu/nvidia_av_data
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
from data_parsing.pre_extracted import make_pre_extracted_loader
from model_components.auto_e2e import AutoE2E
from model_components.losses import TrajectoryImitationLoss
from training.dataset_policy import NVIDIA_TRAINING_POLICY


def build_shards(dataset, out_dir, max_samples, image_size=256):
    """Pack a few dataset samples into one WebDataset shard + manifest, applying
    the map/camera split (real cams -> cam_i.jpg, nav-map -> distinct map.jpg)."""
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((image_size, image_size))
    tar_path = os.path.join(out_dir, "train-000000.tar")
    n = min(max_samples, len(dataset))
    visual = None
    has_map = False
    with tarfile.open(tar_path, "w") as tar:
        for si in range(n):
            sample = dataset[si]
            visual = sample["visual_tiles"]          # (V, 3, H, W) real cameras
            map_tile = sample.get("map_tile")        # (3, H, W) nav-map
            ego = sample["egomotion_history"]
            traj = sample["trajectory_target"]
            ego_data = np.concatenate([
                ego.numpy() if torch.is_tensor(ego) else np.asarray(ego),
                traj.numpy() if torch.is_tensor(traj) else np.asarray(traj),
            ]).astype(np.float32)
            key = f"s{si:08d}"

            def _write(member, frame):
                f = to_pil(frame.cpu().clamp(0, 1) if frame.dtype.is_floating_point else frame.cpu())
                f = resize(f)
                b = io.BytesIO()
                f.save(b, format="JPEG", quality=90)
                jpg = b.getvalue()
                ti = tarfile.TarInfo(name=f"{key}.{member}")
                ti.size = len(jpg)
                tar.addfile(ti, io.BytesIO(jpg))

            for cam_i in range(visual.shape[0]):
                _write(f"cam_{cam_i}.jpg", visual[cam_i])
            if map_tile is not None:
                _write("map.jpg", map_tile)
                has_map = True

            eb = ego_data.tobytes()
            ti = tarfile.TarInfo(name=f"{key}.ego.npy")
            ti.size = len(eb)
            tar.addfile(ti, io.BytesIO(eb))
            m = json.dumps({"idx": si}).encode()
            ti = tarfile.TarInfo(name=f"{key}.meta.json")
            ti.size = len(m)
            tar.addfile(ti, io.BytesIO(m))

    manifest = {"total_samples": n, "shards": 1, "image_size": image_size,
                "num_views": int(visual.shape[0]), "has_map": has_map,
                "geometry_type": "pseudo"}
    # Attach real f-theta calibration if the dataset can build it.
    spec = dataset.projection_spec(image_size) if hasattr(dataset, "projection_spec") else None
    if spec is not None:
        manifest["projection"] = spec
        manifest["geometry_type"] = spec.get("type", "pinhole")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return n, int(visual.shape[0]), manifest["geometry_type"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--clip-uuid", default=None)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[e2e] device={device}")

    # 1. Dataset — must yield 7 real cams + a separate map_tile.
    clip_uuids = [args.clip_uuid] if args.clip_uuid else None
    # The dataset is a raw pre-extraction source; the shard packer owns the
    # single geometry-aware resize and the loader owns normalization.
    ds = NvidiaAVDataset(data_root=args.data_root, clip_uuids=clip_uuids)
    s0 = ds[0]
    V = s0["visual_tiles"].shape[0]
    assert "map_tile" in s0, "dataset must emit a separate map_tile"
    assert V == 7, f"expected 7 real cameras, got {V}"
    assert s0["map_tile"].shape[0] == 3, "map_tile must be (3,H,W)"
    print(f"[e2e] dataset OK: visual_tiles {tuple(s0['visual_tiles'].shape)}, "
          f"map_tile {tuple(s0['map_tile'].shape)}")

    # 2. Shard packing (map/camera split + manifest).
    out_dir = tempfile.mkdtemp()
    n, vpacked, geom = build_shards(ds, out_dir, args.samples)
    print(f"[e2e] packed {n} samples, V={vpacked}, geometry={geom}")

    # 3. Loader — reconstructs the projection operator + map_input.
    loader = make_pre_extracted_loader(out_dir, batch_size=args.batch_size,
                                       num_workers=0, shuffle=0)
    projection = getattr(loader, "projection", None)
    geometry_type = getattr(loader, "geometry_type", "pseudo")
    if projection is not None:
        projection = projection.to(device)
    batch = next(iter(loader))
    assert batch["visual_tiles"].shape[1] == V, "loader V mismatch"
    assert "map_input" in batch and batch["map_input"].shape[1] == 3
    assert "camera_params" not in batch, "geometry must be a loader attr, not per-sample"
    print(f"[e2e] loader OK: visual_tiles {tuple(batch['visual_tiles'].shape)}, "
          f"map_input {tuple(batch['map_input'].shape)}, geometry={geometry_type}, "
          f"projection={'real' if projection is not None else 'pseudo'}")

    # 4. Model forward + loss + backward through the projection ABI.
    model = AutoE2E(backbone="swin_v2_tiny", num_views=V, is_pretrained=False).to(device)
    model.train()
    loss_fn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=NVIDIA_TRAINING_POLICY.temporal_decay,
        signal_scales=NVIDIA_TRAINING_POLICY.signal_scales,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    visual = batch["visual_tiles"].to(device)
    map_input = batch["map_input"].to(device)
    vis_hist = torch.zeros(visual.shape[0], 896, device=device)
    ego_hist = batch["egomotion_history"].to(device)
    target = batch["trajectory_target"].to(device)

    opt.zero_grad()
    pred = model(visual, map_input, vis_hist, ego_hist,
                 projection=projection, geometry_type=geometry_type,
                 mode="train", trajectory_target=target)
    loss = loss_fn(pred, target)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    assert torch.isfinite(loss), f"non-finite loss {loss.item()}"
    assert grad_norm > 0, "no gradient reached the model"
    print(f"[e2e] forward+backward OK: pred {tuple(pred.shape)}, "
          f"loss={loss.item():.4f}, grad_norm={float(grad_norm):.3f}")
    print("[e2e] PASSED — full real-data pipeline runs through the projection ABI")


if __name__ == "__main__":
    main()
