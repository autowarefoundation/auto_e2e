#!/usr/bin/env python3
"""Download NVIDIA PhysicalAI-AV data and unpack into the directory layout expected
by NvidiaAVDataset.

Prerequisites
-------------
1. pip install physical_ai_av   (already done if you ran the auto_e2e setup)
2. Agree to the dataset license at:
   https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
3. Login: huggingface-cli login   (or set HF_TOKEN env var)

Usage
-----
    # Download 1 clip (smoke test) — about 2-4 GB per clip depending on cameras
    python download_dataset.py --out /home/ubuntu/nvidia_av_data --clips 1

    # Download specific clip UUIDs
    python download_dataset.py --out /home/ubuntu/nvidia_av_data \
        --clip-uuids fd1d1b6b-59bf-4292-8295-5028aa6aa5e3

    # Download N random clips
    python download_dataset.py --out /home/ubuntu/nvidia_av_data --clips 5

After download, pass ``--out`` path as ``data_root`` to NvidiaAVDataset.

Expected output structure
-------------------------
    <out>/
      camera/<cam_name>/<uuid>.<cam_name>.mp4
      camera/<cam_name>/<uuid>.<cam_name>.timestamps.parquet
      labels/egomotion/<uuid>.egomotion.parquet
"""

from __future__ import annotations

import argparse
import io
import logging
import pathlib
import zipfile


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CAMERAS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]


def parse_args():
    p = argparse.ArgumentParser(description="Download NVIDIA PhysicalAI-AV subset")
    p.add_argument("--out", type=pathlib.Path, required=True,
                   help="Output directory (becomes data_root for NvidiaAVDataset)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--clips", type=int,
                   help="Number of random clips to download")
    g.add_argument("--clip-uuids", nargs="+",
                   help="Specific clip UUIDs to download")
    return p.parse_args()


def unpack_camera_zip(zip_bytes: bytes, clip_id: str, cam_name: str, out: pathlib.Path):
    """Extract camera video and timestamps from a chunk zip into parser layout."""
    cam_dir = out / "camera" / cam_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()
        for name in names:
            if clip_id not in name:
                continue
            data = zf.read(name)
            if name.endswith(".mp4"):
                dest = cam_dir / f"{clip_id}.{cam_name}.mp4"
            elif name.endswith(".parquet") and "timestamp" in name.lower():
                dest = cam_dir / f"{clip_id}.{cam_name}.timestamps.parquet"
            else:
                continue
            dest.write_bytes(data)
            logger.info("  %s (%d KB)", dest.relative_to(out), len(data) // 1024)


def unpack_egomotion_zip(zip_bytes: bytes, clip_id: str, out: pathlib.Path):
    """Extract egomotion parquet from a chunk zip into parser layout."""
    ego_dir = out / "labels" / "egomotion"
    ego_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()
        for name in names:
            if clip_id not in name:
                continue
            if name.endswith(".parquet"):
                data = zf.read(name)
                dest = ego_dir / f"{clip_id}.egomotion.parquet"
                dest.write_bytes(data)
                logger.info("  %s (%d KB)", dest.relative_to(out), len(data) // 1024)
                break


def main():
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from physical_ai_av import PhysicalAIAVDatasetInterface

    logger.info("Initializing SDK (downloads metadata from HF)...")
    ds = PhysicalAIAVDatasetInterface(
        local_dir=str(out / ".hf_cache"),
        confirm_download_threshold_gb=float("inf"),
    )

    if args.clip_uuids:
        clip_ids = args.clip_uuids
    else:
        clip_ids = ds.clip_index.index.tolist()[:args.clips]

    logger.info("Will download %d clip(s): %s", len(clip_ids), clip_ids)

    features_to_dl = CAMERAS + ["egomotion"]

    for i, clip_id in enumerate(clip_ids, 1):
        logger.info("[%d/%d] Downloading clip %s ...", i, len(clip_ids), clip_id)

        # Download chunk zips via SDK
        ds.download_clip_features(clip_id, features=features_to_dl)

        # Extract each camera
        for cam in CAMERAS:
            chunk_file = ds.features.get_chunk_feature_filename(
                ds.get_clip_chunk(clip_id), cam
            )
            with ds.open_file(chunk_file, maybe_stream=True) as f:
                unpack_camera_zip(f.read(), clip_id, cam, out)

        # Extract egomotion
        chunk_file = ds.features.get_chunk_feature_filename(
            ds.get_clip_chunk(clip_id), "egomotion"
        )
        with ds.open_file(chunk_file, maybe_stream=True) as f:
            unpack_egomotion_zip(f.read(), clip_id, out)

    logger.info("Done. data_root = %s", out)
    logger.info("Test with: NvidiaAVDataset(data_root='%s', clip_uuids=%s)", out, clip_ids)


if __name__ == "__main__":
    main()
