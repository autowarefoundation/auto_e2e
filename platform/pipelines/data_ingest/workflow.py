"""Flyte data ingest workflow: adapter-driven parallel episode processing.

Converts raw datasets into WebDataset shards on S3 via pluggable adapters.
Each episode is processed independently (map_task parallelism).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import List

import boto3
from flytekit import dynamic, task, workflow, Resources, current_context

from .adapters import get_adapter
from .shard_writer import ShardWriter

logger = logging.getLogger(__name__)

DATA_PREP_IMAGE = "{ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/data-prep:latest"


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="4", mem="16Gi"),
    cache=True,
    cache_version="1",
)
def ingest_episode(
    adapter_name: str,
    episode_id: str,
    output_bucket: str,
    dataset_name: str,
    version: str,
) -> dict:
    """Process one episode: download → extract valid samples → pack shards → S3."""
    adapter = get_adapter(adapter_name)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        from .adapters.protocol import EpisodeRef
        ref = EpisodeRef(episode_id=episode_id)

        # 1. Download episode
        episode_path = adapter.download_episode(ref, work_dir)

        # 2. Compute valid sample points
        samples = adapter.compute_valid_samples(episode_path)
        if not samples:
            logger.warning(f"No valid samples for episode {episode_id}")
            return {"episode_id": episode_id, "num_samples": 0, "shards": []}

        # 3. Extract frames + pack into shards
        shard_dir = work_dir / "shards"
        writer = ShardWriter(shard_dir, prefix=f"ep-{episode_id}")

        for idx, sample in enumerate(samples):
            # Extract all camera frames
            camera_jpegs = []
            for cam_idx in range(len(adapter.camera_names)):
                jpeg = adapter.extract_frame(episode_path, sample, cam_idx)
                camera_jpegs.append(jpeg)

            writer.add_sample(
                sample_id=f"{episode_id}_{idx:06d}",
                camera_jpegs=camera_jpegs,
                ego_history=sample.ego_history,
                ego_future=sample.ego_future,
                metadata={"episode_id": episode_id, "frame_idx": sample.frame_idx},
            )

        shard_paths = writer.close()

        # 4. Upload shards to S3
        s3 = boto3.client("s3")
        s3_prefix = f"{dataset_name}/{version}/shards"
        uploaded = []
        for shard_path in shard_paths:
            key = f"{s3_prefix}/{shard_path.name}"
            s3.upload_file(str(shard_path), output_bucket, key)
            uploaded.append(key)

    return {
        "episode_id": episode_id,
        "num_samples": writer.total_samples,
        "shards": uploaded,
    }


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="1Gi"),
)
def build_manifest(
    adapter_name: str,
    episode_results: List[dict],
    output_bucket: str,
    dataset_name: str,
    version: str,
) -> str:
    """Build manifest.json and upload to S3."""
    adapter = get_adapter(adapter_name)
    total_samples = sum(r["num_samples"] for r in episode_results)
    all_shards = [s for r in episode_results for s in r["shards"]]

    manifest = {
        "dataset": dataset_name,
        "version": version,
        "num_samples": total_samples,
        "num_episodes": len([r for r in episode_results if r["num_samples"] > 0]),
        "cameras": adapter.camera_names,
        "num_cameras": len(adapter.camera_names),
        "frame_size": [256, 256],
        "egomotion_hz": 10,
        "history_steps": 64,
        "future_steps": 64,
        "shards": all_shards,
    }

    s3 = boto3.client("s3")
    key = f"{dataset_name}/{version}/manifest.json"
    s3.put_object(
        Bucket=output_bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode(),
    )
    return f"s3://{output_bucket}/{key}"


@dynamic
def ingest_dataset(
    adapter_name: str = "l2d",
    output_bucket: str = "auto-e2e-platform-datasets-381491877296",
    dataset_name: str = "l2d",
    version: str = "v1.0",
    episode_limit: int = 0,
) -> str:
    """Full ingest: list episodes → parallel process → manifest."""
    adapter = get_adapter(adapter_name)
    episodes = adapter.list_episodes(limit=episode_limit)

    results = []
    for ep in episodes:
        result = ingest_episode(
            adapter_name=adapter_name,
            episode_id=ep.episode_id,
            output_bucket=output_bucket,
            dataset_name=dataset_name,
            version=version,
        )
        results.append(result)

    return build_manifest(
        adapter_name=adapter_name,
        episode_results=results,
        output_bucket=output_bucket,
        dataset_name=dataset_name,
        version=version,
    )
