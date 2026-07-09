"""S3-backed, per-sample reasoning-label cache (issue #98 / #113).

The teacher (Cosmos3-Nano behind the OpenAI-compatible endpoint) is expensive
(~seconds/sample) and its output depends ONLY on (sample frames, prompt_version,
teacher). So a label is computed at most ONCE in the dataset's lifetime and
reused across every re-pack / re-run: the cache key is

    reasoning_labels_cache/dataset=<name>/teacher=<t>/prompt_version=<v>/{sample_id}.json

Changing dataset, teacher, or prompt_version yields a different prefix → a fresh
cache (never a stale mix). This is the "generate once, reference forever" layer
the strategy audit asked for; it decouples the slow external teacher call from
deterministic shard packing.

Runtime-safe / train-only: lives under data_processing, imports no teacher at
module load; boto3/S3 only touched when a cache root is configured.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from .schema import ReasoningLabelRecord
from .targets import record_from_json


def cache_prefix(dataset: str, teacher: str, prompt_version: str) -> str:
    """The key prefix that isolates one (dataset, teacher, prompt) cache."""
    safe_dataset = dataset.replace("/", "_")
    return (
        f"reasoning_labels_cache/dataset={safe_dataset}"
        f"/teacher={teacher}/prompt_version={prompt_version}"
    )


class LabelCache:
    """Get-or-compute reasoning labels, persisted per sample in S3.

    Args:
        bucket: S3 bucket for the cache (None disables caching → always compute).
        dataset / teacher / prompt_version: cache-key components (see module doc).
        s3_client: injectable boto3 S3 client (None → make one lazily). Tests
            pass a stub so no network/credentials are needed.
    """

    def __init__(
        self,
        bucket: Optional[str],
        dataset: str,
        teacher: str,
        prompt_version: str,
        s3_client=None,
    ) -> None:
        self.bucket = bucket
        self.dataset = dataset
        self.teacher = teacher
        self.prompt_version = prompt_version
        self._prefix = cache_prefix(dataset, teacher, prompt_version)
        self._client = s3_client
        self.hits = 0
        self.misses = 0

    def _s3(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3")
        return self._client

    def _key(self, sample_id: str) -> str:
        return f"{self._prefix}/{sample_id}.json"

    def get(self, sample_id: str) -> Optional[ReasoningLabelRecord]:
        """Return the cached record for ``sample_id`` or None on a miss."""
        if not self.bucket:
            return None
        try:
            obj = self._s3().get_object(Bucket=self.bucket, Key=self._key(sample_id))
            payload = json.loads(obj["Body"].read().decode("utf-8"))
            self.hits += 1
            return record_from_json(payload)
        except Exception:
            # Miss (NoSuchKey) or any read error → recompute; never serve a
            # partial/corrupt cache entry.
            self.misses += 1
            return None

    def put(self, sample_id: str, record: ReasoningLabelRecord) -> None:
        """Persist ``record`` for ``sample_id`` (no-op if caching disabled)."""
        if not self.bucket:
            return
        body = json.dumps(asdict(record)).encode("utf-8")
        self._s3().put_object(Bucket=self.bucket, Key=self._key(sample_id), Body=body)

    def get_or_compute(self, sample_id: str, compute) -> ReasoningLabelRecord:
        """Return the cached record, else call ``compute()`` and cache it.

        ``compute`` is a zero-arg callable returning a ReasoningLabelRecord (the
        teacher call). Only invoked on a cache miss — the whole point of the layer.
        """
        cached = self.get(sample_id)
        if cached is not None:
            return cached
        record = compute()
        self.put(sample_id, record)
        return record
