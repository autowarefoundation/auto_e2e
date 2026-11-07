"""DatasetSnapshot, the cache-provenance object threaded through every fan-out stage.

Flyte's task-cache key is (task interface, input literal values, cache_version).
`FlyteDirectory` inputs hash by URI, and `cache_version="v1"` alone is a blunt
global switch, so a code/spec change that leaves the raw inputs untouched would
silently serve a stale cached shard. To make the cache key reflect the real
determinants of a stage's output, every fan-out stage takes a DatasetSnapshot as
an explicit input:

    dataset          which corpus            ("yaak-ai/L2D")
    source_revision  pins the raw bytes      (HF commit sha / SDK snapshot id)
    uid_schema_version / parser_version / shard_schema_version / geometry_version
                     the code CONTRACTS      (from contract_versions.py)
    metadata_digest  the resolved group set  (blake2b of the sorted group ids)

Because it's a frozen dataclass of primitives, Flyte serializes it into the cache
key by value: bump ``source_revision`` (new data), a contract version (code
change), or change the group set, and the affected ranges miss the cache;
leave them alone and a re-run is a cache no-op, so extending from 20 to 50 to
all episodes only runs the new ranges.

Kept Flyte-free (pure dataclass + hashlib) so it is importable from the parsers
and tests without pulling in flytekit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Sequence

from data_processing.contract_versions import (
    GEOMETRY_VERSION,
    PARSER_VERSION,
    SHARD_SCHEMA_VERSION,
    UID_SCHEMA_VERSION,
)

# Split groups with this prefix fall in bucket 0, which every non-empty
# validation split contains, so a geographic hold-out (nuPlan Pittsburgh)
# never reaches training whatever the validation fraction.
HELDOUT_SPLIT_GROUP_PREFIX: Final = "heldout-"


def group_metadata_digest(group_ids: Sequence[int | str]) -> str:
    """Stable digest of a resolved group-id set (episodes / clip uuids).

    Order-independent (the set of groups determines the output, not the order we
    happened to enumerate them) and type-tagged so int episode 12 and the string
    "12" never collide. Deterministic across processes (blake2b, not the
    process-salted builtin hash) so re-runs reproduce the same key → cache hit.
    """
    # Sort by (type-name, str value) for a canonical, order-independent form. A
    # single dataset's group ids are homogeneous (all int, or all str), but the
    # type tag keeps the digest well-defined even if they were mixed.
    canon = sorted((type(g).__name__, str(g)) for g in group_ids)
    h = hashlib.blake2b(digest_size=16)
    for tname, val in canon:
        h.update(tname.encode())
        h.update(b"\x1f")          # unit separator: unambiguous field boundary
        h.update(val.encode())
        h.update(b"\x1e")          # record separator between groups
    return h.hexdigest()


def split_bucket(split_group_uid: str, buckets: int = 10) -> int:
    """Return the stable train/validation bucket for an episode or clip."""
    if not split_group_uid:
        raise ValueError("split_group_uid must not be empty")
    if buckets < 2:
        raise ValueError("buckets must be at least 2")
    if split_group_uid.startswith(HELDOUT_SPLIT_GROUP_PREFIX):
        return 0
    digest = hashlib.blake2b(
        split_group_uid.encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % buckets


def shard_partition_id(group_ids: Sequence[int | str] | None) -> str:
    """Stable publication prefix for one independently packed partition."""
    if group_ids is None:
        return ""
    return f"part-{group_metadata_digest(group_ids)[:16]}"


def published_shard_name(
    group_ids: Sequence[int | str] | None,
    shard_index: int,
) -> str:
    """Return a globally unique, deterministic tar name for publication."""
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    partition_id = shard_partition_id(group_ids)
    prefix = f"{partition_id}-" if partition_id else ""
    return f"{prefix}train-{shard_index:06d}.tar"


@dataclass(frozen=True)
class DatasetSnapshot:
    """Immutable provenance of a dataset slice, the cache-key determinant.

    Every fan-out stage takes one as an input so its Flyte cache key reflects the
    data revision + code contracts + the exact group set it processed, not just
    the (URI-hashed) FlyteDirectory. See module docstring.
    """

    dataset: str                 # "yaak-ai/L2D" / "nvidia/PhysicalAI-Autonomous-Vehicles"
    source_revision: str         # HF commit sha (L2D) / SDK snapshot id (NVIDIA); pins raw bytes
    metadata_digest: str         # group_metadata_digest of the resolved group ids
    uid_schema_version: str = UID_SCHEMA_VERSION
    parser_version: str = PARSER_VERSION
    shard_schema_version: str = SHARD_SCHEMA_VERSION
    geometry_version: str = GEOMETRY_VERSION

    @classmethod
    def build(
        cls, dataset: str, source_revision: str, group_ids: Sequence[int | str],
    ) -> "DatasetSnapshot":
        """Construct from a resolved group-id set, computing the digest.

        Contract versions default from contract_versions.py (the single source),
        so a stage never inlines them — bumping there flows into every snapshot.
        """
        return cls(dataset=dataset, source_revision=source_revision,
                   metadata_digest=group_metadata_digest(group_ids))

    def cache_key_fields(self) -> dict:
        """The provenance as a flat dict — for logging and for asserting the cache
        key carries exactly the intended determinants (no runtime/tuning knob)."""
        return {
            "dataset": self.dataset,
            "source_revision": self.source_revision,
            "metadata_digest": self.metadata_digest,
            "uid_schema_version": self.uid_schema_version,
            "parser_version": self.parser_version,
            "shard_schema_version": self.shard_schema_version,
            "geometry_version": self.geometry_version,
        }
