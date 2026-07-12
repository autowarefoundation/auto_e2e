"""DatasetSnapshot — the cache-provenance object threaded through every fan-out
stage (#121 §3.4a).

Flyte's task-cache key is (task interface, input literal values, cache_version).
`FlyteDirectory` inputs hash by URI, and `cache_version="v1"` alone is a blunt
global switch, so a code/spec change that leaves the raw inputs untouched would
silently serve a STALE cached shard. To make the cache key reflect the REAL
determinants of a stage's output, every fan-out stage takes a DatasetSnapshot as
an explicit input:

    dataset          which corpus            ("yaak-ai/L2D")
    source_revision  pins the raw bytes      (HF commit sha / SDK snapshot id)
    uid_schema_version / parser_version / shard_schema_version / geometry_version
                     the code CONTRACTS      (from contract_versions.py)
    metadata_digest  the resolved group set  (blake2b of the sorted group ids)

Because it's a frozen dataclass of primitives, Flyte serializes it into the cache
key by value: bump ``source_revision`` (new data), a contract version (code
change, §3.4c), or change the group set, and the affected ranges correctly MISS;
leave them alone and a re-run is a cache no-op. This is what makes "extend from
20 → 50 → all episodes" cheap — only the NEW ranges run (§3.4a).

Kept Flyte-free (pure dataclass + hashlib) so it is unit-testable and importable
from the parsers/tests without pulling in flytekit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from data_processing.contract_versions import (
    GEOMETRY_VERSION,
    PARSER_VERSION,
    SHARD_SCHEMA_VERSION,
    UID_SCHEMA_VERSION,
)


def group_metadata_digest(group_ids: Sequence[int | str]) -> str:
    """Stable digest of a resolved group-id set (episodes / clip uuids).

    Order-independent (the SET of groups determines the output, not the order we
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


@dataclass(frozen=True)
class DatasetSnapshot:
    """Immutable provenance of a dataset slice — the cache-key determinant (§3.4a).

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
