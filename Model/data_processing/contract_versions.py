"""Single source of truth for the data-pipeline CONTRACT versions (#121 §3.4c).

These version strings are the only sanctioned way to invalidate the Flyte task
cache / reasoning-label S3 cache. Each is defined here ONCE and imported wherever
it is needed — never written as an inline literal — so a `grep` over this file
shows every cache-invalidating knob and its blast radius.

Bump a version ONLY in a reviewed change, with a one-line reason, when the thing
it names actually changes. The stability test asserts these are single constants
and that no runtime/tuning knob (episode count, num_workers, resource limits, …)
ever enters a cached task's input signature.

| Version              | Bump ONLY when…                                        | Re-runs         |
|----------------------|--------------------------------------------------------|-----------------|
| UID_SCHEMA_VERSION   | the sample_uid / split_group_uid FORMAT changes        | label + pack    |
| PARSER_VERSION       | sample ENUMERATION or per-sample fields change         | ingest+label+pack|
| SHARD_SCHEMA_VERSION | the packed tar member layout changes                   | pack            |
| GEOMETRY_VERSION     | the calibration/projection encoding changes            | pack            |
| REASONING_LABEL_POLICY_VERSION | the reasoning sample selection changes     | label + pack    |

Source revision (HF commit), teacher model revision, and the prompt body hash are
NOT constants here — they are resolved at run time (from HF / teacher config /
prompt text) and threaded through as inputs; see DatasetSnapshot (§3.4a).
"""

from __future__ import annotations

# Format of sample_uid ("l2d-v1-e…-f…") and split_group_uid. Bump if the id
# STRING format changes (not when new episodes are added).
UID_SCHEMA_VERSION = "v1"

# Sample enumeration + per-sample field contract of the parsers
# (_build_sample_index, egomotion extraction, WM window offsets). Bump if which
# frames are valid, or the fields a sample carries, change.
PARSER_VERSION = "v2"

# Packed WebDataset shard member layout (cam_i.jpg / map.jpg / hist_*/fut_* /
# ego.npy / meta.json / calib.json / reasoning.json). Bump if the member set,
# names, or encoding change.
# v2 (dedup pack, #121 §3.4d): added `window_index.json` per-sample member and
# a SIBLING `pool/{frame_id}.jpg` directory replacing legacy `hist_*/fut_*` tar
# members. Loader now requires pool/ when window_index.json is present.
# v3: added pose.npy (absolute lat/lon/heading/timestamp) and gps.npy (current +
# 64 future lat/lon points) for datasets with geospatial source fields.
# v4: emits dataset-level geo paths, sample-pose parquet, and privacy-filtered
# heatmaps for every GPS-capable parser, including KITScenes partitions.
SHARD_SCHEMA_VERSION = "v4"

# Calibration / projection spec encoding and raster-map coordinate semantics.
# v2 queries KITScenes maps in the scene-local pose frame and applies the map
# origin exactly once when publishing absolute geographic coordinates.
GEOMETRY_VERSION = "v2"

# Selection policy for the sparse reasoning-label subset. v2 adds the first
# valid sample of every split group to the regular frame-index grid so even a
# short scene receives supervision.
REASONING_LABEL_POLICY_VERSION = "v2"


def contract_versions() -> dict:
    """All contract versions as one dict — used in the DatasetSnapshot / cache key
    and asserted single-sourced by the stability test."""
    return {
        "uid_schema_version": UID_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "reasoning_label_policy_version": REASONING_LABEL_POLICY_VERSION,
    }
