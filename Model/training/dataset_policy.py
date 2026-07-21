"""Dataset-specific training semantics for the fixed AutoE2E contract.

AutoE2E owns the input and output horizons. This module contains only corpus
properties that must not silently inherit values measured on L2D.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch


AUTO_E2E_TIMESTEPS = 64
AUTO_E2E_TEMPORAL_DECAY = 0.95
EGOMOTION_SIGNALS = 4
ACCELERATION_INDEX = 1

L2D_DATASET_NAME = "yaak-ai/L2D"
NVIDIA_DATASET_NAME = "nvidia/PhysicalAI-Autonomous-Vehicles"
KITSCENES_DATASET_NAME = "KIT-MRT/KITScenes-Multimodal"
VALIDATION_SCOPE_FULL = "full"
VALIDATION_SCOPE_SUBSET = "subset"
SUBSET_EXACT_GROUP_STRATEGY = "subset_exact_group_fraction"


@dataclass(frozen=True)
class DatasetTrainingPolicy:
    """Corpus-specific values used inside the fixed AutoE2E training ABI."""

    dataset_name: str
    temporal_decay: float
    signal_scales: tuple[float, float]
    mask_latest_history_acceleration: bool = False
    validation_strategy: str = "hash_buckets"
    validation_split_id: str = "legacy_hash_buckets_v1"
    validation_manifest: str | None = None
    validation_manifest_schema: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.temporal_decay <= 1.0:
            raise ValueError("temporal_decay must be in (0, 1]")
        if len(self.signal_scales) != 2 or any(
            value <= 0.0 for value in self.signal_scales
        ):
            raise ValueError("signal_scales must contain two positive values")
        if self.validation_strategy not in {
            "hash_buckets",
            "exact_group_fraction",
            SUBSET_EXACT_GROUP_STRATEGY,
        }:
            raise ValueError(
                f"unsupported validation_strategy {self.validation_strategy!r}"
            )
        if not self.validation_split_id:
            raise ValueError("validation_split_id must not be empty")
        if (
            self.validation_strategy
            in {"exact_group_fraction", SUBSET_EXACT_GROUP_STRATEGY}
            and (
                not self.validation_manifest
                or not self.validation_manifest_schema
            )
        ):
            raise ValueError(
                "exact group validation requires a manifest and schema"
            )

    def metadata(self) -> dict[str, object]:
        return asdict(self)


# L2D keeps its established 6.4-second training contract.
L2D_TRAINING_POLICY = DatasetTrainingPolicy(
    dataset_name=L2D_DATASET_NAME,
    temporal_decay=AUTO_E2E_TEMPORAL_DECAY,
    signal_scales=(0.79, 0.12),
)

# NVIDIA has not yet had a corpus-specific scale audit. Preserve its prior
# behavior explicitly instead of reaching it through an unknown-dataset fallback.
NVIDIA_TRAINING_POLICY = DatasetTrainingPolicy(
    dataset_name=NVIDIA_DATASET_NAME,
    temporal_decay=AUTO_E2E_TEMPORAL_DECAY,
    signal_scales=(0.79, 0.12),
)

# AutoE2E retains its 64-step input/output horizon and temporal weighting for
# KITScenes. Only signal scales are corpus statistics: population standard
# deviations measured over the exact frozen internal-train tensors in the
# 533-partition v2.2 pack (38,847 samples x 64 target rows). The separate
# benchmark evaluator applies the KITScenes four-second observation and
# three-/five-second reporting protocol without changing training.
KITSCENES_TRAINING_POLICY = DatasetTrainingPolicy(
    dataset_name=KITSCENES_DATASET_NAME,
    temporal_decay=AUTO_E2E_TEMPORAL_DECAY,
    signal_scales=(0.778, 0.0350),
    # KITScenes v2 derives acceleration with centered finite differences. Its
    # final history acceleration indirectly reads frame i+1, so mask that one
    # value until a causal parser revision is repacked.
    mask_latest_history_acceleration=True,
    validation_strategy="exact_group_fraction",
    validation_split_id="kitscenes_train_dev_v1",
    validation_manifest="splits/kitscenes_train_dev_v1.json",
    validation_manifest_schema="kitscenes_train_dev_split_v1",
)

# Checkpoints produced before dataset policies were recorded used L2D signal
# scales and did not sanitize the KITScenes finite-difference feature. Keep that
# interpretation explicit so loading an active legacy run does not change.
LEGACY_KITSCENES_TRAINING_POLICY = DatasetTrainingPolicy(
    dataset_name=KITSCENES_DATASET_NAME,
    temporal_decay=AUTO_E2E_TEMPORAL_DECAY,
    signal_scales=(0.79, 0.12),
)

_POLICIES = {
    policy.dataset_name: policy
    for policy in (
        L2D_TRAINING_POLICY,
        NVIDIA_TRAINING_POLICY,
        KITSCENES_TRAINING_POLICY,
    )
}

_LEGACY_POLICIES = {
    L2D_DATASET_NAME: L2D_TRAINING_POLICY,
    NVIDIA_DATASET_NAME: NVIDIA_TRAINING_POLICY,
    KITSCENES_DATASET_NAME: LEGACY_KITSCENES_TRAINING_POLICY,
}


def training_policy_for_dataset(
    dataset_name: str,
    *,
    validation_scope: str = VALIDATION_SCOPE_FULL,
) -> DatasetTrainingPolicy:
    """Return an explicit policy; unknown datasets must be audited first."""
    try:
        policy = _POLICIES[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"no training policy is defined for dataset {dataset_name!r}"
        ) from error
    if validation_scope == VALIDATION_SCOPE_FULL:
        return policy
    if validation_scope != VALIDATION_SCOPE_SUBSET:
        raise ValueError(
            f"unsupported validation scope {validation_scope!r}"
        )
    if dataset_name != KITSCENES_DATASET_NAME:
        raise ValueError(
            "subset validation scope is defined only for KITScenes"
        )
    return replace(
        policy,
        validation_strategy=SUBSET_EXACT_GROUP_STRATEGY,
        validation_split_id="kitscenes_smoke_subset_v1",
    )


def training_policy_from_config(
    config: Mapping[str, object],
    dataset_name: str,
) -> DatasetTrainingPolicy:
    """Load a checkpoint policy without reinterpreting legacy checkpoints."""
    payload = config.get("trajectory_training_policy")
    if payload is None:
        try:
            return _LEGACY_POLICIES[dataset_name]
        except KeyError as error:
            raise ValueError(
                "legacy checkpoint has no training policy and no explicit "
                f"legacy contract for dataset {dataset_name!r}"
            ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint trajectory_training_policy must be a mapping")
    policy = DatasetTrainingPolicy(**dict(payload))
    if policy.dataset_name != dataset_name:
        raise ValueError(
            "checkpoint training policy dataset does not match evaluation "
            f"dataset: {policy.dataset_name!r} != {dataset_name!r}"
        )
    return policy


def _load_validation_manifest(
    policy: DatasetTrainingPolicy,
) -> dict[str, object]:
    manifest_path = Path(__file__).parent / str(policy.validation_manifest)
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not load validation manifest {manifest_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("validation manifest must be a JSON object")
    return payload


def _validation_manifest_split_id(
    policy: DatasetTrainingPolicy,
) -> str:
    """Return the corpus manifest ID used to validate an exact split policy."""
    if policy.validation_strategy == SUBSET_EXACT_GROUP_STRATEGY:
        return KITSCENES_TRAINING_POLICY.validation_split_id
    return policy.validation_split_id


def _manifest_count(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"validation manifest {key} must be a non-negative integer"
        )
    return value


def _manifest_fraction(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"validation manifest {key} must be a number"
        )
    return float(value)


def _manifest_digest(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"validation manifest {key} must be a lowercase SHA-256"
        )
    return value


def validation_sample_identity(
    policy: DatasetTrainingPolicy,
) -> tuple[int, str]:
    """Return the frozen validation sample count and UID digest."""
    if policy.validation_strategy != "exact_group_fraction":
        raise ValueError(
            "validation sample identity requires an exact group split"
        )
    payload = _load_validation_manifest(policy)
    if (
        payload.get("schema_version") != policy.validation_manifest_schema
        or payload.get("dataset_name") != policy.dataset_name
        or payload.get("split_id") != policy.validation_split_id
    ):
        raise ValueError(
            "validation manifest identity does not match training policy"
        )
    return (
        _manifest_count(payload, "validation_sample_count"),
        _manifest_digest(payload, "validation_sample_uid_digest"),
    )


def validation_group_uids(
    group_uids: Sequence[str],
    *,
    val_fraction: float,
    policy: DatasetTrainingPolicy,
    source_revision: str | None = None,
    packed_dataset_version: str | None = None,
    packed_contract_digest: str | None = None,
    packed_partition_count: int | None = None,
    empty_partition_count: int | None = None,
    packed_sample_count: int | None = None,
    packed_sample_uid_digest: str | None = None,
) -> tuple[str, ...] | None:
    """Validate and return a frozen holdout, or None for legacy bucketing."""
    if policy.validation_strategy == "hash_buckets":
        return None
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    normalized = sorted(set(str(uid) for uid in group_uids))
    if len(normalized) != len(group_uids):
        raise ValueError("split group UIDs must be unique")
    if len(normalized) < 2 or any(not uid for uid in normalized):
        raise ValueError("at least two non-empty split group UIDs are required")

    required_provenance = {
        "source_revision": source_revision,
        "packed_dataset_version": packed_dataset_version,
        "packed_contract_digest": packed_contract_digest,
        "packed_partition_count": packed_partition_count,
        "empty_partition_count": empty_partition_count,
        "packed_sample_count": packed_sample_count,
        "packed_sample_uid_digest": packed_sample_uid_digest,
    }
    missing = [
        key for key, value in required_provenance.items()
        if value is None or (isinstance(value, str) and not value)
    ]
    if missing:
        raise ValueError(
            "exact validation splitting requires packed provenance: "
            f"{sorted(missing)}"
        )

    payload = _load_validation_manifest(policy)
    if payload.get("schema_version") != policy.validation_manifest_schema:
        raise ValueError(
            "validation manifest schema does not match training policy"
        )
    if payload.get("dataset_name") != policy.dataset_name:
        raise ValueError(
            "validation manifest dataset does not match training policy"
        )
    if payload.get("split_id") != _validation_manifest_split_id(policy):
        raise ValueError(
            "validation manifest split ID does not match training policy"
        )
    if payload.get("source_revision") != source_revision:
        raise ValueError(
            "validation manifest source revision does not match packed shards"
        )
    if payload.get("dataset_version") != packed_dataset_version:
        raise ValueError(
            "validation manifest dataset version does not match packed shards"
        )
    if payload.get("packed_contract_digest") != packed_contract_digest:
        raise ValueError(
            "validation manifest contract digest does not match packed shards"
        )
    _manifest_digest(payload, "source_artifact_set_sha256")
    if _manifest_fraction(payload, "validation_fraction") != val_fraction:
        raise ValueError(
            "requested val_fraction differs from the frozen validation manifest"
        )
    if policy.validation_strategy == SUBSET_EXACT_GROUP_STRATEGY:
        full_partition_count = _manifest_count(
            payload,
            "available_scene_count",
        )
        if (
            not isinstance(packed_partition_count, int)
            or isinstance(packed_partition_count, bool)
            or not 0 < packed_partition_count < full_partition_count
        ):
            raise ValueError(
                "KITScenes subset validation requires a proper partition "
                "subset of the frozen corpus"
            )
        if (
            not isinstance(empty_partition_count, int)
            or isinstance(empty_partition_count, bool)
            or not 0 <= empty_partition_count < packed_partition_count
        ):
            raise ValueError(
                "KITScenes subset empty-partition count is invalid"
            )
        if len(normalized) != (
            packed_partition_count - empty_partition_count
        ):
            raise ValueError(
                "KITScenes subset group count differs from its non-empty "
                "partition count"
            )
        if (
            not isinstance(packed_sample_count, int)
            or isinstance(packed_sample_count, bool)
            or packed_sample_count <= 0
        ):
            raise ValueError(
                "KITScenes subset sample count must be positive"
            )
        if (
            not isinstance(packed_sample_uid_digest, str)
            or len(packed_sample_uid_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in packed_sample_uid_digest
            )
        ):
            raise ValueError(
                "KITScenes subset sample UID digest must be a lowercase SHA-256"
            )

        validation_count = max(
            1,
            min(
                len(normalized) - 1,
                round(val_fraction * len(normalized)),
            ),
        )
        ranked = sorted(
            normalized,
            key=lambda uid: (
                hashlib.sha256(
                    (
                        f"{policy.validation_split_id}\0{uid}"
                    ).encode("utf-8")
                ).digest(),
                uid,
            ),
        )
        return tuple(sorted(ranked[:validation_count]))

    if packed_partition_count != (
        _manifest_count(payload, "available_scene_count")
    ):
        raise ValueError(
            "packed KITScenes partition count differs from the frozen split"
        )
    if empty_partition_count != (
        _manifest_count(payload, "excluded_empty_scene_count")
    ):
        raise ValueError(
            "packed KITScenes empty-partition count differs from the frozen split"
        )
    eligible_sample_count = _manifest_count(
        payload,
        "eligible_sample_count",
    )
    if packed_sample_count != eligible_sample_count:
        raise ValueError(
            "packed KITScenes sample count differs from the frozen split"
        )
    if packed_sample_uid_digest != _manifest_digest(
        payload,
        "eligible_sample_uid_digest",
    ):
        raise ValueError(
            "packed KITScenes sample identity differs from the frozen split"
        )
    training_sample_count = _manifest_count(
        payload,
        "training_sample_count",
    )
    validation_sample_count = _manifest_count(
        payload,
        "validation_sample_count",
    )
    if (
        training_sample_count + validation_sample_count
        != eligible_sample_count
    ):
        raise ValueError(
            "validation manifest train/validation sample counts are inconsistent"
        )
    _manifest_digest(payload, "training_sample_uid_digest")
    _manifest_digest(payload, "validation_sample_uid_digest")

    expected_count = _manifest_count(payload, "eligible_group_count")
    expected_digest_value = payload.get("eligible_group_uid_digest")
    if not isinstance(expected_digest_value, str):
        raise ValueError(
            "validation manifest eligible_group_uid_digest must be a string"
        )
    expected_digest = expected_digest_value
    actual_digest = group_uid_digest(normalized)
    if len(normalized) != expected_count or actual_digest != expected_digest:
        raise ValueError(
            "packed KITScenes group inventory differs from the frozen split: "
            f"expected_count={expected_count} actual_count={len(normalized)} "
            f"expected_digest={expected_digest} actual_digest={actual_digest}"
        )

    selected_payload = payload.get("validation_group_uids")
    if not isinstance(selected_payload, list):
        raise ValueError(
            "validation manifest validation_group_uids must be a list"
        )
    selected_values: list[str] = []
    for uid in selected_payload:
        if not isinstance(uid, str):
            raise ValueError(
                "validation manifest group UIDs must be strings"
            )
        selected_values.append(uid)
    selected = tuple(selected_values)
    if (
        list(selected) != sorted(set(selected))
        or any(not uid for uid in selected)
    ):
        raise ValueError(
            "validation manifest group UIDs must be sorted and unique"
        )
    if not set(selected) < set(normalized):
        raise ValueError(
            "validation groups must be a proper subset of eligible groups"
        )
    if len(selected) != _manifest_count(
        payload,
        "validation_group_count",
    ):
        raise ValueError("validation manifest group count is inconsistent")
    expected_validation_count = max(
        1,
        min(
            expected_count - 1,
            round(val_fraction * expected_count),
        ),
    )
    if len(selected) != expected_validation_count:
        raise ValueError(
            "validation manifest group count differs from its fraction"
        )
    selected_digest = group_uid_digest(selected)
    if selected_digest != payload.get("validation_group_uid_digest"):
        raise ValueError("validation manifest group digest is inconsistent")
    return selected


def group_uid_digest(group_uids: Sequence[str]) -> str:
    """Hash one sorted, unique group inventory using canonical JSON."""
    normalized = [str(uid) for uid in group_uids]
    if normalized != sorted(set(normalized)) or any(
        not uid for uid in normalized
    ):
        raise ValueError("group UIDs must be sorted, unique, and non-empty")
    return hashlib.sha256(
        json.dumps(
            normalized,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def adapt_egomotion_history(
    history: torch.Tensor,
    policy: DatasetTrainingPolicy,
) -> torch.Tensor:
    """Remove corpus-specific invalid values without changing AutoE2E history."""
    expected_width = AUTO_E2E_TIMESTEPS * EGOMOTION_SIGNALS
    if history.ndim != 2 or history.shape[1] != expected_width:
        raise ValueError(
            f"egomotion history must have shape [batch, {expected_width}]"
        )

    if not policy.mask_latest_history_acceleration:
        return history

    adapted = history.reshape(
        history.shape[0], AUTO_E2E_TIMESTEPS, EGOMOTION_SIGNALS
    ).clone()
    adapted[:, -1, ACCELERATION_INDEX] = 0
    return adapted.reshape(history.shape[0], expected_width)
