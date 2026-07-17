import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import training.dataset_policy as dataset_policy
from model_components.losses import TrajectoryImitationLoss
from training.dataset_policy import (
    AUTO_E2E_TEMPORAL_DECAY,
    AUTO_E2E_TIMESTEPS,
    KITSCENES_TRAINING_POLICY,
    L2D_TRAINING_POLICY,
    adapt_egomotion_history,
    group_uid_digest,
    training_policy_from_config,
    training_policy_for_dataset,
    validation_group_uids,
    validation_sample_identity,
)


class TestTrajectoryImitationLoss:
    def test_output_is_scalar(self):
        loss_fn = TrajectoryImitationLoss()
        pred = torch.randn(4, 128)
        target = torch.randn(4, 128)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_gradient_flows_to_input(self):
        loss_fn = TrajectoryImitationLoss()
        pred = torch.randn(4, 128, requires_grad=True)
        target = torch.randn(4, 128)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == (4, 128)

    def test_temporal_weighting_changes_loss(self):
        pred = torch.randn(4, 128)
        target = torch.randn(4, 128)

        uniform_loss = TrajectoryImitationLoss(temporal_decay=1.0)(pred, target)
        decayed_loss = TrajectoryImitationLoss(temporal_decay=0.9)(pred, target)

        assert uniform_loss.item() != decayed_loss.item()

    def test_zero_input_produces_zero_loss(self):
        loss_fn = TrajectoryImitationLoss()
        pred = torch.zeros(4, 128)
        target = torch.zeros(4, 128)
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_smooth_l1_vs_mse_differ(self):
        pred = torch.randn(4, 128)
        target = torch.randn(4, 128)

        l1_loss = TrajectoryImitationLoss(loss_type="smooth_l1")(pred, target)
        mse_loss = TrajectoryImitationLoss(loss_type="mse")(pred, target)

        assert l1_loss.item() != mse_loss.item()

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            TrajectoryImitationLoss(loss_type="l1")

    def test_per_signal_normalization_gives_curvature_gradient(self):
        # Regression for the flat-ADE bug: curvature (signal 1) is ~40x smaller
        # than accel (signal 0). Without per-signal normalization, a small
        # curvature error sits in SmoothL1's quadratic regime and produces a
        # near-zero gradient, so the planner never learns curvature. After
        # normalization, the per-element curvature gradient must be comparable
        # in magnitude to the accel gradient for equal-in-std errors.
        loss_fn = TrajectoryImitationLoss(signal_scales=(0.54, 0.014))
        # A pred that is off by ~1 std on BOTH signals (accel +0.54, curv +0.014).
        pred = torch.zeros(1, 128, requires_grad=True)
        target = torch.zeros(1, 64, 2)
        target[..., 0] = 0.54    # accel target
        target[..., 1] = 0.014   # curvature target
        loss = loss_fn(pred, target.view(1, 128))
        loss.backward()
        g = pred.grad.view(64, 2)
        accel_g = g[:, 0].abs().mean().item()
        curv_g = g[:, 1].abs().mean().item()
        # Comparable within 2x (would be ~40x apart without normalization).
        assert curv_g > 0.3 * accel_g, (accel_g, curv_g)

    def test_signal_scales_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="signal_scales must have"):
            TrajectoryImitationLoss(num_signals=2, signal_scales=(1.0,))

    def test_auto_e2e_tail_remains_supervised(self):
        loss_fn = TrajectoryImitationLoss(signal_scales=(1.0, 1.0))
        pred = torch.zeros(1, 64, 2)
        target = torch.zeros(1, 64, 2)
        target[:, 50:, :] = 1.0

        assert loss_fn(pred.flatten(1), target.flatten(1)).item() > 0.0

    def test_kitscenes_only_overrides_corpus_specific_values(self):
        policy = training_policy_for_dataset(
            "KIT-MRT/KITScenes-Multimodal"
        )

        assert policy is KITSCENES_TRAINING_POLICY
        assert AUTO_E2E_TIMESTEPS == 64
        assert policy.temporal_decay == AUTO_E2E_TEMPORAL_DECAY
        assert policy.signal_scales == pytest.approx((0.778, 0.0350))
        assert not hasattr(policy, "observation_steps")
        assert not hasattr(policy, "supervised_steps")
        assert not hasattr(policy, "evaluation_steps")

    def test_l2d_policy_preserves_established_contract(self):
        policy = training_policy_for_dataset("yaak-ai/L2D")

        assert policy is L2D_TRAINING_POLICY
        assert policy.temporal_decay == AUTO_E2E_TEMPORAL_DECAY
        assert policy.signal_scales == pytest.approx((0.79, 0.12))

    def test_unknown_dataset_cannot_inherit_l2d_policy(self):
        with pytest.raises(ValueError, match="no training policy"):
            training_policy_for_dataset("example/future-dataset")

    def test_legacy_kitscenes_checkpoint_keeps_recorded_run_semantics(self):
        policy = training_policy_from_config(
            {},
            "KIT-MRT/KITScenes-Multimodal",
        )

        assert policy is not KITSCENES_TRAINING_POLICY
        assert policy.temporal_decay == AUTO_E2E_TEMPORAL_DECAY
        assert policy.signal_scales == pytest.approx((0.79, 0.12))
        assert policy.validation_strategy == "hash_buckets"

    def test_kitscenes_history_adapter_only_masks_noncausal_acceleration(self):
        history = torch.arange(2 * 64 * 4, dtype=torch.float32).reshape(
            2, 64 * 4
        )

        adapted = adapt_egomotion_history(
            history,
            KITSCENES_TRAINING_POLICY,
        ).reshape(2, 64, 4)

        assert torch.equal(
            adapted[:, :-1],
            history.reshape(2, 64, 4)[:, :-1],
        )
        assert torch.equal(
            adapted[:, -1, [0, 2, 3]],
            history.reshape(2, 64, 4)[:, -1, [0, 2, 3]],
        )
        assert torch.count_nonzero(adapted[:, -1, 1]) == 0

    def test_kitscenes_holdout_uses_frozen_scene_manifest(
        self,
        monkeypatch,
    ):
        groups = [
            f"kitscenes-scene-{index:03d}"
            for index in range(17)
        ]
        selected_manifest = sorted((groups[2], groups[11]))
        monkeypatch.setattr(
            dataset_policy,
            "_load_validation_manifest",
            lambda policy: {
                "dataset_name": policy.dataset_name,
                "schema_version": policy.validation_manifest_schema,
                "split_id": policy.validation_split_id,
                "source_revision": "revision-a",
                "source_artifact_set_sha256": "b" * 64,
                "dataset_version": "v2.2",
                "packed_contract_digest": "contract-a",
                "validation_fraction": 0.1,
                "available_scene_count": 20,
                "excluded_empty_scene_count": 3,
                "eligible_group_count": len(groups),
                "eligible_group_uid_digest": group_uid_digest(groups),
                "eligible_sample_count": 123,
                "eligible_sample_uid_digest": "a" * 64,
                "training_sample_count": 111,
                "training_sample_uid_digest": "d" * 64,
                "validation_group_count": len(selected_manifest),
                "validation_group_uid_digest": group_uid_digest(
                    selected_manifest
                ),
                "validation_group_uids": selected_manifest,
                "validation_sample_count": 12,
                "validation_sample_uid_digest": "c" * 64,
            },
        )

        selected = validation_group_uids(
            groups,
            val_fraction=0.1,
            policy=KITSCENES_TRAINING_POLICY,
            source_revision="revision-a",
            packed_dataset_version="v2.2",
            packed_contract_digest="contract-a",
            packed_partition_count=20,
            empty_partition_count=3,
            packed_sample_count=123,
            packed_sample_uid_digest="a" * 64,
        )
        reordered = validation_group_uids(
            list(reversed(groups)),
            val_fraction=0.1,
            policy=KITSCENES_TRAINING_POLICY,
            source_revision="revision-a",
            packed_dataset_version="v2.2",
            packed_contract_digest="contract-a",
            packed_partition_count=20,
            empty_partition_count=3,
            packed_sample_count=123,
            packed_sample_uid_digest="a" * 64,
        )

        assert selected == reordered
        assert selected == tuple(selected_manifest)

    def test_kitscenes_holdout_rejects_changed_scene_inventory(self):
        with pytest.raises(
            ValueError,
            match="differs from the frozen split",
        ):
            validation_group_uids(
                ["kitscenes-a", "kitscenes-b"],
                val_fraction=0.1,
                policy=KITSCENES_TRAINING_POLICY,
                source_revision=(
                    "6fde0034446669e2ed7235e4c7fe323cd23d599d"
                ),
                packed_dataset_version="v2.2",
                packed_contract_digest=(
                    "a0bf504e37b448b42135e9292b307d7e"
                    "a3087cb6ec9554e52cb1d4db7b696224"
                ),
                packed_partition_count=533,
                empty_partition_count=129,
                packed_sample_count=42667,
                packed_sample_uid_digest=(
                    "d169a7ac79a30586e213e6b1f4ac4377"
                    "c038bd4edbeef39895e526160a00e286"
                ),
            )

    def test_committed_kitscenes_holdout_manifest_is_self_consistent(self):
        payload = dataset_policy._load_validation_manifest(
            KITSCENES_TRAINING_POLICY
        )
        selected = payload["validation_group_uids"]

        assert selected == sorted(set(selected))
        assert len(selected) == payload["validation_group_count"] == 40
        assert payload["eligible_group_count"] == 404
        assert payload["eligible_sample_count"] == 42667
        assert payload["eligible_sample_uid_digest"] == (
            "d169a7ac79a30586e213e6b1f4ac4377"
            "c038bd4edbeef39895e526160a00e286"
        )
        assert payload["training_sample_count"] == 38847
        assert payload["validation_sample_count"] == 3820
        assert payload["dataset_version"] == "v2.2"
        assert payload["packed_contract_digest"] == (
            "a0bf504e37b448b42135e9292b307d7e"
            "a3087cb6ec9554e52cb1d4db7b696224"
        )
        assert group_uid_digest(selected) == (
            payload["validation_group_uid_digest"]
        )
        assert validation_sample_identity(
            KITSCENES_TRAINING_POLICY
        ) == (
            3820,
            "62ea79c5f45b1ac47dab3cfeab604244"
            "38cd0c09994b6f44c215a473f9e31f04",
        )

    def test_l2d_holdout_retains_legacy_hash_buckets(self):
        assert validation_group_uids(
            [],
            val_fraction=0.1,
            policy=L2D_TRAINING_POLICY,
        ) is None

    def test_exact_holdout_rejects_duplicate_scene_uids(self):
        with pytest.raises(ValueError, match="must be unique"):
            validation_group_uids(
                ["kitscenes-a", "kitscenes-a"],
                val_fraction=0.1,
                policy=KITSCENES_TRAINING_POLICY,
            )
