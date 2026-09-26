"""Training lifecycle and recovered-workflow contracts."""

from __future__ import annotations

import ast
import gc
import hashlib
import inspect
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flytekit")

from evaluation.checkpoint_selection import SELECTOR_POLICY_VERSION
from evaluation.reconstruction_audit import AUDIT_SCHEMA_VERSION
from Platform.pipelines import workflows


CANONICAL_METRIC_CONTRACT = {
    "version": "rollout_validation_v2",
    "horizon_seconds": 3.0,
    "horizon_steps": 30,
    "target_source": "logged_xy",
    "aggregation": "scene_balanced",
}


class _SceneProjection:
    def __init__(self, scene_index):
        self.scene_index = scene_index

    def to(self, device):
        return SimpleNamespace(
            scene_index=self.scene_index,
            device=device,
        )


class _MetricModel:
    def __init__(self):
        self.training = True
        self.reset_count = 0
        self.last_egomotion_history = None
        self.initial_noise_calls = []

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def reset_visual_history(self):
        self.reset_count += 1

    def __call__(self, visual, *args, **kwargs):
        self.last_egomotion_history = args[2]
        self.initial_noise_calls.append(
            kwargs["initial_noise"].detach().clone()
        )
        return torch.zeros((visual.shape[0], 128), dtype=torch.float32)


class _RouteSensitiveMetricModel(_MetricModel):
    def __call__(self, visual, *args, **kwargs):
        self.initial_noise_calls.append(
            kwargs["initial_noise"].detach().clone()
        )
        route_mask = kwargs["route_mask"]
        batch_size, _, _, width = route_mask.shape
        columns = torch.arange(
            width,
            dtype=route_mask.dtype,
            device=route_mask.device,
        )
        corridor = route_mask[:, 0]
        mass = corridor.sum(dim=(1, 2)).clamp_min(1.0)
        centroid = (
            corridor.sum(dim=1) * columns
        ).sum(dim=1) / mass
        curvature = (width / 2.0 - centroid) * 1e-3
        output = torch.zeros(
            (batch_size, 64, 2),
            dtype=visual.dtype,
            device=visual.device,
        )
        output[:, :, 1] = curvature[:, None]
        return output.reshape(batch_size, 128)


def _validation_batch(sample_uids):
    batch_size = len(sample_uids)
    ego = torch.zeros((batch_size, 256), dtype=torch.float32)
    ego[:, -4] = 2.0
    return {
        "sample_uid": list(sample_uids),
        "visual_tiles": torch.zeros(
            (batch_size, 7, 3, 2, 2), dtype=torch.float32
        ),
        "map_context": torch.zeros(
            (batch_size, 3, 2, 2), dtype=torch.float32
        ),
        "route_mask": torch.zeros(
            (batch_size, 2, 2, 2), dtype=torch.float32
        ),
        "map_valid": torch.ones(batch_size, dtype=torch.bool),
        "route_valid": torch.zeros(batch_size, dtype=torch.bool),
        "egomotion_history": ego,
        "visual_history": torch.zeros(
            (batch_size, 896), dtype=torch.float32
        ),
        "trajectory_target": torch.zeros(
            (batch_size, 128), dtype=torch.float32
        ),
    }


def _navigation_validation_batch(
    sample_uid,
    route_id,
    lateral_m,
    maneuver="straight",
):
    from navigation.geometry import (
        DEFAULT_NAVIGATION_GEOMETRY,
        MapChannel,
        RouteChannel,
    )

    geometry = DEFAULT_NAVIGATION_GEOMETRY
    batch = _validation_batch([sample_uid])
    batch["map_context"] = torch.zeros(
        (1, 14, geometry.height_px, geometry.width_px),
        dtype=torch.float32,
    )
    batch["map_context"][:, MapChannel.KNOWN_MAP_AREA] = 1.0
    route = torch.zeros(
        (1, 2, geometry.height_px, geometry.width_px),
        dtype=torch.float32,
    )
    points = torch.stack([
        torch.arange(0.0, 65.0),
        torch.full((65,), float(lateral_m)),
    ], dim=1).numpy()
    pixels = geometry.ego_to_pixel(points)
    for row, col in torch.from_numpy(pixels).round().to(torch.int64):
        route[
            0,
            RouteChannel.SELECTED_CORRIDOR,
            max(0, int(row) - 1):int(row) + 2,
            max(0, int(col) - 1):int(col) + 2,
        ] = 1.0
    row, col = torch.from_numpy(pixels[-1]).round().to(torch.int64)
    route[
        0,
        RouteChannel.DESTINATION,
        max(0, int(row) - 1):int(row) + 2,
        max(0, int(col) - 1):int(col) + 2,
    ] = 1.0
    batch["route_mask"] = route
    batch["route_valid"] = torch.ones(1, dtype=torch.bool)
    batch["navigation_metadata"] = {
        "route_id": [route_id],
        "route_maneuver": [maneuver],
        "route_intersection": torch.zeros(1, dtype=torch.bool),
        "destination_visible": torch.ones(1, dtype=torch.bool),
        "route_confidence": torch.full((1,), 0.9),
    }
    return batch


def _rollout_selector_validation_batch(sample_uid="sample-a"):
    from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

    batch = _validation_batch([sample_uid])
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    field = torch.zeros(
        1,
        geometry.height_px,
        geometry.width_px,
        dtype=torch.float32,
    )
    batch.update({
        "split_group_uid": ["scene-a"],
        "route_mask": torch.zeros(
            1,
            2,
            geometry.height_px,
            geometry.width_px,
            dtype=torch.float32,
        ),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "pose_current": torch.tensor(
            [[49.0, 8.0, 0.0]],
            dtype=torch.float64,
        ),
        "gps_future": torch.tensor(
            [[[49.0, 8.0]] * 65],
            dtype=torch.float64,
        ),
        "route_supervision": {
            "distance_to_corridor_m": field,
            "distance_to_drivable_m": field,
            "destination_xy_m": torch.zeros(1, 2),
            "destination_visible": torch.tensor([False]),
            "available": torch.tensor([True]),
            "drivable_available": torch.tensor([True]),
        },
        "navigation_metadata": {
            "route_intersection": torch.tensor([False]),
        },
    })
    return batch


def test_epoch_evaluation_restores_mode_and_hashes_fixed_uids():
    model = _MetricModel()
    loader = [
        (_validation_batch(["sample-b", "sample-a"]), None, "pseudo")
    ]

    metrics = workflows._evaluate_open_loop(
        model, loader, torch.device("cpu")
    )

    expected_digest = hashlib.sha256(
        b"sample-a\nsample-b"
    ).hexdigest()
    assert metrics == {
        "ade": 0.0,
        "fde": 0.0,
        "evaluation_steps": 30,
        "prediction_steps": 64,
        "sample_count": 2,
        "sample_uid_digest": expected_digest,
        "metric_contract": {
            "version": "control_rollout_validation_v2",
            "horizon_seconds": 3.0,
            "horizon_steps": 30,
            "target_source": "target_control_rollout",
            "aggregation": "sample_mean",
        },
        "horizons": {
            "1s": {"steps": 10, "ade": 0.0, "fde": 0.0},
            "2s": {"steps": 20, "ade": 0.0, "fde": 0.0},
            "3s": {"steps": 30, "ade": 0.0, "fde": 0.0},
        },
    }
    assert model.training is True
    assert model.reset_count == 2


def test_epoch_evaluation_builds_logged_xy_selector_records():
    model = _MetricModel()
    loader = [
        (
            _rollout_selector_validation_batch(),
            None,
            "pseudo",
        )
    ]

    metrics = workflows._evaluate_open_loop(
        model,
        loader,
        torch.device("cpu"),
        include_rollout_selector_records=True,
    )

    record = metrics["rollout_selector_records"][0]
    assert record["sample_uid"] == "sample-a"
    assert record["split_group_uid"] == "scene-a"
    assert record["ade_3s_m"] > 0.0
    assert record["fde_3s_m"] > record["ade_3s_m"]
    assert record["comfort_excess"] == 0.0
    assert record["offroad_excess"] == 0.0
    assert record["route_gap"] == 0.0
    assert metrics["ade"] == record["ade_3s_m"]
    assert metrics["fde"] == record["fde_3s_m"]
    assert metrics["metric_contract"] == CANONICAL_METRIC_CONTRACT


def test_evaluation_noise_is_stable_by_sample_uid():
    forward = workflows._stable_evaluation_noise(
        ["sample-a", "sample-b"],
        128,
        torch.float32,
    )
    reverse = workflows._stable_evaluation_noise(
        ["sample-b", "sample-a"],
        128,
        torch.float32,
    )

    torch.testing.assert_close(forward[0], reverse[1])
    torch.testing.assert_close(forward[1], reverse[0])
    assert not torch.equal(forward[0], forward[1])


def test_epoch_evaluation_rejects_duplicate_uids():
    model = _MetricModel()
    loader = [
        (_validation_batch(["sample-a", "sample-a"]), None, "pseudo")
    ]

    with pytest.raises(ValueError, match="duplicate sample UIDs"):
        workflows._evaluate_open_loop(
            model, loader, torch.device("cpu")
        )


def test_training_projection_cache_cannot_alias_404_scene_calibrations():
    device = torch.device("cpu")
    cache = workflows._ProjectionDeviceCache(device)
    source_refs = []
    converted_scenes = []

    for scene_index in range(404):
        source = _SceneProjection(scene_index)
        source_refs.append(weakref.ref(source))
        converted = cache.get(source)
        assert cache.get(source) is converted
        converted_scenes.append(converted.scene_index)
        del converted
        del source

    gc.collect()
    assert converted_scenes == list(range(404))
    assert all(source_ref() is None for source_ref in source_refs)
    assert len(cache) == 0

    training_source = inspect.getsource(workflows.train_il.task_function)
    assert "_ProjectionDeviceCache(device)" in training_source
    assert "_proj_cache.get(batch_proj)" in training_source
    assert "id(batch_proj)" not in training_source


def test_exact_split_alone_requires_one_explicit_source_revision():
    same_revision = {
        "a": {"source_revision": "revision-a"},
        "b": {"source_revision": "revision-a"},
    }
    mixed_revisions = {
        "a": {"source_revision": "revision-a"},
        "b": {"source_revision": "revision-b"},
    }

    assert workflows._training_source_revision(
        same_revision,
        require_single=True,
    ) == "revision-a"
    assert workflows._training_source_revision(
        mixed_revisions,
        require_single=False,
    ) == ""
    with pytest.raises(ValueError, match="one explicit packed"):
        workflows._training_source_revision(
            mixed_revisions,
            require_single=True,
        )
    with pytest.raises(ValueError, match="one explicit packed"):
        workflows._training_source_revision(
            {"a": {}, "b": {"source_revision": "revision-a"}},
            require_single=True,
        )


def test_exact_evaluation_rejects_packed_provenance_drift(tmp_path):
    from Platform.pipelines.training_checkpoint import stable_digest

    contracts = {"geometry": "v2", "shard": "v2"}
    shard_dir = tmp_path / "partition"
    shard_dir.mkdir()
    manifest_path = shard_dir / "manifest.json"
    manifest = {
        "dataset": "KIT-MRT/KITScenes-Multimodal",
        "source_revision": "revision-a",
        "dataset_version": "v2.2",
        "contracts": contracts,
    }
    manifest_path.write_text(json.dumps(manifest))

    kwargs = {
        "dataset_name": "KIT-MRT/KITScenes-Multimodal",
        "source_revision": "revision-a",
        "dataset_version": "v2.2",
        "contract_digest": stable_digest(contracts),
    }
    workflows._validate_evaluation_shard_provenance(
        [str(shard_dir)],
        **kwargs,
    )

    manifest["contracts"]["geometry"] = "v3"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance differs"):
        workflows._validate_evaluation_shard_provenance(
            [str(shard_dir)],
            **kwargs,
        )


def test_training_wires_dataset_specific_trajectory_policy():
    training_source = inspect.getsource(workflows.train_il.task_function)

    assert "training_policy_for_dataset" in training_source
    assert "dataset.value" in training_source
    assert "signal_scales=training_policy.signal_scales" in training_source
    assert "temporal_decay=training_policy.temporal_decay" in training_source
    assert (
        "temporal_weight_normalization=("
        in training_source
    )
    assert "supervised_timesteps" not in training_source
    assert "AUTO_E2E_TIMESTEPS" in training_source
    assert "adapt_egomotion_history" in training_source
    assert "discover_split_inventory" in training_source
    assert "select_validation_group_uids" in training_source
    assert "validation_group_uids=fixed_validation_groups" in (
        training_source
    )
    assert "decode_future_frames=False" in training_source
    assert '"trajectory_training_policy": training_policy.metadata()' in (
        training_source
    )
    assert '"validation_split": validation_split_contract' in (
        training_source
    )
    assert '"training_objective_version": training_objective_version' in (
        training_source
    )
    assert "route_consistency_loss_fn(" in training_source
    assert "route_consistency_weight" in training_source
    assert "rollout_aligned_loss_fn(" in training_source
    assert "wgs84_trajectory_to_ego_xy(" in training_source
    assert "logged_positions = torch.from_numpy(" in training_source
    assert "rollout-aligned loss requires packed pose and GPS" in (
        training_source
    )
    assert (
        "initial_speed,\n"
        "                        logged_positions,\n"
        "                        route_supervision,"
    ) in training_source
    assert '0.5 * rollout_terms["rollout"]' in training_source
    assert '0.05 * rollout_terms["constraint"]' in training_source
    assert '"rollout_aligned_loss": rollout_aligned_config' in (
        training_source
    )
    assert '"objective_term_gradient_norms": None' in training_source
    assert '"weighted_jepa": (' in training_source
    assert "weighted JEPA produced no World Model gradient" in training_source
    assert "reconstruction audit identity differs from training" in (
        training_source
    )
    assert '"rollout_policy_version": ROLLOUT_POLICY_VERSION' in (
        training_source
    )
    assert "reconstruction_audit_decision != \"go\"" in training_source
    assert "if selector_enabled:" in training_source
    assert (
        "composite-selector training requires a reconstruction audit"
        in training_source
    )
    assert "target rollout reconstruction thresholds failed" not in (
        training_source
    )
    assert '"position_target_source": (' in training_source
    assert '"packed_logged_xy" if objective_v2 else "not_applicable"' in (
        training_source
    )
    assert "audit_report.get(\"thresholds\") != expected_thresholds" in (
        training_source
    )
    assert '"p95_fde_3s_limit_m": P95_FDE_3S_LIMIT_M' in training_source
    assert (
        '"p95_fde_full_limit_m": P95_FDE_FULL_LIMIT_M'
        in training_source
    )
    assert "threshold exception requires current-model" not in training_source
    assert '"reconstruction_audit": reconstruction_audit_contract' in (
        training_source
    )
    assert (
        "!= ROUTE_SUPERVISION_ARTIFACT_VERSION"
        in training_source
    )
    assert "route-enabled epoch produced no eligible route sample" in (
        training_source
    )

    evaluation_source = inspect.getsource(workflows._run_evaluation)
    assert "validation_group_uids=fixed_validation_groups" in (
        evaluation_source
    )
    assert "decode_future_frames=False" in evaluation_source
    assert "validation group manifest digest mismatch" in evaluation_source
    assert "checkpoint has no exact validation_split contract" in (
        evaluation_source
    )

    offline_rl_source = inspect.getsource(
        workflows.train_offline_rl.task_function
    )
    assert "refusing to train on one shard" in offline_rl_source


def test_reconstruction_audit_uses_training_group_digest_contract():
    audit_function = (
        workflows.audit_kitscenes_target_reconstruction.task_function
    )
    source = inspect.getsource(audit_function)
    signature = inspect.signature(audit_function)

    assert "group_uid_digest(validation_group_uids)" in source
    assert '"\\n".join(validation_group_uids)' not in source
    assert "discover_split_inventory(shard_dirs)" in source
    assert "training_policy_for_dataset(" in source
    assert "select_validation_group_uids(" in source
    assert "packed_partition_count=len(shard_identities)" in source
    assert "packed_sample_count=split_inventory.sample_count" in source
    assert (
        "packed_sample_uid_digest=split_inventory.sample_uid_digest"
        in source
    )
    assert "expected_validation_sample_uid_digest" not in (
        signature.parameters
    )
    assert "validation_group_uids" not in signature.parameters
    assert signature.parameters["val_fraction"].default == 0.1
    assert signature.parameters["validation_scope"].default == "full"


def test_recovered_reconstruction_audit_binds_exact_repack_scope():
    nodes = (
        workflows.wf_audit_recovered_kitscenes_target_reconstruction.nodes
    )
    assert [
        getattr(node.flyte_entity, "name", "")
        for node in nodes
    ] == [
        workflows.wf_repack_existing_kitscenes.name,
        workflows.audit_kitscenes_target_reconstruction.name,
    ]
    repack_node, audit_node = nodes
    audit_bindings = {
        binding.var: binding.binding.promise
        for binding in audit_node.bindings
    }
    assert audit_bindings["packed_shards"].node_id == repack_node.id


def test_reconstruction_audit_allows_only_empty_partitions_without_gps(
    tmp_path,
    monkeypatch,
):
    from evaluation import reconstruction_audit
    from training import dataset_policy
    from data_parsing import pre_extracted

    source_revision = "a" * 40
    sample_digest = "b" * 64
    metric_distribution = {
        "mean": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }
    metrics = {
        name: {
            "natural": dict(metric_distribution),
            "scene_mean_distribution": dict(metric_distribution),
        }
        for name in (
            "ade_3s_m",
            "fde_3s_m",
            "ade_full_m",
            "fde_full_m",
        )
    }

    class _Shard:
        def __init__(self, path):
            self.path = path
            self.remote_source = f"s3://test/{path.name}"

        def download(self):
            return str(self.path)

    class _Inventory:
        group_uids = ("scene-a", "scene-empty")
        sample_count = 1
        sample_uid_digest = sample_digest

        @staticmethod
        def sample_identity_for_groups(group_uids):
            assert group_uids == ("scene-a",)
            return 1, sample_digest

    def write_manifest(path, *, total_samples, has_gps):
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({
            "contracts": {"packed_schema": "v8"},
            "dataset": workflows.Dataset.KITSCENES.value,
            "dataset_version": "v3.3",
            "has_gps": has_gps,
            "hz": 10,
            "partition_id": path.name,
            "shard_names": (
                ["samples.tar"] if total_samples else []
            ),
            "source_revision": source_revision,
            "total_samples": total_samples,
        }))

    non_empty = tmp_path / "non-empty"
    empty = tmp_path / "empty"
    write_manifest(non_empty, total_samples=1, has_gps=True)
    write_manifest(empty, total_samples=0, has_gps=False)

    monkeypatch.setattr(
        pre_extracted,
        "discover_split_inventory",
        lambda shard_dirs: _Inventory(),
    )
    monkeypatch.setattr(
        dataset_policy,
        "training_policy_for_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            validation_split_id="subset-test",
        ),
    )
    monkeypatch.setattr(
        dataset_policy,
        "validation_group_uids",
        lambda *args, **kwargs: ("scene-a",),
    )
    monkeypatch.setattr(
        reconstruction_audit,
        "load_packed_reconstruction_inputs",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        reconstruction_audit,
        "audit_packed_target_rollout_reconstruction",
        lambda inputs: {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "sample_count": 1,
            "scene_count": 1,
            "sample_uid_digest": sample_digest,
            "split_group_uid_digest": "c" * 64,
            "horizons": {
                "three_second_steps": 30,
                "full_horizon_steps": 64,
                "dt": 0.1,
            },
            "thresholds": {
                "p95_fde_3s_limit_m": 1.0,
                "p95_fde_full_limit_m": 2.0,
            },
            "thresholds_pass": True,
            "decision": {
                "status": "pending_review",
                "automatic_recommendation": "go",
                "rationale": None,
            },
            "input_quality": {
                "missing_sample_count": 0,
                "non_finite_sample_count": 0,
            },
            "metrics": metrics,
            "heading_alignment": {
                "valid_step_count": 64,
                "full_horizon": {
                    "mean": 0.0,
                    "p50": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                },
            },
            "error_by_step": [],
            "worst_scenes": {},
            "scenes": [],
            "records": [{
                "sample_uid": "sample-a",
                "split_group_uid": "scene-a",
                "ade_3s_m": 0.0,
                "fde_3s_m": 0.0,
                "ade_full_m": 0.0,
                "fde_full_m": 0.0,
            }],
        },
    )
    monkeypatch.setenv("AUTO_E2E_EVAL_IMAGE", "eval@sha256:test")

    result = workflows.audit_kitscenes_target_reconstruction.task_function(
        packed_shards=[_Shard(non_empty), _Shard(empty)],
        audit_code_revision="d" * 40,
        expected_dataset_version="v3.3",
        val_fraction=0.1,
        validation_scope="subset",
    )

    assert result.thresholds_pass
    manifest = json.loads((non_empty / "manifest.json").read_text())
    manifest["has_gps"] = False
    (non_empty / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no pose-grounded trajectory"):
        workflows.audit_kitscenes_target_reconstruction.task_function(
            packed_shards=[_Shard(non_empty), _Shard(empty)],
            audit_code_revision="d" * 40,
            expected_dataset_version="v3.3",
            val_fraction=0.1,
            validation_scope="subset",
        )


def test_training_seed_controls_comparable_navigation_runs():
    training_function = workflows.train_il.task_function
    training_source = inspect.getsource(training_function)
    signature = inspect.signature(training_function)

    assert signature.parameters["training_seed"].default == 149
    assert "random.seed(training_seed)" in training_source
    assert "np.random.seed(training_seed)" in training_source
    assert "torch.manual_seed(training_seed)" in training_source
    assert "torch.cuda.manual_seed_all(training_seed)" in training_source
    assert "torch.backends.cudnn.benchmark = False" in training_source
    assert "torch.backends.cudnn.deterministic = True" in training_source
    assert '"training_seed": training_seed' in training_source
    assert '"train/seed": training_seed' in training_source


def test_selector_preflight_requires_frozen_validation_identity():
    workflows._validate_selector_preflight_identity(
        {
            "sample_count": 2,
            "sample_uid_digest": "a" * 64,
        },
        expected_sample_count=2,
        expected_sample_uid_digest="a" * 64,
    )

    with pytest.raises(ValueError, match="preflight validation identity"):
        workflows._validate_selector_preflight_identity(
            {
                "sample_count": 2,
                "sample_uid_digest": "b" * 64,
            },
            expected_sample_count=2,
            expected_sample_uid_digest="a" * 64,
        )


def test_navigation_objective_wiring_is_train_only_and_versioned():
    source = inspect.getsource(workflows.train_il.task_function)
    tree = ast.parse(source)
    loader_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "make_multi_dataset_loader"
    ]
    assert len(loader_calls) == 2
    by_root = {
        call.args[0].id: {
            keyword.arg: keyword.value for keyword in call.keywords
        }
        for call in loader_calls
    }
    assert "navigation_repeat_policy" not in by_root["shard_dirs"]
    training_repeat = by_root["training_shard_dirs"][
        "navigation_repeat_policy"
    ]
    assert isinstance(training_repeat, ast.Name)
    assert training_repeat.id == "navigation_repeat_policy"

    module = ast.parse(inspect.getsource(workflows))
    recovered = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "wf_recovered_kitscenes_full_run"
    )
    names = [argument.arg for argument in recovered.args.args]
    defaults = dict(zip(
        names[-len(recovered.args.defaults):],
        recovered.args.defaults,
        strict=True,
    ))
    assert ast.literal_eval(defaults["epochs"]) == 20
    assert isinstance(defaults["training_objective_version"], ast.Name)
    assert defaults["training_objective_version"].id == (
        "ROLLOUT_ALIGNED_OBJECTIVE_VERSION"
    )
    assert ast.literal_eval(defaults["enable_junction_sampling"]) is False
    assert ast.literal_eval(defaults["enable_route_consistency"]) is False
    train_call = next(
        call
        for call in ast.walk(recovered)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "train_il"
    )
    train_keywords = {
        keyword.arg: keyword.value for keyword in train_call.keywords
    }
    for field in (
        "training_objective_version",
        "enable_junction_sampling",
        "enable_route_consistency",
        "route_consistency_weight",
        "reconstruction_audit",
        "reconstruction_audit_decision",
        "reconstruction_audit_rationale",
        "allow_resume_policy_transition",
    ):
        assert isinstance(train_keywords[field], ast.Name)
        assert train_keywords[field].id == field


def test_sharded_full_run_forwards_composite_selector_audit():
    module = ast.parse(inspect.getsource(workflows))
    full_run = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "wf_sharded_full_run"
    )
    train_call = next(
        call
        for call in ast.walk(full_run)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "train_il"
    )
    train_keywords = {
        keyword.arg: keyword.value for keyword in train_call.keywords
    }
    for field in (
        "reconstruction_audit",
        "reconstruction_audit_decision",
        "reconstruction_audit_rationale",
    ):
        assert isinstance(train_keywords[field], ast.Name)
        assert train_keywords[field].id == field


def test_rollout_control_arm_uses_composite_selector_without_rollout_loss():
    source = inspect.getsource(workflows.train_il.task_function)

    assert (
        "selector_enabled = objective_v2 or objective_v2_control"
        in source
    )
    assert "if selector_enabled and not manifest.get(\"has_gps\"" in source
    assert (
        "if selector_enabled and (\n"
        "            not manifest.get(\"has_route_supervision\""
        in source
    )
    assert (
        "rollout_aligned_loss_fn = RolloutAlignedLoss().to(device)"
        in source
    )
    assert '"enabled": objective_v2' in source
    assert (
        workflows.ROLLOUT_ALIGNED_CONTROL_OBJECTIVE_VERSION
        == "rollout_aligned_control_v1"
    )
    assert "build_selector_calibration_report" in source
    assert '"calibration_report"' in source
    assert '"selection/effective_weight/{name}"' in source
    assert '"selection/component/{name}"' in source
    assert '"selection/calibration/min_rank_correlation"' in source
    assert '"top_level_weights": dict(TOP_LEVEL_WEIGHTS)' in source
    assert '"utility_scales": dict(UTILITY_SCALES)' in source
    assert '"train/checkpoint_selector_weight_{name}"' in source
    assert '"train/checkpoint_selector_scale_{name}"' in source
    assert '"train/loss_rollout"' in source
    assert '"train/loss_comfort_jerk"' in source
    assert '"train/loss_comfort_lateral_acceleration"' in source
    assert '"train/loss_comfort_lateral"' in source
    assert '"train/loss_map_route"' in source
    assert '"train/loss_map_drivable"' in source
    assert '"train/loss_total"' in source
    assert '"validation/{aggregate_name}/{metric_name}"' in source
    assert '"validation/scene_distribution/"' in source
    assert '"validation/coverage/{metric_name}/eligible_samples"' in source
    assert '"audit/reconstruction/sample_count"' in source
    assert "torch.cuda.synchronize(device)" in source
    assert '"train_wall_seconds": training_wall_seconds' in source
    assert '"epoch_compute_wall_seconds": epoch_compute_wall_seconds' in source
    assert '"samples_per_second": (' in source
    assert '"optimizer_steps_per_second": (' in source
    assert '"throughput": throughput' in source
    assert '"throughput": throughput_summary' in source
    assert '"train/throughput/samples_per_second"' in source
    assert '"train/throughput/optimizer_steps_per_second"' in source


def test_rollout_epoch_diagnostics_use_eligible_sample_weights():
    sums = {
        "rollout": 0.0,
        "map": 0.0,
        "route": 0.0,
        "drivable": 0.0,
    }
    weights = {name: 0 for name in sums}

    workflows._accumulate_rollout_epoch_terms(
        sums,
        weights,
        {
            "rollout": torch.tensor(2.0),
            "map": torch.tensor(8.0),
            "route": torch.tensor(4.0),
            "drivable": torch.tensor(0.0),
            "map_sample_count": torch.tensor(1),
            "route_sample_count": torch.tensor(1),
            "drivable_sample_count": torch.tensor(0),
        },
        batch_sample_count=4,
    )
    workflows._accumulate_rollout_epoch_terms(
        sums,
        weights,
        {
            "rollout": torch.tensor(4.0),
            "map": torch.tensor(2.0),
            "route": torch.tensor(0.0),
            "drivable": torch.tensor(3.0),
            "map_sample_count": torch.tensor(2),
            "route_sample_count": torch.tensor(0),
            "drivable_sample_count": torch.tensor(2),
        },
        batch_sample_count=2,
    )

    assert sums == {
        "rollout": 16.0,
        "map": 12.0,
        "route": 4.0,
        "drivable": 6.0,
    }
    assert weights == {
        "rollout": 6,
        "map": 3,
        "route": 1,
        "drivable": 2,
    }


def test_kitscenes_evaluation_keeps_prediction_but_scores_three_seconds():
    from training.dataset_policy import KITSCENES_TRAINING_POLICY

    model = _MetricModel()
    batch = _validation_batch(["sample-a"])
    history = batch["egomotion_history"].reshape(1, 64, 4)
    history[:, :, :] = 1.0
    history[:, -1, 0] = 2.0
    target = batch["trajectory_target"].reshape(1, 64, 2)
    target[:, 50:, :] = 100.0

    metrics = workflows._evaluate_open_loop(
        model,
        [(batch, None, "pseudo")],
        torch.device("cpu"),
        training_policy=KITSCENES_TRAINING_POLICY,
    )

    assert metrics["ade"] == 0.0
    assert metrics["fde"] == 0.0
    assert metrics["evaluation_steps"] == 30
    assert metrics["prediction_steps"] == 64
    adapted = model.last_egomotion_history.reshape(1, 64, 4)
    assert torch.count_nonzero(adapted[:, :24]) == 24 * 4
    assert adapted[0, -1, 0].item() == 2.0
    assert adapted[0, -1, 1].item() == 0.0


def test_standalone_navigation_evaluation_runs_cross_scene_route_swap():
    from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

    model = _RouteSensitiveMetricModel()
    loader = [
        (
            _navigation_validation_batch(
                "sample-a",
                "route-a",
                20.0,
                maneuver="left",
            ),
            None,
            "pseudo",
        ),
        (
            _navigation_validation_batch(
                "sample-b",
                "route-b",
                -20.0,
                maneuver="right",
            ),
            None,
            "pseudo",
        ),
    ]

    metrics = workflows._evaluate_open_loop(
        model,
        loader,
        torch.device("cpu"),
        navigation_geometry=DEFAULT_NAVIGATION_GEOMETRY,
        route_swap_counterfactual=True,
        include_navigation_records=True,
    )

    report = metrics["navigation"]
    records = metrics["navigation_records"]
    assert [record["sample_uid"] for record in records] == [
        "sample-a",
        "sample-b",
    ]
    assert report["slices"]["overall"]["sample_count"] == 2
    assert report["slices"]["route_valid"]["sample_count"] == 2
    counterfactual = report["route_swap_counterfactual"]
    assert counterfactual["sample_count"] == 1
    assert counterfactual["different_maneuver_sample_count"] == 1
    assert counterfactual["endpoint_delta_m"]["mean"] > 0.0
    assert (
        counterfactual["maneuver_direction_consistent"]["mean"]
        == 1.0
    )
    assert "right_to_left" in counterfactual["maneuver_pairs"]
    assert (
        report["slices"]["overall"]["route_quality"][
            "route_confidence"
        ]["p50"]
        == pytest.approx(0.9)
    )
    assert len(model.initial_noise_calls) == 3
    torch.testing.assert_close(
        model.initial_noise_calls[1],
        model.initial_noise_calls[2],
    )


def test_terminal_resume_state_allows_finalization():
    assert workflows._resume_terminal_state(
        completed_epoch=10,
        bad_epochs=1,
        requested_epochs=10,
        patience=3,
    ) == (True, False)
    assert workflows._resume_terminal_state(
        completed_epoch=6,
        bad_epochs=3,
        requested_epochs=10,
        patience=3,
    ) == (True, True)
    assert workflows._resume_terminal_state(
        completed_epoch=6,
        bad_epochs=2,
        requested_epochs=10,
        patience=3,
    ) == (False, False)

    with pytest.raises(ValueError, match="beyond requested"):
        workflows._resume_terminal_state(
            completed_epoch=11,
            bad_epochs=0,
            requested_epochs=10,
            patience=3,
        )


def test_resume_policy_transition_enables_repeat_and_resets_patience():
    transition = workflows._resume_policy_transition(
        saved_config={
            "junction_sampling": {"enabled": False, "policy": None},
            "early_stopping_patience": 3,
        },
        requested_config={
            "junction_sampling": {
                "enabled": True,
                "policy": {"version": "navigation_repeat_v1"},
            },
            "early_stopping_patience": 8,
        },
    )

    assert transition["policy_version"] == "dual_best_resume_transition_v1"
    assert transition["junction_sampling"]["from"]["enabled"] is False
    assert transition["junction_sampling"]["to"]["enabled"] is True
    assert transition["junction_sampling"]["changed"] is True
    assert transition["early_stopping_patience"] == {
        "from": 3,
        "to": 8,
    }
    assert transition["bad_epochs_after_reset"] == 0
    assert transition["scheduler_state_action"] == (
        "reset_plateau_state_preserve_optimizer_lr"
    )
    assert transition["best_checkpoint_scope"] == "full_history"

    score_best = {
        "epoch": 1,
        "uri": "s3://checkpoints/run/epoch-0001.pt",
        "sha256": "a" * 64,
        "selection": {
            "score": 0.52,
            "components": {"trajectory": 0.40},
        },
    }
    trajectory_best = {
        "epoch": 2,
        "uri": "s3://checkpoints/run/epoch-0002.pt",
        "sha256": "b" * 64,
        "selection": {
            "score": 0.51,
            "components": {"trajectory": 0.45},
        },
    }
    bad_epochs, best, best_trajectory = (
        workflows._transition_resume_selection_state(
            transition,
            bad_epochs=2,
            best_checkpoint=score_best,
            best_trajectory_checkpoint=trajectory_best,
        )
    )

    assert bad_epochs == 0
    assert best == score_best
    assert best_trajectory == trajectory_best
    assert transition["bad_epochs_before_reset"] == 2
    assert transition["best_before_resume"] == {
        "epoch": 1,
        "uri": "s3://checkpoints/run/epoch-0001.pt",
        "sha256": "a" * 64,
        "selection_score": 0.52,
    }
    assert transition["best_trajectory_before_resume"] == {
        "epoch": 2,
        "uri": "s3://checkpoints/run/epoch-0002.pt",
        "sha256": "b" * 64,
        "trajectory_utility": 0.45,
    }


def test_resume_policy_transition_allows_patience_only_continuation():
    transition = workflows._resume_policy_transition(
        saved_config={
            "junction_sampling": {"enabled": False, "policy": None},
            "early_stopping_patience": 3,
        },
        requested_config={
            "junction_sampling": {"enabled": False, "policy": None},
            "early_stopping_patience": 5,
        },
    )

    assert transition["junction_sampling"]["changed"] is False
    assert transition["early_stopping_patience"] == {"from": 3, "to": 5}


@pytest.mark.parametrize(
    (
        "candidate_score",
        "candidate_trajectory",
        "expected_score",
        "expected_trajectory",
        "expected_patience_improvement",
    ),
    [
        (0.61, 0.49, True, False, True),
        (0.59, 0.51, False, True, True),
        (0.59, 0.49, False, False, False),
    ],
)
def test_dual_best_improvement_controls_patience(
    candidate_score,
    candidate_trajectory,
    expected_score,
    expected_trajectory,
    expected_patience_improvement,
):
    score_improved, trajectory_improved = (
        workflows._dual_best_improvements(
            {
                "score": candidate_score,
                "components": {"trajectory": candidate_trajectory},
            },
            best_selection={
                "score": 0.60,
                "components": {"trajectory": 0.40},
            },
            best_trajectory_selection={
                "score": 0.50,
                "components": {"trajectory": 0.50},
            },
            min_delta=0.0005,
        )
    )

    assert score_improved is expected_score
    assert trajectory_improved is expected_trajectory
    assert (
        score_improved or trajectory_improved
    ) is expected_patience_improvement


def test_first_selection_establishes_both_best_tracks():
    assert workflows._dual_best_improvements(
        {
            "score": 0.60,
            "components": {"trajectory": 0.50},
        },
        best_selection=None,
        best_trajectory_selection=None,
        min_delta=0.0005,
    ) == (True, True)


def test_trajectory_best_is_reconstructed_from_old_metric_history():
    history = [
        {
            "epoch": 1,
            "val_ade": 4.35,
            "val_fde": 11.76,
            "checkpoint_uri": "s3://checkpoints/run/epoch-0001.pt",
            "checkpoint_sha256": "a" * 64,
            "validation_metric_contract": CANONICAL_METRIC_CONTRACT,
            "checkpoint_selection": {
                "policy_version": SELECTOR_POLICY_VERSION,
                "score": 0.53,
                "components": {"trajectory": 0.40},
            },
        },
        {
            "epoch": 2,
            "val_ade": 4.31,
            "val_fde": 11.72,
            "checkpoint_uri": "s3://checkpoints/run/epoch-0002.pt",
            "checkpoint_sha256": "b" * 64,
            "validation_metric_contract": CANONICAL_METRIC_CONTRACT,
            "checkpoint_selection": {
                "policy_version": SELECTOR_POLICY_VERSION,
                "score": 0.52,
                "components": {"trajectory": 0.45},
            },
        },
        {
            "epoch": 3,
            "val_ade": 5.28,
            "val_fde": 14.01,
            "checkpoint_uri": "s3://checkpoints/run/epoch-0003.pt",
            "checkpoint_sha256": "c" * 64,
            "validation_metric_contract": CANONICAL_METRIC_CONTRACT,
            "checkpoint_selection": {
                "policy_version": SELECTOR_POLICY_VERSION,
                "score": 0.50,
                "components": {"trajectory": 0.4504},
            },
        },
    ]

    best = workflows._best_trajectory_checkpoint_from_history(
        history,
        expected_policy_version=SELECTOR_POLICY_VERSION,
        min_delta=0.0005,
    )

    assert best["epoch"] == 2
    assert best["uri"].endswith("epoch-0002.pt")
    assert best["sha256"] == "b" * 64


def test_trajectory_best_reconstruction_rejects_invalid_identity_and_policy():
    entry = {
        "epoch": 1,
        "val_ade": 4.35,
        "val_fde": 11.76,
        "checkpoint_uri": "s3://checkpoints/run/epoch-0001.pt",
        "checkpoint_sha256": None,
        "validation_metric_contract": CANONICAL_METRIC_CONTRACT,
        "checkpoint_selection": {
            "policy_version": SELECTOR_POLICY_VERSION,
            "score": 0.53,
            "components": {"trajectory": 0.40},
        },
    }

    with pytest.raises(ValueError, match="immutable checkpoint identity"):
        workflows._best_trajectory_checkpoint_from_history(
            [entry],
            expected_policy_version=SELECTOR_POLICY_VERSION,
            min_delta=0.0005,
        )

    with pytest.raises(ValueError, match="policy differs"):
        workflows._best_trajectory_checkpoint_from_history(
            [{
                **entry,
                "checkpoint_sha256": "a" * 64,
                "checkpoint_selection": {
                    **entry["checkpoint_selection"],
                    "policy_version": "different-selector",
                },
            }],
            expected_policy_version=SELECTOR_POLICY_VERSION,
            min_delta=0.0005,
        )


def test_resume_transition_wiring_resets_scheduler_and_preserves_bests():
    source = inspect.getsource(workflows.train_il.task_function)

    envelope_validation = source.index(
        "validate_resume_envelope(resume_payload)"
    )
    transition_parsing = source.index(
        "if allow_resume_policy_transition:"
    )
    payload_validation = source.index(
        "validate_resume_payload("
    )
    optimization_restore = source.index(
        "_restore_resume_optimization_state("
    )
    selection_transition = source.index(
        "_transition_resume_selection_state("
    )
    active_pointer = source.index(
        "if best_checkpoint is not None:\n"
        "            update_best_pointer("
    )

    assert envelope_validation < transition_parsing < payload_validation
    assert payload_validation < optimization_restore < selection_transition
    assert selection_transition < active_pointer
    assert '"resume_plateau_state_reset": "true"' in source
    assert '"resume_optimizer_lr_preserved": "true"' in source
    assert '"resume_best_checkpoints_preserved": "true"' in source


def test_resume_transition_preserves_optimizer_lr_and_resets_plateau():
    source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = torch.optim.AdamW(
        [source_parameter],
        lr=0.01,
    )
    source_optimizer.param_groups[0]["lr"] = 0.0025
    source_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        source_optimizer,
        mode="max",
        patience=1,
    )
    source_scheduler.step(0.5)
    source_scheduler.step(0.4)
    payload = {
        "optimizer_state_dict": source_optimizer.state_dict(),
        "scheduler_state_dict": source_scheduler.state_dict(),
    }

    resumed_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    resumed_optimizer = torch.optim.AdamW(
        [resumed_parameter],
        lr=0.1,
    )
    resumed_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        resumed_optimizer,
        mode="max",
        patience=1,
    )
    state = workflows._restore_resume_optimization_state(
        resumed_optimizer,
        resumed_scheduler,
        payload,
        transition={"policy_version": "transition-v1"},
    )

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        0.0025
    )
    assert resumed_scheduler.state_dict()["best"] == -float("inf")
    assert resumed_scheduler.state_dict()["num_bad_epochs"] == 0
    assert state == {
        "optimizer_lr": [0.0025],
        "optimizer_lr_preserved": True,
        "plateau_state_restored": False,
    }


@pytest.mark.parametrize(
    ("saved_enabled", "requested_enabled", "saved_patience", "new_patience"),
    [
        (True, False, 3, 8),
        (False, True, 3, 3),
        (False, True, 8, 3),
        (False, False, 3, 3),
    ],
)
def test_resume_policy_transition_rejects_unsupported_changes(
    saved_enabled,
    requested_enabled,
    saved_patience,
    new_patience,
):
    with pytest.raises(ValueError, match="resume policy transition"):
        workflows._resume_policy_transition(
            saved_config={
                "junction_sampling": {
                    "enabled": saved_enabled,
                    "policy": None,
                },
                "early_stopping_patience": saved_patience,
            },
            requested_config={
                "junction_sampling": {
                    "enabled": requested_enabled,
                    "policy": {"version": "navigation_repeat_v1"},
                },
                "early_stopping_patience": new_patience,
            },
        )


def test_resume_record_recovers_self_digest_and_metrics(tmp_path):
    checkpoint = tmp_path / "epoch-0003.pt"
    checkpoint.write_bytes(b"trusted-checkpoint")
    payload = {
        "epoch": 3,
        "training_state": {
            "current_checkpoint_uri": (
                "s3://checkpoints/imitation-learning/run/epoch-0003.pt"
            ),
            "metric_history": [
                {"epoch": 3, "val_ade": 1.25, "val_fde": 2.5}
            ],
        },
    }

    record = workflows._resumed_checkpoint_record(
        payload, str(checkpoint)
    )

    assert record["epoch"] == 3
    assert record["ade"] == 1.25
    assert record["fde"] == 2.5
    assert record["size"] == len(b"trusted-checkpoint")
    assert record["sha256"] == hashlib.sha256(
        b"trusted-checkpoint"
    ).hexdigest()


def test_resume_record_recovers_composite_selection(tmp_path):
    checkpoint = tmp_path / "epoch-0003.pt"
    checkpoint.write_bytes(b"trusted-checkpoint")
    selection = {
        "policy_version": SELECTOR_POLICY_VERSION,
        "score": 0.75,
    }
    payload = {
        "epoch": 3,
        "training_state": {
            "current_checkpoint_uri": (
                "s3://checkpoints/imitation-learning/run/epoch-0003.pt"
            ),
            "metric_history": [{
                "epoch": 3,
                "val_ade": 1.25,
                "val_fde": 2.5,
                "validation_metric_contract": CANONICAL_METRIC_CONTRACT,
                "checkpoint_selection": selection,
            }],
        },
    }

    record = workflows._resumed_checkpoint_record(
        payload,
        str(checkpoint),
    )

    assert record["selection"] == selection
    assert record["metric_contract"] == CANONICAL_METRIC_CONTRACT


class _RegistryClient:
    def __init__(self):
        self.registered = False
        self.versions = []
        self.tags = {}

    def get_registered_model(self, name):
        if not self.registered:
            raise KeyError(name)
        return SimpleNamespace(name=name)

    def create_registered_model(self, name):
        self.registered = True
        return SimpleNamespace(name=name)

    def search_model_versions(self, query):
        return list(self.versions)

    def create_model_version(self, *, name, source, run_id):
        version = SimpleNamespace(
            version=str(len(self.versions) + 1),
            source=source,
            run_id=run_id,
        )
        self.versions.append(version)
        return version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(name, version, key)] = value


def test_registry_reuses_one_version_for_all_checkpoint_roles():
    client = _RegistryClient()
    kwargs = {
        "run_id": "run-1",
        "roles": ["final", "best_trajectory", "best"],
        "epoch": 4,
        "checkpoint_uri": "s3://checkpoints/run-1/epoch-0004.pt",
        "checkpoint_sha256": "a" * 64,
        "ade": 1.0,
        "fde": 2.0,
        "metric_contract": CANONICAL_METRIC_CONTRACT,
    }

    first = workflows._register_checkpoint_version(client, **kwargs)
    retry = workflows._register_checkpoint_version(client, **kwargs)

    assert first == retry == "1"
    assert len(client.versions) == 1
    assert client.tags[
        ("auto-e2e-driving-policy", "1", "checkpoint_role")
    ] == "best,best_trajectory,final"


def test_registry_records_composite_checkpoint_selection():
    client = _RegistryClient()
    selection = {
        "policy_version": SELECTOR_POLICY_VERSION,
        "score": 0.75,
    }

    version = workflows._register_checkpoint_version(
        client,
        run_id="run-1",
        roles=["best"],
        epoch=4,
        checkpoint_uri="s3://checkpoints/run-1/epoch-0004.pt",
        checkpoint_sha256="a" * 64,
        ade=1.0,
        fde=2.0,
        metric_contract=CANONICAL_METRIC_CONTRACT,
        selection=selection,
    )

    assert client.tags[
        ("auto-e2e-driving-policy", version, "checkpoint_selector_policy")
    ] == selection["policy_version"]
    assert client.tags[
        ("auto-e2e-driving-policy", version, "checkpoint_composite_score")
    ] == str(selection["score"])
    assert client.tags[
        ("auto-e2e-driving-policy", version, "validation_ade_3s_m")
    ] == "1.0"
    assert client.tags[
        (
            "auto-e2e-driving-policy",
            version,
            "validation_metric_target_source",
        )
    ] == "logged_xy"


def test_recovery_graph_never_calls_ingest_or_cosmos():
    static_entities = [
        getattr(node.flyte_entity, "name", "")
        for node in workflows.wf_recovered_kitscenes_full_run.nodes
    ]
    assert static_entities == [
        workflows.wf_repack_existing_kitscenes.name,
        workflows.audit_kitscenes_navigation_quality.name,
        workflows.train_il.name,
        workflows.evaluate_il_policy.name,
    ]
    audit_node = workflows.wf_recovered_kitscenes_full_run.nodes[1]
    train_node = workflows.wf_recovered_kitscenes_full_run.nodes[2]
    train_bindings = {
        binding.var: binding.binding.promise
        for binding in train_node.bindings
    }
    assert (
        train_bindings["navigation_quality_audit"].node_id
        == audit_node.id
    )

    dynamic_tree = ast.parse(
        inspect.getsource(
            workflows._map_recovered_kitscenes_artifacts.task_function
        )
    )
    referenced_names = {
        node.id for node in ast.walk(dynamic_tree)
        if isinstance(node, ast.Name)
    }
    assert "data_processing" in referenced_names
    assert "data_ingest" not in referenced_names
    assert "generate_reasoning_labels" not in referenced_names


def test_navigation_comparison_graph_reuses_one_repack():
    nodes = workflows.wf_compare_recovered_kitscenes_navigation.nodes
    assert [
        getattr(node.flyte_entity, "name", "")
        for node in nodes
    ] == [
        workflows.wf_repack_existing_kitscenes.name,
        workflows.evaluate_navigation_records.name,
        workflows.evaluate_navigation_records.name,
        workflows.compare_navigation_record_artifacts.name,
    ]
    comparison_bindings = {
        binding.var: binding.binding.promise.node_id
        for binding in nodes[3].bindings
    }
    assert comparison_bindings == {
        "conditioned_records": nodes[1].id,
        "baseline_records": nodes[2].id,
    }


def test_shared_pack_maps_bind_optional_strict_count_to_none():
    tree = ast.parse(
        inspect.getsource(workflows._map_dataset_partitions.task_function)
    )
    pack_partials = []
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "partial"
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "data_processing"
        ):
            continue
        pack_partials.append(call)

    assert len(pack_partials) == 2
    for partial in pack_partials:
        keywords = {item.arg: item.value for item in partial.keywords}
        assert isinstance(
            keywords["expected_reasoning_label_count"], ast.Constant
        )
        assert keywords["expected_reasoning_label_count"].value is None


def test_resume_load_keeps_rng_tensors_on_cpu():
    tree = ast.parse(inspect.getsource(workflows.train_il.task_function))
    loads = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        )
    ]
    resume_load = next(
        node
        for node in loads
        if node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "resume_path"
    )
    keywords = {item.arg: item.value for item in resume_load.keywords}
    assert ast.literal_eval(keywords["map_location"]) == "cpu"
    assert ast.literal_eval(keywords["weights_only"]) is False


def test_camera_bev_grid_override_changes_only_the_grid():
    """The 6 GB workaround must not silently change what the BEV covers.

    Shrinking the grid is safe only if pc_range is left alone: the BEV then
    covers the same ground area with coarser cells, and the map BEV stays
    aligned. If pc_range moved too, the model would be looking at a smaller
    patch of the world and the metric would not be comparable.
    """
    from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

    base = DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()
    resized = workflows._camera_bev_kwargs_with_grid(base, 64)

    assert resized["bev_h"] == 64
    assert resized["bev_w"] == 64
    changed = {k for k in base if base[k] != resized.get(k)}
    assert changed == {"bev_h", "bev_w"}, (
        f"resizing the grid must not touch anything else, but changed {changed}"
    )
    assert base["bev_h"] != 64, "fixture would be vacuous if the default were 64"
    # The caller must not have its own dict mutated underneath it.
    assert base["bev_h"] == DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()["bev_h"]


@pytest.mark.parametrize("bad", [0, -1, -256])
def test_camera_bev_grid_override_rejects_non_positive(bad):
    from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

    base = DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()
    with pytest.raises(ValueError, match="camera_bev_size must be positive"):
        workflows._camera_bev_kwargs_with_grid(base, bad)


def test_camera_bev_grid_defaults_to_the_geometry():
    """With the parameter unset the run must be byte-identical to before."""
    source = inspect.getsource(workflows.train_il.task_function)
    assert "camera_bev_size: Optional[int] = None" in source
    assert "if camera_bev_size is not None:" in source, (
        "the override must be opt-in; an unconditional call would change the "
        "default grid for every existing run"
    )
