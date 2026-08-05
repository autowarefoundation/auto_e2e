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

from Platform.pipelines import workflows


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
        "evaluation_steps": 64,
        "sample_count": 2,
        "sample_uid_digest": expected_digest,
    }
    assert model.training is True
    assert model.reset_count == 2


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


def test_kitscenes_epoch_evaluation_preserves_auto_e2e_horizon():
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

    assert metrics["ade"] > 0.0
    assert metrics["fde"] > 0.0
    assert metrics["evaluation_steps"] == 64
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


def test_registry_reuses_one_version_when_best_is_final():
    client = _RegistryClient()
    kwargs = {
        "run_id": "run-1",
        "roles": ["final", "best"],
        "epoch": 4,
        "checkpoint_uri": "s3://checkpoints/run-1/epoch-0004.pt",
        "checkpoint_sha256": "a" * 64,
        "ade": 1.0,
        "fde": 2.0,
    }

    first = workflows._register_checkpoint_version(client, **kwargs)
    retry = workflows._register_checkpoint_version(client, **kwargs)

    assert first == retry == "1"
    assert len(client.versions) == 1
    assert client.tags[
        ("auto-e2e-driving-policy", "1", "checkpoint_role")
    ] == "best,final"


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


class _ReasoningMetricModel(_MetricModel):
    """`_MetricModel` with a bypassable reasoning head. The trajectory picks up a
    fixed bump while ``Reactive_E2E.ReasoningHead`` is present, so the intervention
    (which sets the head to None) moves the trajectory by exactly that bump — the
    quantity the gate metric must report."""

    _BUMP = 1e-3

    def __init__(self):
        super().__init__()
        self.Reactive_E2E = SimpleNamespace(ReasoningHead=object())

    def __call__(self, visual, *args, **kwargs):
        out = super().__call__(visual, *args, **kwargs)
        if self.Reactive_E2E.ReasoningHead is not None:
            out = out + self._BUMP
        return out


def test_intervention_delta_reported_when_opted_in():
    model = _ReasoningMetricModel()
    loader = [
        (_validation_batch(["sample-b", "sample-a"]), None, "pseudo")
    ]

    metrics = workflows._evaluate_open_loop(
        model, loader, torch.device("cpu"), report_intervention=True
    )

    assert "reasoning_intervention_delta" in metrics
    # a fixed initial_noise is threaded into both runs, so the delta reflects only
    # the reasoning bump (1e-3 across 128 signals): ||1e-3||_2 = 1e-3 * sqrt(128).
    assert metrics["reasoning_intervention_delta"] == pytest.approx(
        _ReasoningMetricModel._BUMP * 128 ** 0.5, abs=1e-4
    )
    assert model.training is True  # eval mode restored after the intervention


def test_no_intervention_delta_without_reasoning_head():
    model = _MetricModel()  # no Reactive_E2E.ReasoningHead
    loader = [(_validation_batch(["sample-a"]), None, "pseudo")]

    metrics = workflows._evaluate_open_loop(
        model, loader, torch.device("cpu"), report_intervention=True
    )

    assert "reasoning_intervention_delta" not in metrics


def test_no_intervention_delta_by_default():
    model = _ReasoningMetricModel()
    loader = [(_validation_batch(["sample-a"]), None, "pseudo")]

    # report_intervention defaults to False -> no cost, no metric (training path).
    metrics = workflows._evaluate_open_loop(
        model, loader, torch.device("cpu")
    )

    assert "reasoning_intervention_delta" not in metrics
