"""Training lifecycle and recovered-workflow contracts."""

from __future__ import annotations

import ast
import gc
import hashlib
import inspect
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

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def reset_visual_history(self):
        self.reset_count += 1

    def __call__(self, visual, *args, **kwargs):
        return torch.zeros((visual.shape[0], 128), dtype=torch.float32)


def _validation_batch(sample_uids):
    batch_size = len(sample_uids)
    ego = torch.zeros((batch_size, 256), dtype=torch.float32)
    ego[:, -4] = 2.0
    return {
        "sample_uid": list(sample_uids),
        "visual_tiles": torch.zeros(
            (batch_size, 7, 3, 2, 2), dtype=torch.float32
        ),
        "map_input": torch.zeros(
            (batch_size, 3, 2, 2), dtype=torch.float32
        ),
        "egomotion_history": ego,
        "visual_history": torch.zeros(
            (batch_size, 896), dtype=torch.float32
        ),
        "trajectory_target": torch.zeros(
            (batch_size, 128), dtype=torch.float32
        ),
    }


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
        "sample_count": 2,
        "sample_uid_digest": expected_digest,
    }
    assert model.training is True
    assert model.reset_count == 2


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
        workflows.train_il.name,
        workflows.evaluate_il_policy.name,
    ]

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
