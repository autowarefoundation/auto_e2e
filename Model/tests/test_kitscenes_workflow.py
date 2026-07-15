"""Flyte wiring tests for the KITScenes scene fan-out."""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

pytest.importorskip("flytekit")

from flytekit import map_task
from flytekit.configuration import ImageConfig, SerializationSettings

from Platform.pipelines import workflows
from data_parsing.kit_scenes.source import InventoryResolution, SceneArchive


_REPO_ROOT = Path(__file__).resolve().parents[2]


class _ReasoningSelectionDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def frame_index(self, sample_index):
        return self.samples[sample_index][1]

    def split_group_uid(self, sample_index):
        return self.samples[sample_index][0]


def test_inventory_preflight_emits_one_scene_per_partition(monkeypatch):
    scene_ids = ("scene-a", "scene-c")
    inventory = InventoryResolution(
        split="train",
        expected_scene_ids=("scene-a", "scene-b", "scene-c"),
        selected_scene_ids=scene_ids,
        missing_scene_ids=("scene-b",),
        total_size_bytes=20,
        source_revision=workflows.KITSCENES_SOURCE_REVISION,
    )
    archives = {
        scene_id: SceneArchive(
            scene_id=scene_id,
            split="train",
            filename=f"data/train/{scene_id}.tar",
            sha256="a" * 64,
            size_bytes=10,
        )
        for scene_id in scene_ids
    }
    monkeypatch.setattr(
        "data_parsing.kit_scenes.source.fetch_archive_manifest",
        lambda *args, **kwargs: archives,
    )
    monkeypatch.setattr(
        "data_parsing.kit_scenes.source.resolve_inventory",
        lambda *args, **kwargs: inventory,
    )

    partitions = workflows.plan_fanout_partitions.task_function(
        dataset=workflows.Dataset.KITSCENES,
        source_revision=workflows.KITSCENES_SOURCE_REVISION,
        episodes=0,
        start_ep=-1,
        end_ep=-1,
        partition_size=1,
        max_partitions=600,
        max_missing_scenes=1,
        split="train",
    )

    assert partitions == [["scene-a"], ["scene-c"]]


def test_ingest_map_binds_scalars_and_maps_only_group_ids():
    mapped = map_task(
        functools.partial(
            workflows.data_ingest,
            dataset=workflows.Dataset.KITSCENES,
            source_revision=workflows.KITSCENES_SOURCE_REVISION,
            episodes=0,
        ),
        concurrency=60,
    )

    assert mapped.bound_inputs == {"dataset", "source_revision", "episodes"}
    assert mapped.concurrency == 60
    assert set(mapped.python_interface.inputs) == {
        "dataset",
        "source_revision",
        "episodes",
        "group_ids",
    }


def test_data_prep_tasks_serialize_karpenter_disruption_protection():
    settings = SerializationSettings(
        image_config=ImageConfig.auto_default_image(),
        project="auto-e2e",
        domain="development",
        version="test",
    )
    expected = {"karpenter.sh/do-not-disrupt": "true"}

    for task in (
        workflows.data_ingest,
        workflows.generate_reasoning_labels,
        workflows.data_processing,
    ):
        assert task.get_k8s_pod(settings).metadata.annotations == expected

    mapped = map_task(
        functools.partial(
            workflows.data_ingest,
            dataset=workflows.Dataset.KITSCENES,
            source_revision=workflows.KITSCENES_SOURCE_REVISION,
            episodes=0,
        ),
        concurrency=60,
    )
    assert mapped.get_k8s_pod(settings).metadata.annotations == expected


def test_large_shm_tasks_serialize_karpenter_disruption_protection():
    settings = SerializationSettings(
        image_config=ImageConfig.auto_default_image(),
        project="auto-e2e",
        domain="development",
        version="test",
    )
    expected = {"karpenter.sh/do-not-disrupt": "true"}

    for task in (
        workflows.train_il,
        workflows.evaluate_il_policy,
        workflows.evaluate_rl_policy,
    ):
        assert task.get_k8s_pod(settings).metadata.annotations == expected


def test_contract_version_import_is_fail_closed():
    tree = ast.parse(Path(workflows.__file__).read_text())
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "data_processing.contract_versions"
    ]

    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        "GEOMETRY_VERSION",
        "PARSER_VERSION",
        "REASONING_LABEL_POLICY_VERSION",
        "SHARD_SCHEMA_VERSION",
        "UID_SCHEMA_VERSION",
    }


def test_current_contracts_reuse_deployed_cache_versions():
    assert workflows._cache_versions_for_contracts(
        uid="v1",
        parser="v2",
        shard="v4",
        geometry="v2",
        label_policy="v2",
    ) == {
        "ingest": "ingest-v1",
        "label": "label-v1-v1-v1",
        "pack": "pack-v2-v1-v4-v2",
    }
    assert workflows.INGEST_CACHE_VERSION == "ingest-v1"
    assert workflows.LABEL_CACHE_VERSION == "label-v1-v1-v1"
    assert workflows.PACK_CACHE_VERSION == "pack-v2-v1-v4-v2"


def test_old_geometry_pack_cache_is_not_aliased():
    versions = workflows._cache_versions_for_contracts(
        uid="v1",
        parser="v2",
        shard="v4",
        geometry="v1",
        label_policy="v2",
    )

    assert versions["pack"] == "pack-v2-v1-v4-v1"


@pytest.mark.parametrize(
    ("dataset", "row_count", "expected"),
    (
        (workflows.Dataset.KITSCENES, 1, 1),
        (workflows.Dataset.KITSCENES, 2, 2),
        (workflows.Dataset.KITSCENES, 10_000, 2),
        (workflows.Dataset.L2D, 10_000, 16),
    ),
)
def test_row_decode_workers_bound_kitscenes_memory(
    dataset, row_count, expected
):
    assert workflows._row_decode_worker_count(dataset, row_count) == expected


def test_future_contracts_get_new_cache_versions():
    assert workflows._cache_versions_for_contracts(
        uid="v1",
        parser="v3",
        shard="v5",
        geometry="v2",
        label_policy="v3",
    ) == {
        "ingest": "ingest-v3",
        "label": "label-v3-v1-v3",
        "pack": "pack-v3-v1-v5-v2",
    }


def test_training_num_views_come_from_consistent_manifests():
    manifests = {
        "kit-a": {"dataset": "kitscenes", "num_views": 6},
        "kit-b": {"dataset": "kitscenes", "num_views": 6},
        "nv-a": {"dataset": "nvidia", "num_views": 7},
    }

    assert workflows._training_num_views_from_manifests(
        manifests, list(manifests)
    ) == 7


def test_training_rejects_inconsistent_partition_num_views():
    manifests = {
        "kit-a": {"dataset": "kitscenes", "num_views": 6},
        "kit-b": {"dataset": "kitscenes", "num_views": 7},
    }

    with pytest.raises(ValueError, match="inconsistent num_views"):
        workflows._training_num_views_from_manifests(
            manifests, list(manifests)
        )


def test_training_rejects_invalid_manifest_num_views():
    manifests = {"kit-a": {"dataset": "kitscenes", "num_views": 0}}

    with pytest.raises(ValueError, match="invalid num_views"):
        workflows._training_num_views_from_manifests(
            manifests, list(manifests)
        )


def test_loader_wiring_avoids_training_peek_and_bounds_eval_prefetch():
    tree = ast.parse(Path(workflows.__file__).read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    train = functions["train_il"]
    merged_peeks = [
        call
        for call in ast.walk(train)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "next"
        and call.args
        and isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Name)
        and call.args[0].func.id == "iter"
        and call.args[0].args
        and isinstance(call.args[0].args[0], ast.Name)
        and call.args[0].args[0].id == "merged"
    ]
    assert not merged_peeks

    evaluation = functions["_run_evaluation"]
    loader_call = next(
        call
        for call in ast.walk(evaluation)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "make_multi_dataset_loader"
    )
    keywords = {keyword.arg: keyword.value for keyword in loader_call.keywords}
    assert ast.literal_eval(keywords["max_active_loaders"]) == 1
    assert ast.literal_eval(keywords["prefetch_factor"]) == 1


def test_dataset_dynamic_propagates_the_pinned_data_prep_image():
    assert workflows._map_dataset_partitions.container_image == (
        workflows.DATA_PREP_IMAGE
    )
    assert workflows._map_dataset_partitions.environment == {
        "AUTO_E2E_DATA_PREP_IMAGE": workflows.DATA_PREP_IMAGE,
    }


@pytest.mark.parametrize(
    "buildspec_name",
    (
        "buildspec-register.yml",
        "buildspec-launch-fullrun.yml",
        "buildspec-launch-recovery.yml",
    ),
)
def test_remote_registration_buildspecs_pin_runtime_contracts(buildspec_name):
    buildspec = (_REPO_ROOT / "Platform" / buildspec_name).read_text()

    assert "flytekit==1.14.9" in buildspec
    assert (
        'export PYTHONPATH="${CODEBUILD_SRC_DIR}/Model:'
        '${CODEBUILD_SRC_DIR}:${PYTHONPATH:-}"'
    ) in buildspec
    assert "aws ecr batch-get-image" in buildspec
    assert "aws ecr describe-images" not in buildspec
    for variable in (
        "AUTO_E2E_TRAINING_IMAGE",
        "AUTO_E2E_EVAL_IMAGE",
        "AUTO_E2E_OFFLINE_RL_IMAGE",
        "AUTO_E2E_DATA_PREP_IMAGE",
    ):
        assert variable in buildspec
    assert '--image "${AUTO_E2E_TRAINING_IMAGE}"' in buildspec


def test_recovery_launcher_requires_audited_artifacts_and_skips_source_stages():
    buildspec = (
        _REPO_ROOT / "Platform" / "buildspec-launch-recovery.yml"
    ).read_text()

    assert "shell: bash" in buildspec
    assert 'test -n "${ARTIFACT_SET_SHA256}"' in buildspec
    assert "--recovery_manifest" in buildspec
    assert "--artifact_set_sha256" in buildspec
    assert "wf_recovered_kitscenes_full_run" in buildspec
    assert "wf_sharded_full_run" not in buildspec
    assert "--reasoning_teacher" not in buildspec


def test_reasoning_selection_bootstraps_short_scenes():
    dataset = _ReasoningSelectionDataset([
        ("scene-a", 64),
        ("scene-a", 65),
        ("scene-b", 64),
        ("scene-b", 70),
        ("scene-b", 71),
    ])

    assert workflows._reasoning_label_indices(dataset, 10) == [0, 2, 3]
    assert workflows._reasoning_label_indices(dataset, 1) == list(range(5))


def test_shard_selection_skips_empty_partitions(tmp_path):
    class _Shard:
        def __init__(self, path):
            self.path = path

        def download(self):
            return str(self.path)

    shards = []
    for name, total_samples in (("empty", 0), ("nonempty", 2)):
        shard_dir = tmp_path / name
        shard_dir.mkdir()
        (shard_dir / "manifest.json").write_text(
            '{"dataset":"KIT-MRT/KITScenes-Multimodal",'
            f'"total_samples":{total_samples}}}'
        )
        shards.append(_Shard(shard_dir))

    selected = workflows._select_shard_dirs(
        shards, workflows.Dataset.KITSCENES
    )

    assert selected == [str(tmp_path / "nonempty")]
