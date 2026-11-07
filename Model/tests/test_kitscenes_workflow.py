"""Flyte wiring tests for the KITScenes scene fan-out."""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path

import pytest

pytest.importorskip("flytekit")

from flytekit import map_task
from flytekit.configuration import ImageConfig, SerializationSettings
from flytekit.core.context_manager import FlyteContextManager

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
        "source_split",
        "data_role",
    }


def test_direct_test_pack_keeps_raw_data_in_the_same_task(monkeypatch):
    raw_data = object()
    packed_data = object()
    calls = {}

    def fake_ingest(**kwargs):
        calls["ingest"] = kwargs
        return raw_data

    def fake_process(**kwargs):
        calls["process"] = kwargs
        return packed_data

    monkeypatch.setattr(
        workflows.data_ingest,
        "_task_function",
        fake_ingest,
    )
    monkeypatch.setattr(
        workflows.data_processing,
        "_task_function",
        fake_process,
    )

    result = (
        workflows.ingest_and_process_kitscenes_test_partition
        .task_function(
            group_ids=["scene-a"],
            source_revision=workflows.KITSCENES_SOURCE_REVISION,
            dataset_version=workflows.KITSCENES_TEST_DATASET_VERSION,
            image_size=workflows.REACTIVE_CAMERA_IMAGE_SIZE,
        )
    )

    assert result is packed_data
    assert calls["ingest"] == {
        "dataset": workflows.Dataset.KITSCENES,
        "source_revision": workflows.KITSCENES_SOURCE_REVISION,
        "episodes": 0,
        "group_ids": ["scene-a"],
        "source_split": "test",
        "data_role": "benchmark",
    }
    assert calls["process"] == {
        "raw_data": raw_data,
        "dataset": workflows.Dataset.KITSCENES,
        "source_revision": workflows.KITSCENES_SOURCE_REVISION,
        "dataset_version": workflows.KITSCENES_TEST_DATASET_VERSION,
        "hz": 10,
        "image_size": workflows.REACTIVE_CAMERA_IMAGE_SIZE,
        "episodes": 0,
        "world_model": False,
        "reasoning_labels": None,
        "group_ids": ["scene-a"],
        "sample_limit": 0,
        "source_split": "test",
        "data_role": "benchmark",
    }


def test_local_flyte_directory_download_is_a_noop(tmp_path):
    directory = workflows.FlyteDirectory(str(tmp_path))

    assert workflows._download_flyte_directory(directory) == str(tmp_path)
    assert directory.remote_source is None


def test_direct_test_pack_rejects_non_singleton_partition():
    with pytest.raises(ValueError, match="exactly one scene"):
        (
            workflows.ingest_and_process_kitscenes_test_partition
            .task_function(
                group_ids=["scene-a", "scene-b"],
                source_revision=workflows.KITSCENES_SOURCE_REVISION,
                dataset_version=workflows.KITSCENES_TEST_DATASET_VERSION,
                image_size=workflows.REACTIVE_CAMERA_IMAGE_SIZE,
            )
        )


def test_direct_test_pack_map_binds_only_group_ids():
    context = FlyteContextManager.current_context()
    with FlyteContextManager.with_context(
        context.with_new_compilation_state()
    ) as compilation_context:
        workflows._map_kitscenes_test_partitions.task_function(
            partitions=[["scene-a"]],
            dataset_version=workflows.KITSCENES_TEST_DATASET_VERSION,
            image_size=workflows.REACTIVE_CAMERA_IMAGE_SIZE,
            ingest_concurrency=8,
            pack_concurrency=5,
        )
        mapped_node = compilation_context.compilation_state.nodes[-1]

    assert mapped_node.flyte_entity.bound_inputs == {
        "dataset_version",
        "image_size",
        "source_revision",
    }
    assert mapped_node.flyte_entity.concurrency == 5
    assert {
        binding.var for binding in mapped_node.bindings
    } == {
        "dataset_version",
        "group_ids",
        "image_size",
        "source_revision",
    }


def test_direct_test_pack_map_rejects_wrong_camera_contract():
    with pytest.raises(ValueError, match="512px camera contract"):
        workflows._map_kitscenes_test_partitions.task_function(
            partitions=[["scene-a"]],
            dataset_version=workflows.KITSCENES_TEST_DATASET_VERSION,
            image_size=256,
            ingest_concurrency=1,
            pack_concurrency=1,
        )


def test_kitscenes_data_roles_keep_benchmark_splits_out_of_training():
    workflows._validate_kitscenes_data_role(
        data_role="training",
        source_split="train",
    )
    workflows._validate_kitscenes_data_role(
        data_role="benchmark",
        source_split="val",
    )
    workflows._validate_kitscenes_data_role(
        data_role="benchmark",
        source_split="overlap_train_val",
    )
    workflows._validate_kitscenes_data_role(
        data_role="benchmark",
        source_split="test",
    )
    with pytest.raises(ValueError, match="training accepts only"):
        workflows._validate_kitscenes_data_role(
            data_role="training",
            source_split="val",
        )
    with pytest.raises(
        ValueError,
        match="labeled evaluation preparation accepts",
    ):
        workflows._validate_kitscenes_data_role(
            data_role="benchmark",
            source_split="train",
        )
    with pytest.raises(ValueError, match="unsupported KITScenes data_role"):
        workflows._validate_kitscenes_data_role(
            data_role="evaluation",
            source_split="test_e2e",
        )


def test_benchmark_inventory_accepts_only_official_eval_splits(monkeypatch):
    inventory = InventoryResolution(
        split="val",
        expected_scene_ids=("scene-a",),
        selected_scene_ids=("scene-a",),
        missing_scene_ids=(),
        total_size_bytes=10,
        source_revision=workflows.KITSCENES_SOURCE_REVISION,
    )
    archives = {
        "scene-a": SceneArchive(
            scene_id="scene-a",
            split="val",
            filename="data/val/scene-a.tar",
            sha256="a" * 64,
            size_bytes=10,
        )
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
        max_partitions=200,
        max_missing_scenes=0,
        split="val",
        data_role="benchmark",
    )

    assert partitions == [["scene-a"]]
    with pytest.raises(ValueError, match="training accepts only"):
        workflows.plan_fanout_partitions.task_function(
            dataset=workflows.Dataset.KITSCENES,
            source_revision=workflows.KITSCENES_SOURCE_REVISION,
            episodes=0,
            start_ep=-1,
            end_ep=-1,
            partition_size=1,
            max_partitions=200,
            max_missing_scenes=0,
            split="val",
            data_role="training",
        )


def test_dataset_dynamic_propagates_the_pinned_data_prep_image():
    assert workflows._map_dataset_partitions.container_image == (
        workflows.DATA_PREP_IMAGE
    )
    assert workflows._map_dataset_partitions.environment == {
        "AUTO_E2E_DATA_PREP_IMAGE": workflows.DATA_PREP_IMAGE,
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
        workflows.audit_kitscenes_benchmark_inventory,
        workflows.data_ingest,
        workflows.generate_reasoning_labels,
        workflows.data_processing,
        workflows.audit_kitscenes_navigation_quality,
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


def test_data_processing_uses_module_scoped_path():
    tree = ast.parse(Path(workflows.__file__).read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "data_processing"
    )
    local_path_imports = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        and node.module == "pathlib"
        and any(alias.name == "Path" for alias in node.names)
    ]
    module_path_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "pathlib"
        and any(alias.name == "Path" for alias in node.names)
    ]

    assert not local_path_imports
    assert len(module_path_imports) == 1


def test_navigation_contracts_invalidate_old_pack_caches():
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
    assert workflows.INGEST_CACHE_VERSION == "ingest-v3"
    assert workflows.LABEL_CACHE_VERSION == "label-v3-v1-v2"
    assert (
        workflows.PACK_CACHE_VERSION
        == "pack-v3-v1-v13-v7-camera512"
    )


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
        (workflows.Dataset.KITSCENES, 2, 1),
        (workflows.Dataset.KITSCENES, 10_000, 1),
        (workflows.Dataset.L2D, 10_000, 12),
    ),
)
def test_row_decode_workers_bound_dataset_memory(
    dataset, row_count, expected
):
    assert workflows._row_decode_worker_count(dataset, row_count) == expected


def test_legacy_l2d_pack_uses_dataset_decoder_limit():
    source = inspect.getsource(workflows.data_processing.task_function)

    assert "pack_workers = _row_decode_worker_count(" in source
    assert "max_workers_cap = 16" not in source


def test_row_decode_initializer_matches_worker_abi():
    source = Path(workflows.__file__).read_text(encoding="utf-8")

    assert (
        "inspect.signature(parallel_pack.init_row_worker).bind(*row_init)"
        in source
    )
    assert (
        "            source_split,\n"
        "            source_revision,\n"
        "            benchmark_protocol,\n"
        "            mapless_kitscenes_test,\n"
        "        )"
    ) in source


@pytest.mark.parametrize(
    (
        "dataset",
        "has_samples",
        "world_model",
        "reactive_targets",
        "expected",
    ),
    (
        (workflows.Dataset.L2D, True, False, True, True),
        (workflows.Dataset.L2D, True, True, False, True),
        # L2D always needs parent assembly for its OSM navigation.
        (workflows.Dataset.L2D, True, False, False, True),
        (workflows.Dataset.KITSCENES, True, False, False, True),
        (
            workflows.Dataset.NVIDIA_PHYSICAL_AI,
            True,
            True,
            True,
            False,
        ),
        (workflows.Dataset.L2D, False, True, True, False),
    ),
)
def test_parent_assembly_pack_selection(
    dataset,
    has_samples,
    world_model,
    reactive_targets,
    expected,
):
    assert (
        workflows._use_parent_assembly_pack(
            dataset,
            has_samples=has_samples,
            world_model=world_model,
            reactive_targets=reactive_targets,
        )
        is expected
    )


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


@pytest.mark.parametrize(
    "buildspec_name",
    (
        "buildspec-register.yml",
    ),
)
def test_remote_registration_buildspecs_pin_runtime_contracts(buildspec_name):
    buildspec = (_REPO_ROOT / "Platform" / buildspec_name).read_text()

    assert "flytekit==1.16.24" in buildspec
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


@pytest.mark.parametrize(
    "buildspec_name",
    (
        "buildspec-register.yml",
    ),
)
def test_registration_buildspecs_disable_fast_source_copy(buildspec_name):
    buildspec = (_REPO_ROOT / "Platform" / buildspec_name).read_text()

    assert "--copy none" in buildspec


def test_all_flyte_runtime_installations_pin_the_shared_sdk_version():
    candidates = tuple((_REPO_ROOT / "Platform").glob("buildspec*.yml")) + tuple(
        (_REPO_ROOT / "Platform" / "docker").glob("*/Dockerfile")
    )
    runtime_files = []
    for path in candidates:
        # Import checks (`python -c "import flytekit"`) are not installations.
        source = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if "python -c" not in line
        )
        if "flytekit" in source:
            runtime_files.append(path)
            assert "flytekit==1.16.24" in source, path
            assert "flytekit==1.14.9" not in source, path
    assert runtime_files


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
