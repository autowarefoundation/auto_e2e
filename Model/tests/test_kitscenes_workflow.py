"""Flyte wiring tests for the KITScenes scene fan-out."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import io
import json
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("flytekit")

from flytekit import map_task
from flytekit.configuration import ImageConfig, SerializationSettings

from Platform.pipelines import workflows
from data_parsing.kit_scenes.source import InventoryResolution, SceneArchive
from data_parsing.kit_scenes.temporal_contract import (
    kitscenes_temporal_contract,
)


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


def test_kitscenes_data_roles_keep_benchmark_splits_out_of_training():
    assert (
        workflows.KITSCENES_BENCHMARK_DATASET_VERSION
        == "v3.3-benchmark-v3"
    )
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

    with pytest.raises(ValueError, match="training accepts only"):
        workflows._validate_kitscenes_data_role(
            data_role="training",
            source_split="val",
        )
    with pytest.raises(ValueError, match="benchmark preparation accepts"):
        workflows._validate_kitscenes_data_role(
            data_role="benchmark",
            source_split="train",
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


def test_benchmark_manifest_task_scans_only_packed_metadata(tmp_path):
    class _Shard:
        def __init__(self, path):
            self.path = path
            self.remote_source = f"s3://benchmark/{path.name}"

        def download(self):
            return str(self.path)

        def __str__(self):
            return self.remote_source

    def build_shard(
        name,
        source_split,
        scene_id,
        *,
        empty=False,
        num_views=None,
        malformed_empty=False,
    ):
        shard_dir = tmp_path / name
        shard_dir.mkdir()
        tar_name = "shard-000000.tar"
        if not empty:
            with tarfile.open(shard_dir / tar_name, "w") as archive:
                for index in range(100):
                    frame_index = 64 + index * 90
                    sample_uid = (
                        f"kitscenes-v1-{scene_id}-f{frame_index:06d}"
                    )
                    payload = json.dumps({
                        "frame_idx": frame_index,
                        "sample_uid": sample_uid,
                        "split_group_uid": f"kitscenes-{scene_id}",
                    }).encode("ascii")
                    info = tarfile.TarInfo(f"{sample_uid}.meta.json")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        shard_names = [] if empty else [tar_name]
        shard_count = 1 if malformed_empty else len(shard_names)
        shard_sample_counts = (
            {tar_name: 1}
            if malformed_empty
            else ({tar_name: 100} if not empty else {})
        )
        manifest = {
            "data_role": "benchmark",
            "dataset": workflows.Dataset.KITSCENES.value,
            "dataset_version": (
                workflows.KITSCENES_BENCHMARK_DATASET_VERSION
            ),
            "has_gps": not empty,
            "has_map": not empty,
            "has_navigation": not empty,
            "hz": 10,
            "num_views": (
                num_views
                if num_views is not None
                else (0 if empty else 6)
            ),
            "partition_id": name,
            "shard_names": shard_names,
            "shard_sample_counts": shard_sample_counts,
            "shards": shard_count,
            "source_revision": workflows.KITSCENES_SOURCE_REVISION,
            "source_split": source_split,
            "temporal_sampling": kitscenes_temporal_contract(
                benchmark_protocol=True,
            ),
            "total_samples": 0 if empty else 100,
        }
        (shard_dir / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="ascii",
        )
        return _Shard(shard_dir)

    val = build_shard(
        "val-scene",
        "val",
        "01234567-89ab-cdef-0123-456789abcdef",
    )
    overlap = build_shard(
        "overlap-scene",
        "overlap_train_val",
        "fedcba98-7654-3210-fedc-ba9876543210",
    )
    empty_val = build_shard(
        "empty-val-scene",
        "val",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        empty=True,
    )

    result = (
        workflows.create_kitscenes_paper_approximation_manifest.task_function(
            val_shards=[val, empty_val],
            overlap_shards=[overlap],
            release_id="test-paper-approx-v1",
        )
    )
    manifest_path = Path(result.manifest.path)
    payload = json.loads(manifest_path.read_text(encoding="ascii"))

    assert payload["protocol_status"] == "paper_protocol_approximation"
    assert payload["sample_count"] == 200
    assert payload["selection"]["candidate_count"] == 200
    assert payload["selection"]["metric_or_target_values_read"] is False
    assert payload["selection"]["packed_future_steps"] == 64
    assert payload["selection"]["packed_history_steps"] == 64
    assert payload["selection"]["sampling_future_steps"] == 50
    assert payload["selection"]["sampling_history_steps"] == 40
    assert payload["selection"]["empty_partition_count"] == 1
    assert payload["selection"]["empty_partition_count_by_split"] == {
        "overlap-train-val": 0,
        "val": 1,
    }
    assert payload["selection"]["empty_partition_ids_by_split"] == {
        "overlap-train-val": [],
        "val": ["empty-val-scene"],
    }
    assert payload["selection"]["selected_count_by_split"] == {
        "overlap-train-val": 100,
        "val": 100,
    }
    assert {
        source["partition_id"]: source["empty"]
        for source in payload["packed_sources"]["val"]
    } == {
        "empty-val-scene": True,
        "val-scene": False,
    }
    assert result.manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_benchmark_manifest_rejects_nonempty_zero_view_partition(tmp_path):
    class _Shard:
        def __init__(self, path):
            self.path = path
            self.remote_source = f"s3://benchmark/{path.name}"

        def download(self):
            return str(self.path)

    shard_dir = tmp_path / "nonempty-zero-view"
    shard_dir.mkdir()
    tar_name = "shard-000000.tar"
    with tarfile.open(shard_dir / tar_name, "w") as archive:
        payload = json.dumps({
            "frame_idx": 64,
            "sample_uid": (
                "kitscenes-v1-01234567-89ab-cdef-0123-456789abcdef-"
                "f000064"
            ),
            "split_group_uid": (
                "kitscenes-01234567-89ab-cdef-0123-456789abcdef"
            ),
        }).encode("ascii")
        info = tarfile.TarInfo("sample.meta.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    (shard_dir / "manifest.json").write_text(
        json.dumps({
            "data_role": "benchmark",
            "dataset": workflows.Dataset.KITSCENES.value,
            "dataset_version": (
                workflows.KITSCENES_BENCHMARK_DATASET_VERSION
            ),
            "has_gps": True,
            "has_map": True,
            "has_navigation": True,
            "hz": 10,
            "num_views": 0,
            "partition_id": "nonempty-zero-view",
            "shard_names": [tar_name],
            "shard_sample_counts": {tar_name: 1},
            "shards": 1,
            "source_revision": workflows.KITSCENES_SOURCE_REVISION,
            "source_split": "val",
            "temporal_sampling": kitscenes_temporal_contract(
                benchmark_protocol=True,
            ),
            "total_samples": 1,
        }),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="num_views=0, expected=6"):
        workflows.create_kitscenes_paper_approximation_manifest.task_function(
            val_shards=[_Shard(shard_dir)],
            overlap_shards=[_Shard(shard_dir)],
        )


def test_benchmark_manifest_rejects_malformed_empty_partition(tmp_path):
    class _Shard:
        def __init__(self, path):
            self.path = path
            self.remote_source = f"s3://benchmark/{path.name}"

        def download(self):
            return str(self.path)

    shard_dir = tmp_path / "malformed-empty"
    shard_dir.mkdir()
    (shard_dir / "manifest.json").write_text(
        json.dumps({
            "data_role": "benchmark",
            "dataset": workflows.Dataset.KITSCENES.value,
            "dataset_version": (
                workflows.KITSCENES_BENCHMARK_DATASET_VERSION
            ),
            "has_gps": False,
            "has_map": False,
            "has_navigation": False,
            "hz": 10,
            "num_views": 0,
            "partition_id": "malformed-empty",
            "shard_names": [],
            "shard_sample_counts": {"ghost.tar": 1},
            "shards": 1,
            "source_revision": workflows.KITSCENES_SOURCE_REVISION,
            "source_split": "val",
            "temporal_sampling": kitscenes_temporal_contract(
                benchmark_protocol=True,
            ),
            "total_samples": 0,
        }),
        encoding="ascii",
    )

    with pytest.raises(
        ValueError,
        match="empty benchmark partition differs",
    ):
        workflows.create_kitscenes_paper_approximation_manifest.task_function(
            val_shards=[_Shard(shard_dir)],
            overlap_shards=[_Shard(shard_dir)],
        )


def test_dataset_dynamic_propagates_the_pinned_data_prep_image():
    assert workflows._map_dataset_partitions.container_image == (
        workflows.DATA_PREP_IMAGE
    )
    assert workflows._map_dataset_partitions.environment == {
        "AUTO_E2E_DATA_PREP_IMAGE": workflows.DATA_PREP_IMAGE,
    }


def test_full_run_overlay_workflow_wires_exact_model_lineage():
    resolver, publisher = workflows.wf_publish_full_run_overlays.nodes
    assert set(workflows.wf_publish_full_run_overlays.python_interface.outputs) == {
        "overlay_result",
        "manifest_key",
        "manifest_sha256",
    }
    assert resolver.flyte_entity.name == (
        "Platform.pipelines.overlay_tasks.resolve_overlay_model_version"
    )

    resolver_bindings = {
        binding.var: binding.binding.promise
        for binding in resolver.bindings
    }
    assert resolver_bindings["train_execution_id"].var == (
        "full_run_execution_id"
    )

    publisher_bindings = {
        binding.var: binding.binding.promise
        for binding in publisher.bindings
    }
    assert publisher_bindings["model_version"].node_id == resolver.id
    assert publisher_bindings["expected_train_execution_id"].var == (
        "full_run_execution_id"
    )
    assert publisher_bindings["shards"].var == "shards"


def test_selected_checkpoint_overlay_workflow_wires_exact_epoch_lineage():
    registrar, publisher = (
        workflows.wf_publish_selected_checkpoint_overlays.nodes
    )
    interface = (
        workflows.wf_publish_selected_checkpoint_overlays.python_interface
    )
    assert set(interface.outputs) == {
        "overlay_result",
        "manifest_key",
        "manifest_sha256",
    }
    assert registrar.flyte_entity.name == (
        "Platform.pipelines.overlay_tasks.register_selected_overlay_checkpoint"
    )

    registrar_bindings = {
        binding.var: binding.binding.promise
        for binding in registrar.bindings
    }
    assert registrar_bindings["run_id"].var == "mlflow_run_id"
    assert registrar_bindings["checkpoint_uri"].var == "checkpoint_uri"
    assert registrar_bindings["checkpoint_sha256"].var == (
        "checkpoint_sha256"
    )
    assert registrar_bindings["checkpoint_epoch"].var == "checkpoint_epoch"
    assert registrar_bindings["train_execution_id"].var == (
        "full_run_execution_id"
    )

    publisher_bindings = {
        binding.var: binding.binding.promise
        for binding in publisher.bindings
    }
    assert publisher_bindings["model_version"].node_id == registrar.id
    assert publisher_bindings["expected_train_execution_id"].var == (
        "full_run_execution_id"
    )
    assert publisher_bindings["shards"].var == "shards"


def test_overlay_precompute_loads_one_checkpoint_for_the_fullset():
    tree = ast.parse(Path(workflows.__file__).read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "wf_precompute_overlays"
    )
    calls = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "precompute_overlay_partition"
    ]

    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords["shard_dirs"], ast.Name)
    assert keywords["shard_dirs"].id == "shards"
    assert not any(isinstance(node, ast.For) for node in ast.walk(function))


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
        workflows.create_kitscenes_paper_approximation_manifest,
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


def test_kitscenes_benchmark_launcher_is_evaluation_only():
    buildspec = (
        _REPO_ROOT
        / "Platform"
        / "buildspec-launch-kitscenes-benchmark.yml"
    ).read_text()

    assert "flytekit==1.16.24" in buildspec
    assert "wf_audit_kitscenes_benchmark_inventory" in buildspec
    assert "wf_prepare_kitscenes_paper_approximation" in buildspec
    assert "VAL_SCENE_LIMIT" in buildspec
    assert "OVERLAP_SCENE_LIMIT" in buildspec
    assert "INGEST_CONCURRENCY" in buildspec
    assert "PACK_CONCURRENCY" in buildspec
    assert 'printf \'%s\' "${MODE}"' in buildspec
    assert "wf_train" not in buildspec
    assert "checkpoint" not in buildspec.lower()


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
        == "pack-v3-v1-v10-v6-camera512"
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
        (workflows.Dataset.L2D, 10_000, 4),
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
        '            "train",\n'
        "            source_revision,\n"
        "            False,\n"
        "        )"
    ) in source


def test_recovery_pack_maps_explicit_unbounded_sample_limits():
    tree = ast.parse(
        inspect.getsource(
            workflows._map_recovered_kitscenes_artifacts.task_function
        )
    )
    sample_limits = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "sample_limits"
            for target in node.targets
        )
    )
    assert isinstance(sample_limits.value, ast.BinOp)
    assert ast.literal_eval(sample_limits.value.left) == [0]
    assert isinstance(sample_limits.value.op, ast.Mult)

    pack_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "pack"
    )
    keywords = {keyword.arg: keyword.value for keyword in pack_call.keywords}
    assert isinstance(keywords["sample_limit"], ast.Name)
    assert keywords["sample_limit"].id == "sample_limits"


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
        (workflows.Dataset.L2D, True, False, False, False),
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


def test_navigation_quality_selects_only_accepted_optimizer_partitions(
    monkeypatch,
):
    shard_dirs = ["excluded", "accepted"]
    manifests = {
        "excluded": {"partition_id": "part-b"},
        "accepted": {"partition_id": "part-a"},
    }
    report = {
        "accepted_partition_ids": ["part-a"],
        "excluded_partition_ids": ["part-b"],
    }
    monkeypatch.setattr(
        "navigation.quality.verify_packed_navigation_quality_audit",
        lambda supplied, paths: report,
    )

    selected, verified = (
        workflows._verified_navigation_training_shard_dirs(
            shard_dirs,
            manifests,
            report,
        )
    )

    assert selected == ["accepted"]
    assert verified == report


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

    training_loader_calls = [
        call
        for call in ast.walk(train)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "make_multi_dataset_loader"
        and call.args
        and isinstance(call.args[0], ast.Name)
    ]
    assert {
        call.args[0].id for call in training_loader_calls
    } == {"shard_dirs", "training_shard_dirs"}

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


@pytest.mark.parametrize(
    "buildspec_name",
    (
        "buildspec-register.yml",
        "buildspec-launch-fullrun.yml",
        "buildspec-launch-recovery.yml",
        "buildspec-launch-reconstruction-audit.yml",
    ),
)
def test_remote_registration_buildspecs_pin_runtime_contracts(buildspec_name):
    buildspec = (_REPO_ROOT / "Platform" / buildspec_name).read_text()

    expected_flytekit = (
        "flytekit==1.16.24"
        if buildspec_name == "buildspec-register.yml"
        else "flytekit==1.14.9"
    )
    assert expected_flytekit in buildspec
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
    assert "DATASET_VERSION: v3.4" in buildspec
    assert 'EPOCHS: "20"' in buildspec
    assert (
        "TRAINING_OBJECTIVE_VERSION: "
        "rollout_aligned_planner_v1"
    ) in buildspec
    assert 'ENABLE_JUNCTION_SAMPLING: "false"' in buildspec
    assert 'ENABLE_ROUTE_CONSISTENCY: "false"' in buildspec
    assert 'ALLOW_RESUME_POLICY_TRANSITION: "false"' in buildspec
    assert "--max_partitions" in buildspec
    assert "--validation_scope" in buildspec
    assert "--training_objective_version" in buildspec
    assert "--enable_junction_sampling" in buildspec
    assert "--enable_route_consistency" in buildspec
    assert "--allow_resume_policy_transition" in buildspec
    assert "--no_allow_resume_policy_transition" in buildspec
    assert "--route_consistency_weight" in buildspec
    assert 'RECONSTRUCTION_AUDIT_URI: ""' in buildspec
    assert "--reconstruction_audit" in buildspec
    assert "--reconstruction_audit_decision" in buildspec
    assert "--reconstruction_audit_rationale" in buildspec
    assert "wf_recovered_kitscenes_full_run" in buildspec
    assert "wf_sharded_full_run" not in buildspec
    assert "--reasoning_teacher" not in buildspec
    assert "--enable_route_conditioning" in buildspec
    assert "--no_enable_route_conditioning" in buildspec
    assert (
        '--enable_route_conditioning "${ENABLE_ROUTE_CONDITIONING}"'
        not in buildspec
    )


def test_reconstruction_audit_launcher_defaults_to_smoke_subset():
    buildspec = (
        _REPO_ROOT
        / "Platform"
        / "buildspec-launch-reconstruction-audit.yml"
    ).read_text()

    assert "shell: bash" in buildspec
    assert 'MAX_PARTITIONS: "10"' in buildspec
    assert "VALIDATION_SCOPE: subset" in buildspec
    assert 'test -n "${ARTIFACT_SET_SHA256}"' in buildspec
    assert 'test -n "${AUDIT_CODE_REVISION}"' in buildspec
    assert (
        "wf_audit_recovered_kitscenes_target_reconstruction"
        in buildspec
    )
    assert "--audit_code_revision" in buildspec
    assert "--max_partitions" in buildspec
    assert "--validation_scope" in buildspec


def test_overlay_launcher_guards_selected_recovery_checkpoints():
    buildspec = (
        _REPO_ROOT / "Platform" / "buildspec-launch-overlay.yml"
    ).read_text()

    assert "DATASET_VERSION: v2.4" in buildspec
    assert (
        'PYTHONPATH="${CODEBUILD_SRC_DIR}/Model:${CODEBUILD_SRC_DIR}:'
        '${PYTHONPATH:-}"'
    ) in buildspec
    for variable in (
        "SELECTED_MLFLOW_RUN_ID",
        "SELECTED_CHECKPOINT_URI",
        "SELECTED_CHECKPOINT_SHA256",
        "SELECTED_CHECKPOINT_EPOCH",
    ):
        assert variable in buildspec
    assert "--allow-running-recovery" in buildspec
    assert "wf_publish_selected_checkpoint_overlays" in buildspec
    assert '--mlflow_run_id "${SELECTED_MLFLOW_RUN_ID}"' in buildspec
    assert '--checkpoint_uri "${SELECTED_CHECKPOINT_URI}"' in buildspec
    assert (
        '--checkpoint_sha256 "${SELECTED_CHECKPOINT_SHA256}"'
        in buildspec
    )
    assert '--checkpoint_epoch "${SELECTED_CHECKPOINT_EPOCH}"' in buildspec


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
