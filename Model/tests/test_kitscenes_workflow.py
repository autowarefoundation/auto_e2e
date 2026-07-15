"""Flyte wiring tests for the KITScenes scene fan-out."""

from __future__ import annotations

import functools

import pytest

pytest.importorskip("flytekit")

from flytekit import map_task
from flytekit.configuration import ImageConfig, SerializationSettings

from Platform.pipelines import workflows
from data_parsing.kit_scenes.source import InventoryResolution, SceneArchive


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


def test_dataset_dynamic_propagates_the_pinned_data_prep_image():
    assert workflows._map_dataset_partitions.container_image == (
        workflows.DATA_PREP_IMAGE
    )
    assert workflows._map_dataset_partitions.environment == {
        "AUTO_E2E_DATA_PREP_IMAGE": workflows.DATA_PREP_IMAGE,
    }


def test_full_run_overlay_workflow_wires_exact_model_lineage():
    resolver, publisher = workflows.wf_publish_full_run_overlays.nodes
    assert resolver.flyte_entity.name.endswith(
        "overlay_tasks.resolve_overlay_model_version"
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
