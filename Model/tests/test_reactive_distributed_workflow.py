"""Flyte wiring for distributed Reactive Stage A and Stage B."""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import re
import subprocess
import sys
import types
import zipfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from flytekit.core.context_manager import FlyteContextManager
from flytekit.types.directory import FlyteDirectory

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
)
from Platform.pipelines import (
    distributed_training,
    nuplan_dataset,
    workflows,
)


def test_distributed_workflow_import_is_path_order_independent():
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_paths = [
        str(repository_root / "Model"),
        str(repository_root),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import reactive_training_contracts as contracts; "
                "assert 'torch' not in sys.modules; "
                "assert 'numpy' not in sys.modules; "
                "import distributed_training as module; "
                "assert module.BEV_POS_WEIGHT_CAP == 64.0; "
                "print(module.BEV_POS_WEIGHT_CAP)"
            ),
        ],
        cwd=repository_root / "Platform" / "pipelines",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "64.0"


def test_shared_pack_cache_includes_camera_resolution_contract():
    assert (
        workflows.PACK_CACHE_VERSION
        == "pack-v3-v1-v12-v6-camera512"
    )


def test_l2d_workflow_defaults_use_reactive_camera_contract():
    source = Path(workflows.__file__).read_text(encoding="utf-8")

    assert workflows.DATASET_PACK_VERSION == "v2.4"
    assert workflows.KITSCENES_NAVIGATION_DATASET_VERSION == "v3.4"
    assert "image_size: int = 256" not in source
    assert (
        source.count(
            "image_size: int = REACTIVE_CAMERA_IMAGE_SIZE"
        )
        == 10
    )
    for name, dataset_version in (
        ("buildspec-launch-sharded.yml", "v3.4"),
        ("buildspec-launch-fullrun.yml", "v3.4"),
        ("buildspec-launch-recovery.yml", "v3.4"),
        ("buildspec-launch-overlay.yml", "v2.4"),
        ("buildspec-launch-reconstruction-audit.yml", "v3.4"),
    ):
        buildspec = (
            Path(workflows.__file__).parents[1] / name
        ).read_text(encoding="utf-8")
        assert f"DATASET_VERSION: {dataset_version}" in buildspec
        assert 'IMAGE_SIZE: "512"' in buildspec


def test_l2d_bounded_sampling_retains_every_nonempty_group():
    class Dataset:
        groups = ["a"] * 4 + ["b"] * 2 + ["c"] * 3

        def __len__(self):
            return len(self.groups)

        def split_group_uid(self, index):
            return self.groups[index]

    selected = workflows._bounded_group_coverage_indices(
        Dataset(),
        sample_limit=6,
    )

    assert len(selected) == 6
    assert len(set(selected)) == 6
    assert {
        Dataset.groups[index] for index in selected
    } == {"a", "b", "c"}
    assert workflows._allocate_partition_sample_limits(
        [["0", "1"], ["2"], ["3", "4"]],
        total_sample_limit=8,
    ) == [3, 2, 3]


def test_flyte_directory_download_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    import fsspec
    from flytekit.exceptions.system import FlyteDownloadDataException

    class Directory:
        attempts = 0

        def download(self):
            self.attempts += 1
            if self.attempts < 3:
                raise FlyteDownloadDataException("transient S3 failure")
            return "/tmp/downloaded"

    delays: list[float] = []
    test_config = {
        "gather_batch_size": 4096,
        "nofiles_gather_batch_size": -1,
    }
    monkeypatch.setattr(fsspec.config, "conf", test_config)
    monkeypatch.setattr(workflows.time, "sleep", delays.append)
    directory = Directory()

    assert workflows._download_flyte_directory(directory) == (
        "/tmp/downloaded"
    )
    assert directory.attempts == 3
    assert delays == [15.0, 30.0]
    assert test_config == {
        "gather_batch_size": 16,
        "nofiles_gather_batch_size": 16,
    }


def test_l2d_bounded_sampling_rejects_insufficient_partition():
    class Dataset:
        groups = ["a", "b"]

        def __len__(self):
            return len(self.groups)

        def split_group_uid(self, index):
            return self.groups[index]

    with pytest.raises(ValueError, match="exceeds available samples"):
        workflows._bounded_group_coverage_indices(
            Dataset(),
            sample_limit=3,
        )


def test_l2d_bounded_sampling_rejects_empty_partition():
    class Dataset:
        def __len__(self):
            return 0

        def split_group_uid(self, index):
            raise AssertionError(index)

    with pytest.raises(ValueError, match="found no valid samples"):
        workflows._bounded_group_coverage_indices(
            Dataset(),
            sample_limit=1,
        )


def test_l2d_sharded_workflow_binds_total_sample_limit():
    _, mapped = workflows.wf_create_dataset_sharded.nodes
    bindings = {
        binding.var: binding.binding
        for binding in mapped.bindings
    }

    assert (
        bindings["total_sample_limit"].promise.var
        == "total_sample_limit"
    )
    assert "sample_limit" in workflows.data_processing.python_interface.inputs

    full_run_nodes = workflows.wf_sharded_full_run.nodes
    dataset_node = next(
        node
        for node in full_run_nodes
        if node.flyte_entity.name.endswith("wf_create_dataset_sharded")
    )
    full_run_bindings = {
        binding.var: binding.binding
        for binding in dataset_node.bindings
    }
    assert (
        full_run_bindings["total_sample_limit"].promise.var
        == "total_sample_limit"
    )

    for filename in (
        "buildspec-launch-sharded.yml",
        "buildspec-launch-fullrun.yml",
    ):
        buildspec = (
            Path(workflows.__file__).parents[1] / filename
        ).read_text(encoding="utf-8")
        assert "DATASET: KIT-MRT/KITScenes-Multimodal" in buildspec
        assert 'TOTAL_SAMPLE_LIMIT: "0"' in buildspec
        assert '--dataset "$DATASET"' in buildspec
        assert "--total_sample_limit $TOTAL_SAMPLE_LIMIT" in buildspec


def test_l2d_bounded_sampling_rejects_reasoning_labels():
    with pytest.raises(
        ValueError,
        match="requires reasoning_teacher='none'",
    ):
        workflows._map_dataset_partitions.task_function(
            partitions=[["0"]],
            dataset=workflows.Dataset.L2D,
            source_revision=workflows.L2D_SOURCE_REVISION,
            dataset_version=workflows.DATASET_PACK_VERSION,
            image_size=512,
            world_model=False,
            reasoning_teacher="mock",
            prompt_version="test",
            label_stride=10,
            label_workers=1,
            ingest_concurrency=1,
            label_concurrency=1,
            pack_concurrency=1,
            total_sample_limit=1,
        )


@pytest.mark.parametrize("reasoning_teacher", ["none", "mock"])
def test_l2d_large_partition_resources_apply_to_pack_arrays(
    reasoning_teacher: str,
):
    context = FlyteContextManager.current_context()
    with FlyteContextManager.with_context(
        context.with_new_compilation_state()
    ) as compilation_context:
        workflows._map_dataset_partitions.task_function(
            partitions=[[str(index) for index in range(50)]],
            dataset=workflows.Dataset.L2D,
            source_revision=workflows.L2D_SOURCE_REVISION,
            dataset_version=workflows.DATASET_PACK_VERSION,
            image_size=512,
            world_model=False,
            reasoning_teacher=reasoning_teacher,
            prompt_version="test",
            label_stride=10,
            label_workers=1,
            ingest_concurrency=1,
            label_concurrency=1,
            pack_concurrency=1,
            total_sample_limit=0,
        )
        pack_node = compilation_context.compilation_state.nodes[-1]

    assert pack_node._resources is not None
    expected = {
        1: "15",
        3: "128Gi",
        5: "800Gi",
    }
    assert {
        entry.name: entry.value
        for entry in pack_node._resources.requests
    } == expected
    assert {
        entry.name: entry.value
        for entry in pack_node._resources.limits
    } == expected


def test_flyte_entrypoints_do_not_use_mutable_defaults():
    pipelines_root = Path(workflows.__file__).parent
    mutable_defaults = []

    source_paths = [
        pipelines_root / "distributed_training.py",
        pipelines_root / "nuplan_dataset.py",
    ]
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            decorators = set()
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(
                    decorator,
                    ast.Call,
                ) else decorator
                if isinstance(target, ast.Name):
                    decorators.add(target.id)
                elif isinstance(target, ast.Attribute):
                    decorators.add(target.attr)
            if not decorators.intersection(
                {"task", "workflow", "dynamic"},
            ):
                continue

            arguments = node.args.posonlyargs + node.args.args
            defaults = [None] * (
                len(arguments) - len(node.args.defaults)
            ) + list(node.args.defaults)
            for argument, default in zip(arguments, defaults):
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    mutable_defaults.append(
                        (
                            source_path.name,
                            node.name,
                            argument.arg,
                        )
                    )

    assert mutable_defaults == []


def test_single_gpu_flyte_tasks_target_validation_capacity():
    template = workflows._large_shm_pod_template()
    pod_spec = template.pod_spec
    assert pod_spec.node_selector == {
        "workload-type": "gpu-validation"
    }
    assert [
        (
            toleration.key,
            toleration.operator,
            toleration.effect,
        )
        for toleration in pod_spec.tolerations
    ] == [("nvidia.com/gpu", "Exists", "NoSchedule")]

    source_path = Path(workflows.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    missing_templates = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            target = decorator.func
            if not isinstance(target, ast.Name) or target.id != "task":
                continue
            keywords = {
                keyword.arg: keyword.value
                for keyword in decorator.keywords
                if keyword.arg is not None
            }
            requests = keywords.get("requests")
            if not isinstance(requests, ast.Call):
                continue
            resource_keywords = {
                keyword.arg: keyword.value
                for keyword in requests.keywords
                if keyword.arg is not None
            }
            gpu = resource_keywords.get("gpu")
            if not (
                isinstance(gpu, ast.Constant)
                and gpu.value == "1"
            ):
                continue
            if "pod_template" not in keywords:
                missing_templates.append(node.name)
    assert missing_templates == []


def test_reactive_bev_evaluation_uses_validation_gpu_contract():
    template = distributed_training._bev_evaluation_pod_template()
    assert template.pod_spec.node_selector == {
        "workload-type": "gpu-validation"
    }
    assert template.pod_spec.volumes[0].empty_dir.size_limit == "16Gi"
    assert {
        "checkpoint",
        "shards",
        "dataset",
        "benchmark_inventory",
        "split",
        "val_fraction",
        "batch_size",
        "num_loader_workers",
        "probability_bins",
    } == set(
        distributed_training.evaluate_reactive_bev_checkpoint
        .python_interface.inputs
    )
    evaluation_node, publication_node = (
        distributed_training
        .wf_evaluate_reactive_bev_checkpoint
        .nodes
    )
    assert evaluation_node.flyte_entity is (
        distributed_training.evaluate_reactive_bev_checkpoint
    )
    assert publication_node.flyte_entity is (
        distributed_training.publish_reactive_bev_evaluation
    )
    source = inspect.getsource(
        distributed_training.evaluate_reactive_bev_checkpoint.task_function
    )
    assert "checkpoint_bev_probability_bins(config)" in source
    assert "expected_sample_uids=expected_sample_uids" in source
    assert "expected_sample_count=expected_sample_count" in source
    assert '"evaluation_sample_count": expected_sample_count' in source
    assert "validate_kitscenes_benchmark_inventory_coverage" in source
    assert '"evaluation_sample_uid_sha256"' in source
    assert "_log_reactive_bev_evaluation_to_mlflow" not in source
    publish_source = inspect.getsource(
        distributed_training.publish_reactive_bev_evaluation.task_function
    )
    assert "_log_reactive_bev_evaluation_to_mlflow" in publish_source
    assert "checkpoint_hasher.hexdigest()" in publish_source
    assert (
        distributed_training
        .publish_reactive_bev_evaluation
        .environment["MLFLOW_TRACKING_URI"]
        == distributed_training.MLFLOW_URI
    )
    resources = (
        distributed_training.evaluate_reactive_bev_checkpoint.resources
    )
    assert resources.requests.cpu == "6"
    assert resources.requests.mem == "28Gi"
    assert resources.requests.gpu == "1"
    assert resources.requests.ephemeral_storage == "420Gi"
    assert resources.limits == resources.requests
    assert set(
        distributed_training.ReactiveBEVEvaluationOutput.__annotations__
    ) == {
        "report",
        "report_sha256",
        "checkpoint_sha256",
        "checkpoint_epoch",
    }
    assert set(
        distributed_training
        .ReactiveBEVEvaluationWorkflowOutput
        .__annotations__
    ) == {
        "report",
        "report_sha256",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "mlflow_run_id",
        "registered_model_name",
        "registered_model_version",
    }


def test_reactive_bev_evaluation_metrics_flatten_numeric_values():
    metrics = distributed_training._reactive_bev_evaluation_metrics({
        "sample_count": 4,
        "evaluation_valid": True,
        "classes": {
            "vehicle": {
                "average_precision": 0.75,
                "availability": "computed",
                "nonfinite": float("nan"),
            },
        },
        "class_order": ["vehicle"],
    })

    assert metrics == {
        "eval/sample_count": 4.0,
        "eval/classes/vehicle/average_precision": 0.75,
    }


def test_reactive_bev_model_registration_is_idempotent():
    version_tags = {}
    created_models = []
    created_versions = []

    class Client:
        def get_registered_model(self, name):
            if not created_models:
                raise RuntimeError("not found")
            assert name == "auto-e2e-bev-segmentation"

        def create_registered_model(self, name):
            created_models.append(name)

        def search_model_versions(self, query):
            assert query == "name='auto-e2e-bev-segmentation'"
            return created_versions

        def create_model_version(self, *, name, source, run_id):
            version = SimpleNamespace(
                version="7",
                source=source,
                run_id=run_id,
                tags={},
            )
            created_versions.append(version)
            return version

        def set_model_version_tag(self, name, version, key, value):
            assert name == "auto-e2e-bev-segmentation"
            assert version == "7"
            version_tags[key] = value
            created_versions[0].tags[key] = value

    report = {
        "dataset": "nuplan/nuplan-v1.1",
        "macro_average_precision_supported_classes": 0.42,
        "sample_count": 4096,
        "schema_version": "bev_segmentation_evaluation_v7",
    }
    client = Client()
    first = distributed_training._register_reactive_bev_model_version(
        client,
        run_id="a" * 32,
        checkpoint_artifact_uri="runs:/run/model/checkpoint.pt",
        checkpoint_source_uri="s3://bucket/checkpoint.pt",
        checkpoint_sha256="b" * 64,
        checkpoint_epoch=2,
        report=report,
        report_sha256="c" * 64,
        source_training_mlflow_run_id="d" * 32,
    )
    second = distributed_training._register_reactive_bev_model_version(
        client,
        run_id="a" * 32,
        checkpoint_artifact_uri="runs:/run/model/checkpoint.pt",
        checkpoint_source_uri="s3://bucket/checkpoint.pt",
        checkpoint_sha256="b" * 64,
        checkpoint_epoch=2,
        report=report,
        report_sha256="c" * 64,
        source_training_mlflow_run_id="d" * 32,
    )

    assert first == second == "7"
    assert created_models == ["auto-e2e-bev-segmentation"]
    assert len(created_versions) == 1
    assert version_tags["checkpoint_sha256"] == "b" * 64
    assert version_tags["model_role"] == (
        "bev_segmentation_candidate"
    )
    assert version_tags[
        "nuplan_nuplan_v1_1_macro_average_precision"
    ] == "0.42"
    assert version_tags["source_training_mlflow_run_id"] == "d" * 32


def test_reviewed_ray_topologies_have_fixed_worker_groups():
    assert (
        distributed_training.RAY_1.worker_node_config[0].replicas
        == 1
    )
    assert (
        distributed_training.RAY_2.worker_node_config[0].replicas
        == 2
    )
    assert (
        distributed_training.RAY_8.worker_node_config[0].replicas
        == 1
    )
    for config in (
        distributed_training.RAY_1,
        distributed_training.RAY_2,
        distributed_training.RAY_4,
        distributed_training.RAY_REACTIVE_4,
        distributed_training.RAY_8,
    ):
        workers = config.worker_node_config[0]
        assert workers.min_replicas == workers.replicas
        assert workers.max_replicas == workers.replicas
        assert config.enable_autoscaling is False
    smoke_worker = distributed_training.RAY_1.worker_node_config[0]
    assert smoke_worker.ray_start_params == {
        "num-cpus": "3",
        "num-gpus": "1",
    }
    assert smoke_worker.pod_template.pod_spec.containers[
        0
    ].resources.requests == {
        "cpu": "3",
        "ephemeral-storage": "100Gi",
        "memory": "24Gi",
        "nvidia.com/gpu": "1",
    }
    assert (
        distributed_training.train_reactive_nuplan_bev_ray_1_smoke
        .metadata.labels["kueue.x-k8s.io/queue-name"]
        == "gpu-validation"
    )
    assert distributed_training._reactive_worker_cpus(1) == 3
    smoke_source = inspect.getsource(
        distributed_training.train_reactive_nuplan_bev_ray_1_smoke
        .task_function
    )
    assert "allow_single_worker_smoke=True" in smoke_source
    assert (
        "checkpoint_interval_steps=min(4, steps_per_epoch)"
        in smoke_source
    )
    assert (
        distributed_training.RAY_REACTIVE_4.worker_node_config[0]
        .ray_start_params["num-gpus"]
        == "4"
    )
    assert (
        distributed_training.RAY_8.worker_node_config[0]
        .ray_start_params["num-gpus"]
        == "8"
    )


def test_four_rank_performance_capacity_matches_ray_contract():
    head_spec = (
        distributed_training.RAY_REACTIVE_4.head_node_config
        .pod_template.pod_spec
    )
    worker = distributed_training.RAY_REACTIVE_4.worker_node_config[0]
    worker_spec = worker.pod_template.pod_spec
    assert head_spec.node_selector is None
    assert not head_spec.tolerations
    assert worker.replicas == 1
    assert worker.ray_start_params == {
        "num-cpus": "48",
        "num-gpus": "4",
    }
    assert worker_spec.node_selector == {
        "workload-type": "p5en-capacity-block"
    }
    assert worker_spec.affinity is None
    assert worker.pod_template.annotations == {
        "karpenter.sh/do-not-disrupt": "true"
    }
    assert worker_spec.containers[0].resources.requests == {
        "cpu": "48",
        "memory": "512Gi",
        "nvidia.com/gpu": "4",
        "ephemeral-storage": "500Gi",
    }
    assert (
        distributed_training.train_reactive_stage_ray_4.metadata.labels[
            "kueue.x-k8s.io/queue-name"
        ]
        == "p5en-capacity-block"
    )

    platform_root = Path(distributed_training.__file__).parents[1]
    node_classes = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodeclass.yaml"
            ).read_text()
        )
    }
    assert set(node_classes) == {
        "auto-e2e-gpu-validation",
        "auto-e2e-p5en-capacity-block",
    }
    validation_class = node_classes[
        "auto-e2e-gpu-validation"
    ]["spec"]
    assert validation_class["ephemeralStorage"] == {
        "size": "500Gi",
        "iops": 3000,
        "throughput": 125,
    }
    assert validation_class["capacityReservationSelectorTerms"] == [
        {
            "ownerID": "REPLACE_WITH_AWS_ACCOUNT_ID",
            "tags": {"Name": "auto-e2e-gpu-validation"},
        }
    ]
    capacity_block_class = node_classes[
        "auto-e2e-p5en-capacity-block"
    ]["spec"]
    assert capacity_block_class[
        "capacityReservationSelectorTerms"
    ] == [
        {
            "ownerID": "REPLACE_WITH_AWS_ACCOUNT_ID",
            "tags": {"Name": "auto-e2e-p5en-capacity-block"},
        }
    ]
    assert "placementGroupSelector" not in capacity_block_class
    assert capacity_block_class["ephemeralStorage"] == {
        "size": "2Ti",
        "iops": 16000,
        "throughput": 1000,
    }
    assert capacity_block_class["subnetSelectorTerms"] == [
        {"tags": {"Name": "auto-e2e-platform-private-us-west-2a"}},
        {"tags": {"Name": "auto-e2e-platform-private-us-west-2b"}},
        {"tags": {"Name": "auto-e2e-platform-private-us-west-2c"}},
    ]

    node_pools = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodepool.yaml"
            ).read_text()
        )
    }
    assert set(node_pools) == {
        "gpu-validation",
        "p5en-capacity-block",
    }
    validation_pool = node_pools["gpu-validation"]["spec"]
    assert validation_pool["limits"] == {
        "cpu": "16",
        "memory": "128Gi",
        "nodes": "2",
        "nvidia.com/gpu": "2",
    }
    assert validation_pool["template"]["metadata"]["labels"] == {
        "workload-type": "gpu-validation"
    }
    validation_template = validation_pool["template"]["spec"]
    assert validation_template["expireAfter"] == "456h"
    assert validation_template["terminationGracePeriod"] == "24h"
    capacity_block_pool = node_pools["p5en-capacity-block"]["spec"]
    capacity_block_template = capacity_block_pool["template"]["spec"]
    assert capacity_block_template["expireAfter"] == "503h"
    assert capacity_block_template["terminationGracePeriod"] == "5m"
    assert capacity_block_pool["disruption"] == {
        "budgets": [{"nodes": "0"}],
        "consolidationPolicy": "WhenEmpty",
        "consolidateAfter": "Never",
    }
    requirements = {
        item["key"]: item["values"]
        for item in capacity_block_template["requirements"]
    }
    assert requirements["node.kubernetes.io/instance-type"] == [
        "p5en.48xlarge",
        "p5.48xlarge",
    ]
    assert requirements["topology.kubernetes.io/zone"] == [
        "us-west-2a",
        "us-west-2b",
        "us-west-2c",
    ]
    assert requirements["karpenter.sh/capacity-type"] == ["reserved"]
    assert capacity_block_pool["limits"] == {
        "cpu": "384",
        "memory": "4Ti",
        "nodes": "2",
        "nvidia.com/gpu": "16",
    }

    queue_objects = {
        (
            item["kind"],
            item["metadata"]["name"],
            item["metadata"].get("namespace"),
        ): item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/kueue-config/kueue-objects.yaml"
            ).read_text()
        )
    }
    performance_queue = queue_objects[
        ("ClusterQueue", "p5en-capacity-block-queue", None)
    ]["spec"]
    assert performance_queue["queueingStrategy"] == "BestEffortFIFO"
    assert performance_queue["namespaceSelector"] == {
        "matchExpressions": [
            {
                "key": "kubernetes.io/metadata.name",
                "operator": "In",
                "values": ["auto-e2e-development"],
            }
        ]
    }
    compute_group = next(
        group
        for group in performance_queue["resourceGroups"]
        if group["coveredResources"]
        == ["cpu", "memory", "ephemeral-storage"]
    )
    assert compute_group["flavors"][0]["resources"] == [
        {"name": "cpu", "nominalQuota": "256"},
        {"name": "memory", "nominalQuota": "3Ti"},
        {"name": "ephemeral-storage", "nominalQuota": "1500Gi"},
    ]
    gpu_group = next(
        group
        for group in performance_queue["resourceGroups"]
        if group["coveredResources"] == ["nvidia.com/gpu"]
    )
    assert gpu_group["flavors"][0]["resources"] == [
        {"name": "nvidia.com/gpu", "nominalQuota": "16"}
    ]
    assert (
        "LocalQueue",
        "p5en-capacity-block",
        "auto-e2e-development",
    ) in queue_objects
    for namespace in (
        "auto-e2e-staging",
        "auto-e2e-production",
    ):
        p5en_queue = queue_objects[
            "LocalQueue",
            "p5en-capacity-block",
            namespace,
        ]
        validation_local_queue = queue_objects[
            "LocalQueue",
            "gpu-validation",
            namespace,
        ]
        assert p5en_queue["spec"]["stopPolicy"] == "HoldAndDrain"
        assert (
            validation_local_queue["spec"]["stopPolicy"]
            == "HoldAndDrain"
        )
    assert (
        "LocalQueue",
        "gpu-validation",
        "auto-e2e-training",
    ) in queue_objects
    validation_queue = queue_objects[
        ("ClusterQueue", "gpu-validation-queue", None)
    ]["spec"]
    assert validation_queue["namespaceSelector"] == {
        "matchExpressions": [
            {
                "key": "kubernetes.io/metadata.name",
                "operator": "In",
                "values": [
                    "auto-e2e-development",
                    "auto-e2e-training",
                ],
            }
        ]
    }
    validation_compute_group = next(
        group
        for group in validation_queue["resourceGroups"]
        if group["coveredResources"]
        == ["cpu", "memory", "ephemeral-storage"]
    )
    assert validation_compute_group["flavors"][0]["resources"] == [
        {"name": "cpu", "nominalQuota": "12"},
        {"name": "memory", "nominalQuota": "56Gi"},
        {"name": "ephemeral-storage", "nominalQuota": "1000Gi"},
    ]
    assert (
        "ClusterQueue",
        "training-queue",
        None,
    ) not in queue_objects
    training_queue = queue_objects[
        ("LocalQueue", "gpu-validation", "auto-e2e-training")
    ]
    assert "annotations" not in training_queue["metadata"]

    deploy_script = (platform_root / "infra/post-apply.sh").read_text()
    render_index = deploy_script.index(
        "s/REPLACE_WITH_AWS_ACCOUNT_ID/${ACCOUNT}/g"
    )
    node_pool_index = deploy_script.index(
        "karpenter-nodepools/gpu-nodepool.yaml"
    )
    assert render_index < node_pool_index
    assert "kueue-config/kueue-objects.yaml" in deploy_script
    assert "nodepool/gpu-validation" in deploy_script
    assert "nodeclass/auto-e2e-p5en-capacity-block" in deploy_script
    assert "kueue-manager-config" in deploy_script
    assert "wait_for_gpu_quota auto-e2e-development 18" in deploy_script
    assert "wait_for_gpu_quota auto-e2e-staging 0" in deploy_script
    assert "wait_for_gpu_quota auto-e2e-production 0" in deploy_script
    assert '"limits.nvidia.com/gpu" not in hard' in deploy_script
    assert "limits.nvidia.com~1gpu" in deploy_script
    assert "--timeout=900s" in deploy_script
    assert "kubectl describe" in deploy_script
    for framework in (
        "batch/job",
        "kubeflow.org/pytorchjob",
        "ray.io/rayjob",
        "pod",
    ):
        assert framework in deploy_script
    assert "nodepool/gpu-training" in deploy_script
    assert "nodeclass/auto-e2e-gpu-training" in deploy_script
    assert re.search(r"\b[0-9]{12}\b", deploy_script) is None
    kueue_values = yaml.safe_load(
        (platform_root / "helm-values/kueue.yaml").read_text()
    )
    controller_manager = kueue_values["controllerManager"]
    assert controller_manager["replicas"] == 2
    assert controller_manager["podDisruptionBudget"] == {
        "enabled": True,
        "minAvailable": 1,
    }
    assert controller_manager["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "DoNotSchedule",
            "labelSelector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "kueue",
                    "app.kubernetes.io/instance": "kueue",
                    "control-plane": "controller-manager",
                }
            },
        }
    ]
    assert controller_manager["livenessProbe"] == {
        "initialDelaySeconds": 120
    }
    assert controller_manager["manager"]["podAnnotations"] == {
        "karpenter.sh/do-not-disrupt": "true"
    }
    assert (
        controller_manager["manager"]["priorityClassName"]
        == "system-cluster-critical"
    )
    assert controller_manager["manager"]["resources"]["requests"] == {
        "cpu": "500m",
        "memory": "512Mi",
    }
    manager_config = kueue_values["managerConfig"][
        "controllerManagerConfigYaml"
    ]
    for framework in (
        "batch/job",
        "kubeflow.org/pytorchjob",
        "ray.io/rayjob",
        "pod",
    ):
        assert f"- {framework}" in manager_config
    assert "manageJobsWithoutQueueName: false" in manager_config
    for namespace in (
        "auto-e2e-development",
        "auto-e2e-staging",
        "auto-e2e-production",
        "auto-e2e-training",
    ):
        assert f"- {namespace}" in manager_config
    kueue_terraform = (
        platform_root / "infra/modules/kueue/main.tf"
    ).read_text()
    assert "controller.manager.configuration" not in kueue_terraform
    for relative_path in re.findall(
        r"\.\./k8s/[A-Za-z0-9_./-]+\.yaml",
        deploy_script,
    ):
        assert (
            platform_root
            / "infra"
            / relative_path
        ).resolve().is_file()

    smoke_test = yaml.safe_load(
        (platform_root / "k8s/gpu-smoke-test.yaml").read_text()
    )
    smoke_resources = smoke_test["spec"]["containers"][0]["resources"]
    assert smoke_resources["requests"] == {
        "cpu": "2",
        "memory": "16Gi",
        "nvidia.com/gpu": "1",
    }
    assert smoke_resources["limits"] == smoke_resources["requests"]


def test_reactive_ray_cpu_contract_has_one_source_of_truth():
    expected_actor_capacity = (
        (distributed_training.RAY_2, 2, 3),
        (distributed_training.RAY_REACTIVE_4, 4, 12),
        (distributed_training.RAY_8, 8, 12),
    )
    for config, actor_count, actor_cpus in expected_actor_capacity:
        worker = config.worker_node_config[0]
        cpu = worker.ray_start_params["num-cpus"]
        resources = (
            worker.pod_template.pod_spec
            .containers[0].resources
        )
        assert resources.requests["cpu"] == cpu
        assert resources.limits["cpu"] == cpu
        assert int(cpu) * worker.replicas == actor_count * actor_cpus


def test_ray_gpu_workers_avoid_detail_process_group_wrapper():
    for config in (
        distributed_training.RAY_2,
        distributed_training.RAY_REACTIVE_4,
        distributed_training.RAY_8,
    ):
        worker = config.worker_node_config[0]
        environment = {
            item.name: item.value
            for item in worker.pod_template.pod_spec.containers[0].env
        }
        assert environment["NCCL_DEBUG"] == "INFO"
        assert environment["TORCH_DISTRIBUTED_DEBUG"] == "INFO"


def test_ray_tasks_serialize_the_resolved_storage_path():
    expected_environment = {
        "AWS_DEFAULT_REGION": "us-west-2",
        "AUTO_E2E_RAY_STORAGE_PATH": (
            distributed_training.RAY_STORAGE_PATH
        ),
        "MLFLOW_TRACKING_URI": distributed_training.MLFLOW_URI,
        "RAY_TRAIN_V2_ENABLED": "1",
    }

    assert distributed_training.ray_ddp_smoke_4.environment == (
        expected_environment
    )
    assert distributed_training.train_reactive_stage_ray_2.environment == (
        expected_environment
    )
    assert distributed_training.train_reactive_stage_ray_4.environment == (
        expected_environment
    )
    assert distributed_training.train_reactive_stage_ray_8.environment == (
        expected_environment
    )


def test_reactive_mlflow_helpers_record_stable_numeric_history():
    config = {
        "backbone": "res_net_50",
        "bev_encoder_learning_rate": 1e-5,
        "bev_weight": 1.0,
        "checkpoint_interval_steps": 256,
        "epochs": 5,
        "freeze_bevformer": False,
        "gradient_accumulation_steps": 1,
        "is_pretrained": True,
        "learning_rate": 1e-4,
        "num_loader_workers": 4,
        "num_workers": 8,
        "per_rank_batch_size": 4,
        "precision": "bf16",
        "route_weight": 0.0,
        "source_uris": ["s3://dataset/part-0", "s3://dataset/part-1"],
        "stage": "nuplan_full",
        "training_scope": "bev_only",
        "training_seed": 149,
        "trajectory_weight": 0.0,
        "val_fraction": 0.1,
        "validation_sample_limit": 4096,
        "weight_decay": 1e-2,
    }

    digest = distributed_training._reactive_mlflow_config_sha256(config)
    assert len(digest) == 64
    assert digest == distributed_training._reactive_mlflow_config_sha256(
        dict(reversed(tuple(config.items())))
    )
    params = distributed_training._reactive_mlflow_params(
        config,
        execution_name="reactive-test",
    )
    assert params["ctx/flyte_execution_id"] == "reactive-test"
    assert params["data/source_partition_count"] == 2
    assert params["train/training_scope"] == "bev_only"
    assert params["train/world_size"] == 8
    assert distributed_training._reactive_mlflow_metrics({
        "epoch": 2,
        "finite": 0.25,
        "boolean": True,
        "digest": "abc",
        "nan": float("nan"),
        "infinite": float("inf"),
    }) == {
        "epoch": 2.0,
        "finite": 0.25,
    }


def test_reactive_ray_task_persists_mlflow_result_and_failure_state():
    source = inspect.getsource(
        distributed_training._run_reactive_stage_task
    )

    assert "_start_reactive_mlflow_run" in source
    assert "_log_reactive_mlflow_result" in source
    assert "_mark_reactive_mlflow_failed" in source
    assert "config=training_config" in source
    assert '"mlflow_run_id": mlflow_run_id' in source
    assert (
        distributed_training.REACTIVE_MLFLOW_EXPERIMENT
        == "reactive-training"
    )


def test_reactive_mlflow_run_is_reused_for_flyte_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "backbone": "res_net_50",
        "bev_encoder_learning_rate": 1e-5,
        "bev_weight": 0.0,
        "checkpoint_interval_steps": 256,
        "epochs": 3,
        "freeze_bevformer": True,
        "gradient_accumulation_steps": 1,
        "is_pretrained": True,
        "learning_rate": 1e-4,
        "num_loader_workers": 2,
        "num_workers": 8,
        "per_rank_batch_size": 4,
        "precision": "bf16",
        "route_weight": 1.0,
        "source_uris": ["s3://dataset/part-0"],
        "stage": "nuplan_full",
        "training_scope": "multitask",
        "training_seed": 149,
        "trajectory_weight": 1.0,
        "val_fraction": 0.1,
        "validation_sample_limit": 1024,
        "weight_decay": 1e-2,
    }
    created_runs = []
    logged_params = {}
    tags = {}

    class Client:
        def search_runs(self, **kwargs):
            assert kwargs["max_results"] == 2
            return created_runs

        def create_run(self, *, experiment_id, start_time, tags):
            assert experiment_id == "11"
            assert start_time > 0
            run = SimpleNamespace(
                info=SimpleNamespace(run_id="a" * 32),
                data=SimpleNamespace(tags=dict(tags)),
            )
            created_runs.append(run)
            return run

        def log_param(self, run_id, name, value):
            assert run_id == "a" * 32
            logged_params[name] = value

        def set_tag(self, run_id, name, value):
            assert run_id == "a" * 32
            tags[name] = value

    client = Client()
    mlflow_module = types.ModuleType("mlflow")
    mlflow_module.set_tracking_uri = lambda uri: tags.setdefault(
        "tracking_uri",
        uri,
    )
    mlflow_module.set_experiment = lambda name: SimpleNamespace(
        experiment_id="11",
        name=name,
    )
    tracking_module = types.ModuleType("mlflow.tracking")
    tracking_module.MlflowClient = lambda: client
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.tracking",
        tracking_module,
    )
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow.test:5000",
    )

    first_client, first_run_id = (
        distributed_training._start_reactive_mlflow_run(
            config,
            execution_name="reactive-test",
            run_name="reactive-test-nuplan_full-ray-8-full",
        )
    )
    second_client, second_run_id = (
        distributed_training._start_reactive_mlflow_run(
            config,
            execution_name="reactive-test",
            run_name="reactive-test-nuplan_full-ray-8-full",
        )
    )

    assert first_client is second_client is client
    assert first_run_id == second_run_id == "a" * 32
    assert len(created_runs) == 1
    assert logged_params["train/world_size"] == 8
    assert tags["tracking_uri"] == "http://mlflow.test:5000"
    assert tags["flyte_retry_reused"] == "true"
    assert tags["task_status"] == "RUNNING"


def test_reactive_mlflow_failure_recovers_latest_s3_history():
    history = [
        {
            "checkpoint_sha256": "a" * 64,
            "epoch": 1,
            "train_total": 0.9,
        },
        {
            "checkpoint_sha256": "b" * 64,
            "epoch": 2,
            "train_total": 0.7,
        },
    ]
    snapshot = {
        "latest_checkpoint_result": {
            "checkpoint_dir_name": "checkpoint_0002",
            "metrics": {
                "checkpoint_sha256": "c" * 64,
                "epoch": 3,
                "executed_optimizer_steps": 7168,
            },
        },
    }

    class Client:
        def __init__(self):
            self.requested = []

        def get_object(self, *, Bucket, Key):
            self.requested.append((Bucket, Key))
            payload = (
                snapshot
                if Key.endswith("checkpoint_manager_snapshot.json")
                else history
            )
            return {
                "Body": io.BytesIO(
                    json.dumps(payload).encode("utf-8")
                )
            }

    client = Client()
    recovered = (
        distributed_training._recover_reactive_mlflow_checkpoint(
            {
                "run_name": "reactive-test",
                "storage_path": "s3://checkpoint-bucket/ray-train",
            },
            s3_client=client,
        )
    )

    assert recovered == {
        "checkpoint_uri": (
            "s3://checkpoint-bucket/ray-train/reactive-test/"
            "checkpoint_0002/checkpoint.pt"
        ),
        "history": history,
        "metrics": snapshot["latest_checkpoint_result"]["metrics"],
    }
    assert client.requested == [
        (
            "checkpoint-bucket",
            "ray-train/reactive-test/"
            "checkpoint_manager_snapshot.json",
        ),
        (
            "checkpoint-bucket",
            "ray-train/reactive-test/checkpoint_0002/history.json",
        ),
    ]


def test_distributed_program_passes_stage_a_checkpoint_to_stage_b():
    stage_a, stage_b = (
        distributed_training.wf_train_reactive_nuplan_l2d_ray_8.nodes
    )
    assert stage_a.flyte_entity.name.endswith(
        "train_reactive_stage_ray_8"
    )
    assert stage_b.flyte_entity.name.endswith(
        "train_reactive_stage_ray_8"
    )
    stage_a_bindings = {
        binding.var: binding.binding for binding in stage_a.bindings
    }
    stage_b_bindings = {
        binding.var: binding.binding for binding in stage_b.bindings
    }
    assert stage_a_bindings["stage"].scalar.primitive.string_value == (
        "nuplan_full"
    )
    assert stage_b_bindings["stage"].scalar.primitive.string_value == (
        "l2d_continuation"
    )
    assert stage_a_bindings[
        "freeze_bevformer"
    ].scalar.primitive.boolean
    assert stage_b_bindings[
        "freeze_bevformer"
    ].scalar.primitive.boolean
    assert (
        stage_a_bindings[
            "parent_checkpoint"
        ].scalar.union.value.scalar.none_type
        is not None
    )
    parent_promise = stage_b_bindings["parent_checkpoint"].promise
    assert parent_promise.node_id == stage_a.id
    assert parent_promise.var == "checkpoint"


def test_remote_dataset_inputs_are_required():
    remote = FlyteDirectory("s3://datasets/nuplan/reactive")
    assert distributed_training._flyte_remote_uri(remote) == (
        "s3://datasets/nuplan/reactive"
    )

    with pytest.raises(ValueError, match="immutable S3"):
        distributed_training._flyte_remote_uri(
            FlyteDirectory("/tmp/reactive")
        )


def test_distributed_workflow_source_has_no_deployment_account_id():
    source = Path(distributed_training.__file__).read_text()

    assert re.search(r"\b[0-9]{12}\b", source) is None
    assert "cr-" not in source
    assert "pg-" not in source
    assert "bev_pos_weights" not in source
    assert "bev_pos_weights" not in (
        distributed_training.train_reactive_stage_ray_4
        .python_interface.inputs
    )
    for task in (
        distributed_training.train_reactive_stage_ray_2,
        distributed_training.train_reactive_stage_ray_4,
        distributed_training.train_reactive_stage_ray_8,
    ):
        assert {
            "trajectory_weight",
            "freeze_bevformer",
        } <= set(task.python_interface.inputs)


def test_four_rank_workflow_runs_one_frozen_multitask_stage():
    node, = distributed_training.wf_train_reactive_nuplan_ray_4.nodes
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith("train_reactive_stage_ray_4")
    assert bindings["epochs"].promise.var == "epochs"
    assert bindings["trajectory_weight"].promise.var == "trajectory_weight"
    assert bindings["bev_weight"].promise.var == "bev_weight"
    assert bindings["route_weight"].promise.var == "route_weight"
    assert bindings["resume_checkpoint"].promise.var == (
        "resume_checkpoint"
    )
    assert (
        bindings["per_rank_batch_size"].promise.var
        == "per_rank_batch_size"
    )
    assert (
        bindings["capacity_block_end_utc"].promise.var
        == "capacity_block_end_utc"
    )
    assert (
        bindings["checkpoint_interval_steps"].promise.var
        == "checkpoint_interval_steps"
    )
    assert bindings["freeze_bevformer"].scalar.primitive.boolean
    assert (
        distributed_training.train_reactive_stage_ray_4.metadata.retries
        == 2
    )
    for task in (
        distributed_training.train_reactive_stage_ray_4,
        distributed_training.train_reactive_stage_ray_8,
    ):
        assert task.metadata.timeout == 0
        assert task.metadata.labels == {
            "kueue.x-k8s.io/queue-name": "p5en-capacity-block",
            "kueue.x-k8s.io/priority-class": "production-high",
        }
    assert distributed_training.ray_ddp_smoke_4.metadata.labels == {
        "kueue.x-k8s.io/queue-name": "p5en-capacity-block",
        "kueue.x-k8s.io/priority-class": "research-low",
    }
    assert (
        distributed_training.ray_ddp_smoke_4.metadata.timeout
        == timedelta(minutes=30)
    )
    assert "capacity_block_end_utc" in (
        distributed_training.ray_ddp_smoke_4.python_interface.inputs
    )


def test_eight_rank_workflow_runs_one_frozen_trajectory_route_stage():
    node, = distributed_training.wf_train_reactive_nuplan_ray_8.nodes
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith("train_reactive_stage_ray_8")
    assert bindings["epochs"].promise.var == "epochs"
    assert bindings["trajectory_weight"].promise.var == "trajectory_weight"
    assert bindings["bev_weight"].promise.var == "bev_weight"
    assert bindings["route_weight"].promise.var == "route_weight"
    assert bindings["resume_checkpoint"].promise.var == (
        "resume_checkpoint"
    )
    assert (
        bindings["capacity_block_end_utc"].promise.var
        == "capacity_block_end_utc"
    )
    assert (
        bindings["checkpoint_interval_steps"].promise.var
        == "checkpoint_interval_steps"
    )
    assert bindings["freeze_bevformer"].scalar.primitive.boolean

    parameters = inspect.signature(
        distributed_training.wf_train_reactive_nuplan_ray_8
    ).parameters
    assert parameters["trajectory_weight"].default == 1.0
    assert parameters["bev_weight"].default == 0.0
    assert parameters["route_weight"].default == 1.0
    assert parameters["per_rank_batch_size"].default == 4
    task_parameters = inspect.signature(
        distributed_training.train_reactive_stage_ray_8.task_function
    ).parameters
    assert task_parameters["per_rank_batch_size"].default == 4
    multistage_parameters = inspect.signature(
        distributed_training.wf_train_reactive_nuplan_l2d_ray_8
    ).parameters
    assert multistage_parameters["per_rank_batch_size"].default == 4


def test_eight_rank_bev_workflow_is_locked_to_segmentation_only():
    node, = distributed_training.wf_train_reactive_nuplan_bev_ray_8.nodes
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith("train_reactive_stage_ray_8")
    assert bindings["epochs"].promise.var == "epochs"
    assert bindings["learning_rate"].promise.var == "learning_rate"
    assert (
        bindings["bev_encoder_learning_rate"].promise.var
        == "bev_encoder_learning_rate"
    )
    assert bindings["trajectory_weight"].scalar.primitive.float_value == 0.0
    assert bindings["bev_weight"].scalar.primitive.float_value == 1.0
    assert bindings["route_weight"].scalar.primitive.float_value == 0.0
    assert not bindings["freeze_bevformer"].scalar.primitive.boolean
    assert bindings["training_scope"].scalar.primitive.string_value == (
        "bev_only"
    )
    assert (
        bindings[
            "bev_repeat_frequency_threshold"
        ].scalar.primitive.float_value
        == pytest.approx(0.05)
    )
    assert (
        bindings["validation_sample_limit"].scalar.primitive.integer
        == 4096
    )
    parameters = inspect.signature(
        distributed_training.wf_train_reactive_nuplan_bev_ray_8
    ).parameters
    assert parameters["epochs"].default == 5
    assert parameters["learning_rate"].default == 1e-4
    assert parameters["bev_encoder_learning_rate"].default == 1e-5
    assert parameters["num_loader_workers"].default == 4
    assert parameters["per_rank_batch_size"].default == 4


def test_two_rank_real_bev_canary_uses_the_production_objective():
    node, = (
        distributed_training
        .wf_train_reactive_nuplan_bev_ray_2_canary
        .nodes
    )
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith("train_reactive_stage_ray_2")
    assert bindings["epochs"].promise.var == "epochs"
    assert bindings["steps_per_epoch"].promise.var == "steps_per_epoch"
    assert bindings["per_rank_batch_size"].scalar.primitive.integer == 1
    assert bindings["precision"].scalar.primitive.string_value == "bf16"
    assert bindings["trajectory_weight"].scalar.primitive.float_value == 0.0
    assert bindings["bev_weight"].scalar.primitive.float_value == 1.0
    assert bindings["route_weight"].scalar.primitive.float_value == 0.0
    assert not bindings["freeze_bevformer"].scalar.primitive.boolean
    assert bindings["training_scope"].scalar.primitive.string_value == (
        "bev_only"
    )


def test_eight_rank_real_bev_canary_matches_production_topology():
    nodes = (
        distributed_training
        .wf_train_reactive_nuplan_bev_ray_8_canary
        .nodes
    )
    node = next(
        item
        for item in nodes
        if item.flyte_entity.name.endswith("train_reactive_stage_ray_8")
    )
    gate = next(
        item
        for item in nodes
        if item.flyte_entity.name.endswith(
            "verify_reactive_bev_canary_training"
        )
    )
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith("train_reactive_stage_ray_8")
    assert bindings["epochs"].promise.var == "epochs"
    assert bindings["steps_per_epoch"].promise.var == "steps_per_epoch"
    assert (
        bindings["capacity_block_end_utc"].promise.var
        == "capacity_block_end_utc"
    )
    assert bindings["per_rank_batch_size"].scalar.primitive.integer == 4
    assert bindings["num_loader_workers"].scalar.primitive.integer == 4
    assert bindings["precision"].scalar.primitive.string_value == "bf16"
    assert bindings["trajectory_weight"].scalar.primitive.float_value == 0.0
    assert bindings["bev_weight"].scalar.primitive.float_value == 1.0
    assert bindings["route_weight"].scalar.primitive.float_value == 0.0
    assert not bindings["freeze_bevformer"].scalar.primitive.boolean
    assert bindings["training_scope"].scalar.primitive.string_value == (
        "bev_only"
    )
    assert bindings[
        "allow_bounded_bev_canary"
    ].scalar.primitive.boolean
    assert (
        bindings["validation_sample_limit"].promise.var
        == "validation_sample_limit"
    )
    assert {item.id for item in gate.upstream_nodes} == {node.id}


def test_nuplan_trajectory_workflow_consumes_a_frozen_bev_parent():
    node, = (
        distributed_training
        .wf_train_reactive_nuplan_from_bev_ray_8
        .nodes
    )
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith("train_reactive_stage_ray_8")
    assert bindings["parent_checkpoint"].promise.var == "bev_checkpoint"
    assert bindings["trajectory_weight"].scalar.primitive.float_value == 1.0
    assert bindings["bev_weight"].scalar.primitive.float_value == 0.0
    assert bindings["route_weight"].scalar.primitive.float_value == 1.0
    assert bindings["freeze_bevformer"].scalar.primitive.boolean
    assert bindings["training_scope"].scalar.primitive.string_value == (
        "multitask"
    )


def test_production_checkpoint_interval_defaults_to_256_steps():
    for task in (
        distributed_training.train_reactive_stage_ray_4,
        distributed_training.train_reactive_stage_ray_8,
    ):
        parameters = inspect.signature(task.task_function).parameters
        assert parameters["checkpoint_interval_steps"].default == 256

    for workflow in (
        distributed_training.wf_train_reactive_nuplan_ray_4,
        distributed_training.wf_train_reactive_nuplan_ray_8,
        distributed_training.wf_train_reactive_nuplan_bev_ray_8,
    ):
        workflow_parameters = inspect.signature(workflow).parameters
        assert (
            workflow_parameters["checkpoint_interval_steps"].default
            == 256
        )


def test_gpu_tasks_use_separate_training_and_validation_capacity():
    for task in (workflows.train_il, workflows.train_offline_rl):
        assert task.pod_template.pod_spec.node_selector == {
            "workload-type": "gpu-validation"
        }
        assert task.pod_template.annotations == {
            "karpenter.sh/do-not-disrupt": "true"
        }
        assert task.metadata.labels == {
            "kueue.x-k8s.io/queue-name": "gpu-validation",
            "kueue.x-k8s.io/priority-class": "research-low",
        }

    for task in (
        workflows.evaluate_il_policy,
        workflows.evaluate_navigation_records,
        workflows.evaluate_rl_policy,
        workflows.evaluate_kitscenes_benchmark_checkpoint,
    ):
        assert task.pod_template.pod_spec.node_selector == {
            "workload-type": "gpu-validation"
        }
        assert task.pod_template.annotations == {
            "karpenter.sh/do-not-disrupt": "true"
        }
        assert task.metadata.labels[
            "kueue.x-k8s.io/queue-name"
        ] == "gpu-validation"


def test_canary_launcher_is_idempotent_and_retries_flyte_admin():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-distributed-canary.yml"
    ).read_text()

    assert "remote.fetch_execution(" in buildspec
    assert "remote.sync_execution(" in buildspec
    assert "FlyteEntityAlreadyExistsException" in buildspec
    assert "FlyteEntityNotExistException" in buildspec
    assert "grpc.StatusCode.UNAVAILABLE" in buildspec
    assert "FLYTE_ADMIN_TRANSIENT_RETRY=" in buildspec
    assert "remote.wait(" not in buildspec


def test_nuplan_acquisition_launcher_uses_private_manifest_and_retries_admin():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-nuplan-acquisition.yml"
    ).read_text()

    assert '"source_manifest": FlyteFile(' in buildspec
    assert 'os.environ["SOURCE_MANIFEST_URI"]' in buildspec
    assert '"datasets_bucket": os.environ["DATASETS_BUCKET"]' in buildspec
    assert "remote.fetch_execution(" in buildspec
    assert "remote.sync_execution(" in buildspec
    assert "FlyteEntityAlreadyExistsException" in buildspec
    assert "FlyteEntityNotExistException" in buildspec
    assert "grpc.StatusCode.UNAVAILABLE" in buildspec
    assert 'WAIT_FOR_COMPLETION: "true"' in buildspec
    assert 'os.environ["WAIT_FOR_COMPLETION"] == "true"' in buildspec
    assert "FLYTE_EXECUTION_DETACHED=true" in buildspec
    assert "remote.wait(" not in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None
    workflow_source = Path(nuplan_dataset.__file__).read_text()
    assert "authorized HTTPS source returned" in workflow_source
    assert "authorized HTTPS source connection failed" in workflow_source
    assert "copy_s3_object_multipart(" in workflow_source
    assert "source_s3.get_object(" not in workflow_source


def test_nuplan_acquisition_workflow_binds_one_dynamic_import_program():
    node, = nuplan_dataset.wf_acquire_nuplan_raw_snapshot.nodes

    assert node.flyte_entity.name.endswith(
        "_acquire_nuplan_raw_snapshot"
    )
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }
    assert bindings["source_manifest"].promise.var == "source_manifest"
    assert bindings["datasets_bucket"].promise.var == "datasets_bucket"
    assert bindings["concurrency"].promise.var == "concurrency"


def test_nuplan_snapshot_pack_uses_front_camera_cache_and_full_default():
    node, = nuplan_dataset.wf_pack_nuplan_snapshot_reactive_dataset.nodes
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith(
        "pack_nuplan_snapshot_reactive_dataset"
    )
    assert node.flyte_entity.metadata.cache_version == (
        "nuplan-snapshot-pack-v13-manifest-v10"
    )
    assert node.flyte_entity.metadata.retries == 1
    assert (
        bindings["limit_total_scenarios"].promise.var
        == "limit_total_scenarios"
    )
    buildspec = (
        Path(nuplan_dataset.__file__).parents[1]
        / "buildspec-launch-nuplan-pack.yml"
    ).read_text()
    assert 'LIMIT_TOTAL_SCENARIOS: "2048"' in buildspec
    assert 'IMAGE_SIZE: "512"' in buildspec
    assert "Platform.pipelines.nuplan_dataset." in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None


def test_nuplan_full_pack_plan_covers_every_train_sensor_group():
    archives = [
        {
            "archive_id": "maps-v1.1",
            "component": "maps",
            "filename": "nuplan-maps-v1.1.zip",
        },
        {
            "archive_id": "db-train_a",
            "component": "database",
            "filename": "a.zip",
        },
        {
            "archive_id": "db-train_b",
            "component": "database",
            "filename": "b.zip",
        },
    ]
    for group_index in range(
        nuplan_dataset.NUPLAN_FULL_TRAIN_GROUP_COUNT
    ):
        for modality in ("camera", "lidar"):
            archives.append({
                "archive_id": (
                    "sensor-train-train_"
                    f"{modality}_{group_index}"
                ),
                "component": "sensor_blobs",
                "filename": f"{modality}_{group_index}.zip",
            })
    sensor_groups = {
        group_index: (
            f"log-{group_index}-0",
            f"log-{group_index}-1",
        )
        for group_index in range(
            nuplan_dataset.NUPLAN_FULL_TRAIN_GROUP_COUNT
        )
    }
    database_logs = {
        "db-train_a": tuple(
            log_name
            for group_index, log_names in sensor_groups.items()
            if group_index % 2 == 0
            for log_name in log_names
        ),
        "db-train_b": tuple(
            log_name
            for group_index, log_names in sensor_groups.items()
            if group_index % 2 == 1
            for log_name in log_names
        ),
    }

    archive_sets, limits = (
        nuplan_dataset._build_nuplan_train_pack_plan(
            {
                "archives": archives,
                "map_version": "nuplan-maps-v1.1",
            },
            sensor_groups,
            database_logs,
            total_scenario_limit=131_072,
        )
    )

    assert len(archive_sets) == 43
    assert archive_sets[0] == [
        "maps-v1.1",
        "db-train_a",
        "sensor-train-train_camera_0",
        "sensor-train-train_lidar_0",
    ]
    assert archive_sets[1][1] == "db-train_b"
    assert len(limits) == 43
    assert sum(limits) == 131_072
    assert max(limits) - min(limits) <= 1


def test_nuplan_group_budget_requires_and_preserves_log_coverage():
    log_counts = [1] + [100] * (
        nuplan_dataset.NUPLAN_FULL_TRAIN_GROUP_COUNT - 1
    )
    total_logs = sum(log_counts)

    with pytest.raises(ValueError, match="each of the"):
        nuplan_dataset._allocate_nuplan_group_scenario_limits(
            log_counts,
            total_logs - 1,
        )

    limits = nuplan_dataset._allocate_nuplan_group_scenario_limits(
        log_counts,
        total_logs,
    )
    assert limits == log_counts


def test_nuplan_s3_range_reader_supports_zip_central_directory():
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("log-a.db", b"sqlite-a")
        archive.writestr("nested/log-b.db", b"sqlite-b")
    payload = archive_buffer.getvalue()

    class Body:
        def __init__(self, data: bytes):
            self.data = data

        def read(self) -> bytes:
            return self.data

    class S3:
        ranges: list[str] = []

        def head_object(self, *, Bucket: str, Key: str):
            assert (Bucket, Key) == ("datasets", "db.zip")
            return {"ContentLength": len(payload)}

        def get_object(
            self,
            *,
            Bucket: str,
            Key: str,
            Range: str,
        ):
            assert (Bucket, Key) == ("datasets", "db.zip")
            self.ranges.append(Range)
            start_text, end_text = Range.removeprefix("bytes=").split("-")
            return {
                "Body": Body(
                    payload[int(start_text):int(end_text) + 1]
                )
            }

    s3 = S3()
    reader = nuplan_dataset._S3RangeReader(
        s3,
        bucket="datasets",
        key="db.zip",
        size=len(payload),
        archive_id="db-train-test",
    )
    with zipfile.ZipFile(reader) as archive:
        assert archive.namelist() == [
            "log-a.db",
            "nested/log-b.db",
        ]
        assert archive.read("nested/log-b.db") == b"sqlite-b"

    assert s3.ranges
    assert all(value.startswith("bytes=") for value in s3.ranges)


def test_nuplan_s3_range_reader_rejects_invalid_object_reads():
    class Body:
        def read(self) -> bytes:
            return b"x"

    class S3:
        content_length = 4

        def head_object(self, **_kwargs):
            return {"ContentLength": self.content_length}

        def get_object(self, **_kwargs):
            return {"Body": Body()}

    s3 = S3()
    with pytest.raises(ValueError, match="archive size changed"):
        nuplan_dataset._S3RangeReader(
            s3,
            bucket="datasets",
            key="db.zip",
            size=5,
            archive_id="db-train-test",
        )

    reader = nuplan_dataset._S3RangeReader(
        s3,
        bucket="datasets",
        key="db.zip",
        size=4,
        archive_id="db-train-test",
    )
    with pytest.raises(ValueError, match="negative S3 range seek"):
        reader.seek(-1)
    with pytest.raises(ValueError, match="unsupported seek mode"):
        reader.seek(0, 99)
    with pytest.raises(OSError, match="short S3 range read"):
        reader.read(2)


def test_nuplan_train_inventory_parser_rejects_incomplete_groups():
    payload = "\n".join(
        f"File group: {index}\nlog-{index}"
        for index in range(
            nuplan_dataset.NUPLAN_FULL_TRAIN_GROUP_COUNT
        )
    )
    parsed = nuplan_dataset._parse_nuplan_train_sensor_inventory(
        payload
    )
    assert parsed[0] == ("log-0",)
    assert parsed[42] == ("log-42",)

    with pytest.raises(ValueError, match="empty group"):
        nuplan_dataset._parse_nuplan_train_sensor_inventory(
            payload.rsplit("\n", 1)[0]
        )


def test_nuplan_full_pack_workflow_binds_sharded_dynamic_program():
    node, = (
        nuplan_dataset
        .wf_pack_nuplan_snapshot_reactive_dataset_sharded
        .nodes
    )
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith(
        "_pack_nuplan_snapshot_reactive_dataset_sharded"
    )
    assert (
        bindings["total_scenario_limit"].promise.var
        == "total_scenario_limit"
    )
    assert bindings["concurrency"].promise.var == "concurrency"
    assert (
        nuplan_dataset.NUPLAN_FULL_PACK_EPHEMERAL_STORAGE
        == "1200Gi"
    )


def test_reactive_nuplan_launcher_uses_registered_eight_rank_workflow():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-reactive-nuplan.yml"
    ).read_text(encoding="utf-8")

    assert (
        "Platform.pipelines.distributed_training."
        "wf_train_reactive_nuplan_bev_ray_8"
    ) in buildspec
    assert 'EPOCHS: "5"' in buildspec
    assert 'CANARY_EPOCHS: "2"' in buildspec
    assert 'test "${CANARY_EPOCHS}" = "2"' in buildspec
    assert 'STEPS_PER_EPOCH: "128"' in buildspec
    assert 'VALIDATION_SAMPLE_LIMIT: "1024"' in buildspec
    assert 'LEARNING_RATE: "1e-4"' in buildspec
    assert 'BEV_ENCODER_LEARNING_RATE: "1e-5"' in buildspec
    assert 'PRECISION: "bf16"' in buildspec
    assert 'VAL_FRACTION: "0.1"' in buildspec
    assert 'NUM_LOADER_WORKERS: "4"' in buildspec
    assert 'PER_RANK_BATCH_SIZE: "4"' in buildspec
    assert "'^(1|2|4)$'" in buildspec
    assert 'RESUME_CHECKPOINT_URI: ""' in buildspec
    assert 'TRAJECTORY_WEIGHT: "0.0"' in buildspec
    assert 'BEV_WEIGHT: "1.0"' in buildspec
    assert 'ROUTE_WEIGHT: "0.0"' in buildspec
    assert "NUPLAN_DATASET_URIS_URI" in buildspec
    assert "len(dataset_uris) != 43" in buildspec
    assert "len(set(dataset_uris)) != 43" in buildspec
    assert "FlyteDirectory(uri) for uri in dataset_uris" in buildspec
    assert "describe-capacity-reservations" in buildspec
    assert "CAPACITY_BLOCK_MINIMUM_REMAINING_SECONDS" in buildspec
    assert "Name=state,Values=active" in buildspec
    assert "AvailableInstanceCount" not in buildspec
    assert "exactly one active tagged P5 Capacity Block" in buildspec
    assert 'capacity_block["ReservationType"] != "capacity-block"' in buildspec
    assert "active P5 Capacity Block is missing EndDate" in buildspec
    assert "Capacity Block instance type must be p5en or p5" in buildspec
    assert "P5 Capacity Block must be in us-west-2a" in buildspec
    assert "42 * 60 * 60" in buildspec
    assert '"capacity_block_end_utc": capacity_block_end_utc' in buildspec
    assert (
        '"per_rank_batch_size": int('
        in buildspec
    )
    assert '"resume_checkpoint": (' in buildspec
    assert (
        '"wf_train_reactive_nuplan_bev_ray_8_canary"'
        in buildspec
    )
    assert '"steps_per_epoch": int(' in buildspec
    assert '"validation_sample_limit": int(' in buildspec
    assert '"epochs": int(os.environ["CANARY_EPOCHS"])' in buildspec
    codebuild_terraform = (
        Path(distributed_training.__file__).parents[1]
        / "infra/modules/codebuild/main.tf"
    ).read_text(encoding="utf-8")
    assert '"ec2:DescribeCapacityReservations"' in codebuild_terraform
    assert '"ec2:PurchaseCapacityBlock"' not in codebuild_terraform
    assert "NUPLAN_DATASET_URI:" not in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None


def test_flyte_resource_quota_tracks_all_gpu_capacity():
    values_path = (
        Path(distributed_training.__file__).parents[1]
        / "helm-values/flyte-core-eks.yaml"
    )
    values_text = values_path.read_text(encoding="utf-8")
    values = yaml.safe_load(values_text)
    custom_data = values["cluster_resource_manager"]["config"][
        "cluster_resources"
    ]["customData"]
    domains = {
        domain: {
            entry_name: entry["value"]
            for item in settings
            for entry_name, entry in item.items()
        }
        for item in custom_data
        for domain, settings in item.items()
    }

    assert domains["development"]["projectQuotaGpu"] == "18"
    assert domains["staging"]["projectQuotaGpu"] == "0"
    assert domains["production"]["projectQuotaGpu"] == "0"
    assert "limits.nvidia.com/gpu: {{ projectQuotaGpu }}" not in values_text
    assert "requests.nvidia.com/gpu: {{ projectQuotaGpu }}" in values_text


def test_nuplan_pack_worker_count_caps_full_and_limited():
    assert nuplan_dataset._nuplan_pack_worker_count(2, 0) == 2
    assert nuplan_dataset._nuplan_pack_worker_count(20, 0) == 8
    assert nuplan_dataset._nuplan_pack_worker_count(20, 64) == 8
    assert nuplan_dataset._nuplan_pack_worker_count(20, 4) == 4


def test_two_rank_canary_wires_both_stages_and_gate():
    nodes = distributed_training.wf_reactive_multistage_ray_2_canary.nodes

    assert len(nodes) == 5
    stage_a = nodes[2]
    stage_b = nodes[3]
    gate = nodes[4]
    stage_b_bindings = {
        binding.var: binding.binding
        for binding in stage_b.bindings
    }
    stage_a_bindings = {
        binding.var: binding.binding
        for binding in stage_a.bindings
    }
    assert (
        stage_a_bindings["trajectory_weight"]
        .scalar.primitive.float_value
        == 0.1
    )
    assert (
        stage_a_bindings["bev_weight"].scalar.primitive.float_value
        == 1.0
    )
    assert (
        stage_a_bindings["route_weight"].scalar.primitive.float_value
        == 0.1
    )
    assert stage_a_bindings[
        "freeze_bevformer"
    ].scalar.primitive.boolean
    assert stage_b_bindings[
        "freeze_bevformer"
    ].scalar.primitive.boolean
    assert stage_b_bindings["parent_checkpoint"].promise.node_id == (
        stage_a.id
    )
    assert {node.id for node in gate.upstream_nodes} == {
        stage_a.id,
        stage_b.id,
    }


def test_canary_gate_requires_loss_decrease_and_stage_b_bev_off(tmp_path):
    def metadata(history, name):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"history": history}))
        return distributed_training.FlyteFile(str(path))

    common = {
        "train_gradient_front_gate_pre_clip_norm": 0.1,
        "train_route_reconstruction": 0.2,
        "train_trajectory": 1.0,
        "validation_ade_6p4s_m": 2.0,
        "validation_selection_score": 0.4,
    }
    stage_a_metrics = {
        **{
            f"bev_pos_weight_{index}": float(index + 2)
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"validation_bev_{class_name}_{suffix}": value
            for class_name in BEV_SEGMENTATION_CLASSES
            for suffix, value in (
                    ("average_precision", 0.5),
                    ("positive_cells", 10.0),
                    ("recall_at_0p5", 0.5),
                )
        },
    }
    stage_a = metadata(
        [
            {
                **common,
                **stage_a_metrics,
                "train_bev_segmentation": 0.5,
                "train_bev_segmentation_bce": 0.6,
                "train_bev_segmentation_dice": 0.4,
                "train_total": 1.7,
            },
            {
                **common,
                **stage_a_metrics,
                "train_bev_segmentation": 0.4,
                "train_bev_segmentation_bce": 0.5,
                "train_bev_segmentation_dice": 0.3,
                "train_total": 1.5,
            },
        ],
        "stage-a",
    )
    stage_b = metadata(
        [
            {
                **common,
                "train_bev_segmentation": 0.0,
                "train_bev_segmentation_bce": 0.0,
                "train_bev_segmentation_dice": 0.0,
                "train_total": 1.2,
            },
            {
                **common,
                "train_bev_segmentation": 0.0,
                "train_bev_segmentation_bce": 0.0,
                "train_bev_segmentation_dice": 0.0,
                "train_total": 1.1,
            },
        ],
        "stage-b",
    )

    report = (
        distributed_training.verify_reactive_canary_training.task_function(
            stage_a_metadata=stage_a,
            stage_b_metadata=stage_b,
        )
    )

    assert json.loads(Path(report.path).read_text())["thresholds_pass"]


def test_bev_canary_gate_requires_learning_and_class_coverage(tmp_path):
    class_metrics = {
        f"validation_bev_{class_name}_{suffix}": value
        for class_name in BEV_SEGMENTATION_CLASSES
        for suffix, value in (
            ("ap_lift", 0.2),
            ("ap_lift_bootstrap_lower_95", 0.1),
            ("ap_lift_bootstrap_upper_95", 0.3),
            ("average_precision", 0.3),
            ("best_iou_on_validation_set", 0.25),
            ("best_iou_precision_on_validation_set", 0.4),
            ("best_iou_recall_on_validation_set", 0.5),
            ("best_iou_threshold_on_validation_set", 0.6),
            ("positive_cells", 10.0),
            ("positive_prevalence", 0.01),
            ("supported", 1.0),
        )
    }
    class_metrics[
        "validation_bev_vulnerable_road_user_ap_lift"
    ] = 0.028
    class_metrics[
        "validation_bev_vulnerable_road_user_"
        "ap_lift_bootstrap_lower_95"
    ] = 0.01
    class_metrics[
        "validation_bev_other_obstacle_ap_lift"
    ] = 0.0015
    class_metrics[
        "validation_bev_other_obstacle_ap_lift_bootstrap_lower_95"
    ] = 0.0002
    for class_name, iou, precision, prevalence in (
        ("vulnerable_road_user", 0.004, 0.006, 0.0008),
        ("other_obstacle", 0.001, 0.0015, 0.0004),
    ):
        class_metrics[
            f"validation_bev_{class_name}_best_iou_on_validation_set"
        ] = iou
        class_metrics[
            f"validation_bev_{class_name}_"
            "best_iou_precision_on_validation_set"
        ] = precision
        class_metrics[
            f"validation_bev_{class_name}_positive_prevalence"
        ] = prevalence
    common = {
        **class_metrics,
        **{
            f"bev_pos_weight_{index}": float(index + 2)
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"bev_class_weight_{index}": (
                2.5 if index >= 6 else 0.5
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"bev_positive_pair_frequency_{index}": (
                0.02 if index >= 6 else 0.5
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"bev_repeat_factor_{index}": (
                4 if index >= 6 else 1
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"train_bev_logit_gradient_l1_class_{index}": 0.1
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"train_bev_logit_gradient_share_class_{index}": 0.125
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        "train_bev_logit_gradient_diagnostic_batches": 16.0,
        "bounded_bev_canary": 1.0,
        "train_bev_segmentation_bce": 0.6,
        "train_bev_segmentation_dice": 0.7,
        "train_gradient_camera_pre_clip_norm": 0.2,
        "train_gradient_front_gate_pre_clip_norm": 0.1,
        "train_loader_restarts": 0.0,
        "bev_rank_min_full_microbatch_capacity": 32.0,
        "bev_rank_max_full_microbatch_capacity": 36.0,
        "bev_rank_min_drop_last_fraction": 0.0,
        "bev_rank_max_drop_last_fraction": 0.01,
        "bev_rank_min_importance_scale": 0.8,
        "bev_rank_max_importance_scale": 1.2,
        "bev_rank_min_optimizer_tail_fraction": 0.0,
        "bev_rank_max_optimizer_tail_fraction": 0.09,
        "bev_rank_min_truncation_fraction": 0.0,
        "bev_rank_max_truncation_fraction": 0.1,
        "validation_bev_all_classes_supported": 1.0,
        "validation_bev_dynamic_macro_ap_lift": 0.2,
        "validation_bev_min_ap_lift": 0.0015,
        "validation_bev_static_macro_ap_lift": 0.2,
        "validation_selection_score": 0.2,
    }
    improved_class_metrics = {
        key: (
            value + 0.01
            if key.endswith("_ap_lift")
            else value + 0.12
            if key.endswith("_best_iou_threshold_on_validation_set")
            else value
        )
        for key, value in class_metrics.items()
    }
    improved_class_metrics[
        "validation_bev_vulnerable_road_user_ap_lift"
    ] = 0.027
    improved_class_metrics[
        "validation_bev_other_obstacle_ap_lift"
    ] = 0.0014
    path = tmp_path / "bev-canary.json"
    path.write_text(
        json.dumps({
            "history": [
                {
                    **common,
                    "epoch": 1,
                    "is_best": 1,
                    "train_bev_segmentation": 0.8,
                },
                {
                    **common,
                    **improved_class_metrics,
                    "epoch": 2,
                    "is_best": 1,
                    "train_bev_segmentation": 0.7,
                    "validation_selection_score": 0.201,
                },
            ]
        })
    )

    report = (
        distributed_training
        .verify_reactive_bev_canary_training
        .task_function(
            metadata=distributed_training.FlyteFile(str(path)),
        )
    )

    payload = json.loads(Path(report.path).read_text())
    assert payload["thresholds_pass"]
    assert payload["all_classes_supported"]
    assert payload["all_classes_beat_prevalence"]
    assert payload["all_classes_have_useful_operating_points"]
    assert payload["schema_version"] == "reactive_bev_canary_report_v7"
    assert payload["operating_point_requirement"] == (
        "beats_prevalence_with_positive_recall_v1"
    )
    assert payload["production_quality_guard_deferred"] is True
    assert payload["class_order"] == list(BEV_SEGMENTATION_CLASSES)
    assert payload["second_epoch_selection_gain"] == pytest.approx(0.001)
    assert payload["selection_gain_minimum"] == pytest.approx(0.001)
    assert payload["threshold_stability_gate_enforced"] is False
    assert payload[
        "ap_lift_absolute_regression_tolerance"
    ] == pytest.approx(0.001)
    assert payload[
        "rare_ap_lift_relative_regression_tolerance"
    ] == pytest.approx(0.1)
    assert payload["selected_checkpoint_epoch"] == 2
    assert payload["positive_weights_by_class"] == {
        class_name: float(index + 2)
        for index, class_name in enumerate(BEV_SEGMENTATION_CLASSES)
    }
    assert payload["repeat_factors_by_class"] == {
        class_name: (4 if index >= 6 else 1)
        for index, class_name in enumerate(BEV_SEGMENTATION_CLASSES)
    }
    assert payload["ap_lift_regression_tolerance_by_class"] == {
        "drivable_area": pytest.approx(0.001),
        "lane_boundary": pytest.approx(0.001),
        "intersection": pytest.approx(0.001),
        "crosswalk": pytest.approx(0.001),
        "stop_line": pytest.approx(0.001),
        "vehicle": pytest.approx(0.001),
        "vulnerable_road_user": pytest.approx(0.001),
        "other_obstacle": pytest.approx(0.00015),
    }


@pytest.mark.parametrize(
    (
        "selection_score",
        "regressed_class",
        "first_class_lift",
        "second_class_lift",
        "threshold_drift",
        "match",
    ),
    (
        (
            0.2005,
            None,
            None,
            None,
            0.0,
            "selection score did not improve enough",
        ),
        (0.21, "vehicle", 0.2, 0.19, 0.0, "AP lift regressed"),
        (
            0.21,
            "other_obstacle",
            0.02,
            0.0175,
            0.0,
            "AP lift regressed",
        ),
        (
            0.21,
            None,
            None,
            None,
            0.5,
            "invalid calibrated threshold",
        ),
    ),
)
def test_bev_canary_gate_rejects_weak_second_epoch(
    tmp_path,
    selection_score,
    regressed_class,
    first_class_lift,
    second_class_lift,
    threshold_drift,
    match,
):
    def epoch(*, loss, selection, lift, threshold):
        values = {
            "epoch": 1,
            "is_best": 1,
            "bounded_bev_canary": 1.0,
            "train_bev_segmentation": loss,
            "train_bev_segmentation_bce": 0.6,
            "train_bev_segmentation_dice": 0.7,
            "train_bev_logit_gradient_diagnostic_batches": 16.0,
            "train_gradient_camera_pre_clip_norm": 0.2,
            "train_gradient_front_gate_pre_clip_norm": 0.1,
            "train_loader_restarts": 0.0,
            "bev_rank_min_full_microbatch_capacity": 32.0,
            "bev_rank_max_full_microbatch_capacity": 36.0,
            "bev_rank_min_drop_last_fraction": 0.0,
            "bev_rank_max_drop_last_fraction": 0.01,
            "bev_rank_min_importance_scale": 0.8,
            "bev_rank_max_importance_scale": 1.2,
            "bev_rank_min_optimizer_tail_fraction": 0.0,
            "bev_rank_max_optimizer_tail_fraction": 0.09,
            "bev_rank_min_truncation_fraction": 0.0,
            "bev_rank_max_truncation_fraction": 0.1,
            "validation_bev_all_classes_supported": 1.0,
            "validation_bev_dynamic_macro_ap_lift": lift,
            "validation_bev_min_ap_lift": lift,
            "validation_bev_static_macro_ap_lift": lift,
            "validation_selection_score": selection,
        }
        for index, class_name in enumerate(BEV_SEGMENTATION_CLASSES):
            values.update({
                f"bev_pos_weight_{index}": float(index + 2),
                f"bev_class_weight_{index}": (
                    2.5 if index >= 6 else 0.5
                ),
                f"bev_positive_pair_frequency_{index}": (
                    0.02 if index >= 6 else 0.5
                ),
                f"bev_repeat_factor_{index}": 1,
                f"train_bev_logit_gradient_l1_class_{index}": 0.1,
                f"train_bev_logit_gradient_share_class_{index}": 0.125,
                f"validation_bev_{class_name}_ap_lift": lift,
                f"validation_bev_{class_name}_"
                "ap_lift_bootstrap_lower_95": lift * 0.5,
                f"validation_bev_{class_name}_"
                "ap_lift_bootstrap_upper_95": lift * 1.5,
                f"validation_bev_{class_name}_average_precision": 0.3,
                f"validation_bev_{class_name}_"
                "best_iou_on_validation_set": 0.25,
                f"validation_bev_{class_name}_"
                "best_iou_precision_on_validation_set": 0.4,
                f"validation_bev_{class_name}_"
                "best_iou_recall_on_validation_set": 0.5,
                f"validation_bev_{class_name}_"
                "best_iou_threshold_on_validation_set": threshold,
                f"validation_bev_{class_name}_positive_cells": 10.0,
                f"validation_bev_{class_name}_positive_prevalence": 0.01,
                f"validation_bev_{class_name}_supported": 1.0,
            })
        return values

    first = epoch(
        loss=0.8,
        selection=0.2,
        lift=0.2,
        threshold=0.6,
    )
    second = epoch(
        loss=0.7,
        selection=selection_score,
        lift=0.21,
        threshold=0.6 + threshold_drift,
    )
    second["epoch"] = 2
    if regressed_class is not None:
        metric_name = (
            f"validation_bev_{regressed_class}_ap_lift"
        )
        first[metric_name] = first_class_lift
        second[metric_name] = second_class_lift
    path = tmp_path / "weak-bev-canary.json"
    path.write_text(json.dumps({"history": [first, second]}))

    with pytest.raises(ValueError, match=match):
        (
            distributed_training.verify_reactive_bev_canary_training
            .task_function(
                metadata=distributed_training.FlyteFile(str(path)),
            )
        )


@pytest.mark.parametrize(
    ("epoch_numbers", "match"),
    (
        ((1, 1), "unique and contiguous"),
        ((2, 3), "unique and contiguous"),
        ((1, 3), "unique and contiguous"),
        ((True, 2), "invalid epoch number"),
        ((1.5, 2), "invalid epoch number"),
        ((None, 2), "invalid epoch number"),
    ),
)
def test_bev_canary_gate_rejects_invalid_epoch_history(
    tmp_path,
    epoch_numbers,
    match,
):
    class_metrics = {
        f"validation_bev_{class_name}_{suffix}": value
        for class_name in BEV_SEGMENTATION_CLASSES
        for suffix, value in (
            ("ap_lift", 0.2),
            ("ap_lift_bootstrap_lower_95", 0.1),
            ("ap_lift_bootstrap_upper_95", 0.3),
            ("average_precision", 0.3),
            ("best_iou_on_validation_set", 0.25),
            ("best_iou_precision_on_validation_set", 0.4),
            ("best_iou_recall_on_validation_set", 0.5),
            ("best_iou_threshold_on_validation_set", 0.6),
            ("positive_cells", 10.0),
            ("supported", 1.0),
        )
    }
    common = {
        **class_metrics,
        **{
            f"bev_pos_weight_{index}": float(index + 2)
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"bev_class_weight_{index}": (
                2.5 if index >= 6 else 0.5
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"bev_positive_pair_frequency_{index}": (
                0.02 if index >= 6 else 0.5
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"bev_repeat_factor_{index}": 1
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"train_bev_logit_gradient_l1_class_{index}": 0.1
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"train_bev_logit_gradient_share_class_{index}": 0.125
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        "is_best": 1,
        "train_bev_logit_gradient_diagnostic_batches": 16.0,
        "train_bev_segmentation_bce": 0.6,
        "train_bev_segmentation_dice": 0.7,
        "train_gradient_camera_pre_clip_norm": 0.2,
        "train_gradient_front_gate_pre_clip_norm": 0.1,
        "train_loader_restarts": 0.0,
        "bev_rank_min_full_microbatch_capacity": 32.0,
        "bev_rank_max_full_microbatch_capacity": 36.0,
        "bev_rank_min_drop_last_fraction": 0.0,
        "bev_rank_max_drop_last_fraction": 0.01,
        "bev_rank_min_importance_scale": 0.8,
        "bev_rank_max_importance_scale": 1.2,
        "bev_rank_min_optimizer_tail_fraction": 0.0,
        "bev_rank_max_optimizer_tail_fraction": 0.09,
        "bev_rank_min_truncation_fraction": 0.0,
        "bev_rank_max_truncation_fraction": 0.1,
        "validation_bev_all_classes_supported": 1.0,
        "validation_bev_dynamic_macro_ap_lift": 0.2,
        "validation_bev_min_ap_lift": 0.2,
        "validation_bev_static_macro_ap_lift": 0.2,
        "validation_selection_score": 0.2,
    }
    history = [
        {
            **common,
            "epoch": epoch_numbers[0],
            "train_bev_segmentation": 0.8,
        },
        {
            **common,
            "epoch": epoch_numbers[1],
            "train_bev_segmentation": 0.7,
            "validation_selection_score": 0.21,
        },
    ]
    path = tmp_path / "invalid-epoch-bev-canary.json"
    path.write_text(json.dumps({"history": history}))

    with pytest.raises(ValueError, match=match):
        (
            distributed_training.verify_reactive_bev_canary_training
            .task_function(
                metadata=distributed_training.FlyteFile(str(path)),
            )
        )
