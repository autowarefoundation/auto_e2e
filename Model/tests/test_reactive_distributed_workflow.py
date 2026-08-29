"""Flyte wiring for distributed Reactive Stage A and Stage B."""

from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

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
                "assert module.BEV_POS_WEIGHT_CAP == 2048.0; "
                "print(module.BEV_POS_WEIGHT_CAP)"
            ),
        ],
        cwd=repository_root / "Platform" / "pipelines",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "2048.0"


def test_shared_pack_cache_includes_camera_resolution_contract():
    assert (
        workflows.PACK_CACHE_VERSION
        == "pack-v3-v1-v10-v6-camera512"
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


def test_reviewed_ray_topologies_have_fixed_worker_groups():
    assert (
        distributed_training.RAY_2.worker_node_config[0].replicas
        == 2
    )
    assert (
        distributed_training.RAY_8.worker_node_config[0].replicas
        == 8
    )
    for config in (
        distributed_training.RAY_2,
        distributed_training.RAY_4,
        distributed_training.RAY_REACTIVE_4,
        distributed_training.RAY_8,
    ):
        workers = config.worker_node_config[0]
        assert workers.min_replicas == workers.replicas
        assert workers.max_replicas == workers.replicas
        assert config.enable_autoscaling is False


def test_four_rank_performance_capacity_matches_ray_contract():
    worker = distributed_training.RAY_REACTIVE_4.worker_node_config[0]
    worker_spec = worker.pod_template.pod_spec
    assert worker.replicas == 4
    assert worker.ray_start_params["num-cpus"] == "3"
    assert worker_spec.node_selector == {
        "workload-type": "gpu-performance"
    }
    assert worker_spec.containers[0].resources.requests == {
        "cpu": "3",
        "memory": "12Gi",
        "nvidia.com/gpu": "1",
    }
    assert (
        distributed_training.train_reactive_stage_ray_4.metadata.labels[
            "kueue.x-k8s.io/queue-name"
        ]
        == "gpu-performance"
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
    reserved_class = node_classes[
        "auto-e2e-gpu-performance-reserved"
    ]["spec"]
    assert reserved_class["capacityReservationSelectorTerms"] == [
        {
            "ownerID": "REPLACE_WITH_AWS_ACCOUNT_ID",
            "tags": {"Name": "auto-e2e-gpu-canary"},
        }
    ]
    assert reserved_class["placementGroupSelector"] == {
        "name": "auto-e2e-distributed-training-pg"
    }
    assert "capacityReservationSelectorTerms" not in node_classes[
        "auto-e2e-gpu-performance-ondemand"
    ]["spec"]
    training_class = node_classes["auto-e2e-gpu-training"]["spec"]
    assert training_class["capacityReservationSelectorTerms"] == [{
        "ownerID": "REPLACE_WITH_AWS_ACCOUNT_ID",
        "tags": {"Name": "auto-e2e-distributed-training"},
    }]

    node_pools = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodepool.yaml"
            ).read_text()
        )
    }
    reserved_pool = node_pools["gpu-performance-reserved"]["spec"]
    assert reserved_pool["weight"] == 100
    assert reserved_pool["limits"] == {
        "cpu": "32",
        "memory": "256Gi",
        "nodes": "4",
        "nvidia.com/gpu": "4",
    }
    assert reserved_pool["template"]["metadata"]["labels"] == {
        "workload-type": "gpu-performance"
    }
    requirements = {
        item["key"]: item["values"]
        for item in reserved_pool["template"]["spec"]["requirements"]
    }
    assert requirements["node.kubernetes.io/instance-type"] == [
        "g6.2xlarge"
    ]
    assert requirements["karpenter.sh/capacity-type"] == ["reserved"]
    training_pool = node_pools["gpu-training"]["spec"]
    assert training_pool["template"]["spec"]["nodeClassRef"]["name"] == (
        "auto-e2e-gpu-training"
    )
    training_requirements = {
        item["key"]: item["values"]
        for item in training_pool["template"]["spec"]["requirements"]
    }
    assert training_requirements["karpenter.sh/capacity-type"] == [
        "reserved"
    ]
    assert training_pool["limits"] == {
        "cpu": "64",
        "memory": "512Gi",
        "nodes": "4",
        "nvidia.com/gpu": "4",
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
        ("ClusterQueue", "gpu-performance-queue", None)
    ]["spec"]
    gpu_group = next(
        group
        for group in performance_queue["resourceGroups"]
        if group["coveredResources"] == ["nvidia.com/gpu"]
    )
    assert gpu_group["flavors"][0]["resources"] == [
        {"name": "nvidia.com/gpu", "nominalQuota": "4"}
    ]
    assert (
        "LocalQueue",
        "gpu-performance",
        "auto-e2e-development",
    ) in queue_objects

    deploy_script = (platform_root / "infra/post-apply.sh").read_text()
    render_index = deploy_script.index(
        "s/REPLACE_WITH_AWS_ACCOUNT_ID/${ACCOUNT}/g"
    )
    node_pool_index = deploy_script.index(
        "karpenter-nodepools/gpu-nodepool.yaml"
    )
    assert render_index < node_pool_index
    assert "kueue-config/kueue-objects.yaml" in deploy_script
    assert "nodepool/gpu-performance-reserved" in deploy_script
    assert re.search(r"\b[0-9]{12}\b", deploy_script) is None
    for relative_path in re.findall(
        r"\.\./k8s/[A-Za-z0-9_./-]+\.yaml",
        deploy_script,
    ):
        assert (
            platform_root
            / "infra"
            / relative_path
        ).resolve().is_file()


def test_reactive_ray_cpu_contract_has_one_source_of_truth():
    for config in (
        distributed_training.RAY_2,
        distributed_training.RAY_REACTIVE_4,
        distributed_training.RAY_8,
    ):
        worker = config.worker_node_config[0]
        cpu = worker.ray_start_params["num-cpus"]
        resources = (
            worker.pod_template.pod_spec
            .containers[0].resources
        )
        assert resources.requests["cpu"] == cpu
        assert resources.limits["cpu"] == cpu
        assert int(cpu) == distributed_training._reactive_worker_cpus(
            worker.replicas
        )


def test_ray_tasks_serialize_the_resolved_storage_path():
    expected_environment = {
        "AWS_DEFAULT_REGION": "us-west-2",
        "AUTO_E2E_RAY_STORAGE_PATH": (
            distributed_training.RAY_STORAGE_PATH
        ),
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
    assert bindings["freeze_bevformer"].scalar.primitive.boolean


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


def test_reactive_nuplan_launcher_uses_registered_four_rank_workflow():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-reactive-nuplan.yml"
    ).read_text(encoding="utf-8")

    assert (
        "Platform.pipelines.distributed_training."
        "wf_train_reactive_nuplan_ray_4"
    ) in buildspec
    assert 'EPOCHS: "3"' in buildspec
    assert 'PRECISION: "bf16"' in buildspec
    assert "NUPLAN_DATASET_URI" in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None


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
                ("recall", 0.5),
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
