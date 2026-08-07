"""Flyte wiring for distributed Reactive Stage A and Stage B."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

pytest.importorskip("flytekit")

from flytekit.types.directory import FlyteDirectory

from Platform.pipelines import distributed_training, workflows


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
        distributed_training.RAY_8,
    ):
        workers = config.worker_node_config[0]
        assert workers.min_replicas == workers.replicas
        assert workers.max_replicas == workers.replicas
        assert config.enable_autoscaling is False


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
        binding.var: binding.binding
        for binding in stage_a.bindings
    }
    stage_b_bindings = {
        binding.var: binding.binding
        for binding in stage_b.bindings
    }
    assert stage_a_bindings["stage"].scalar.primitive.string_value == (
        "nuplan_full"
    )
    assert stage_b_bindings["stage"].scalar.primitive.string_value == (
        "l2d_continuation"
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

    assert "381491877296" not in source
    assert "cr-" not in source
    assert "pg-" not in source


def test_l2d_reactive_pack_workflow_binds_osm_and_target_contract():
    node, = workflows.wf_pack_l2d_reactive_dataset.nodes
    assert node.flyte_entity.name.endswith("wf_create_dataset_sharded")
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }
    assert bindings["dataset"].scalar.primitive.string_value == (
        workflows.Dataset.L2D.value
    )
    assert bindings[
        "reactive_targets"
    ].scalar.primitive.boolean is True
    assert bindings["osm_graph_snapshot"].promise.var == (
        "osm_graph_snapshot"
    )


def test_l2d_osm_builder_workflow_is_one_offline_task():
    node, = workflows.wf_build_l2d_osm_graph_artifact.nodes

    assert node.flyte_entity.name.endswith(
        "build_l2d_osm_graph_artifact"
    )
    assert node.bindings[0].binding.promise is not None


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
        "train_route_reconstruction": 0.2,
        "train_trajectory": 1.0,
        "validation_ade_6p4s_m": 2.0,
    }
    stage_a = metadata(
        [
            {
                **common,
                "train_bev_segmentation": 0.5,
                "train_total": 1.7,
            },
            {
                **common,
                "train_bev_segmentation": 0.4,
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
                "train_total": 1.2,
            },
            {
                **common,
                "train_bev_segmentation": 0.0,
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
