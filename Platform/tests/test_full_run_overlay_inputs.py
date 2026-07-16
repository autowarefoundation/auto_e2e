"""Tests for the completed Full Run shard extractor."""

from types import SimpleNamespace

import pytest

from Platform.scripts.extract_full_run_overlay_inputs import (
    build_overlay_inputs,
    extract_shard_uris,
    validate_full_run_inputs,
    validate_recovery_inputs,
)


def _inputs(**overrides):
    values = {
        "dataset": "KIT-MRT/KITScenes-Multimodal",
        "dataset_version": "v2.1",
        "episodes": 0,
        "reasoning_teacher": "openai_compatible",
        "enable_reasoning": True,
        "enable_world_model": True,
    }
    values.update(overrides)
    return values


def _literal_map(*uris):
    literals = [
        SimpleNamespace(
            scalar=SimpleNamespace(blob=SimpleNamespace(uri=uri))
        )
        for uri in uris
    ]
    return SimpleNamespace(
        literals={
            "o0": SimpleNamespace(
                collection=SimpleNamespace(literals=literals)
            )
        }
    )


def _recovery_inputs(**overrides):
    values = {
        "recovery_manifest": "s3://checkpoints/recovery/manifest.json",
        "artifact_set_sha256": "a" * 64,
        "dataset_version": "v2.2",
    }
    values.update(overrides)
    return values


class _Client:
    def __init__(self, node, data):
        self.node = node
        self.data = data
        self.list_calls = 0

    def list_node_executions(self, execution_id, limit, token):
        self.list_calls += 1
        assert execution_id == "workflow-id"
        assert limit == 100
        assert token is None
        return [self.node], ""

    def get_node_execution_data(self, node_id):
        assert node_id is self.node.id
        return self.data


class _Remote:
    def __init__(self, execution, node, literal_map):
        self.execution = execution
        self.client = _Client(node, object())
        self.literal_map = literal_map

    def fetch_execution(self, name):
        assert name == "a1234567890123456789"
        return self.execution

    def _get_output_literal_map(self, data):
        assert data is self.client.data
        return self.literal_map


def _remote(*uris, phase=4, node_phase=3, inputs=None):
    compiled_node = SimpleNamespace(
        id="n0",
        flyte_entity=SimpleNamespace(
            name="pipelines.workflows.wf_create_dataset_sharded"
        ),
        metadata=SimpleNamespace(name="wf_create_dataset_sharded"),
    )
    execution = SimpleNamespace(
        id="workflow-id",
        closure=SimpleNamespace(phase=phase),
        inputs=inputs or _inputs(),
        flyte_workflow=SimpleNamespace(
            id=SimpleNamespace(
                name="pipelines.workflows.wf_sharded_full_run"
            ),
            flyte_nodes=[compiled_node],
        ),
    )
    node = SimpleNamespace(
        id=SimpleNamespace(node_id="n0"),
        closure=SimpleNamespace(phase=node_phase),
    )
    return _Remote(execution, node, _literal_map(*uris))


def _recovery_remote(*uris, phase=2, node_phase=3, inputs=None):
    remote = _remote(
        *uris,
        phase=phase,
        node_phase=node_phase,
        inputs=inputs or _recovery_inputs(),
    )
    remote.execution.flyte_workflow.id.name = (
        "pipelines.workflows.wf_recovered_kitscenes_full_run"
    )
    compiled_node = remote.execution.flyte_workflow.flyte_nodes[0]
    compiled_node.flyte_entity.name = (
        "pipelines.workflows.wf_repack_existing_kitscenes"
    )
    compiled_node.metadata.name = "wf_repack_existing_kitscenes"
    return remote


def test_build_overlay_inputs_extracts_the_dataset_subworkflow_output():
    remote = _remote("s3://artifacts/partition-a", "s3://artifacts/partition-b")

    result = build_overlay_inputs(
        remote,
        execution_id="a1234567890123456789",
        expected_dataset="KIT-MRT/KITScenes-Multimodal",
        expected_dataset_version="v2.1",
    )

    assert result == {
        "full_run_execution_id": "a1234567890123456789",
        "shards": [
            "s3://artifacts/partition-a",
            "s3://artifacts/partition-b",
        ],
    }


def test_build_overlay_inputs_extracts_a_completed_running_recovery_repack():
    remote = _recovery_remote(
        "s3://artifacts/partition-a",
        "s3://artifacts/partition-b",
    )

    result = build_overlay_inputs(
        remote,
        execution_id="a1234567890123456789",
        expected_dataset="KIT-MRT/KITScenes-Multimodal",
        expected_dataset_version="v2.2",
        allow_running_recovery=True,
    )

    assert result == {
        "full_run_execution_id": "a1234567890123456789",
        "shards": [
            "s3://artifacts/partition-a",
            "s3://artifacts/partition-b",
        ],
    }


def test_running_recovery_requires_explicit_opt_in():
    remote = _recovery_remote("s3://artifacts/partition-a")

    with pytest.raises(ValueError, match="is not SUCCEEDED"):
        build_overlay_inputs(
            remote,
            execution_id="a1234567890123456789",
            expected_dataset="KIT-MRT/KITScenes-Multimodal",
            expected_dataset_version="v2.2",
        )
    assert remote.client.list_calls == 0


def test_running_recovery_requires_a_completed_repack_node():
    remote = _recovery_remote(
        "s3://artifacts/partition-a",
        node_phase=2,
    )

    with pytest.raises(ValueError, match="dataset node .* is not SUCCEEDED"):
        build_overlay_inputs(
            remote,
            execution_id="a1234567890123456789",
            expected_dataset="KIT-MRT/KITScenes-Multimodal",
            expected_dataset_version="v2.2",
            allow_running_recovery=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"episodes": 10}, "episodes=0"),
        ({"reasoning_teacher": "none"}, "did not generate reasoning labels"),
        ({"enable_reasoning": False}, "without reasoning supervision"),
        ({"enable_world_model": False}, "without the world-model branch"),
        ({"dataset_version": "v2.0"}, "expected 'v2.1'"),
    ],
)
def test_validate_full_run_inputs_rejects_non_production_contracts(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_full_run_inputs(
            _inputs(**overrides),
            expected_dataset="KIT-MRT/KITScenes-Multimodal",
            expected_dataset_version="v2.1",
            allow_partial=False,
        )


def test_build_overlay_inputs_rejects_an_incomplete_execution():
    remote = _remote("s3://artifacts/partition-a", phase=2)

    with pytest.raises(ValueError, match="is not SUCCEEDED"):
        build_overlay_inputs(
            remote,
            execution_id="a1234567890123456789",
            expected_dataset="KIT-MRT/KITScenes-Multimodal",
            expected_dataset_version="v2.1",
        )
    assert remote.client.list_calls == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset_version": "v2.1"}, "expected 'v2.2'"),
        ({"artifact_set_sha256": "bad"}, "not a lowercase SHA-256"),
        ({"recovery_manifest": "/tmp/manifest.json"}, "not an immutable S3"),
    ],
)
def test_validate_recovery_inputs_rejects_unpinned_artifacts(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_recovery_inputs(
            _recovery_inputs(**overrides),
            expected_dataset_version="v2.2",
        )


def test_validate_recovery_inputs_uses_flytefile_remote_source():
    manifest = SimpleNamespace(
        remote_source="s3://checkpoints/recovery/manifest.json",
        path="/tmp/flyte/local/manifest.json",
    )

    validate_recovery_inputs(
        _recovery_inputs(recovery_manifest=manifest),
        expected_dataset_version="v2.2",
    )


def test_extract_shard_uris_rejects_duplicate_directories():
    with pytest.raises(ValueError, match="duplicate"):
        extract_shard_uris(
            _literal_map(
                "s3://artifacts/partition-a",
                "s3://artifacts/partition-a",
            )
        )
