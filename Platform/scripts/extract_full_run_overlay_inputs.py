#!/usr/bin/env python3
"""Build overlay inputs from a completed dataset-producing subworkflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_EXECUTION_ID_RE = re.compile(r"^a[a-z0-9]{19}$")
_FULL_RUN_WORKFLOW = "pipelines.workflows.wf_sharded_full_run"
_DATASET_WORKFLOW = "pipelines.workflows.wf_create_dataset_sharded"
_RECOVERY_WORKFLOW = "pipelines.workflows.wf_recovered_kitscenes_full_run"
_REPACK_WORKFLOW = "pipelines.workflows.wf_repack_existing_kitscenes"
_WORKFLOW_RUNNING = 2
_WORKFLOW_SUCCEEDED = 4
_NODE_SUCCEEDED = 3


def _plain_value(value: Any) -> Any:
    remote_source = getattr(value, "remote_source", None)
    if remote_source:
        return remote_source
    return getattr(value, "value", value)


def _required_input(inputs: Mapping[str, Any], name: str) -> Any:
    value = inputs.get(name)
    if value is None:
        raise ValueError(f"Full Run has no {name!r} input")
    return _plain_value(value)


def validate_full_run_inputs(
    inputs: Mapping[str, Any],
    *,
    expected_dataset: str,
    expected_dataset_version: str,
    allow_partial: bool,
) -> None:
    """Reject runs whose packed labels/shards are not the production contract."""
    dataset = str(_required_input(inputs, "dataset"))
    if dataset != expected_dataset:
        raise ValueError(
            f"Full Run dataset is {dataset!r}, expected {expected_dataset!r}"
        )
    dataset_version = str(_required_input(inputs, "dataset_version"))
    if dataset_version != expected_dataset_version:
        raise ValueError(
            "Full Run dataset_version is "
            f"{dataset_version!r}, expected {expected_dataset_version!r}"
        )
    if not allow_partial and int(_required_input(inputs, "episodes")) != 0:
        raise ValueError("production publication requires a Full Run with episodes=0")
    if str(_required_input(inputs, "reasoning_teacher")) == "none":
        raise ValueError("Full Run did not generate reasoning labels")
    if not bool(_required_input(inputs, "enable_reasoning")):
        raise ValueError("Full Run model was trained without reasoning supervision")
    if not bool(_required_input(inputs, "enable_world_model")):
        raise ValueError("Full Run model was trained without the world-model branch")


def validate_recovery_inputs(
    inputs: Mapping[str, Any],
    *,
    expected_dataset_version: str,
) -> None:
    """Reject recovery runs that do not identify audited KITScenes artifacts."""
    dataset_version = str(_required_input(inputs, "dataset_version"))
    if dataset_version != expected_dataset_version:
        raise ValueError(
            "recovery dataset_version is "
            f"{dataset_version!r}, expected {expected_dataset_version!r}"
        )
    artifact_set_sha256 = str(
        _required_input(inputs, "artifact_set_sha256")
    )
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_set_sha256):
        raise ValueError(
            "recovery artifact_set_sha256 is not a lowercase SHA-256"
        )
    recovery_manifest = str(_required_input(inputs, "recovery_manifest"))
    if not recovery_manifest.startswith("s3://"):
        raise ValueError("recovery_manifest is not an immutable S3 artifact")


def dataset_node_id(execution: Any) -> str:
    """Find the dataset subworkflow node without depending on its compiled ID."""
    workflow = execution.flyte_workflow
    workflow_name = str(getattr(getattr(workflow, "id", None), "name", ""))
    if workflow_name == _FULL_RUN_WORKFLOW:
        expected_entity = _DATASET_WORKFLOW
        expected_metadata = "wf_create_dataset_sharded"
    elif workflow_name == _RECOVERY_WORKFLOW:
        expected_entity = _REPACK_WORKFLOW
        expected_metadata = "wf_repack_existing_kitscenes"
    else:
        raise ValueError(
            "execution workflow is "
            f"{workflow_name!r}, expected {_FULL_RUN_WORKFLOW!r} or "
            f"{_RECOVERY_WORKFLOW!r}"
        )

    matches = []
    for node in workflow.flyte_nodes:
        entity_name = str(getattr(getattr(node, "flyte_entity", None), "name", ""))
        metadata_name = str(getattr(getattr(node, "metadata", None), "name", ""))
        if (
            entity_name == expected_entity
            or metadata_name == expected_metadata
        ):
            matches.append(str(node.id))
    if len(matches) != 1:
        raise ValueError(
            f"{workflow_name} must contain exactly one {expected_metadata} "
            f"node, found {matches}"
        )
    return matches[0]


def iter_node_executions(remote: Any, execution: Any) -> Iterable[Any]:
    token = None
    while True:
        nodes, token = remote.client.list_node_executions(
            execution.id,
            limit=100,
            token=token,
        )
        yield from nodes
        if not token:
            return


def extract_shard_uris(literal_map: Any) -> list[str]:
    """Decode the dataset node's bare List[FlyteDirectory] output."""
    output = literal_map.literals.get("o0")
    collection = getattr(output, "collection", None)
    literals = getattr(collection, "literals", None)
    if not literals:
        raise ValueError("dataset node output o0 is not a non-empty collection")

    uris = []
    for index, literal in enumerate(literals):
        scalar = getattr(literal, "scalar", None)
        blob = getattr(scalar, "blob", None)
        uri = str(getattr(blob, "uri", ""))
        if not uri.startswith("s3://"):
            raise ValueError(
                f"dataset node output o0[{index}] is not an S3 FlyteDirectory"
            )
        uris.append(uri)
    if len(set(uris)) != len(uris):
        raise ValueError("dataset node returned duplicate FlyteDirectory URIs")
    return uris


def build_overlay_inputs(
    remote: Any,
    *,
    execution_id: str,
    expected_dataset: str,
    expected_dataset_version: str,
    allow_partial: bool = False,
    allow_running_recovery: bool = False,
) -> dict[str, Any]:
    if not _EXECUTION_ID_RE.fullmatch(execution_id):
        raise ValueError(f"invalid Flyte execution ID {execution_id!r}")

    execution = remote.fetch_execution(name=execution_id)
    workflow_name = str(
        getattr(
            getattr(execution.flyte_workflow, "id", None),
            "name",
            "",
        )
    )
    phase = int(execution.closure.phase)
    running_recovery = (
        workflow_name == _RECOVERY_WORKFLOW
        and allow_running_recovery
        and phase == _WORKFLOW_RUNNING
    )
    if phase != _WORKFLOW_SUCCEEDED and not running_recovery:
        raise ValueError(
            f"dataset-producing workflow {execution_id} is not SUCCEEDED "
            f"(phase={execution.closure.phase})"
        )
    if workflow_name == _RECOVERY_WORKFLOW:
        validate_recovery_inputs(
            execution.inputs,
            expected_dataset_version=expected_dataset_version,
        )
    elif workflow_name == _FULL_RUN_WORKFLOW:
        validate_full_run_inputs(
            execution.inputs,
            expected_dataset=expected_dataset,
            expected_dataset_version=expected_dataset_version,
            allow_partial=allow_partial,
        )
    else:
        raise ValueError(f"unsupported workflow {workflow_name!r}")
    wanted_node_id = dataset_node_id(execution)

    matching_nodes = [
        node
        for node in iter_node_executions(remote, execution)
        if str(node.id.node_id) == wanted_node_id
    ]
    if len(matching_nodes) != 1:
        raise ValueError(
            f"dataset node {wanted_node_id!r} has "
            f"{len(matching_nodes)} executions"
        )
    node = matching_nodes[0]
    if int(node.closure.phase) != _NODE_SUCCEEDED:
        raise ValueError(
            f"dataset node {wanted_node_id!r} is not SUCCEEDED "
            f"(phase={node.closure.phase})"
        )

    data = remote.client.get_node_execution_data(node.id)
    literal_map = remote._get_output_literal_map(data)
    shard_uris = extract_shard_uris(literal_map)
    return {
        "full_run_execution_id": execution_id,
        "shards": shard_uris,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--project", default="auto-e2e")
    parser.add_argument("--domain", default="development")
    parser.add_argument(
        "--expected-dataset",
        default="KIT-MRT/KITScenes-Multimodal",
    )
    parser.add_argument("--expected-dataset-version", default="v2.2")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow episodes>0 for smoke validation only.",
    )
    parser.add_argument(
        "--allow-running-recovery",
        action="store_true",
        help=(
            "Allow a running recovery parent after its repack subworkflow "
            "has succeeded."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from flytekit.configuration import Config
    from flytekit.remote import FlyteRemote

    remote = FlyteRemote(
        config=Config.auto(config_file=args.config),
        default_project=args.project,
        default_domain=args.domain,
    )
    payload = build_overlay_inputs(
        remote,
        execution_id=args.execution_id,
        expected_dataset=args.expected_dataset,
        expected_dataset_version=args.expected_dataset_version,
        allow_partial=args.allow_partial,
        allow_running_recovery=args.allow_running_recovery,
    )
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Wrote {len(payload['shards'])} Full Run shard directories to {output}"
    )


if __name__ == "__main__":
    main()
