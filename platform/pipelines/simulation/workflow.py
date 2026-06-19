"""Flyte simulation workflow: CARLA closed-loop evaluation.

Provisions CARLA server pod, runs scenarios, collects results.
"""

from __future__ import annotations

import os
from typing import List

from flytekit import task, workflow, Resources

_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "381491877296")
_REGION = os.environ.get("AWS_REGION", "us-west-2")

TRAINING_IMAGE = f"{_ACCOUNT_ID}.dkr.ecr.{_REGION}.amazonaws.com/auto-e2e/training:latest"
CARLA_IMAGE = "carlasim/carla:0.9.15"


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
    pod_labels={
        "kueue.x-k8s.io/queue-name": "gpu-queue",
    },
    node_selector={"workload-type": "simulation"},
    tolerations=[{"key": "nvidia.com/gpu-sim", "operator": "Exists", "effect": "NoSchedule"}],
)
def run_carla_scenarios(checkpoint_s3: str, scenarios: str = "S01,S02,S03") -> dict:
    """Start CARLA server (sidecar) and run closed-loop scenarios.

    NOTE: In production, CARLA server runs as a separate pod (sidecar or
    pre-provisioned). For simplicity, this task assumes CARLA is reachable
    at carla-server:2000 (deployed separately via K8s manifest).
    """
    import json
    import subprocess
    import boto3

    # Download checkpoint
    s3 = boto3.client("s3")
    bucket, key = checkpoint_s3.replace("s3://", "").split("/", 1)
    local_ckpt = "/tmp/checkpoint.pt"
    s3.download_file(bucket, key, local_ckpt)

    # Run closed-loop evaluation
    result = subprocess.run([
        "python", "-m", "evaluation.closed_loop_runner",
        "--carla-host", "carla-server",
        "--checkpoint", local_ckpt,
        "--scenarios", scenarios,
        "--output", "/tmp/results.json",
        "--device", "cpu",
    ], capture_output=True, text=True, cwd="/workspace/Model")

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        return {"error": result.stderr, "scenarios": []}

    results = json.loads(open("/tmp/results.json").read())
    return {"scenarios": results, "all_passed": all(r["success"] for r in results)}


@workflow
def closed_loop_eval(
    checkpoint_s3: str = "s3://auto-e2e-platform-checkpoints-381491877296/staging.pt",
    scenarios: str = "S01,S02,S03",
) -> dict:
    """Run CARLA closed-loop evaluation on staging checkpoint."""
    return run_carla_scenarios(checkpoint_s3=checkpoint_s3, scenarios=scenarios)
