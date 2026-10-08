"""Contract for the NodePools that serve the Kueue gpu-overlay flavor."""

from __future__ import annotations

from pathlib import Path

import yaml

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "Platform/k8s/karpenter-nodepools/gpu-overlay-nodepool.yaml"
)
L4_INSTANCE_FAMILIES = ("g6.", "gr6.")


def _pools() -> dict[str, dict]:
    return {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(MANIFEST.read_text())
    }


def _requirement(pool: dict, key: str) -> list[str]:
    requirement, = [
        item
        for item in pool["spec"]["template"]["spec"]["requirements"]
        if item["key"] == key
    ]
    assert requirement["operator"] == "In"
    return requirement["values"]


def test_overlay_pools_target_the_kueue_flavor_label():
    pools = _pools()

    assert set(pools) == {"gpu-overlay", "gpu-overlay-reserved"}
    for pool in pools.values():
        template = pool["spec"]["template"]
        assert template["metadata"]["labels"] == {
            "workload-type": "gpu-overlay"
        }
        assert _requirement(pool, "workload-type") == ["gpu-overlay"]
        assert template["spec"]["taints"] == [
            {"key": "nvidia.com/gpu", "effect": "NoSchedule"}
        ]
        assert pool["spec"]["disruption"]["consolidationPolicy"] == (
            "WhenEmpty"
        )


def test_overlay_pools_keep_one_gpu_model():
    for pool in _pools().values():
        instance_types = _requirement(
            pool,
            "node.kubernetes.io/instance-type",
        )
        assert instance_types
        assert all(
            instance_type.startswith(L4_INSTANCE_FAMILIES)
            for instance_type in instance_types
        )


def test_reserved_overlay_pool_prefers_the_validation_reservation():
    pools = _pools()
    reserved = pools["gpu-overlay-reserved"]

    assert reserved["spec"]["weight"] > pools["gpu-overlay"]["spec"].get(
        "weight", 0
    )
    assert reserved["spec"]["template"]["spec"]["nodeClassRef"]["name"] == (
        "auto-e2e-gpu-validation"
    )
    assert _requirement(reserved, "karpenter.sh/capacity-type") == [
        "reserved"
    ]
    assert reserved["spec"]["limits"]["nvidia.com/gpu"] == "1"
