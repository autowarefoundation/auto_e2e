import re
from pathlib import Path


def test_flyte_role_can_prune_only_ray_training_checkpoints():
    terraform = (
        Path(__file__).parents[1]
        / "infra"
        / "modules"
        / "flyte"
        / "main.tf"
    ).read_text(encoding="utf-8")

    delete_statements = re.findall(
        r"""
        \{
        \s*Effect\s*=\s*"Allow"
        \s*Action\s*=\s*\["s3:DeleteObject"\]
        \s*Resource\s*=\s*\[
        \s*"([^"]+)"
        \s*,?\s*\]
        \s*\}
        """,
        terraform,
        flags=re.VERBOSE,
    )

    assert delete_statements == [
        "arn:aws:s3:::${var.checkpoints_bucket}/ray-train/*"
    ]
