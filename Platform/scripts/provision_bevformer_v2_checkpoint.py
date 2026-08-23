#!/usr/bin/env python3
"""Provision the verified BEVFormer V2 t1 checkpoint in the platform account."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Model"))

from model_components.bevformer_v2_pretrained import (  # noqa: E402
    BEVFORMER_V2_SOURCE_REPOSITORY,
    BEVFORMER_V2_T1_CHECKPOINT_MIRROR_KEY,
    BEVFORMER_V2_T1_CHECKPOINT_SHA256,
    BEVFORMER_V2_TRAINING_DATA_LICENSE_SPDX,
    BEVFORMER_V2_WEIGHT_LICENSE_SPDX,
    bevformer_v2_t1_checkpoint_mirror_uri,
    sha256_file,
)


def _head_or_none(client, *, bucket: str, key: str):
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        status = int(error.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode",
            0,
        ))
        if status == 404:
            return None
        raise


def provision(
    source: str | Path,
    *,
    account_id: str | None = None,
    cluster_name: str = "auto-e2e-platform",
) -> dict[str, object]:
    source_path = Path(source)
    digest = sha256_file(source_path)
    if digest != BEVFORMER_V2_T1_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {digest} != "
            f"{BEVFORMER_V2_T1_CHECKPOINT_SHA256}"
        )
    resolved_account = account_id or str(
        boto3.client("sts").get_caller_identity()["Account"]
    )
    uri = bevformer_v2_t1_checkpoint_mirror_uri(
        resolved_account,
        cluster_name=cluster_name,
    )
    bucket = uri.removeprefix("s3://").split("/", 1)[0]
    key = BEVFORMER_V2_T1_CHECKPOINT_MIRROR_KEY
    metadata = {
        "sha256": digest,
        "source-repository": BEVFORMER_V2_SOURCE_REPOSITORY,
        "weight-license-spdx": BEVFORMER_V2_WEIGHT_LICENSE_SPDX,
        "training-data-license-spdx": (
            BEVFORMER_V2_TRAINING_DATA_LICENSE_SPDX
        ),
    }
    client = boto3.client("s3")
    existing = _head_or_none(client, bucket=bucket, key=key)
    if existing is not None:
        if (
            int(existing["ContentLength"]) != source_path.stat().st_size
            or existing.get("Metadata", {}).get("sha256") != digest
        ):
            raise RuntimeError(
                "existing checkpoint mirror differs; refusing to overwrite"
            )
        head = existing
        uploaded = False
    else:
        client.upload_file(
            str(source_path),
            bucket,
            key,
            ExtraArgs={
                "Metadata": metadata,
                "ServerSideEncryption": "AES256",
            },
        )
        head = client.head_object(Bucket=bucket, Key=key)
        uploaded = True
    return {
        "content_length": int(head["ContentLength"]),
        "sha256": digest,
        "uploaded": uploaded,
        "uri": uri,
        "version_id": head.get("VersionId"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--account-id")
    parser.add_argument(
        "--cluster-name",
        default="auto-e2e-platform",
    )
    args = parser.parse_args()
    print(json.dumps(
        provision(
            args.source,
            account_id=args.account_id,
            cluster_name=args.cluster_name,
        ),
        allow_nan=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
