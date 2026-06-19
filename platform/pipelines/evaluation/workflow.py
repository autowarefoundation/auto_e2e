"""Flyte evaluation workflow: checkpoint → inference → metrics → gate → promote.

Uses the training image (no extra deps needed). Runs on GPU for inference,
CPU for metric computation.
"""

from __future__ import annotations

from flytekit import task, workflow, Resources

import os

_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "381491877296")
_REGION = os.environ.get("AWS_REGION", "us-west-2")

DATA_PREP_IMAGE = f"{_ACCOUNT_ID}.dkr.ecr.{_REGION}.amazonaws.com/auto-e2e/data-prep:latest"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://172.20.240.62:5000")


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="6", mem="40Gi", gpu="1"),
    limits=Resources(gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI, "AWS_DEFAULT_REGION": "us-west-2"},
    labels={"kueue.x-k8s.io/queue-name": "gpu-queue"},
)
def run_eval(checkpoint_s3: str, val_shard_dir: str) -> dict:
    """Load checkpoint, run inference on val shards, compute metrics."""
    import json
    import numpy as np
    import torch
    import boto3
    import webdataset as wds
    from pathlib import Path

    # Download checkpoint
    s3 = boto3.client("s3")
    bucket, key = checkpoint_s3.replace("s3://", "").split("/", 1)
    local_ckpt = "/tmp/checkpoint.pt"
    s3.download_file(bucket, key, local_ckpt)

    # Load model
    import sys
    sys.path.insert(0, "/workspace/Model")
    from model_components.auto_e2e import AutoE2E
    from evaluation.metrics import compute_open_loop_metrics

    ckpt = torch.load(local_ckpt, map_location="cuda", weights_only=False)
    # Infer model config from checkpoint keys
    model = AutoE2E(backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                    fusion_mode="concat", is_pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.cuda().eval()

    # Load val shards
    from data_parsing.pre_extracted import make_pre_extracted_loader
    loader = make_pre_extracted_loader(val_shard_dir, batch_size=4, num_workers=2)

    # Inference
    all_pred_accel, all_pred_curv = [], []
    all_gt_accel, all_gt_curv = [], []
    all_init_speed = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            visual = batch["visual_tiles"].cuda()
            vis_hist = batch["visual_history"].cuda()
            ego_hist = batch["egomotion_history"].cuda()
            target = batch["trajectory_target"]  # (B, 128)

            trajectory, _, _ = model(visual, vis_hist, ego_hist, mode="eval")
            # trajectory: (B, 128) = 64 steps × 2 signals [accel, curv]
            pred = trajectory.cpu().numpy().reshape(-1, 64, 2)
            gt = target.numpy().reshape(-1, 64, 2)

            all_pred_accel.append(pred[:, :, 0])
            all_pred_curv.append(pred[:, :, 1])
            all_gt_accel.append(gt[:, :, 0])
            all_gt_curv.append(gt[:, :, 1])

            # Initial speed from last step of ego history
            ego_np = ego_hist.cpu().numpy().reshape(-1, 64, 4)
            all_init_speed.append(ego_np[:, -1, 0])  # speed is signal 0

    pred_accel = np.concatenate(all_pred_accel)
    pred_curv = np.concatenate(all_pred_curv)
    gt_accel = np.concatenate(all_gt_accel)
    gt_curv = np.concatenate(all_gt_curv)
    init_speed = np.concatenate(all_init_speed)

    metrics = compute_open_loop_metrics(pred_accel, pred_curv, gt_accel, gt_curv, init_speed)
    print(f"Eval metrics: {json.dumps(metrics, indent=2)}")
    return metrics


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="1Gi"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def promote_if_passed(metrics: dict, checkpoint_s3: str, mlflow_run_id: str = "") -> bool:
    """Gate check + MLflow promotion if passed."""
    import sys
    sys.path.insert(0, "/workspace/Model")
    from evaluation.metrics import gate_check

    passed = gate_check(metrics)
    print(f"Gate check: {'PASSED' if passed else 'FAILED'}")

    if passed:
        import mlflow
        client = mlflow.MlflowClient()
        # Log metrics to MLflow
        if mlflow_run_id:
            for k, v in metrics.items():
                client.log_metric(mlflow_run_id, f"eval_{k}", v)
        # Promote: set alias "staging" on latest model version
        try:
            versions = client.search_model_versions("name='auto_e2e'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                client.set_registered_model_alias("auto_e2e", "staging", latest.version)
                print(f"Promoted version {latest.version} to 'staging'")
        except Exception as e:
            print(f"Promotion skipped: {e}")

    return passed


@workflow
def evaluate_checkpoint(
    checkpoint_s3: str = "s3://auto-e2e-platform-checkpoints-381491877296/latest.pt",
    val_shard_dir: str = "/data/shards",
    mlflow_run_id: str = "",
) -> bool:
    """Evaluate a checkpoint and promote to staging if it passes the gate."""
    metrics = run_eval(checkpoint_s3=checkpoint_s3, val_shard_dir=val_shard_dir)
    return promote_if_passed(metrics=metrics, checkpoint_s3=checkpoint_s3,
                             mlflow_run_id=mlflow_run_id)
