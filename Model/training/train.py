"""Minimal training loop for AutoE2E: forward -> imitation loss -> backward -> step.

This is the smallest training entry point that actually updates weights. It wires
together three pieces that already exist and are unit-tested:

  - ``AutoE2E``                  the model (forward pass)
  - ``L2DDataset``               sequential L2D frames -> batched dict
  - ``TrajectoryImitationLoss``  smooth-L1 / MSE over the predicted waypoints

Only the trajectory (imitation) loss is optimized. ``FutureState`` runs during
``mode="train"`` but its output is not yet a training signal (see #13), so it is
left OFF by default here to save memory and compute. Pass
``--enable-future-state`` only to profile the worst-case memory of the full
forward (e.g. BEV at full resolution); it does NOT add a loss term yet.

The backbone, view-fusion mode, and BEV grid resolution are all constructor
arguments. ``--backbone`` and ``--fusion-mode`` are validated against the
component registries (so a newly registered module is selectable without
touching this file), and ``--bev-h/--bev-w`` size the BEV grid. Mixed precision
(``--amp``) runs in bf16, which the target GPUs (L40S/A10G/...) support natively
— no GradScaler needed.

Examples
--------
    # Smoke test: random tensors, no dataset download, reports peak VRAM.
    python train.py --smoke-test --fusion-mode bev --bev-h 450 --bev-w 300 \
        --batch-size 4 --amp

    # Real training on L2D (requires the lerobot package + dataset access).
    python train.py --fusion-mode concat --batch-size 8 --epochs 10 --amp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

# Make Model/ importable so data_parsing and model_components resolve regardless
# of the current working directory (mirrors inference/run_forward_pass.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model_components.auto_e2e import AutoE2E
from model_components.losses import TrajectoryImitationLoss
from model_components.backbones import BACKBONE_REGISTRY
from model_components.view_fusion import FUSION_REGISTRY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal AutoE2E training loop")

    # Model. backbone / fusion-mode choices are pulled live from the component
    # registries, so adding an entry to BACKBONE_REGISTRY or FUSION_REGISTRY
    # makes it selectable here without editing this file.
    p.add_argument("--backbone", default="swin_v2_tiny",
                   choices=sorted(BACKBONE_REGISTRY))
    p.add_argument("--num-views", type=int, default=7,
                   help="L2D ships 7 camera views (6 surround + 1 map render)")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--fusion-mode", default="concat",
                   choices=sorted(FUSION_REGISTRY))
    p.add_argument("--bev-h", type=int, default=450,
                   help="BEV grid height (bev fusion only)")
    p.add_argument("--bev-w", type=int, default=300,
                   help="BEV grid width (bev fusion only)")
    p.add_argument("--num-timesteps", type=int, default=64)
    p.add_argument("--num-signals", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip pretrained backbone weights (offline / fast tests)")
    p.add_argument("--enable-future-state", action="store_true",
                   help="Run FutureState during forward. Memory profiling only — "
                        "its output is not a loss term yet (see #13).")

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max grad norm; 0 disables clipping")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--loss-type", default="smooth_l1",
                   choices=["smooth_l1", "mse"])
    p.add_argument("--temporal-decay", type=float, default=1.0)

    # Precision / device. Target GPUs (g6e/L40S, A10G, ...) all support bf16,
    # so AMP uses bf16 — no GradScaler, no fp16 overflow handling needed.
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision training in bf16")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")

    # Data
    p.add_argument("--repo-id", default="yaak-ai/L2D")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Subset of episode indices; default = all")
    p.add_argument("--dataset-backbone-name", default="swinv2_tiny_window8_256",
                   help="timm name used by L2DDataset to resolve image transforms")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dataset-format", default="lerobot",
                   choices=["lerobot", "pre_extracted"],
                   help="lerobot: on-the-fly decode (EC2 dev). "
                        "pre_extracted: WebDataset shards on local disk (EKS prod).")
    p.add_argument("--shard-dir", default=None,
                   help="Path to WebDataset shards (required for pre_extracted format)")

    # Loop / logging
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-dir", default=None,
                   help="If set, write a checkpoint per epoch (real training only)")

    # MLflow
    p.add_argument("--register-model", action="store_true",
                   help="Register final checkpoint in MLflow Model Registry")
    p.add_argument("--dataset", default=None,
                   help="Dataset name for MLflow tagging (defaults to --repo-id)")

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Train on random tensors (no lerobot/dataset). Reports peak VRAM.")
    p.add_argument("--smoke-steps", type=int, default=5)

    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_model(args: argparse.Namespace, device: torch.device) -> AutoE2E:
    view_fusion_kwargs = None
    if args.fusion_mode == "bev":
        view_fusion_kwargs = {"bev_h": args.bev_h, "bev_w": args.bev_w}

    model = AutoE2E(
        backbone=args.backbone,
        num_views=args.num_views,
        embed_dim=args.embed_dim,
        fusion_mode=args.fusion_mode,
        is_pretrained=not args.no_pretrained,
        view_fusion_kwargs=view_fusion_kwargs,
        num_timesteps=args.num_timesteps,
        num_signals=args.num_signals,
    )
    return model.to(device)


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    if args.dataset_format == "pre_extracted":
        from data_parsing.pre_extracted import make_pre_extracted_loader
        if not args.shard_dir:
            raise ValueError("--shard-dir required for pre_extracted format")
        return make_pre_extracted_loader(
            shard_dir=args.shard_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # Default: lerobot on-the-fly decode
    from data_parsing.l2d import L2DDataset

    dataset = L2DDataset(
        repo_id=args.repo_id,
        episodes=args.episodes,
        backbone_name=args.dataset_backbone_name,
        local_files_only=args.local_files_only,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
        drop_last=True,
    )


def make_smoke_batch(args: argparse.Namespace, device: torch.device) -> dict:
    """A batch of random tensors matching L2DDataset's collated shapes."""
    B, V = args.batch_size, args.num_views
    return {
        "visual_tiles": torch.randn(B, V, 3, 256, 256, device=device),
        "visual_history": torch.randn(B, 896, device=device),
        "egomotion_history": torch.randn(B, 256, device=device),
        "trajectory_target": torch.randn(
            B, args.num_timesteps * args.num_signals, device=device
        ),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def run_training(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    # MLflow: active only when MLFLOW_TRACKING_URI is set AND not smoke-test.
    mlflow_active = (
        os.environ.get("MLFLOW_TRACKING_URI")
        and not args.smoke_test
    )
    if mlflow_active:
        import mlflow
        import mlflow.pytorch
        import subprocess

        # Enable system metrics (GPU utilization, memory)
        try:
            mlflow.enable_system_metrics_logging()
        except Exception:
            pass  # Fails if psutil not available

        # Autolog for PyTorch (epoch metrics auto-captured)
        mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)

        mlflow.set_experiment("auto-e2e/il-training")

        # Git info
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd="/workspace"
        ).stdout.strip() or "unknown"
        git_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd="/workspace"
        ).stdout.strip() or "unknown"

        run_name = f"{args.backbone}-{args.fusion_mode}-e{args.epochs}"
        mlflow.start_run(run_name=run_name)

        # Tags
        mlflow.set_tags({
            "stage": "IL",
            "mlflow.source.git.commit": git_commit,
            "mlflow.source.git.branch": git_branch,
        })

        # Params (namespace separated, UI-friendly)
        mlflow.log_params({
            # Model architecture
            "model/backbone": args.backbone,
            "model/fusion_mode": args.fusion_mode,
            "model/num_timesteps": args.num_timesteps,
            "model/num_signals": args.num_signals,
            # Training hyperparams
            "train/epochs": args.epochs,
            "train/batch_size": args.batch_size,
            "train/lr": args.lr,
            "train/weight_decay": args.weight_decay,
            "train/optimizer": "AdamW",
            "train/amp": args.amp,
            "train/grad_clip": args.grad_clip,
            "train/loss_type": args.loss_type,
            # Data
            "data/dataset": args.dataset or args.repo_id or "unknown",
            "data/shard_dir": args.shard_dir or "",
            "data/format": args.dataset_format,
            # Reproducibility
            "git_commit": git_commit,
        })

    print(f"device={device} | backbone={args.backbone} | fusion={args.fusion_mode} | "
          f"amp={'bf16' if use_amp else 'off'}")
    if args.fusion_mode == "bev":
        print(f"BEV grid = {args.bev_h}x{args.bev_w}")

    model = build_model(args, device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = TrajectoryImitationLoss(
        loss_type=args.loss_type,
        temporal_decay=args.temporal_decay,
        num_timesteps=args.num_timesteps,
        num_signals=args.num_signals,
    ).to(device)

    # mode="train" activates FutureState; any other value skips it (see AutoE2E).
    forward_mode = "train" if args.enable_future_state else "eval"

    # camera_params stays None: concat/cross_attn ignore it, and BEV falls back to
    # its learnable pseudo_projection. Real L2D calibration is future work.
    camera_params = None

    batches: Iterable[Any]
    if args.smoke_test:
        batches = [make_smoke_batch(args, device) for _ in range(args.smoke_steps)]
        epochs = 1
        print(f"SMOKE TEST: {args.smoke_steps} random batches, B={args.batch_size}")
    else:
        loader = build_dataloader(args)
        batches = loader
        epochs = args.epochs
        if hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__'):
            print(f"Dataset: {len(loader.dataset)} samples, {len(loader)} batches/epoch")
        else:
            print(f"WebDataset streaming loader, epochs={epochs}")

    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        running, n = 0.0, 0
        t0 = time.perf_counter()

        for step, batch in enumerate(batches):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                trajectory, _ego_hidden, _future = model(
                    batch["visual_tiles"],
                    batch["visual_history"],
                    batch["egomotion_history"],
                    camera_params=camera_params,
                    mode=forward_mode,
                )
                loss = loss_fn(trajectory, batch["trajectory_target"])

            # bf16 has fp32 dynamic range, so no GradScaler is needed.
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item()
            n += 1
            if mlflow_active:
                mlflow.log_metric("train_loss", loss.item(), step=epoch * 10000 + step)
            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        dt = time.perf_counter() - t0
        msg = f"epoch {epoch} done | mean_loss {running / max(n, 1):.4f} | {dt:.1f}s"
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            msg += f" | peak VRAM {peak:.2f} GB"
        print(msg)

        if mlflow_active:
            mlflow.log_metric("epoch_mean_loss", running / max(n, 1), step=epoch)

        if args.save_dir and not args.smoke_test:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch},
                ckpt,
            )
            print(f"saved {ckpt}")
            if mlflow_active:
                mlflow.log_artifact(ckpt)

    if mlflow_active:
        # Log final metrics
        mlflow.log_metrics({
            "final/train_loss": running / max(n, 1),
            "final/total_epochs": epochs,
        })
        if device.type == "cuda":
            mlflow.log_metric("final/gpu_peak_vram_gb", torch.cuda.max_memory_allocated(device) / 1e9)

        # Register model in Model Registry
        if args.save_dir:
            best_ckpt = os.path.join(args.save_dir, f"epoch_{epochs-1}.pt")
            if os.path.exists(best_ckpt):
                mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
            mlflow.pytorch.log_model(
                model, "model",
                registered_model_name="auto-e2e-driving-policy",
            )

        mlflow.end_run()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
