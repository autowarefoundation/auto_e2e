"""Ray Train DDP canary for the AutoE2E imitation model.

This module is executable inside a KubeRay RayJob and is also called by the
Flyte Ray task wrapper. It proves that one AutoE2E replica per GPU participates
in synchronous DDP, updates parameters, reaches consensus, and reports a
durable Ray checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any


SUPPORTED_WORLD_SIZES = frozenset({2, 4, 8})


def validate_smoke_config(
    *,
    num_workers: int,
    steps: int,
    learning_rate: float,
) -> None:
    if num_workers not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "num_workers must be one of "
            f"{sorted(SUPPORTED_WORLD_SIZES)}, got {num_workers}"
        )
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError(
            "learning_rate must be in (0, 1], got "
            f"{learning_rate}"
        )


def _base_model(model):
    from torch.nn.parallel import DistributedDataParallel

    return model.module if isinstance(model, DistributedDataParallel) else model


def _fixed_rank_batch(
    *,
    rank: int,
    device,
    num_views: int,
    image_size: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rank * 10_007)
    batch_size = 1

    def normal(*shape: int):
        return torch.randn(*shape, generator=generator).to(device)

    return {
        "visual": normal(
            batch_size,
            num_views,
            3,
            image_size,
            image_size,
        ),
        "map_context": normal(
            batch_size,
            3,
            image_size,
            image_size,
        ),
        "visual_history": torch.zeros(
            batch_size,
            896,
            device=device,
        ),
        "egomotion_history": normal(batch_size, 64 * 4),
        "route_mask": normal(
            batch_size,
            2,
            image_size,
            image_size,
        ),
        "map_valid": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        "route_valid": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        "target": normal(batch_size, 64 * 2),
    }


def _parameter_sample(model, *, sample_size: int = 2048):
    import torch

    samples = []
    remaining = sample_size
    for parameter in _base_model(model).parameters():
        if not parameter.requires_grad:
            continue
        flattened = parameter.detach().reshape(-1)
        take = min(remaining, flattened.numel())
        if take:
            samples.append(flattened[:take].float())
            remaining -= take
        if remaining == 0:
            break
    if not samples:
        raise RuntimeError("model has no trainable parameter sample")
    return torch.cat(samples)


def _assert_parameter_consensus(model, *, world_size: int) -> float:
    import torch
    import torch.distributed as dist

    local = _parameter_sample(model)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    maximum_delta = max(
        float((candidate - gathered[0]).abs().max().item())
        for candidate in gathered
    )
    if maximum_delta > 1e-6:
        raise RuntimeError(
            "DDP replicas diverged after optimizer step: "
            f"maximum_parameter_delta={maximum_delta}"
        )
    return maximum_delta


def train_loop_per_worker(config: dict[str, Any]) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    from model_components.auto_e2e import AutoE2E

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    expected_world_size = int(config["num_workers"])
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Ray world size {world_size} != expected {expected_world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("distributed GPU canary requires CUDA")

    device = get_device()
    torch.cuda.set_device(device)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = AutoE2E(
        backbone="swin_v2_tiny",
        num_views=int(config["num_views"]),
        embed_dim=256,
        is_pretrained=False,
        map_context_channels=3,
        route_channels=2,
        enable_route_conditioning=True,
        enable_reasoning=False,
        enable_world_model=False,
    )
    model = prepare_model(
        model,
        parallel_strategy="ddp",
        parallel_strategy_kwargs={
            "find_unused_parameters": True,
            "gradient_as_bucket_view": True,
        },
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=1e-2,
    )

    start_step = 0
    restored = train.get_checkpoint()
    if restored is not None:
        with restored.as_directory() as checkpoint_dir:
            payload = torch.load(
                Path(checkpoint_dir) / "checkpoint.pt",
                map_location=device,
                weights_only=False,
            )
        _base_model(model).load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["step"])

    hostnames: list[str | None] = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    unique_hostnames = sorted({str(host) for host in hostnames})
    if len(unique_hostnames) != world_size:
        raise RuntimeError(
            "one-GPU-per-node invariant failed: "
            f"world_size={world_size} hosts={unique_hostnames}"
        )

    batch = _fixed_rank_batch(
        rank=rank,
        device=device,
        num_views=int(config["num_views"]),
        image_size=int(config["image_size"]),
        seed=seed,
    )
    parameter_before = _parameter_sample(model).clone()
    global_losses: list[float] = []
    maximum_parameter_delta = 0.0
    started = time.perf_counter()

    for step in range(start_step, int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["visual"],
            batch["map_context"],
            batch["visual_history"],
            batch["egomotion_history"],
            route_mask=batch["route_mask"],
            map_valid=batch["map_valid"],
            route_valid=batch["route_valid"],
            projection=None,
            geometry_type="pseudo",
            mode="train",
            trajectory_target=batch["target"],
        )
        trajectory = output[0] if isinstance(output, tuple) else output
        loss = functional.smooth_l1_loss(trajectory, batch["target"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"rank {rank} produced non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError(
                f"rank {rank} produced invalid gradient norm {gradient_norm}"
            )
        optimizer.step()

        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        global_losses.append(float(reduced_loss.item() / world_size))
        maximum_parameter_delta = max(
            maximum_parameter_delta,
            _assert_parameter_consensus(
                model,
                world_size=world_size,
            ),
        )

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    parameter_update_norm = float(
        (_parameter_sample(model) - parameter_before).norm().item()
    )
    if parameter_update_norm <= 0.0:
        raise RuntimeError("optimizer did not update AutoE2E parameters")

    metrics = {
        "backend": dist.get_backend(),
        "elapsed_seconds": elapsed_seconds,
        "final_global_loss": global_losses[-1],
        "hostname_count": len(unique_hostnames),
        "hostnames": unique_hostnames,
        "initial_global_loss": global_losses[0],
        "maximum_parameter_delta": maximum_parameter_delta,
        "parameter_update_norm": parameter_update_norm,
        "steps": int(config["steps"]),
        "world_size": world_size,
    }

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        checkpoint = None
        if rank == 0:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": _base_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": int(config["steps"]),
                    "world_size": world_size,
                    "hostnames": unique_hostnames,
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(metrics, checkpoint=checkpoint)


def run_smoke(
    *,
    num_workers: int,
    steps: int,
    storage_path: str,
    run_name: str,
    learning_rate: float = 1e-4,
    seed: int = 149,
) -> dict[str, Any]:
    validate_smoke_config(
        num_workers=num_workers,
        steps=steps,
        learning_rate=learning_rate,
    )
    if not storage_path.startswith("s3://"):
        raise ValueError("storage_path must be an S3 URI")
    if not run_name or "/" in run_name:
        raise ValueError("run_name must be non-empty and cannot contain '/'")

    import ray
    from ray import train
    from ray.train.torch import TorchTrainer

    if not ray.is_initialized():
        ray.init(address="auto")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "image_size": 256,
            "learning_rate": learning_rate,
            "num_views": 6,
            "num_workers": num_workers,
            "seed": seed,
            "steps": steps,
        },
        scaling_config=train.ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": 4, "GPU": 1},
            placement_strategy="SPREAD",
        ),
        run_config=train.RunConfig(
            name=run_name,
            storage_path=storage_path,
            failure_config=train.FailureConfig(max_failures=2),
            checkpoint_config=train.CheckpointConfig(num_to_keep=3),
        ),
    )
    result = trainer.fit()
    metrics = dict(result.metrics)
    if int(metrics.get("world_size", 0)) != num_workers:
        raise RuntimeError(f"unexpected Ray result metrics: {metrics}")
    return {
        "checkpoint_uri": (
            str(result.checkpoint.path)
            if result.checkpoint is not None
            else None
        ),
        "metrics": metrics,
        "run_name": run_name,
        "storage_path": storage_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument(
        "--storage-path",
        default=os.environ.get(
            "RAY_TRAIN_STORAGE_PATH",
            "s3://auto-e2e-platform-checkpoints/ray-train",
        ),
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get(
            "RAY_TRAIN_RUN_NAME",
            f"auto-e2e-ddp-smoke-{int(time.time())}",
        ),
    )
    parser.add_argument("--output", default="/tmp/ray-ddp-smoke.json")
    args = parser.parse_args()

    result = run_smoke(
        num_workers=args.num_workers,
        steps=args.steps,
        storage_path=args.storage_path,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
