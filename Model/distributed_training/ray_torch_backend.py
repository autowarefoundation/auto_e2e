"""Ray Torch backend compatibility for pinned Ray and PyTorch versions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import ray
import torch
import torch.distributed as dist
from ray import train
from ray.train.torch import TorchConfig, get_device
from ray.train.torch.config import (
    _TorchBackend,
    get_address_and_port,
)


def _prepare_torch_worker() -> None:
    context = train.get_context()
    os.environ["LOCAL_RANK"] = str(context.get_local_rank())
    os.environ["LOCAL_WORLD_SIZE"] = str(context.get_local_world_size())
    os.environ["NODE_RANK"] = str(context.get_node_rank())
    os.environ["RANK"] = str(context.get_world_rank())
    os.environ["WORLD_SIZE"] = str(context.get_world_size())

    device = get_device()
    os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        probe = torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        del probe


def _setup_prepared_torch_process_group(
    *,
    backend: str,
    world_rank: int,
    world_size: int,
    init_method: str,
    timeout_s: int,
) -> None:
    if backend == "nccl":
        os.environ.setdefault(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "1",
        )
        os.environ["NCCL_NVLS_ENABLE"] = "0"
    kwargs = {}
    device = get_device()
    if backend == "nccl" and device.type == "cuda":
        kwargs["device_id"] = device
    if world_rank == 0:
        print(
            "Setting up prepared process group for: "
            f"{init_method} [rank=0, world_size={world_size}, "
            f"device_id={kwargs.get('device_id')}]",
            flush=True,
        )
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=world_rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
        **kwargs,
    )


class _PreparedCudaTorchBackend(_TorchBackend):
    def on_start(self, worker_group, backend_config) -> None:
        worker_group.execute(_prepare_torch_worker)
        backend = backend_config.backend
        if backend is None:
            backend = (
                "nccl"
                if worker_group.num_gpus_per_worker > 0
                else "gloo"
            )
        master_addr, master_port = worker_group.execute_single(
            0,
            get_address_and_port,
        )
        if backend_config.init_method == "env":
            init_method = "env://"

            def set_env_vars(addr, port):
                os.environ["MASTER_ADDR"] = addr
                os.environ["MASTER_PORT"] = str(port)

            worker_group.execute(
                set_env_vars,
                addr=master_addr,
                port=master_port,
            )
        elif backend_config.init_method == "tcp":
            init_method = f"tcp://{master_addr}:{master_port}"
        else:
            raise ValueError(
                "init_method must be either 'env' or 'tcp'"
            )

        futures = [
            worker_group.execute_single_async(
                index,
                _setup_prepared_torch_process_group,
                backend=backend,
                world_rank=index,
                world_size=len(worker_group),
                init_method=init_method,
                timeout_s=backend_config.timeout_s,
            )
            for index in range(len(worker_group))
        ]
        ray.get(futures)


@dataclass
class PreparedCudaTorchConfig(TorchConfig):
    @property
    def backend_cls(self):
        return _PreparedCudaTorchBackend
