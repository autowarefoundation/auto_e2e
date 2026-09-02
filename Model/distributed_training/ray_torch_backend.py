"""Ray Torch backend compatibility for pinned Ray and PyTorch versions."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from ray import train
from ray.train.torch import TorchConfig, get_device
from ray.train.torch.config import _TorchBackend


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


class _PreparedCudaTorchBackend(_TorchBackend):
    def on_start(self, worker_group, backend_config) -> None:
        worker_group.execute(_prepare_torch_worker)
        super().on_start(worker_group, backend_config)


@dataclass
class PreparedCudaTorchConfig(TorchConfig):
    @property
    def backend_cls(self):
        return _PreparedCudaTorchBackend
