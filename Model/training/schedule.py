"""Parameter groups and learning-rate schedules of the two AutoE2E stages.

Stage 1 (AutoBEV alone): the pretrained DINOv2 trains at ``STAGE1_VIT_LR_SCALE``
of the base rate and everything else at the base rate.

Stage 2 (whole model, from a stage 1 checkpoint):

- for the first ``FREEZE_FRACTION`` of the steps AutoBEV is frozen and only the
  connection (navigation encoder, map fusion, pose-error head) and the planner
  learn;
- then AutoBEV thaws: DINOv2's patch embedding and its first
  ``FROZEN_VIT_BLOCKS`` blocks stay frozen, the upper blocks train at
  ``VIT_LR_SCALE`` with a ``VIT_LAYER_DECAY`` per block below the top, the rest
  of AutoBEV (FPN adapters, spatial and temporal attention, depth and heads)
  at ``BEV_ENCODER_LR_SCALE``, and the planner side at the base rate;
- AutoBEV's optimizer state is cleared when it thaws and its rate warms up over
  ``THAW_WARMUP_FRACTION``.

Freezing is a learning-rate factor of zero rather than ``requires_grad``, so
the set of parameters with gradients, and the DDP graph, never changes. The
permanently frozen DINOv2 parameters have ``requires_grad`` off from the start.

Every rate follows a linear warm-up and a cosine decay to ``MIN_LR_FACTOR``.
``autobev_backoff`` halves the AutoBEV factors when the asynchronous validation
reports that its BEV quality fell (at most ``MAX_BACKOFFS`` times).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn

STAGE1_VIT_LR_SCALE: Final = 0.1
# Share of the steps AutoBEV stays frozen, and of its warm-up after it thaws, per stage.
FREEZE_FRACTION: Final = {1: 0.0, 2: 0.05}
FROZEN_VIT_BLOCKS: Final = 6
VIT_LR_SCALE: Final = 0.05
VIT_LAYER_DECAY: Final = 0.8
BEV_ENCODER_LR_SCALE: Final = 0.2
WARMUP_FRACTION: Final = 0.02
THAW_WARMUP_FRACTION: Final = {1: WARMUP_FRACTION, 2: 0.01}
MIN_LR_FACTOR: Final = 0.1
WEIGHT_DECAY: Final = 0.01
AUXILIARY_WEIGHT: Final = (1.0, 0.3)
MAX_BACKOFFS: Final = 3


@dataclass
class Group:
    name: str
    parameters: list[nn.Parameter]
    scale: float
    autobev: bool


def _no_decay(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or "embed" in name or "token" in name


def _split(name: str, named: list[tuple[str, nn.Parameter]], scale: float, autobev: bool) -> list[Group]:
    return [
        Group(f"{name}/{kind}", [p for n, p in named if _no_decay(n, p) == no_decay], scale, autobev)
        for kind, no_decay in (("decay", False), ("no_decay", True))
    ]


def stage1_groups(autobev: nn.Module) -> list[Group]:
    named = [(n, p) for n, p in autobev.named_parameters() if p.requires_grad]
    vit = [(n, p) for n, p in named if n.startswith("backbone.vit.")]
    rest = [(n, p) for n, p in named if not n.startswith("backbone.vit.")]
    return _split("vit", vit, STAGE1_VIT_LR_SCALE, True) + _split("autobev", rest, 1.0, True)


def stage2_groups(model: nn.Module) -> list[Group]:
    """Groups of an ``AutoE2E``; freezes the lower DINOv2 for good (``requires_grad``)."""
    reactive = model.Reactive_E2E
    vit = reactive.AutoBEV.backbone.vit
    for parameter in vit.parameters():
        parameter.requires_grad_(False)
    groups: list[Group] = []
    top = len(vit.blocks) - 1
    for index in range(FROZEN_VIT_BLOCKS, len(vit.blocks)):
        block = list(vit.blocks[index].named_parameters())
        for _, parameter in block:
            parameter.requires_grad_(True)
        groups += _split(f"vit{index}", block, VIT_LR_SCALE * VIT_LAYER_DECAY ** (top - index), True)
    # The final norm sits above the last block and trains with it.
    norm = list(vit.norm.named_parameters())
    for _, parameter in norm:
        parameter.requires_grad_(True)
    groups += _split("vit_norm", norm, VIT_LR_SCALE, True)
    encoder = [(n, p) for n, p in reactive.AutoBEV.named_parameters() if not n.startswith("backbone.vit.")]
    groups += _split("bev_encoder", encoder, BEV_ENCODER_LR_SCALE, True)
    planner = [(n, p) for n, p in reactive.named_parameters() if not n.startswith("AutoBEV.")]
    groups += _split("planner", planner, 1.0, False)
    return [group for group in groups if group.parameters]


def optimizer(groups: list[Group], learning_rate: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": g.parameters, "lr": learning_rate * g.scale, "name": g.name,
             "weight_decay": 0.0 if g.name.endswith("no_decay") else WEIGHT_DECAY}
            for g in groups
        ],
        lr=learning_rate,
    )


def _cosine(step: int, start: int, total: int, warmup: int) -> float:
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return (progress + 1) / warmup
    decay = min(1.0, (progress - warmup + 1) / max(1, total - start - warmup))
    return MIN_LR_FACTOR + 0.5 * (1.0 - MIN_LR_FACTOR) * (1.0 + math.cos(math.pi * decay))


@dataclass
class Schedule:
    """Learning-rate factors and the auxiliary weight at every optimizer step."""

    total_steps: int
    stage: int
    groups: list[Group]
    base_lr: float
    autobev_backoff: int = 0

    @property
    def thaw_step(self) -> int:
        return math.ceil(self.total_steps * FREEZE_FRACTION[self.stage])

    def _warmup(self, fraction: float) -> int:
        return max(1, math.ceil(self.total_steps * fraction))

    def factor(self, step: int, group: Group) -> float:
        if not group.autobev:
            return _cosine(step, 0, self.total_steps, self._warmup(WARMUP_FRACTION))
        thaw_warmup = self._warmup(THAW_WARMUP_FRACTION[self.stage])
        return 0.5 ** self.autobev_backoff * _cosine(step, self.thaw_step, self.total_steps, thaw_warmup)

    def auxiliary_weight(self, step: int) -> float:
        start, end = AUXILIARY_WEIGHT
        return start + (end - start) * min(1.0, step / max(1, self.total_steps - 1))

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Set every group's rate for ``step``; clear AutoBEV's state on the step it thaws.

        The state built from gradients while the learning rate was zero would
        otherwise steer the first thawed steps. Steps run in order, and a
        resumed run starts after its last step, so this happens once.
        """
        thawing = step == self.thaw_step > 0
        for group in self.groups:
            for parameter in group.parameters if group.autobev and thawing else ():
                optimizer.state.pop(parameter, None)
        for param_group, group in zip(optimizer.param_groups, self.groups, strict=True):
            param_group["lr"] = self.base_lr * group.scale * self.factor(step, group)
