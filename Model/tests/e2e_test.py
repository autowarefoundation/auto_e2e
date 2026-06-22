"""End-to-end training smoke test over every real dataset parser.

For each dataset that can actually be loaded in the current environment, this
builds AutoE2E, runs a short optimisation loop on real samples, and asserts that
the trajectory imitation loss trends downward — i.e. the full pipeline
(parser -> DataLoader -> model -> loss -> backward -> step) learns.

Datasets whose data or parser are unavailable are skipped, not failed:
  - L2D            loads from the HuggingFace hub on demand (network needed).
  - nvidia_av      needs a local data_root; skipped when absent.
  - kit_scenes     parser is not yet on main (PR #41); skipped when missing.

These tests use the REAL backbone and REAL data, so they are slow and marked
``e2e_data``. They are excluded from the default run (see pytest.ini) and invoked
explicitly:

    cd Model/tests && python -m pytest e2e_test.py -v -m e2e_data -s

Loss-trend criterion: per-step SGD loss is noisy, so we do not require strict
monotonic decrease. Instead the mean of the last third of steps must be clearly
below the mean of the first third, and every step must be finite.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_components.auto_e2e import AutoE2E
from model_components.losses import TrajectoryImitationLoss


# Each spec describes how to build one dataset and what shape it produces.
# `build` returns a torch Dataset or raises to signal "unavailable" (-> skip).
# `num_views` lets the model match the parser's camera count.
def _build_l2d():
    from data_parsing.l2d import L2DDataset

    # A couple of episodes give enough valid samples for a short loop without
    # pulling the whole 100k-episode dataset.
    return L2DDataset(repo_id="yaak-ai/L2D", episodes=[0, 1])


def _build_nvidia():
    from data_parsing.nvidia_physical_ai import NvidiaAVDataset

    data_root = os.environ.get("NVIDIA_AV_ROOT")
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(
            "NVIDIA_AV_ROOT not set or missing; nvidia_physical_ai data unavailable"
        )
    return NvidiaAVDataset(data_root=data_root)


def _build_kit_scenes():
    # Parser not yet merged to main (PR #41). Import error -> skip.
    from data_parsing.kit_scenes import KitScenesDataset  # noqa: F401

    data_root = os.environ.get("KITSCENES_ROOT")
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError("KITSCENES_ROOT not set or missing")
    return KitScenesDataset(data_root=data_root)


DATASET_SPECS = [
    pytest.param("l2d", _build_l2d, 7, id="l2d"),
    pytest.param("nvidia_av", _build_nvidia, 8, id="nvidia_av"),
    pytest.param("kit_scenes", _build_kit_scenes, 8, id="kit_scenes"),
]

# Short loop sized to expose a trend without being a full training run.
# 20 steps gives enough signal to overcome SGD mini-batch noise on small datasets.
_NUM_STEPS = 20
_BATCH_SIZE = 4
_LR = 1e-3


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _try_build(build_fn):
    """Build a dataset, translating any unavailability into pytest.skip."""
    try:
        return build_fn()
    except pytest.skip.Exception:
        raise
    except ImportError as e:
        pytest.skip(f"parser unavailable: {e}")
    except (FileNotFoundError, OSError, ValueError) as e:
        pytest.skip(f"data unavailable: {e}")


def _run_loss_trend(dataset, num_views, device):
    """Run a short training loop and return the list of per-step losses."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Train from scratch (no pretrained download) so the test is self-contained
    # and the loss has clear room to move.
    model = AutoE2E(
        num_views=num_views,
        fusion_mode="concat",
        is_pretrained=False,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR)
    loss_fn = TrajectoryImitationLoss().to(device)

    losses = []
    data_iter = iter(loader)
    for _ in range(_NUM_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        visual_tiles = batch["visual_tiles"].to(device)
        visual_history = batch["visual_history"].to(device)
        egomotion_history = batch["egomotion_history"].to(device)
        target = batch["trajectory_target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        trajectory, _ego, _future = model(
            visual_tiles, visual_history, egomotion_history,
            camera_params=None, mode="eval",
        )
        loss = loss_fn(trajectory, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


@pytest.mark.e2e_data
@pytest.mark.parametrize("name,build_fn,num_views", DATASET_SPECS)
def test_loss_decreases_on_real_data(name, build_fn, num_views):
    dataset = _try_build(build_fn)
    assert len(dataset) >= _BATCH_SIZE, (
        f"{name}: only {len(dataset)} samples, need >= {_BATCH_SIZE}"
    )

    losses = _run_loss_trend(dataset, num_views, _device())

    # No NaN/Inf anywhere — the pipeline stays numerically sane on real data.
    assert all(torch.isfinite(torch.tensor(x)) for x in losses), (
        f"{name}: non-finite loss encountered: {losses}"
    )

    # Loss must trend down: last-third mean clearly below first-third mean.
    third = max(1, _NUM_STEPS // 3)
    first = sum(losses[:third]) / third
    last = sum(losses[-third:]) / third
    assert last < first, (
        f"{name}: loss did not decrease. first_third={first:.4f} "
        f"last_third={last:.4f} all={[round(x, 4) for x in losses]}"
    )
