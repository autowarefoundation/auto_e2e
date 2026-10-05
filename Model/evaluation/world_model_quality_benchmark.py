#!/usr/bin/env python3
"""CLI: world-model quality benchmark (JEPA recon + Reactive vs Combined ADE).

Sibling to ``Model/speed_benchmark/speed_benchmark.py``, but reports *quality*
metrics instead of FPS.

Default is ``--trained``: train Combined for a few steps, then report JEPA
reconstruction vs the frozen target encoder and Reactive vs Combined ADE/FDE
against the batch's trajectory target. That is the review-facing number.

When you have packed shards + a checkpoint::

    python evaluation/world_model_quality_benchmark.py \\
        --shard-dir /path/to/partition --checkpoint ckpt.pt --train-steps 0
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODEL_ROOT = _HERE.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from evaluation.world_model_quality import (  # noqa: E402
    jepa_reconstruction_metrics,
    measure_jepa_on_batch,
    null_predictor_metrics,
    relative_jepa_improvement,
    summarize_world_model_quality,
    train_world_model_quality,
    world_model_trajectory_impact,
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_MODEL_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_synthetic_jepa(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    target = tuple(torch.randn(4, 16, 8, 8, generator=g) for _ in range(4))
    g2 = torch.Generator().manual_seed(seed + 1)
    pred = tuple(t + 0.25 * torch.randn(t.shape, generator=g2) for t in target)
    jepa = jepa_reconstruction_metrics(pred, target)
    null = null_predictor_metrics(target)
    rel = relative_jepa_improvement(jepa, null)
    return summarize_world_model_quality(jepa={**jepa, **null, **rel})


def _mock_batch(
    device: torch.device, batch: int = 2, views: int = 6, t: int = 4, f: int = 4
) -> dict:
    return {
        "camera_tiles": torch.randn(batch, views, 3, 256, 256, device=device),
        "map_context": torch.randn(batch, 3, 256, 256, device=device),
        "visual_history": torch.zeros(batch, 896, device=device),
        "egomotion_history": torch.randn(batch, 256, device=device),
        "trajectory_target": torch.randn(batch, 128, device=device),
        "history_frames": torch.randn(batch, t, views, 3, 256, 256, device=device),
        "future_frames": torch.randn(batch, f, views, 3, 256, 256, device=device),
    }


def run_trained(device: torch.device, steps: int, batch: int = 2) -> dict:
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E
    from tests.conftest import MockBackbone

    views = 6
    with patch("model_components.reactive_e2e.Backbone", MockBackbone):
        model = AutoE2E(
            num_views=views,
            view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
            enable_world_model=True,
        ).to(device)
    return train_world_model_quality(
        model, _mock_batch(device, batch, views), steps=steps
    )


def run_from_shard(
    shard_dir: Path,
    checkpoint: Path | None,
    device: torch.device,
    steps: int,
) -> dict:
    from data_parsing.pre_extracted import make_pre_extracted_loader
    from model_components.auto_e2e import AutoE2E

    loader = make_pre_extracted_loader(
        str(shard_dir), batch_size=1, num_workers=0, shuffle=0
    )
    raw = next(iter(loader))
    batch = {
        "camera_tiles": raw["visual_tiles"].to(device),
        "map_context": raw["map_context"].to(device),
        "visual_history": raw["visual_history"].to(device),
        "egomotion_history": raw["egomotion_history"].to(device),
        "trajectory_target": raw["trajectory_target"].to(device),
        "history_frames": raw["history_frames"].to(device),
        "future_frames": raw["future_frames"].to(device),
    }
    model = AutoE2E(
        enable_world_model=True,
        num_views=int(batch["camera_tiles"].shape[1]),
    ).to(device)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        payload = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(payload, strict=False)
        if steps <= 0:
            return measure_jepa_on_batch(
                model,
                batch["camera_tiles"],
                batch["map_context"],
                batch["visual_history"],
                batch["egomotion_history"],
                batch["history_frames"],
                batch["future_frames"],
                batch["trajectory_target"],
            )
    return train_world_model_quality(model, batch, steps=max(steps, 1))


def run_impact(backbone: str, device: torch.device, batch: int = 2, views: int = 7) -> dict:
    from model_components.auto_e2e import AutoE2E
    from model_components.view_fusion import PinholeProjection

    model = AutoE2E(
        backbone=backbone,
        num_views=views,
        view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
        enable_world_model=True,
    ).to(device)
    model.eval()
    camera = torch.randn(batch, views, 3, 256, 256, device=device)
    map_input = torch.randn(batch, 3, 256, 256, device=device)
    visual_history = torch.randn(batch, 896, device=device)
    egomotion = torch.randn(batch, 256, device=device)
    projection = PinholeProjection(torch.randn(batch, views, 3, 4, device=device))
    impact = world_model_trajectory_impact(
        model, camera, map_input, visual_history, egomotion,
        projection=projection, geometry_type="pinhole",
    )
    return summarize_world_model_quality(impact=impact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--impact", action="store_true")
    parser.add_argument("--trained", action="store_true",
                        help="Train Combined then report JEPA + ADE/FDE (default)")
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--backbone", default="swin_v2_tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.synthetic and not args.impact and not args.trained and args.shard_dir is None:
        args.trained = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)

    payload = {
        "schema": "auto_e2e_world_model_quality_v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "metrics": {},
    }

    if args.synthetic:
        payload["metrics"].update(run_synthetic_jepa(args.seed))
    if args.impact:
        payload["metrics"].update(run_impact(args.backbone, device, args.batch))
    if args.shard_dir is not None:
        payload["metrics"].update(
            run_from_shard(args.shard_dir, args.checkpoint, device, args.train_steps)
        )
        payload["source"] = "shard"
    elif args.trained:
        payload["metrics"].update(run_trained(device, args.train_steps, args.batch))
        payload["source"] = "trained_mock_backbone"

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
