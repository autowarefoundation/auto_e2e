"""World-model quality evaluation — JEPA reconstruction + reactive vs Combined.

Speed already lives in ``Model/speed_benchmark/`` (Reactive vs Combined FPS).
This module measures *quality*:

1. **JEPA reconstruction** — per-horizon L1 / L2 / cosine between predicted
   future feature maps and frozen-target maps (the World Model's self-supervised
   objective, evaluated on a held-out window).
2. **Reactive vs Combined trajectory impact** — how much enabling the World
   Model changes the open-loop trajectory on identical inputs (and optional
   ADE/FDE vs ground-truth controls when available).

Pure eval helpers: no training loop changes. Unit-testable with synthetic
tensors; optional model-level helpers follow the faithfulness.py ABI.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from .metrics import compute_open_loop_metrics


def _as_list(features: Sequence[torch.Tensor] | tuple) -> list[torch.Tensor]:
    return list(features)


def jepa_reconstruction_metrics(
    predicted_features: Sequence[torch.Tensor],
    target_features: Sequence[torch.Tensor],
) -> dict[str, float]:
    """Per-horizon and mean JEPA reconstruction quality.

    Args:
        predicted_features: list/tuple of ``[B, C, H, W]`` predicted maps.
        target_features: list/tuple of matching target maps (detached).

    Returns:
        Dict with mean ``l1``, ``l2``, ``cosine`` plus per-horizon
        ``l1@h{k}``, ``l2@h{k}``, ``cosine@h{k}`` (1-indexed horizons).
    """
    preds = _as_list(predicted_features)
    targets = _as_list(target_features)
    if len(preds) != len(targets):
        raise ValueError(
            f"predicted/target horizon mismatch: {len(preds)} vs {len(targets)}"
        )
    if not preds:
        raise ValueError("predicted_features must be non-empty")

    out: dict[str, float] = {}
    l1s, l2s, cosines = [], [], []
    for k, (p, t) in enumerate(zip(preds, targets), start=1):
        if p.shape != t.shape:
            raise ValueError(f"horizon {k} shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
        diff = p.detach().float() - t.detach().float()
        l1 = diff.abs().mean().item()
        l2 = (diff.pow(2).mean()).sqrt().item()
        # Cosine similarity over flattened maps, averaged over batch.
        pf = p.detach().float().flatten(1)
        tf = t.detach().float().flatten(1)
        cos = torch.nn.functional.cosine_similarity(pf, tf, dim=1).mean().item()
        out[f"l1@h{k}"] = float(l1)
        out[f"l2@h{k}"] = float(l2)
        out[f"cosine@h{k}"] = float(cos)
        l1s.append(l1)
        l2s.append(l2)
        cosines.append(cos)

    out["l1"] = float(np.mean(l1s))
    out["l2"] = float(np.mean(l2s))
    out["cosine"] = float(np.mean(cosines))
    out["num_horizons"] = float(len(preds))
    return out


def null_predictor_metrics(target_features: Sequence[torch.Tensor]) -> dict[str, float]:
    """Baseline: predict zeros (same shape as targets). Higher L1/L2 than a
    trained WM; cosine near 0. Used to contextualise absolute recon numbers."""
    targets = _as_list(target_features)
    zeros = [torch.zeros_like(t) for t in targets]
    metrics = jepa_reconstruction_metrics(zeros, targets)
    return {f"null_{k}": v for k, v in metrics.items()}


def relative_jepa_improvement(
    model_metrics: dict[str, float],
    null_metrics: dict[str, float],
) -> dict[str, float]:
    """Fractional L1/L2 reduction vs the null (zero) predictor.

    ``1.0`` means perfect reconstruction relative to null; ``0.0`` means no
    better than predicting zeros.
    """
    out: dict[str, float] = {}
    for key in ("l1", "l2"):
        null_v = null_metrics.get(f"null_{key}", 0.0)
        model_v = model_metrics.get(key, 0.0)
        if null_v <= 1e-12:
            out[f"rel_improvement_{key}"] = 0.0
        else:
            out[f"rel_improvement_{key}"] = float(max(0.0, (null_v - model_v) / null_v))
    return out


def trajectory_impact_metrics(
    reactive_traj: torch.Tensor,
    combined_traj: torch.Tensor,
) -> dict[str, float]:
    """How much the World Model changes the reactive trajectory.

    Both tensors are AutoE2E trajectory outputs ``[B, T*2]`` (or ``[B, T, 2]``).
    """
    r = reactive_traj.detach().float()
    c = combined_traj.detach().float()
    if r.shape != c.shape:
        raise ValueError(f"trajectory shape mismatch: {tuple(r.shape)} vs {tuple(c.shape)}")
    diff = c - r
    return {
        "trajectory_l2": float(diff.pow(2).mean().sqrt().item()),
        "trajectory_l1": float(diff.abs().mean().item()),
        "trajectory_max_abs": float(diff.abs().max().item()),
    }


def _split_controls(traj: torch.Tensor, num_timesteps: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """AutoE2E flattens (accel, curv) as ``[B, T*2]`` interleaved or stacked.

    The planners emit ``[B, T, 2]`` with last dim = (accel, curvature) in most
    paths; accept both ``[B, T, 2]`` and ``[B, T*2]`` (accel then curv blocks).
    """
    t = traj.detach().float().cpu().numpy()
    if t.ndim == 3 and t.shape[-1] == 2:
        return t[..., 0], t[..., 1]
    if t.ndim == 2 and t.shape[1] == num_timesteps * 2:
        # Common layout: [accel_0..accel_T-1, curv_0..curv_T-1]
        accel = t[:, :num_timesteps]
        curv = t[:, num_timesteps:]
        return accel, curv
    if t.ndim == 2 and t.shape[1] % 2 == 0:
        # Interleaved [a0,c0,a1,c1,...]
        paired = t.reshape(t.shape[0], -1, 2)
        return paired[..., 0], paired[..., 1]
    raise ValueError(f"unrecognised trajectory shape {t.shape}")


def open_loop_pair_metrics(
    reactive_traj: torch.Tensor,
    combined_traj: torch.Tensor,
    gt_accel: np.ndarray,
    gt_curv: np.ndarray,
    initial_speed: np.ndarray,
    *,
    num_timesteps: int = 64,
) -> dict[str, float]:
    """ADE/FDE for Reactive and Combined against the same GT controls."""
    r_a, r_c = _split_controls(reactive_traj, num_timesteps)
    c_a, c_c = _split_controls(combined_traj, num_timesteps)
    reactive = compute_open_loop_metrics(r_a, r_c, gt_accel, gt_curv, initial_speed)
    combined = compute_open_loop_metrics(c_a, c_c, gt_accel, gt_curv, initial_speed)
    out: dict[str, float] = {}
    for k, v in reactive.items():
        out[f"reactive_{k}"] = v
    for k, v in combined.items():
        out[f"combined_{k}"] = v
    out["ade3s_delta_combined_minus_reactive"] = (
        out["combined_ADE@3s"] - out["reactive_ADE@3s"]
    )
    return out


def _traj(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def world_model_trajectory_impact(
    model: torch.nn.Module,
    camera_tiles: torch.Tensor,
    map_context: torch.Tensor,
    visual_history: torch.Tensor,
    egomotion_history: torch.Tensor,
    *,
    projection=None,
    geometry_type: Optional[str] = None,
) -> dict[str, float]:
    """Run the same batch with WM buffer bypassed vs active (Combined).

    Requires ``model`` built with ``enable_world_model=True``. Compares:
    - reactive path: pass through without updating / using WM history rewrite
      by temporarily disabling the WM forward contribution (zero visual history
      rewrite — uses the caller-supplied ``visual_history`` only);
    - combined path: normal Combined forward with rolling WM history.

    For a fair open-loop comparison on a single tick, both runs use the provided
    ``visual_history``; the Combined run additionally advances the WM buffer so
    subsequent ticks would diverge — on a single tick the impact is the WM's
    rewrite of ``visual_history`` when ``history_frames`` is not supplied.
    """
    if getattr(model, "World_Action_Model_E2E", None) is None:
        raise ValueError("world_model_trajectory_impact requires enable_world_model=True")

    model.eval()
    if hasattr(model, "reset_visual_history"):
        model.reset_visual_history()

    # Combined (WM enabled as constructed)
    combined_out = model(
        camera_tiles, map_context, visual_history, egomotion_history,
        projection=projection, geometry_type=geometry_type, mode="infer",
    )
    combined_traj = _traj(combined_out)

    # Reactive reference: same weights but force WM off for one forward by
    # swapping the module pointer temporarily.
    wam = model.World_Action_Model_E2E
    buf = model.visual_history_buffer
    model.World_Action_Model_E2E = None
    model.visual_history_buffer = None
    try:
        if hasattr(model, "reset_visual_history"):
            pass
        reactive_out = model(
            camera_tiles, map_context, visual_history, egomotion_history,
            projection=projection, geometry_type=geometry_type, mode="infer",
        )
        reactive_traj = _traj(reactive_out)
    finally:
        model.World_Action_Model_E2E = wam
        model.visual_history_buffer = buf

    return trajectory_impact_metrics(reactive_traj, combined_traj)


def summarize_world_model_quality(
    *,
    jepa: Optional[dict[str, float]] = None,
    impact: Optional[dict[str, float]] = None,
    open_loop: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Merge metric dicts under a stable schema for JSON / CLI output."""
    out: dict[str, float] = {}
    if jepa:
        out.update({f"jepa_{k}": float(v) for k, v in jepa.items()})
    if impact:
        out.update({f"impact_{k}": float(v) for k, v in impact.items()})
    if open_loop:
        out.update({f"ol_{k}": float(v) for k, v in open_loop.items()})
    return out


def _jepa_targets_from_frames(
    wam: torch.nn.Module, future_frames: torch.Tensor
) -> list[torch.Tensor]:
    future_obs = [future_frames[:, k] for k in range(wam.num_future_steps)]
    return wam.target(future_obs)


def measure_jepa_on_batch(
    model: torch.nn.Module,
    camera_tiles: torch.Tensor,
    map_context: torch.Tensor,
    visual_history: torch.Tensor,
    egomotion_history: torch.Tensor,
    history_frames: torch.Tensor,
    future_frames: torch.Tensor,
    trajectory_target: torch.Tensor,
) -> dict[str, float]:
    """JEPA recon + Reactive/Combined ADE on one batch (model in eval)."""
    wam = getattr(model, "World_Action_Model_E2E", None)
    if wam is None:
        raise ValueError("measure_jepa_on_batch requires enable_world_model=True")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(
                camera_tiles, map_context, visual_history, egomotion_history,
                mode="train",
                trajectory_target=trajectory_target,
                history_frames=history_frames,
                future_frames=future_frames,
            )
            _traj_pred, aux = out
            pred_maps = aux["future_state_pred"]
            target_maps = _jepa_targets_from_frames(wam, future_frames)
            jepa = jepa_reconstruction_metrics(pred_maps, target_maps)
            null = null_predictor_metrics(target_maps)
            rel = relative_jepa_improvement(jepa, null)

            impact = world_model_trajectory_impact(
                model, camera_tiles, map_context, visual_history, egomotion_history,
            )

            speed = np.full(_traj_pred.shape[0], 5.0)
            gt_a, gt_c = _split_controls(trajectory_target.detach())
            wam_mod = model.World_Action_Model_E2E
            buf = model.visual_history_buffer
            combined_infer = _traj(model(
                camera_tiles, map_context, visual_history, egomotion_history,
                mode="infer",
            ))
            model.World_Action_Model_E2E = None
            model.visual_history_buffer = None
            try:
                reactive_infer = _traj(model(
                    camera_tiles, map_context, visual_history, egomotion_history,
                    mode="infer",
                ))
            finally:
                model.World_Action_Model_E2E = wam_mod
                model.visual_history_buffer = buf
            open_loop = open_loop_pair_metrics(
                reactive_infer, combined_infer, gt_a, gt_c, speed,
            )
    finally:
        if was_training:
            model.train()

    return summarize_world_model_quality(
        jepa={**jepa, **null, **rel},
        impact=impact,
        open_loop=open_loop,
    )


def train_world_model_quality(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    steps: int = 20,
    lr: float = 1e-3,
) -> dict[str, float]:
    """Train Combined (IL + JEPA) for ``steps``, then measure held-in-batch quality.

    This is the review-facing experiment: numbers come from a *trained* model,
    not random init. Pass packed-shard tensors the same way ``gpu_verify_train``
    does when a real ``--shard-dir`` is available.
    """
    wam = getattr(model, "World_Action_Model_E2E", None)
    if wam is None:
        raise ValueError("train_world_model_quality requires enable_world_model=True")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    traj_loss_fn = torch.nn.SmoothL1Loss()
    history: list[float] = []

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch["camera_tiles"], batch["map_context"],
            batch["visual_history"], batch["egomotion_history"],
            mode="train",
            trajectory_target=batch["trajectory_target"],
            history_frames=batch["history_frames"],
            future_frames=batch["future_frames"],
        )
        trajectory, aux = out
        loss = traj_loss_fn(trajectory, batch["trajectory_target"])
        jepa = wam.jepa_loss(aux["future_state_pred"], aux["future_frames"])
        total = loss + jepa
        total.backward()
        opt.step()
        history.append(float(total.detach()))

    metrics = measure_jepa_on_batch(
        model,
        batch["camera_tiles"], batch["map_context"],
        batch["visual_history"], batch["egomotion_history"],
        batch["history_frames"], batch["future_frames"],
        batch["trajectory_target"],
    )
    metrics["train_loss_first"] = history[0]
    metrics["train_loss_last"] = history[-1]
    metrics["train_steps"] = float(steps)
    return metrics
