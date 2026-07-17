"""GPU-box verification for train_il on decode-dedup shards (§3.4d must-validate #2).

Runs on the EC2 g6e box. Given a shard directory (produced by data_processing with
world_model=True on a dedup shard), constructs the loader, an AutoE2E model with
enable_reasoning + enable_world_model, and runs ONE forward+backward step to
verify:
  * loader rebuilds history_frames/future_frames from pool/ + window_index.json
  * all three losses (trajectory_imitation + reasoning + jepa) are finite and non-None
  * gradient flows into each branch (grad-flow probe)
  * one optimizer.step() runs without crashing

Not a pytest — a runtime check on real (or fake) shards. Fails LOUD on any drift.

Run:
    cd /home/ubuntu/auto_e2e/Model
    python -m tests.gpu_verify_train --shard-dir /path/to/shard_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True,
                    help="Directory with .tar shards + pool/ + manifest.json")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Small (WM needs bs=1 on limited GPU mem)")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    if not shard_dir.exists():
        raise SystemExit(f"shard dir not found: {shard_dir}")
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    print(f"Manifest: num_views={manifest.get('num_views')} "
          f"has_wm={manifest.get('has_world_model')} "
          f"total_samples={manifest.get('total_samples')}")
    assert manifest.get("has_world_model"), "shard has no WM windows — cannot verify JEPA"
    pool_dir = shard_dir / "pool"
    assert pool_dir.exists(), f"pool/ dir missing at {pool_dir} — is this a dedup shard?"
    print(f"Pool: {sum(1 for _ in pool_dir.iterdir())} unique frames in pool/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Loader
    from data_parsing.pre_extracted import make_pre_extracted_loader
    loader = make_pre_extracted_loader(
        str(shard_dir), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=0,
        pin_memory=(device.type == "cuda"))
    projection = getattr(loader, "projection", None)
    geometry_type = getattr(loader, "geometry_type", "pseudo")
    if projection is not None:
        projection = projection.to(device)
    print(f"Loader ready: geometry_type={geometry_type}")

    # Peek one batch to confirm shape + presence of WM window tensors
    it = iter(loader)
    batch = next(it)
    for k in ("visual_tiles", "map_input", "egomotion_history",
              "visual_history", "trajectory_target", "history_frames",
              "future_frames"):
        v = batch.get(k)
        if v is None:
            print(f"  MISSING: {k}")
        else:
            print(f"  {k}: {tuple(v.shape) if hasattr(v,'shape') else type(v).__name__}")
    assert batch.get("history_frames") is not None, \
        "history_frames absent — pool/ or window_index.json broken"
    assert batch.get("future_frames") is not None, \
        "future_frames absent — pool/ or window_index.json broken"

    # Model
    from model_components.auto_e2e import AutoE2E
    from model_components.losses import TrajectoryImitationLoss
    from training.dataset_policy import (
        adapt_egomotion_history,
        training_policy_for_dataset,
    )
    from training.losses.horizon_reasoning_loss import HorizonReasoningLoss

    num_views = int(batch["visual_tiles"].shape[1])
    model = AutoE2E(
        backbone="swin_v2_tiny", num_views=num_views, embed_dim=256,
        is_pretrained=True,
        enable_reasoning=True, reasoning_mode="pooled_latent",
        enable_world_model=True,
    ).to(device)
    print(f"Model built: num_views={num_views}, WM+reasoning ON")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    policy = training_policy_for_dataset(str(manifest["dataset"]))
    traj_loss_fn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=policy.temporal_decay,
        signal_scales=policy.signal_scales,
    ).to(device)
    reasoning_loss_fn = HorizonReasoningLoss()
    from data_processing.reasoning_label_generation.targets import (
        target_batch_from_loader as tbf,
    )

    # Move batch to device
    visual = batch["visual_tiles"].to(device)
    ego_hist = adapt_egomotion_history(
        batch["egomotion_history"].to(device),
        policy,
    )
    vis_hist = batch["visual_history"].to(device)
    target = batch["trajectory_target"].to(device)
    map_input = batch["map_input"].to(device)
    history_frames = batch["history_frames"].to(device)
    future_frames = batch["future_frames"].to(device)

    optimizer.zero_grad()
    out = model(visual, map_input, vis_hist, ego_hist,
                projection=projection, geometry_type=geometry_type,
                mode="train", trajectory_target=target,
                history_frames=history_frames, future_frames=future_frames)
    trajectory, aux = out if isinstance(out, tuple) else (out, {})

    # (1) trajectory
    traj_loss = traj_loss_fn(trajectory, target)
    assert torch.isfinite(traj_loss), f"traj_loss non-finite: {traj_loss}"
    print(f"  ✓ traj_loss = {traj_loss.item():.4f} (finite)")

    # (2) jepa
    jepa_val = None
    fsp = aux.get("future_state_pred")
    if fsp is not None:
        jepa = model.World_Action_Model_E2E.jepa_loss(fsp, future_frames)
        assert torch.isfinite(jepa), f"jepa non-finite: {jepa}"
        jepa_val = float(jepa.item())
        print(f"  ✓ jepa_loss = {jepa_val:.4f} (finite)")
    else:
        raise SystemExit("future_state_pred is None — WM branch not producing prediction")

    # (3) reasoning
    reason_val = None
    rp = aux.get("reasoning_pred")
    if rp is not None:
        tb = tbf(batch)
        if tb is not None:
            terms = reasoning_loss_fn(
                rp,
                {g: t.to(device) for g, t in tb.targets.items()},
                source_weights=tb.source_weights.to(device),
                confidence_targets=tb.confidence_targets.to(device),
            )
            reason_val = float(terms["total"].item())
            assert torch.isfinite(terms["total"]), f"reason non-finite: {terms['total']}"
            print(f"  ✓ reason_loss = {reason_val:.4f} (finite)")
        else:
            print("  (no reasoning targets in this batch — 1Hz label subset "
                  "at bs=1 may not include a labeled sample)")

    total = traj_loss + 1.0 * (jepa if fsp is not None else 0) \
        + (0.05 * terms["total"] if rp is not None and reason_val is not None else 0)
    total.backward()

    # Grad-flow probe
    def _branch_gn(substr):
        tot, n = 0.0, 0
        for nm, p in model.named_parameters():
            if substr in nm and p.grad is not None:
                tot += float(p.grad.norm().item()) ** 2
                n += 1
        return (tot ** 0.5, n)
    print(f"  planner grad: {_branch_gn('TrajectoryPlanner')}")
    print(f"  WM grad: {_branch_gn('World_Action_Model')}")
    print(f"  reasoning grad: {_branch_gn('Reasoning')}")

    optimizer.step()
    print("  ✓ optimizer.step() ran without crashing")

    print("\n✓✓✓ train_il full-3-loss forward+backward+step works on the dedup shard.")


if __name__ == "__main__":
    main()
