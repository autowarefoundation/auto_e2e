"""Swappable-planner benchmark harness — flow_matching vs bezier.

Compares the trajectory planners registered in ``PLANNER_REGISTRY`` under
IDENTICAL conditions (same embed_dim / horizon / inputs / device), as requested
by @RyotaYamada in #56 (Zain to lead the Bézier-vs-Flow-Matching decision).

What this harness measures NOW (no trained checkpoint, no dataset, no simulator):
  * inference latency  (p50 / p99 / jitter, ms)
  * parameter count
  * architectural smoothness on the (acceleration, curvature) unicycle output:
      - jerk proxy        = Var(Δ acceleration)   (lower = smoother)
      - curvature change  = Var(Δ curvature)      (lower = smoother)
    These hold even with random weights (Bézier's Bernstein basis is smooth by
    construction), so they are a fair *architectural* comparison.

What it does NOT measure yet (TODO — require a trained checkpoint + data/sim):
  * ADE / FDE                      -> needs ground-truth trajectories (KITScenes / L2D)
  * off-road rate, collision /     -> needs map + agents + a metric/sim
    near-collision
  * closed-loop stability          -> needs NAVSIM / Bench2Drive / HUGSIM

Run:
    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python Model/speed_benchmark/planner_benchmark.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_components.trajectory_planning import PLANNER_REGISTRY, build_planner  # noqa: E402

CONFIG = dict(embed_dim=256, num_timesteps=64, num_signals=2,
              egomotion_dim=256, visual_history_dim=896)
# gru was removed from PLANNER_REGISTRY in the Reactive_E2E refactor; the
# remaining registered planners are flow_matching and bezier. (Any name not in
# the registry is skipped gracefully in main().)
PLANNERS = ["flow_matching", "bezier"]
WARMUP, ITERS, BATCH, H, W = 10, 50, 1, 8, 8


def _make_inputs(device):
    bev = torch.randn(BATCH, CONFIG["embed_dim"], H, W, device=device)
    vis = torch.randn(BATCH, CONFIG["visual_history_dim"], device=device)
    ego = torch.randn(BATCH, CONFIG["egomotion_dim"], device=device)
    return bev, vis, ego


def _smoothness(traj):
    """traj [B, T*S] -> Var(Δaccel), Var(Δcurvature) on the (accel, curv) channels."""
    t = traj.view(traj.shape[0], CONFIG["num_timesteps"], CONFIG["num_signals"])
    accel, curv = t[..., 0], t[..., 1]
    jerk = (accel[:, 1:] - accel[:, :-1]).var().item()
    dcurv = (curv[:, 1:] - curv[:, :-1]).var().item()
    return float(jerk), float(dcurv)


def _bench_one(name, device):
    torch.manual_seed(0)
    planner = build_planner(name, **CONFIG).to(device).eval()
    n_params = sum(p.numel() for p in planner.parameters())
    bev, vis, ego = _make_inputs(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = planner(bev, vis, ego)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = planner(bev, vis, ego)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    # Planners return the trajectory tensor directly ([B, T*S]) — the old
    # (trajectory, ...) tuple contract was removed in the Reactive_E2E refactor.
    jerk, dcurv = _smoothness(out)
    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[min(len(times) - 1, int(round(len(times) * 0.99)) - 1)]
    return {
        "planner": name,
        "params": n_params,
        "latency_p50_ms": round(p50, 3),
        "latency_p99_ms": round(p99, 3),
        "jitter_ms": round(p99 - p50, 3),
        "jerk_var_dAccel": jerk,
        "curv_var_dCurv": dcurv,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in PLANNERS:
        if name not in PLANNER_REGISTRY:
            print(f"skip {name}: not in registry")
            continue
        try:
            rows.append(_bench_one(name, device))
        except Exception as e:  # noqa: BLE001
            rows.append({"planner": name, "error": repr(e)})

    print(f"\nDevice: {device} | torch {torch.__version__} | "
          f"batch={BATCH} warmup={WARMUP} iters={ITERS}\n")
    hdr = ["planner", "params", "latency_p50_ms", "latency_p99_ms",
           "jitter_ms", "jerk_var_dAccel", "curv_var_dCurv"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        if "error" in r:
            print(f"| {r['planner']} | ERROR: {r['error']} |")
            continue
        print("| " + " | ".join(
            f"{r[h]:.3e}" if h.startswith(("jerk", "curv")) else str(r[h])
            for h in hdr) + " |")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"planner_benchmark_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({"device": str(device), "torch": torch.__version__,
                   "config": CONFIG, "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nNOTE: ADE/FDE, off-road, collision and closed-loop require a trained "
          "checkpoint + dataset/simulator — not computed here (see module docstring).")


if __name__ == "__main__":
    main()
