"""Temporal-memory benchmark — no_memory vs one_hz vs bev_queue (Issue #20).

Compares the candidates registered in ``TEMPORAL_MEMORY_REGISTRY`` under
IDENTICAL conditions (same history shape / device), as @RyotaYamada requested
in #20 (benchmark no-memory / 1 Hz / BEV-queue style fusion). Mirrors
``planner_benchmark.py``.

Measures (no trained checkpoint needed):
  * inference latency (p50 / p99 / jitter, ms) on a [B, T, feat] history
  * parameter count
  * output context shape (sanity that the [B,T,feat] -> [B,feat] contract holds)

Run:
    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python Model/speed_benchmark/temporal_memory_benchmark.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_components.temporal_memory import (  # noqa: E402
    TEMPORAL_MEMORY_REGISTRY,
    build_temporal_memory,
)

VISUAL_DIM, EGO_DIM = 896, 256
T, BATCH = 64, 1          # 64 history steps (e.g. 6.4 s @10 Hz)
WARMUP, ITERS = 10, 50
CANDIDATES = ["no_memory", "one_hz", "bev_queue"]


def _make_history(device):
    return (torch.randn(BATCH, T, VISUAL_DIM, device=device),
            torch.randn(BATCH, T, EGO_DIM, device=device))


def _bench_one(name, device):
    torch.manual_seed(0)
    mem = build_temporal_memory(name, visual_dim=VISUAL_DIM,
                                egomotion_dim=EGO_DIM).to(device).eval()
    n_params = sum(p.numel() for p in mem.parameters())
    vis, ego = _make_history(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            v_ctx, e_ctx = mem(vis, ego)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            v_ctx, e_ctx = mem(vis, ego)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[min(len(times) - 1, int(round(len(times) * 0.99)) - 1)]
    return {
        "memory": name,
        "params": n_params,
        "latency_p50_ms": round(p50, 3),
        "latency_p99_ms": round(p99, 3),
        "jitter_ms": round(p99 - p50, 3),
        "visual_ctx_dim": tuple(v_ctx.shape),
        "ego_ctx_dim": tuple(e_ctx.shape),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in CANDIDATES:
        if name not in TEMPORAL_MEMORY_REGISTRY:
            print(f"skip {name}: not in registry")
            continue
        try:
            rows.append(_bench_one(name, device))
        except Exception as e:  # noqa: BLE001
            rows.append({"memory": name, "error": repr(e)})

    print(f"\nDevice: {device} | torch {torch.__version__} | "
          f"history [B={BATCH}, T={T}, vis={VISUAL_DIM}, ego={EGO_DIM}]\n")
    hdr = ["memory", "params", "latency_p50_ms", "latency_p99_ms", "jitter_ms"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        if "error" in r:
            print(f"| {r['memory']} | ERROR: {r['error']} |")
            continue
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"temporal_memory_benchmark_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({"device": str(device), "torch": torch.__version__,
                   "history": {"B": BATCH, "T": T, "visual_dim": VISUAL_DIM,
                               "ego_dim": EGO_DIM}, "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nNOTE: this measures cost only. Which memory helps DRIVING needs a "
          "trained model + open/closed-loop eval (see #66 / #20).")


if __name__ == "__main__":
    main()
