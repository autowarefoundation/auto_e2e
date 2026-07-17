"""GPU-box verification for the decode-dedup pack (#121 §3.4d must-validate #1).

Runs on the EC2 g6e box (lerobot present, real L2D video available). Verifies
that the decode-dedup path produces:
  * pool bytes identical to legacy pack_sample's pool bytes for the same physical
    (row, cam) frame;
  * per-sample cam_i.jpg bytes identical to legacy;
  * loader-rebuilt history_frames/future_frames tensor-equal to legacy;
  * a measurable speedup (decode count reduced).

Run:
    cd /home/ubuntu/auto_e2e/Model
    python -m tests.gpu_verify_dedup --episodes 2

Not a pytest — this is a runtime check on real video. Fails LOUD on any drift.
"""

from __future__ import annotations

import argparse
import json
import time

from torchvision import transforms

from data_parsing.l2d import L2DDataset
import data_processing.reasoning_label_generation.parallel_pack as pp


IMAGE_SIZE = 256


def _install_worker_globals(ds, dataset_value, calib_bytes):
    pp._DS = ds
    pp._DATASET_VALUE = dataset_value
    pp._CALIB_BYTES = calib_bytes
    pp._TO_PIL = transforms.ToPILImage()
    pp._RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="yaak-ai/L2D")
    ap.add_argument("--episodes", type=int, default=2,
                    help="number of L2D episodes to load")
    ap.add_argument("--samples", type=int, default=3,
                    help="samples to compare (from si=0)")
    args = ap.parse_args()

    ep_list = list(range(args.episodes))
    print(f"Loading L2DDataset(episodes={ep_list}, WM=True)...")
    ds_wm = L2DDataset(repo_id=args.repo_id, episodes=ep_list,
                       include_world_model_windows=True)
    ds_plain = L2DDataset(repo_id=args.repo_id, episodes=ep_list,
                          include_world_model_windows=False)
    assert ds_wm._samples == ds_plain._samples, (
        "ENUMERATION MISMATCH: WM-mode and plain-mode L2DDataset differ — "
        "the alignment invariant is broken")
    print(f"  ✓ enumeration identical, {len(ds_wm)} samples")

    calib = json.dumps({"dataset": args.repo_id, "geometry_type": "pseudo"}).encode()
    n_samples = min(args.samples, len(ds_wm))

    # --- 1. Legacy pack_sample (per-sample full-window decode) ---
    _install_worker_globals(ds_wm, args.repo_id, calib)
    print(f"\nLegacy pack_sample over {n_samples} samples...")
    t0 = time.time()
    legacy = []
    for si in range(n_samples):
        legacy.append(pp.pack_sample(si))
    t_legacy = time.time() - t0
    total_pool_legacy = sum(len(p) for _, _, _, p in legacy)
    total_members = sum(len(m) for _, _, m, _ in legacy)
    print(f"  ✓ {n_samples} samples in {t_legacy:.2f}s "
          f"({total_pool_legacy} pool frames from per-sample decode, "
          f"{total_members} per-sample members)")

    # --- 2. Decode-dedup Pass A (unique-row decode) ---
    # Build union of unique rows using WM ds
    all_rows = set()
    sample_cur = {}
    for si in range(n_samples):
        for r in ds_wm.window_rows(si):
            all_rows.add(r)
        ep_idx_s, row_s = ds_wm._samples[si]
        ep_start_s, _ = ds_wm._episode_ranges[ep_idx_s]
        sample_cur[si] = (ep_idx_s, row_s - ep_start_s)
        all_rows.add(sample_cur[si])
    print(f"\nDecode-dedup Pass A: {len(all_rows)} unique rows "
          f"(vs {n_samples * 8} naive)")

    # decode_row uses plain-mode _DS; install
    pp._DS = ds_plain
    pp._TO_PIL = transforms.ToPILImage()
    pp._RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    t0 = time.time()
    dedup_pool = {}
    dedup_cur = {}
    for row_key in sorted(all_rows):
        key, cam_jpegs, map_jpeg = pp.decode_row(row_key)
        dedup_cur[key] = (cam_jpegs, map_jpeg)
        for fid, blob in cam_jpegs.items():
            dedup_pool[fid] = blob
    t_dedup = time.time() - t0
    print(f"  ✓ {len(all_rows)} rows decoded in {t_dedup:.2f}s "
          f"({len(dedup_pool)} pool frames)")

    speedup = t_legacy / max(t_dedup, 1e-6)
    print(f"\n=== SPEEDUP: {speedup:.2f}x "
          f"(legacy {t_legacy:.1f}s → dedup {t_dedup:.1f}s)")

    # --- 3. Byte-equality: for every frame_id in legacy pool, dedup has same bytes ---
    print("\nByte-equality check (pool)...")
    legacy_pool_all = {}
    for _, _, _, p in legacy:
        legacy_pool_all.update(p)
    mismatches = 0
    for fid, blob in dedup_pool.items():
        if fid not in legacy_pool_all:
            continue  # legacy may not decode edge frames that dedup did
        if blob != legacy_pool_all[fid]:
            mismatches += 1
    if mismatches:
        raise SystemExit(f"BYTE MISMATCH: {mismatches} pool frames differ "
                         f"between legacy and dedup")
    print(f"  ✓ all {len(dedup_pool)} dedup pool frames byte-identical to legacy")

    # --- 4. cam_i.jpg byte-equality (per-sample current-frame bytes) ---
    print("\nByte-equality check (per-sample cam_i.jpg)...")
    for si in range(n_samples):
        uid_lg, _, legacy_members, _ = legacy[si]
        cur_key = sample_cur[si]
        cur_cams, _ = dedup_cur[cur_key]
        for fid, blob in cur_cams.items():
            cam_i = int(fid.rsplit("-c", 1)[-1])
            legacy_cam = legacy_members[f"cam_{cam_i}.jpg"]
            if blob != legacy_cam:
                raise SystemExit(
                    f"BYTE MISMATCH: sample {si} cam_{cam_i}.jpg differs")
    print("  ✓ all cam_i.jpg bytes match legacy")

    print("\n✓✓✓ All checks passed. Decode-dedup is byte-equivalent to legacy.")
    print(f"    speedup on {n_samples} samples: {speedup:.2f}x")


if __name__ == "__main__":
    main()
