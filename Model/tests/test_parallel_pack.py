"""Parallel-pack correctness with deduped WM frame pool (#121 §3.4d).

data_processing's shard packing runs in ProcessPool workers
(``parallel_pack.pack_sample``): each worker decodes + JPEG-encodes a sample and
returns per-sample member BYTES PLUS a frame-pool contribution; the parent appends
members to the tar and writes each pool frame_id once. These tests pin:

  * per-sample members (cam_i.jpg, map.jpg, ego.npy, meta.json, calib.json, and —
    for WM — window_index.json) match a serial reference byte-for-byte; the WM
    window PIXELS are NOT per-sample anymore (they move to the pool);
  * the pool holds each window frame keyed by its frame_id, and the bytes are
    byte-identical to the serially-encoded window frames — so the loader rebuilds
    identical history_frames/future_frames;
  * window_index maps (step,view)→frame_id correctly and every frame_id stays in
    the sample's own episode (boundary safety);
  * cross-sample dedup actually collapses overlapping neighbour windows;
  * manifest flags derive from window_index.json presence; reasoning.json is added
    by the PARENT, never the worker.

No lerobot / video: a tiny in-memory fake dataset stands in for L2DDataset.
"""

from __future__ import annotations

import io
import json

import numpy as np
import torch
from torchvision import transforms

import data_processing.reasoning_label_generation.parallel_pack as pp

IMAGE_SIZE = 32
_WM_STRIDE = 10


from data_parsing.l2d.camera import CAMERA_NAMES as _L2D_CAM_NAMES  # type: ignore[misc]
from data_parsing.l2d.camera import MAP_VIEW_NAME as _L2D_MAP_NAME  # type: ignore[misc]


class _FakeLerobot:
    """Fake lerobot_dataset: indexing by local row returns per-camera float tensors
    keyed by the real L2D camera names. Used by decode_row worker tests.
    """

    def __init__(self, float_frames=True):
        self.float_frames = float_frames

    def __getitem__(self, row):
        item = {}
        for i, c in enumerate(_L2D_CAM_NAMES):
            g = torch.Generator().manual_seed(row * 100 + i)
            if self.float_frames:
                item[c] = torch.rand(3, 20, 24, generator=g)
            else:
                item[c] = (torch.rand(3, 20, 24, generator=g) * 255).to(torch.uint8)
        item[_L2D_MAP_NAME] = torch.zeros(3, 20, 24)
        return item


class _FakeDS:
    """Minimal raw pre-extraction source with a deterministic per-(episode,row,cam)
    frame identity, so window_frame_ids, window_rows, decode_row and the pool dedup
    can be exercised.

    Each sample ``si`` is episode 0, row ``ROW0 + si`` (dense 10Hz). Its WM window
    references rows ``row + {-30,-20,-10,0,+10,+20,+30,+40}`` (stride 10), and the
    frame CONTENT is a deterministic function of (row, cam) — so two samples whose
    windows overlap on the same physical row produce byte-identical frames (the
    thing dedup must collapse).
    """

    ROW0 = 100  # first sample's row; >= wm past reach so no clamp
    EP0_START = 0

    def __init__(self, n, num_views=6, with_map=True, wm=False,
                 wm_frames=4, float_frames=True, with_gps=False):
        self.n = n
        self.num_views = num_views
        self.with_map = with_map
        self.wm = wm
        self.wm_frames = wm_frames
        self.float_frames = float_frames
        self.with_gps = with_gps
        self._wm_num_frames = wm_frames
        self._wm_stride = _WM_STRIDE
        # Attributes used by decode_row worker
        self._samples = [(0, self.ROW0 + i) for i in range(n)]
        self._episode_ranges = {0: (self.EP0_START, self.EP0_START + self.ROW0 + n + 200)}
        # Fake lerobot_dataset for decode_row
        self.lerobot_dataset = _FakeLerobot(float_frames=float_frames)

    def __len__(self):
        return self.n

    def _row(self, si):
        return self.ROW0 + si

    def sample_uid(self, si):
        return f"l2d-v1-e000000-f{self._row(si):06d}"

    def split_group_uid(self, si):
        return "l2d-e000000"

    def frame_index(self, si):
        return self._row(si)

    def _cam_frame(self, row, cam):
        """Frame CONTENT keyed by (row, cam) — identical across samples that share
        a physical row, so dedup collapses them."""
        g = torch.Generator().manual_seed(row * 100 + cam)
        if self.float_frames:
            return torch.rand(3, 20, 24, generator=g)
        return (torch.rand(3, 20, 24, generator=g) * 255).to(torch.uint8)

    def window_frame_ids(self, si):
        row = self._row(si)
        n, s = self.wm_frames, self._wm_stride
        hist_off = [-(n - 1 - t) * s for t in range(n)]
        fut_off = [(t + 1) * s for t in range(n)]

        def ids(offsets):
            return [[f"l2d-v1-e000000-r{row + o:06d}-c{v}"
                     for v in range(self.num_views)] for o in offsets]
        return {"history": ids(hist_off), "future": ids(fut_off)}

    def window_rows(self, si):
        """Return (ep_idx, frame_index) for every window row (no decode)."""
        row = self._row(si)
        n, s = self.wm_frames, self._wm_stride
        hist_off = [-(n - 1 - t) * s for t in range(n)]
        fut_off = [(t + 1) * s for t in range(n)]
        ep_start = self.EP0_START
        return [(0, row + o - ep_start) for o in hist_off + fut_off]

    def egomotion_for(self, si):
        """Return (ego_history, trajectory) tensors without video decode."""
        ego_h = torch.arange(256, dtype=torch.float32) + si
        traj = torch.arange(128, dtype=torch.float32) - si
        return ego_h, traj

    def numeric_for(self, si):
        ego_h, traj = self.egomotion_for(si)
        if not self.with_gps:
            return ego_h, traj, None, None
        row = self._row(si)
        pose = {
            "latitude_deg": 49.0 + row / 100000,
            "longitude_deg": 11.0 + row / 100000,
            "heading_deg_cw_from_north": 90.0,
            "timestamp_ns": 1_670_000_000_000_000_000 + row * 100_000_000,
            "gps_accuracy_m": float("nan"),
        }
        gps = np.column_stack([
            49.0 + np.arange(65) / 100000,
            11.0 + np.arange(65) / 100000,
        ])
        return ego_h, traj, pose, gps

    def __getitem__(self, si):
        row = self._row(si)
        sample = {
            "visual_tiles": torch.stack(
                [self._cam_frame(row, v) for v in range(self.num_views)], dim=0),
            "egomotion_history": torch.arange(256, dtype=torch.float32) + si,
            "trajectory_target": torch.arange(128, dtype=torch.float32) - si,
        }
        if self.with_map:
            sample["map_tile"] = self._cam_frame(row, 90)
        if self.wm:
            n, s = self.wm_frames, self._wm_stride
            hist_off = [-(n - 1 - t) * s for t in range(n)]
            fut_off = [(t + 1) * s for t in range(n)]
            sample["history_frames"] = torch.stack([
                torch.stack([self._cam_frame(row + o, v)
                             for v in range(self.num_views)], dim=0)
                for o in hist_off], dim=0)
            sample["future_frames"] = torch.stack([
                torch.stack([self._cam_frame(row + o, v)
                             for v in range(self.num_views)], dim=0)
                for o in fut_off], dim=0)
        if self.with_gps:
            _, _, sample["pose_current"], sample["gps_future"] = self.numeric_for(si)
        return sample


def _ref_jpeg(frame_tensor, resize, to_pil):
    t = frame_tensor.cpu()
    if t.dtype.is_floating_point:
        t = t.clamp(0, 1)
    f = resize(to_pil(t))
    b = io.BytesIO()
    f.save(b, format="JPEG", quality=90)
    return b.getvalue()


def _install_worker_globals(ds, dataset_value, calib_bytes):
    pp._DS = ds
    pp._DATASET_VALUE = dataset_value
    pp._CALIB_BYTES = calib_bytes
    pp._TO_PIL = transforms.ToPILImage()
    pp._RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


# --------------------------------------------------------------------------
# 1. Per-sample members (no WM pixels) byte-identical; imitation-only.
# --------------------------------------------------------------------------
def test_pack_sample_imitation_members_and_no_pool():
    calib = json.dumps({"dataset": "yaak-ai/L2D", "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(3, num_views=6, with_map=True, wm=False, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    resize, to_pil = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToPILImage()

    for si in range(len(ds)):
        uid, nviews, members, frame_pool = pp.pack_sample(si)
        assert uid == ds.sample_uid(si)
        assert nviews == 6
        assert frame_pool == {}                     # no WM → empty pool
        assert "window_index.json" not in members
        assert "reasoning.json" not in members      # parent adds it, not the worker
        # cam + map bytes match the serial reference
        sample = ds[si]
        for cam_i in range(6):
            assert members[f"cam_{cam_i}.jpg"] == _ref_jpeg(
                sample["visual_tiles"][cam_i], resize, to_pil)
        assert members["map.jpg"] == _ref_jpeg(sample["map_tile"], resize, to_pil)


# --------------------------------------------------------------------------
# 2. WM: pixels move to the pool; window_index maps to byte-identical frames.
# --------------------------------------------------------------------------
def test_pack_sample_wm_pool_and_window_index_byte_identical():
    calib = json.dumps({"dataset": "yaak-ai/L2D", "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(1, num_views=6, with_map=True, wm=True, wm_frames=4, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    resize, to_pil = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToPILImage()

    uid, nviews, members, frame_pool = pp.pack_sample(0)
    # NO per-sample hist_/fut_ members anymore — only window_index.json.
    assert not any(k.startswith(("hist_", "fut_")) for k in members)
    assert "window_index.json" in members
    idx = json.loads(members["window_index.json"])
    assert len(idx["history"]) == 4 and len(idx["future"]) == 4
    assert all(len(step) == 6 for step in idx["history"] + idx["future"])

    # Pool holds one entry per distinct window frame_id (4+4 steps × 6 views = 48).
    assert len(frame_pool) == 48
    # Each window_index frame_id resolves in the pool to the serially-encoded frame.
    sample = ds[0]
    for t, step in enumerate(idx["history"]):
        for v, fid in enumerate(step):
            assert frame_pool[fid] == _ref_jpeg(sample["history_frames"][t, v], resize, to_pil)
    for t, step in enumerate(idx["future"]):
        for v, fid in enumerate(step):
            assert frame_pool[fid] == _ref_jpeg(sample["future_frames"][t, v], resize, to_pil)


# --------------------------------------------------------------------------
# 3. Cross-sample dedup: overlapping neighbour windows share frame_ids.
# --------------------------------------------------------------------------
def test_cross_sample_dedup_collapses_overlap():
    """Sample si future +10 row == sample si+1 future +? / history — overlapping
    physical rows must yield the SAME frame_id (so the parent stores them once)."""
    ds = _FakeDS(12, num_views=6, with_map=True, wm=True, wm_frames=4, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", b"{}")
    # Union of all pool frame_ids across samples vs the naive per-sample total.
    seen = set()
    naive_total = 0
    for si in range(len(ds)):
        _, _, _, frame_pool = pp.pack_sample(si)
        naive_total += len(frame_pool)
        seen |= set(frame_pool)
    # Naive per-sample storage is 48/sample; the deduped union is far smaller
    # because consecutive 10Hz samples' stride-10 windows overlap heavily.
    assert naive_total == 12 * 48
    assert len(seen) < naive_total          # dedup actually collapses frames
    # Every frame_id is content-addressed by (episode,row,cam) → dot-free, safe key.
    for fid in seen:
        assert "." not in fid and fid.startswith("l2d-v1-e000000-r")


# --------------------------------------------------------------------------
# 4. Boundary safety: every window frame_id is in the sample's own episode.
# --------------------------------------------------------------------------
def test_window_ids_stay_in_episode():
    ds = _FakeDS(2, num_views=6, with_map=True, wm=True, wm_frames=4, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", b"{}")
    _, _, members, _ = pp.pack_sample(0)
    idx = json.loads(members["window_index.json"])
    for step in idx["history"] + idx["future"]:
        for fid in step:
            # all ids carry the sample's episode (e000000) — never a neighbour clip
            assert fid.startswith("l2d-v1-e000000-r")


# --------------------------------------------------------------------------
# 5. Manifest flags now derive from window_index.json presence.
# --------------------------------------------------------------------------
def _derive_flags(ds, dataset_value, n):
    _install_worker_globals(ds, dataset_value, b"{}")
    num_views, has_map, has_wm, count = 0, False, False, 0
    for si in range(n):
        _, nviews, members, _ = pp.pack_sample(si)
        num_views = nviews
        has_map = has_map or ("map.jpg" in members)
        has_wm = has_wm or ("window_index.json" in members)
        count += 1
    return {
        "num_views": num_views if count else 0,
        "has_map": bool(count) and has_map,
        "has_world_model": bool(count) and has_wm,
    }


def test_manifest_flags_l2d_wm():
    ds = _FakeDS(3, num_views=6, with_map=True, wm=True, float_frames=True)
    assert _derive_flags(ds, "yaak-ai/L2D", len(ds)) == {
        "num_views": 6, "has_map": True, "has_world_model": True}


def test_manifest_flags_nvidia_no_map_no_wm():
    ds = _FakeDS(2, num_views=7, with_map=False, wm=False, float_frames=False)
    assert _derive_flags(ds, "nvidia/PhysicalAI-Autonomous-Vehicles", 2) == {
        "num_views": 7, "has_map": False, "has_world_model": False}


def test_manifest_flags_empty_input():
    ds = _FakeDS(0, num_views=6, with_map=True, wm=True)
    assert _derive_flags(ds, "yaak-ai/L2D", 0) == {
        "num_views": 0, "has_map": False, "has_world_model": False}


# --------------------------------------------------------------------------
# 6. ego.npy + meta.json + global uid unchanged.
# --------------------------------------------------------------------------
def test_ego_meta_uid_unchanged():
    calib = b"{}"
    ds = _FakeDS(1, num_views=6, with_map=True, wm=False, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    uid, _, members, _ = pp.pack_sample(0)
    arr = np.frombuffer(members["ego.npy"], dtype=np.float32)
    assert arr.shape == (256 + 128,)
    np.testing.assert_array_equal(arr[:256], (np.arange(256) + 0).astype(np.float32))
    np.testing.assert_array_equal(arr[256:], (np.arange(128) - 0).astype(np.float32))
    assert json.loads(members["meta.json"]) == {
        "idx": 0, "dataset": "yaak-ai/L2D",
        "sample_uid": uid, "split_group_uid": ds.split_group_uid(0),
        "split_bucket": 2, "frame_idx": ds.frame_index(0),
    }
    assert members["calib.json"] == calib
    assert uid == ds.sample_uid(0) and uid.startswith("l2d-v1-")


def test_pack_sample_adds_pose_and_gps_atomically():
    from data_processing.geospatial import decode_gps_future, decode_pose

    ds = _FakeDS(
        1, num_views=6, with_map=True, wm=False,
        float_frames=True, with_gps=True,
    )
    _install_worker_globals(ds, "yaak-ai/L2D", b"{}")
    _, _, members, _ = pp.pack_sample(0)

    assert set(("pose.npy", "gps.npy")).issubset(members)
    pose = decode_pose(members["pose.npy"])
    gps = decode_gps_future(members["gps.npy"])
    assert pose["timestamp_ns"] == ds.numeric_for(0)[2]["timestamp_ns"]
    np.testing.assert_array_equal(gps, ds.numeric_for(0)[3])


# --------------------------------------------------------------------------
# 7. Decode-dedup: row-level workers (#121 §3.4d decode fix)
# --------------------------------------------------------------------------

def _install_row_worker_globals(ds, calib_bytes):
    """Set globals for decode_row worker (init_row_worker equivalent)."""
    pp._DS = ds
    pp._CALIB_BYTES = calib_bytes
    pp._TO_PIL = transforms.ToPILImage()
    pp._RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def test_decode_row_returns_correct_frame_ids():
    """decode_row produces frame_ids keyed by global (ep,row,cam) identity."""
    ds = _FakeDS(3, num_views=6, with_map=True, wm=True, float_frames=True)
    _install_row_worker_globals(ds, b"{}")
    # decode row frame_index=100 for episode 0
    (ep_idx, fi), cam_jpegs, _ = pp.decode_row((0, 100))
    assert ep_idx == 0 and fi == 100
    assert len(cam_jpegs) == 6
    for v in range(6):
        fid = f"l2d-v1-e000000-r000100-c{v}"
        assert fid in cam_jpegs, f"missing {fid}"


def test_decode_row_bytes_match_pack_sample_pool():
    """THE byte-equality guarantee: decode_row produces the SAME jpeg bytes as
    pack_sample's pool for the same physical (row, cam)."""
    ds = _FakeDS(1, num_views=6, with_map=True, wm=True, float_frames=True)
    _install_row_worker_globals(ds, b"{}")

    # Get pool bytes from pack_sample for si=0 (row=100, offset-0 = hist[-1]).
    _install_worker_globals(ds, "yaak-ai/L2D", b"{}")
    _, _, _, ps_pool = pp.pack_sample(0)

    # Get bytes from decode_row for the same row (frame_index 100 = row 100).
    _install_row_worker_globals(ds, b"{}")
    _, dr_cams, _ = pp.decode_row((0, 100))

    # Frame_id for hist[-1]=offset-0 in pack_sample pool matches decode_row.
    for v in range(6):
        fid = f"l2d-v1-e000000-r000100-c{v}"
        assert fid in ps_pool, f"{fid} not in pack_sample pool"
        assert fid in dr_cams, f"{fid} not in decode_row cams"
        assert ps_pool[fid] == dr_cams[fid], (
            f"byte mismatch for {fid}: pack_sample pool vs decode_row")


def test_window_rows_covers_all_window_offsets():
    """window_rows returns all 8 window offsets for a sample without decoding."""
    ds = _FakeDS(5, wm=True, wm_frames=4)
    rows = ds.window_rows(0)
    assert len(rows) == 8   # 4 hist + 4 fut
    # All in episode 0, no negative frame_index.
    ep_start = ds.EP0_START
    for ep_idx, fi in rows:
        assert ep_idx == 0
        assert fi >= 0, f"negative frame_index {fi} — crossed episode start"


def test_decode_count_is_unique_rows_not_8x():
    """Simulated decode count: union of window_rows across samples << n_samples × 8."""
    n = 12  # 12 consecutive 10Hz samples, windows heavily overlap
    ds = _FakeDS(n, wm=True, wm_frames=4)
    all_rows = set()
    for si in range(n):
        for r in ds.window_rows(si):
            all_rows.add(r)
    naive_total = n * 8   # old path decoded 8 frames × n samples
    assert len(all_rows) < naive_total, (
        f"unique rows {len(all_rows)} should be less than naive {naive_total}")
    print(f"unique rows: {len(all_rows)} vs naive {naive_total} "
          f"(dedup ratio {naive_total/len(all_rows):.1f}x)")


# --------------------------------------------------------------------------
# 8. End-to-end byte-equality: decode-dedup path produces IDENTICAL tensors
#    to the legacy pack_sample path, when reconstructed by the loader.
# --------------------------------------------------------------------------

def _simulate_decode_dedup_shard(ds, dataset_value, calib_bytes):
    """Simulate the decode-dedup Pass A + Pass B parent loop in-process.
    Returns (sample_members_by_uid, pool_bytes_by_frame_id).
    """
    _install_worker_globals(ds, dataset_value, calib_bytes)
    _install_row_worker_globals(ds, calib_bytes)

    # Pass A: unique rows + decode
    all_rows = set()
    sample_cur_rows = {}
    for si in range(len(ds)):
        for r in ds.window_rows(si):
            all_rows.add(r)
        ep_idx_s, row_s = ds._samples[si]
        ep_start_s, _ = ds._episode_ranges[ep_idx_s]
        cur_fi = row_s - ep_start_s
        sample_cur_rows[si] = (ep_idx_s, cur_fi)
        all_rows.add((ep_idx_s, cur_fi))

    row_map = {}
    pool_bytes = {}
    for row_key in sorted(all_rows):
        key, cam_jpegs, map_jpeg = pp.decode_row(row_key)
        row_map[key] = (cam_jpegs, map_jpeg)
        for fid, blob in cam_jpegs.items():
            pool_bytes[fid] = blob

    # Pass B: assemble each sample's members from pool (no decode)
    import numpy as np
    sample_members = {}
    for si in range(len(ds)):
        uid = ds.sample_uid(si)
        members = {}
        ids = ds.window_frame_ids(si)
        members["window_index.json"] = json.dumps(ids).encode()
        cur_key = sample_cur_rows[si]
        cur_cams, cur_map = row_map[cur_key]
        for fid, blob in sorted(cur_cams.items(),
                                key=lambda kv: int(kv[0].rsplit("-c", 1)[-1])):
            cam_i = int(fid.rsplit("-c", 1)[-1])
            members[f"cam_{cam_i}.jpg"] = blob
        if cur_map is not None:
            members["map.jpg"] = cur_map
        ego_h, traj = ds.egomotion_for(si)
        ego_data = np.concatenate([ego_h.numpy(), traj.numpy()]).astype(np.float32)
        members["ego.npy"] = ego_data.tobytes()
        members["meta.json"] = json.dumps({
            "idx": si, "dataset": dataset_value,
            "sample_uid": uid, "split_group_uid": ds.split_group_uid(si),
            "split_bucket": 2, "frame_idx": ds.frame_index(si),
        }).encode()
        members["calib.json"] = calib_bytes
        sample_members[uid] = members
    return sample_members, pool_bytes


def test_end_to_end_dedup_pack_byte_identical_to_legacy():
    """THE ultimate byte-equality guarantee: the decode-dedup path produces the
    SAME per-sample members and SAME pool jpeg bytes as the legacy pack_sample
    path for the same physical frames. (For overlapping frames the pool bytes are
    exactly what legacy pool would have contained.)
    """
    calib = json.dumps({"dataset": "yaak-ai/L2D", "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(3, num_views=6, with_map=True, wm=True, wm_frames=4, float_frames=True)

    # Simulate decode-dedup
    dd_members, dd_pool = _simulate_decode_dedup_shard(ds, "yaak-ai/L2D", calib)

    # Compare per-sample cam_i.jpg + pool frame_ids to legacy pack_sample output
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    legacy_pool_all = {}
    for si in range(len(ds)):
        uid, _, legacy_members, legacy_pool = pp.pack_sample(si)
        # cam_i, ego, meta, calib bytes must match
        for cam_i in range(6):
            k = f"cam_{cam_i}.jpg"
            assert dd_members[uid][k] == legacy_members[k], f"cam mismatch {uid}/{k}"
        assert dd_members[uid]["ego.npy"] == legacy_members["ego.npy"]
        assert dd_members[uid]["meta.json"] == legacy_members["meta.json"]
        # window_index.json byte-identical
        assert dd_members[uid]["window_index.json"] == legacy_members["window_index.json"]
        legacy_pool_all.update(legacy_pool)

    # dd_pool ⊆ legacy_pool_all (dedup keeps fewer bytes, but each frame_id maps
    # to identical jpeg bytes).
    for fid, blob in dd_pool.items():
        assert fid in legacy_pool_all, f"missing frame_id {fid} in legacy pool"
        assert blob == legacy_pool_all[fid], f"pool byte mismatch for {fid}"


def test_end_to_end_dedup_loader_produces_identical_tensors():
    """Given a decode-dedup shard (members + pool), the loader (pre_extracted
    _decode_sample) rebuilds history_frames/future_frames identical to what the
    legacy shard would produce."""
    from data_parsing.pre_extracted import _decode_sample

    calib = json.dumps({"dataset": "yaak-ai/L2D", "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(2, num_views=6, with_map=True, wm=True, wm_frames=4, float_frames=True)

    # Build dedup shard
    dd_members, dd_pool = _simulate_decode_dedup_shard(ds, "yaak-ai/L2D", calib)

    # Loader reads sample with pool accessor
    pool_fn = lambda fid: dd_pool[fid]
    uid = ds.sample_uid(0)
    sample = dict(dd_members[uid])
    sample["__key__"] = uid
    out_dedup = _decode_sample(sample, pool=pool_fn)

    # For comparison, run legacy pack_sample to get pool + rebuild with the loader
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    uid_lg, _, legacy_members, legacy_pool = pp.pack_sample(0)
    sample_lg = dict(legacy_members)
    sample_lg["__key__"] = uid_lg
    pool_lg_fn = lambda fid: legacy_pool[fid]
    out_legacy = _decode_sample(sample_lg, pool=pool_lg_fn)

    # history/future must be tensor-equal
    assert torch.equal(out_dedup["history_frames"], out_legacy["history_frames"])
    assert torch.equal(out_dedup["future_frames"], out_legacy["future_frames"])
    assert torch.equal(out_dedup["visual_tiles"], out_legacy["visual_tiles"])
