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


class _FakeDS:
    """Minimal raw pre-extraction source with a deterministic per-(episode,row,cam)
    frame identity, so window_frame_ids and the pool dedup can be exercised.

    Each sample ``si`` is episode 0, row ``ROW0 + si`` (dense 10Hz). Its WM window
    references rows ``row + {-30,-20,-10,0,+10,+20,+30,+40}`` (stride 10), and the
    frame CONTENT is a deterministic function of (row, cam) — so two samples whose
    windows overlap on the same physical row produce byte-identical frames (the
    thing dedup must collapse).
    """

    ROW0 = 100  # first sample's row; >= wm past reach so no clamp

    def __init__(self, n, num_views=6, with_map=True, wm=False,
                 wm_frames=4, float_frames=True):
        self.n = n
        self.num_views = num_views
        self.with_map = with_map
        self.wm = wm
        self.wm_frames = wm_frames
        self.float_frames = float_frames
        self._wm_num_frames = wm_frames
        self._wm_stride = _WM_STRIDE

    def __len__(self):
        return self.n

    def _row(self, si):
        return self.ROW0 + si

    def sample_uid(self, si):
        return f"l2d-v1-e000000-f{self._row(si):06d}"

    def split_group_uid(self, si):
        return "l2d-e000000"

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
    }
    assert members["calib.json"] == calib
    assert uid == ds.sample_uid(0) and uid.startswith("l2d-v1-")
