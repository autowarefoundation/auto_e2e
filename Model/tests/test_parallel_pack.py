"""Parallel-pack correctness (perf refactor f52a90e / 68a6f6f).

data_processing's shard packing was moved into ProcessPool workers
(``parallel_pack.pack_sample``): each worker decodes + JPEG-encodes a sample and
returns per-member BYTES; the parent appends them to the tar. The refactor claims
the shard is BYTE-IDENTICAL to the old serial packer. These tests pin that:

  * ``pack_sample`` returns exactly the member suffixes the serial packer wrote
    (cam_i.jpg, map.jpg, hist/fut_*.jpg, ego.npy, meta.json, calib.json) with
    byte-identical content — reasoning.json is added by the PARENT, never by the
    worker (labels are not shipped into workers);
  * the manifest flags (num_views / has_map / has_world_model) the parent derives
    from the packed members equal the serial values, and are correct at n=0.

No lerobot / video: a tiny in-memory fake dataset stands in for L2DDataset /
NvidiaAVDataset. ``pack_sample`` reads its inputs from module globals (set by
``init_pack_worker`` in each child); we set those globals directly and call
``pack_sample`` in-process, which exercises the identical decode+encode path the
workers run.
"""

from __future__ import annotations

import io
import json

import numpy as np
import torch
from torchvision import transforms

import data_processing.reasoning_label_generation.parallel_pack as pp

IMAGE_SIZE = 32


# --------------------------------------------------------------------------
# Fakes + a serial reference packer (mirrors the OLD serial loop in
# data_processing, pre-f52a90e) so we can assert byte-identity independently.
# --------------------------------------------------------------------------
class _FakeDS:
    """Minimal raw pre-extraction source: returns deterministic RAW frames.

    ``float_frames`` picks the L2D case (float [0,1] frames, clamped in _jpeg) vs
    the NVIDIA case (uint8 [0,255] passthrough). ``num_views`` / ``with_map`` /
    ``wm`` shape the sample exactly like the real datasets' __getitem__.
    """

    def __init__(self, n, num_views=6, with_map=True, wm=False,
                 wm_frames=4, float_frames=True):
        self.n = n
        self.num_views = num_views
        self.with_map = with_map
        self.wm = wm
        self.wm_frames = wm_frames
        self.float_frames = float_frames

    def __len__(self):
        return self.n

    def sample_uid(self, si):
        # Deterministic global-style uid (mirrors the real parsers' contract).
        return f"l2d-v1-e000000-f{si:06d}"

    def split_group_uid(self, si):
        return "l2d-e000000"

    def _frame(self, seed):
        g = torch.Generator().manual_seed(seed)
        if self.float_frames:
            return torch.rand(3, 20, 24, generator=g)
        return (torch.rand(3, 20, 24, generator=g) * 255).to(torch.uint8)

    def __getitem__(self, si):
        sample = {
            "visual_tiles": torch.stack(
                [self._frame(si * 100 + v) for v in range(self.num_views)], dim=0),
            "egomotion_history": torch.arange(256, dtype=torch.float32) + si,
            "trajectory_target": torch.arange(128, dtype=torch.float32) - si,
        }
        if self.with_map:
            sample["map_tile"] = self._frame(si * 100 + 90)
        if self.wm:
            sample["history_frames"] = torch.stack([
                torch.stack([self._frame(si * 1000 + t * 10 + v)
                             for v in range(self.num_views)], dim=0)
                for t in range(self.wm_frames)], dim=0)
            sample["future_frames"] = torch.stack([
                torch.stack([self._frame(si * 2000 + f * 10 + v)
                             for v in range(self.num_views)], dim=0)
                for f in range(self.wm_frames)], dim=0)
        return sample


def _ref_jpeg(frame_tensor, resize, to_pil):
    """Independent re-implementation of the serial _write_jpeg encoding."""
    t = frame_tensor.cpu()
    if t.dtype.is_floating_point:
        t = t.clamp(0, 1)
    f = resize(to_pil(t))
    b = io.BytesIO()
    f.save(b, format="JPEG", quality=90)
    return b.getvalue()


def _serial_members(ds, si, dataset_value, calib_bytes, resize, to_pil):
    """The exact per-sample member set the OLD serial packer wrote (no reasoning)."""
    sample = ds[si]
    members = {}
    visual = sample["visual_tiles"]
    for cam_i in range(visual.shape[0]):
        members[f"cam_{cam_i}.jpg"] = _ref_jpeg(visual[cam_i], resize, to_pil)
    map_tile = sample.get("map_tile")
    if map_tile is not None:
        members["map.jpg"] = _ref_jpeg(map_tile, resize, to_pil)
    hist = sample.get("history_frames")
    fut = sample.get("future_frames")
    if hist is not None and fut is not None:
        for t in range(hist.shape[0]):
            for v in range(hist.shape[1]):
                members[f"hist_{t}_cam_{v}.jpg"] = _ref_jpeg(hist[t, v], resize, to_pil)
        for fh in range(fut.shape[0]):
            for v in range(fut.shape[1]):
                members[f"fut_{fh}_cam_{v}.jpg"] = _ref_jpeg(fut[fh, v], resize, to_pil)
    ego = np.concatenate([
        sample["egomotion_history"].numpy(),
        sample["trajectory_target"].numpy(),
    ]).astype(np.float32)
    members["ego.npy"] = ego.tobytes()
    members["meta.json"] = json.dumps({
        "idx": si, "dataset": dataset_value,
        "sample_uid": ds.sample_uid(si), "split_group_uid": ds.split_group_uid(si),
    }).encode()
    members["calib.json"] = calib_bytes
    return members


def _install_worker_globals(ds, dataset_value, calib_bytes):
    """Set the per-process globals init_pack_worker would set (fake DS, no lerobot)."""
    pp._DS = ds
    pp._DATASET_VALUE = dataset_value
    pp._CALIB_BYTES = calib_bytes
    pp._TO_PIL = transforms.ToPILImage()
    pp._RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


# --------------------------------------------------------------------------
# 1. Member-set + byte equality vs the serial reference.
# --------------------------------------------------------------------------
def test_pack_sample_byte_identical_l2d_imitation():
    """L2D imitation-only (float frames, 6 cams + map, no WM): members + bytes match."""
    calib = json.dumps({"dataset": "yaak-ai/L2D", "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(3, num_views=6, with_map=True, wm=False, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    to_pil = transforms.ToPILImage()

    for si in range(len(ds)):
        got_uid, nviews, members = pp.pack_sample(si)
        ref = _serial_members(ds, si, "yaak-ai/L2D", calib, resize, to_pil)
        assert got_uid == ds.sample_uid(si)
        assert nviews == 6
        assert set(members) == set(ref)          # no reasoning.json from the worker
        assert "reasoning.json" not in members
        for k in ref:
            assert members[k] == ref[k], f"byte mismatch on {k}"


def test_pack_sample_byte_identical_l2d_world_model():
    """WM branch: hist_/fut_ member names + JPEG bytes match the serial packer."""
    calib = json.dumps({"dataset": "yaak-ai/L2D", "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(2, num_views=6, with_map=True, wm=True, wm_frames=4, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    to_pil = transforms.ToPILImage()

    _, _, members = pp.pack_sample(0)
    ref = _serial_members(ds, 0, "yaak-ai/L2D", calib, resize, to_pil)
    assert set(members) == set(ref)
    # 6 cams + map + 4x6 hist + 4x6 fut + ego + meta + calib
    assert sum(k.startswith("hist_") for k in members) == 24
    assert sum(k.startswith("fut_") for k in members) == 24
    for k in ref:
        assert members[k] == ref[k], f"byte mismatch on {k}"


def test_pack_sample_byte_identical_nvidia_uint8_no_map_no_wm():
    """NVIDIA case: 7 uint8 cams, no map, no WM. uint8 passthrough encoding matches."""
    calib = json.dumps({"dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
                        "geometry_type": "pseudo"}).encode()
    ds = _FakeDS(2, num_views=7, with_map=False, wm=False, float_frames=False)
    _install_worker_globals(ds, "nvidia/PhysicalAI-Autonomous-Vehicles", calib)
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    to_pil = transforms.ToPILImage()

    _, nviews, members = pp.pack_sample(0)
    ref = _serial_members(ds, 0, "nvidia/PhysicalAI-Autonomous-Vehicles",
                          calib, resize, to_pil)
    assert nviews == 7
    assert "map.jpg" not in members
    assert not any(k.startswith(("hist_", "fut_")) for k in members)
    assert set(members) == set(ref)
    for k in ref:
        assert members[k] == ref[k], f"byte mismatch on {k}"


def test_ego_npy_is_float32_history_then_target():
    """ego.npy = float32(egomotion_history ++ trajectory_target), byte-exact."""
    calib = b"{}"
    ds = _FakeDS(1, num_views=6, with_map=True, wm=False, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", calib)
    _, _, members = pp.pack_sample(0)
    arr = np.frombuffer(members["ego.npy"], dtype=np.float32)
    assert arr.shape == (256 + 128,)
    np.testing.assert_array_equal(arr[:256], (np.arange(256) + 0).astype(np.float32))
    np.testing.assert_array_equal(arr[256:], (np.arange(128) - 0).astype(np.float32))
    assert json.loads(members["meta.json"]) == {
        "idx": 0, "dataset": "yaak-ai/L2D",
        "sample_uid": ds.sample_uid(0), "split_group_uid": ds.split_group_uid(0),
    }
    assert members["calib.json"] == calib


# --------------------------------------------------------------------------
# 2. Manifest flag derivation (parent side) from packed members.
# --------------------------------------------------------------------------
def _derive_manifest_flags(ds, dataset_value, calib_bytes, n):
    """Replicate the parent's derivation loop (workflows.py lines 390-414)."""
    _install_worker_globals(ds, dataset_value, calib_bytes)
    num_views, has_map, has_wm, sample_count = 0, False, False, 0
    for si in range(n):
        _, nviews, members = pp.pack_sample(si)
        num_views = nviews
        has_map = has_map or ("map.jpg" in members)
        has_wm = has_wm or any(k.startswith("hist_") for k in members)
        sample_count += 1
    return {
        "num_views": num_views if sample_count else 0,
        "has_map": bool(sample_count) and has_map,
        "has_world_model": bool(sample_count) and has_wm,
    }


def test_manifest_flags_l2d_wm():
    ds = _FakeDS(3, num_views=6, with_map=True, wm=True, float_frames=True)
    flags = _derive_manifest_flags(ds, "yaak-ai/L2D", b"{}", len(ds))
    assert flags == {"num_views": 6, "has_map": True, "has_world_model": True}


def test_manifest_flags_nvidia_no_map_no_wm():
    ds = _FakeDS(2, num_views=7, with_map=False, wm=False, float_frames=False)
    flags = _derive_manifest_flags(ds, "nvidia/PhysicalAI-Autonomous-Vehicles",
                                   b"{}", len(ds))
    assert flags == {"num_views": 7, "has_map": False, "has_world_model": False}


def test_manifest_flags_empty_input():
    """n_samples == 0: all flags collapse to the zero/False defaults."""
    ds = _FakeDS(0, num_views=6, with_map=True, wm=True)
    flags = _derive_manifest_flags(ds, "yaak-ai/L2D", b"{}", 0)
    assert flags == {"num_views": 0, "has_map": False, "has_world_model": False}


# --------------------------------------------------------------------------
# 3. Order + sample_uid (#121 §3.1): pack_sample returns the parser's GLOBAL uid
#    (not a positional s{si}), and meta.json carries uid + split_group_uid.
# --------------------------------------------------------------------------
def test_pack_returns_global_uid_and_meta():
    ds = _FakeDS(3, num_views=6, with_map=True, wm=False, float_frames=True)
    _install_worker_globals(ds, "yaak-ai/L2D", b"{}")
    uids = []
    for si in range(len(ds)):
        uid, _, members = pp.pack_sample(si)
        uids.append(uid)
        assert uid == ds.sample_uid(si)           # returns the parser's uid
        meta = json.loads(members["meta.json"])
        assert meta["idx"] == si
        assert meta["sample_uid"] == uid
        assert meta["split_group_uid"] == ds.split_group_uid(si)
    # Distinct, in order, and in the new global format (no positional s0000…).
    assert uids == [ds.sample_uid(i) for i in range(3)]
    assert all(u.startswith("l2d-v1-") for u in uids)
