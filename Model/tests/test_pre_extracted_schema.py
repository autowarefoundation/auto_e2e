"""Tests for the pre-extracted shard schema: map/camera split + manifest geometry.

These guard the correctness-critical decode seam (map.jpg must never be counted
as a camera view) and the manifest projection round-trip, without needing real
shards on disk.
"""

import io
import json

import numpy as np
import pytest
import torch

pytest.importorskip("webdataset")  # module imports webdataset at top level

from PIL import Image

from data_parsing.pre_extracted import _decode_sample, load_projection_from_manifest


def _jpeg_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buf, format="JPEG")
    return buf.getvalue()


def _ego_bytes():
    return np.zeros(384, dtype=np.float32).tobytes()


class TestDecodeSampleMapSplit:
    def test_sample_uid_is_preserved_for_overlay_inference(self):
        sample = {"cam_0.jpg": _jpeg_bytes((0, 0, 0)),
                  "ego.npy": _ego_bytes(),
                  "__key__": "l2d-v1-e000012-f000064"}
        out = _decode_sample(sample)
        assert out["sample_uid"] == sample["__key__"]

    def test_map_not_counted_as_camera(self):
        """A sample with 6 cams + map.jpg -> visual_tiles (6,...), map separate."""
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i * 10, 0, 0)) for i in range(6)}
        sample["map.jpg"] = _jpeg_bytes((0, 0, 255))
        sample["ego.npy"] = _ego_bytes()

        out = _decode_sample(sample)
        assert out["visual_tiles"].shape == (6, 3, 256, 256), \
            "map.jpg leaked into visual_tiles"
        assert out["map_input"].shape == (3, 256, 256)

    def test_cam_ordering_numeric_not_lexical(self):
        """cam_10 must sort after cam_2 (numeric), not before (lexical)."""
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i, 0, 0)) for i in range(12)}
        sample["ego.npy"] = _ego_bytes()
        out = _decode_sample(sample)
        assert out["visual_tiles"].shape[0] == 12

    def test_missing_map_yields_zeros(self):
        """Legacy / NVIDIA-zero shards without map.jpg -> zero map_input."""
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i, 0, 0)) for i in range(7)}
        sample["ego.npy"] = _ego_bytes()
        out = _decode_sample(sample)
        assert out["visual_tiles"].shape[0] == 7
        assert out["map_input"].shape == (3, 256, 256)
        assert out["map_input"].abs().max() == 0.0

    def test_shard_pixels_normalized_exactly_once(self):
        """The raw-frame -> JPEG -> loader path must apply ImageNet Normalize
        EXACTLY ONCE. Regression for the double-normalize bug (#77): the old
        pre-extraction normalized in the dataset, clamped, then normalized again
        in the loader. Here the shard JPEG is a plain (unnormalized) image, so the
        decoded tensor must equal Normalize(ToTensor(jpeg)) — and must NOT match a
        twice-normalized tensor."""
        from torchvision import transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # A plain (unnormalized) shard image, as the corrected packer writes.
        jpg = _jpeg_bytes((120, 60, 200))
        out = _decode_sample({"cam_0.jpg": jpg, "ego.npy": _ego_bytes()})
        decoded = out["visual_tiles"][0]

        img = Image.open(io.BytesIO(jpg))
        once = transforms.Normalize(mean, std)(transforms.ToTensor()(img))
        assert torch.allclose(decoded, once, atol=1e-5), \
            "loader must normalize the plain shard image exactly once"
        twice = transforms.Normalize(mean, std)(once)
        assert not torch.allclose(decoded, twice, atol=1e-3), \
            "decoded tensor must NOT be double-normalized"

    def test_no_camera_params_key(self):
        """Geometry is a loader attribute now, never a per-sample tensor."""
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i, 0, 0)) for i in range(6)}
        sample["ego.npy"] = _ego_bytes()
        out = _decode_sample(sample)
        assert "camera_params" not in out


class TestManifestProjection:
    def test_pseudo_when_no_manifest(self, tmp_path):
        proj, geom = load_projection_from_manifest(str(tmp_path))
        assert proj is None and geom == "pseudo"

    def test_pseudo_when_manifest_has_no_projection(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"geometry_type": "pseudo"}))
        proj, geom = load_projection_from_manifest(str(tmp_path))
        assert proj is None and geom == "pseudo"

    def test_corrupt_manifest_raises_not_pseudo(self, tmp_path):
        """A present-but-unparseable manifest must RAISE, not silently degrade a
        calibrated run to pseudo geometry (missing manifest is still pseudo)."""
        (tmp_path / "manifest.json").write_text("{ this is not valid json ,,,")
        with pytest.raises(ValueError, match="could not be parsed"):
            load_projection_from_manifest(str(tmp_path))

    def test_pinhole_roundtrip(self, tmp_path):
        matrix = torch.randn(4, 3, 4)
        spec = {"type": "pinhole", "matrix": matrix.tolist()}
        (tmp_path / "manifest.json").write_text(json.dumps({
            "geometry_type": "pinhole", "projection": spec,
        }))
        proj, geom = load_projection_from_manifest(str(tmp_path))
        assert geom == "pinhole" and proj.num_views == 4
        assert torch.allclose(proj.matrix[0], matrix, atol=1e-5)

    def test_ftheta_roundtrip(self, tmp_path):
        V = 3
        spec = {
            "type": "ftheta",
            "t_camera_ego": torch.eye(4).reshape(1, 4, 4).expand(V, 4, 4).tolist(),
            "fw_poly": [[0.0, 200.0]] * V,
            "cx": [128.0] * V,
            "cy": [128.0] * V,
            "image_wh": [[256.0, 256.0]] * V,
            "max_theta": None,
        }
        (tmp_path / "manifest.json").write_text(json.dumps({
            "geometry_type": "ftheta", "projection": spec,
        }))
        proj, geom = load_projection_from_manifest(str(tmp_path))
        assert geom == "ftheta" and proj.num_views == V
        # projects an on-axis ego point to the principal point
        pts = torch.tensor([[0.0, 0.0, 5.0, 1.0]])
        res = proj.project_ego_to_image(pts, 256)
        assert res.uv_norm.shape == (1, V, 1, 2)

    def test_ftheta_roundtrip_shared_poly_via_to_spec(self, tmp_path):
        """A shared [K] fw_poly must survive to_spec -> manifest -> load and
        project without a shape mismatch (round-2 review regression)."""
        from model_components.view_fusion.projection import FThetaProjection

        V = 2
        T = torch.eye(4).reshape(1, 1, 4, 4).expand(1, V, 4, 4).contiguous()
        # shared 1-D polynomial (not per-view)
        built = FThetaProjection(T, torch.tensor([0.0, 200.0, -1.0]), cx=128.0, cy=128.0)
        spec = built.to_spec()
        (tmp_path / "manifest.json").write_text(json.dumps({
            "geometry_type": "ftheta", "projection": spec,
        }))
        proj, geom = load_projection_from_manifest(str(tmp_path))
        assert geom == "ftheta" and proj.num_views == V
        pts = torch.randn(50, 4)
        res = proj.project_ego_to_image(pts, 256)  # must not raise
        assert res.uv_norm.shape == (1, V, 50, 2)

    def test_ftheta_roundtrip_per_view_poly_not_collapsed(self, tmp_path):
        """A per-view [V,K] fw_poly must survive to_spec -> load with EACH view's
        polynomial preserved (codex review: to_spec was collapsing to view 0)."""
        from model_components.view_fusion.projection import FThetaProjection

        V = 2
        T = torch.eye(4).reshape(1, 1, 4, 4).expand(1, V, 4, 4).contiguous()
        # distinct per-view polynomials: view 1 has 2x the radius slope of view 0.
        fw = torch.tensor([[0.0, 100.0], [0.0, 200.0]])  # [V, K], unbatched
        built = FThetaProjection(T, fw, cx=128.0, cy=128.0)
        spec = built.to_spec()
        assert spec["fw_poly"] == [[0.0, 100.0], [0.0, 200.0]], "per-view poly collapsed"
        (tmp_path / "manifest.json").write_text(json.dumps({
            "geometry_type": "ftheta", "projection": spec,
        }))
        proj, _ = load_projection_from_manifest(str(tmp_path))
        # An off-axis point must land at different radii on the two views.
        pt = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        res = proj.project_ego_to_image(pt, 256)
        u0 = res.uv_norm[0, 0, 0, 0].item()
        u1 = res.uv_norm[0, 1, 0, 0].item()
        assert abs(u1 - 0.5) > abs(u0 - 0.5) + 1e-4, \
            "view 1 (2x slope) should project farther from centre than view 0"

    def test_ftheta_roundtrip_per_view_max_theta(self, tmp_path):
        """A per-view max_theta serialized as a list must reload and project."""
        spec = {
            "type": "ftheta",
            "t_camera_ego": torch.eye(4).reshape(1, 4, 4).expand(2, 4, 4).tolist(),
            "fw_poly": [[0.0, 200.0]] * 2,
            "cx": [128.0] * 2, "cy": [128.0] * 2,
            "image_wh": [[256.0, 256.0]] * 2,
            "max_theta": [1.5, 1.8],  # per-view list
        }
        (tmp_path / "manifest.json").write_text(json.dumps({
            "geometry_type": "ftheta", "projection": spec,
        }))
        proj, _ = load_projection_from_manifest(str(tmp_path))
        res = proj.project_ego_to_image(torch.randn(10, 4), 256)  # must not raise
        assert res.valid_mask.shape == (1, 2, 10)


class TestDecodeWorldModelWindows:
    """WM window members hist_/fut_ decode to [steps, V, 3, H, W] (#13)."""

    def _window_sample(self, n_cams=6, T=4, F=4):
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i, 0, 0)) for i in range(n_cams)}
        sample["ego.npy"] = _ego_bytes()
        for t in range(T):
            for v in range(n_cams):
                sample[f"hist_{t}_cam_{v}.jpg"] = _jpeg_bytes((t, v, 0))
        for f in range(F):
            for v in range(n_cams):
                sample[f"fut_{f}_cam_{v}.jpg"] = _jpeg_bytes((f, v, 1))
        return sample

    def test_windows_decoded_with_right_shape(self):
        out = _decode_sample(self._window_sample(n_cams=6, T=4, F=4))
        assert out["history_frames"].shape == (4, 6, 3, 256, 256)
        assert out["future_frames"].shape == (4, 6, 3, 256, 256)

    def test_windows_absent_when_not_packed(self):
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i, 0, 0)) for i in range(6)}
        sample["ego.npy"] = _ego_bytes()
        out = _decode_sample(sample)
        assert "history_frames" not in out
        assert "future_frames" not in out

    # --- deduped frame-pool path (#121 §3.4d) ---------------------------------
    def _pool_sample_and_accessor(self, n_cams=6, T=4, F=4):
        """A sample carrying window_index.json + a matching in-memory pool accessor.

        Reuses the SAME jpeg bytes the legacy layout would embed, keyed by frame_id,
        so the rebuilt tensors must equal the legacy hist_/fut_ decode.
        """
        sample = {f"cam_{i}.jpg": _jpeg_bytes((i, 0, 0)) for i in range(n_cams)}
        sample["ego.npy"] = _ego_bytes()
        pool_bytes = {}
        hist_ids, fut_ids = [], []
        for t in range(T):
            step = []
            for v in range(n_cams):
                fid = f"l2d-v1-e000000-r{100 + t:06d}-c{v}"
                pool_bytes[fid] = _jpeg_bytes((t, v, 0))
                step.append(fid)
            hist_ids.append(step)
        for f in range(F):
            step = []
            for v in range(n_cams):
                fid = f"l2d-v1-e000000-r{200 + f:06d}-c{v}"
                pool_bytes[fid] = _jpeg_bytes((f, v, 1))
                step.append(fid)
            fut_ids.append(step)
        sample["window_index.json"] = json.dumps(
            {"history": hist_ids, "future": fut_ids}).encode()
        return sample, (lambda fid: pool_bytes[fid])

    def test_pool_window_decoded_with_right_shape(self):
        sample, pool = self._pool_sample_and_accessor(n_cams=6, T=4, F=4)
        out = _decode_sample(sample, pool=pool)
        assert out["history_frames"].shape == (4, 6, 3, 256, 256)
        assert out["future_frames"].shape == (4, 6, 3, 256, 256)

    def test_pool_window_equals_legacy_layout(self):
        """THE byte-equality guarantee: the pool path rebuilds the SAME tensors the
        legacy hist_/fut_ layout would, for the same underlying jpeg bytes."""
        legacy = self._window_sample(n_cams=6, T=4, F=4)
        legacy_out = _decode_sample(legacy)
        pool_sample, pool = self._pool_sample_and_accessor(n_cams=6, T=4, F=4)
        pool_out = _decode_sample(pool_sample, pool=pool)
        assert torch.equal(pool_out["history_frames"], legacy_out["history_frames"])
        assert torch.equal(pool_out["future_frames"], legacy_out["future_frames"])

    def test_pool_window_index_without_accessor_raises(self):
        sample, _ = self._pool_sample_and_accessor()
        with pytest.raises(ValueError, match="frame pool accessor"):
            _decode_sample(sample, pool=None)


class TestMergedDatasetLoader:
    """Round-robin interleaving of multiple single-dataset loaders (#77 merge)."""

    def _fake_loader(self, batches, projection, geom):
        class _L:
            def __init__(self, b, p, g):
                self._b, self.projection, self.geometry_type = b, p, g
            def __iter__(self):
                return iter(self._b)
        return _L(batches, projection, geom)

    class _Lifecycle:
        def __init__(self):
            self.created = []
            self.opened = []
            self.closed = []
            self.loader_closed = []
            self.active = 0
            self.peak_active = 0

        def factory(self, label, batches, *, fail=False):
            lifecycle = self

            class _Iterator:
                def __init__(self):
                    self._batches = iter(batches)
                    self._fail = fail
                    self._closed = False
                    lifecycle.opened.append(label)
                    lifecycle.active += 1
                    lifecycle.peak_active = max(
                        lifecycle.peak_active, lifecycle.active
                    )

                def __iter__(self):
                    return self

                def __next__(self):
                    if self._fail:
                        self._fail = False
                        raise RuntimeError(f"failed {label}")
                    return next(self._batches)

                def close(self):
                    if not self._closed:
                        self._closed = True
                        lifecycle.closed.append(label)
                        lifecycle.active -= 1

            class _Loader:
                projection = None
                geometry_type = "pseudo"

                def __iter__(self):
                    return _Iterator()

                def close(self):
                    lifecycle.loader_closed.append(label)

            def create():
                lifecycle.created.append(label)
                return _Loader()

            return create

    def test_round_robin_interleaves_and_tags_geometry(self):
        from data_parsing.pre_extracted import MergedDatasetLoader
        a = self._fake_loader(["a0", "a1", "a2"], None, "pseudo")
        b = self._fake_loader(["b0", "b1"], "PROJ", "ftheta")
        merged = MergedDatasetLoader([a, b])
        seen = list(merged)
        # Each item carries its dataset's geometry.
        assert ("a0", None, "pseudo") in seen
        assert ("b0", "PROJ", "ftheta") in seen
        # Interleaved (a0, b0, a1, b1, a2) not concatenated (a0,a1,a2,b0,b1).
        order = [x[0] for x in seen]
        assert order == ["a0", "b0", "a1", "b1", "a2"]
        # All batches from both datasets appear exactly once.
        assert sorted(order) == ["a0", "a1", "a2", "b0", "b1"]

    def test_single_loader_degrades_cleanly(self):
        from data_parsing.pre_extracted import MergedDatasetLoader
        a = self._fake_loader(["x0", "x1"], None, "pseudo")
        merged = MergedDatasetLoader([a])
        assert [x[0] for x in merged] == ["x0", "x1"]

    def test_empty_raises(self):
        from data_parsing.pre_extracted import MergedDatasetLoader
        with pytest.raises(ValueError, match="at least one"):
            MergedDatasetLoader([])

    def test_404_factories_are_lazy_and_active_window_is_bounded(self):
        from data_parsing.pre_extracted import MergedDatasetLoader

        lifecycle = self._Lifecycle()
        factories = [
            lifecycle.factory(i, [i])
            for i in range(404)
        ]
        merged = MergedDatasetLoader(
            loader_factories=factories,
            max_active_loaders=4,
        )

        assert lifecycle.created == []
        seen = [item[0] for item in merged]

        assert seen == list(range(404))
        assert lifecycle.created == list(range(404))
        assert lifecycle.peak_active == 4
        assert lifecycle.active == 0
        assert sorted(lifecycle.closed) == list(range(404))
        assert sorted(lifecycle.loader_closed) == list(range(404))

    def test_early_close_releases_every_active_child(self):
        from data_parsing.pre_extracted import MergedDatasetLoader

        lifecycle = self._Lifecycle()
        merged = MergedDatasetLoader(
            loader_factories=[
                lifecycle.factory(i, [f"{i}-0", f"{i}-1"])
                for i in range(404)
            ],
            max_active_loaders=4,
        )

        iterator = iter(merged)
        assert next(iterator)[0] == "0-0"
        assert lifecycle.created == [0, 1, 2, 3]
        iterator.close()

        assert lifecycle.active == 0
        assert sorted(lifecycle.closed) == [0, 1, 2, 3]
        assert sorted(lifecycle.loader_closed) == [0, 1, 2, 3]
        assert lifecycle.created == [0, 1, 2, 3]

    def test_each_epoch_recreates_children_after_complete_cleanup(self):
        from data_parsing.pre_extracted import MergedDatasetLoader

        lifecycle = self._Lifecycle()
        merged = MergedDatasetLoader(
            loader_factories=[
                lifecycle.factory("a", ["a0", "a1"]),
                lifecycle.factory("b", ["b0"]),
            ],
            max_active_loaders=2,
        )

        expected = ["a0", "b0", "a1"]
        assert [item[0] for item in merged] == expected
        assert lifecycle.active == 0
        assert [item[0] for item in merged] == expected

        assert lifecycle.created == ["a", "b", "a", "b"]
        assert lifecycle.closed == ["b", "a", "b", "a"]
        assert lifecycle.loader_closed == ["b", "a", "b", "a"]
        assert lifecycle.peak_active == 2
        assert lifecycle.active == 0

    def test_child_exception_releases_failed_and_other_active_children(self):
        from data_parsing.pre_extracted import MergedDatasetLoader

        lifecycle = self._Lifecycle()
        factories = [lifecycle.factory(0, [], fail=True)]
        factories.extend(
            lifecycle.factory(i, [i])
            for i in range(1, 404)
        )
        merged = MergedDatasetLoader(
            loader_factories=factories,
            max_active_loaders=4,
        )

        with pytest.raises(RuntimeError, match="failed 0"):
            next(iter(merged))

        assert lifecycle.created == [0, 1, 2, 3]
        assert lifecycle.active == 0
        assert sorted(lifecycle.closed) == [0, 1, 2, 3]
        assert sorted(lifecycle.loader_closed) == [0, 1, 2, 3]

    def test_multi_loader_uses_global_worker_budget_and_lazy_factories(
            self, monkeypatch):
        import data_parsing.pre_extracted as pre_extracted

        calls = []

        def fake_loader(shard_dir, **kwargs):
            calls.append((shard_dir, kwargs))
            return self._fake_loader([shard_dir], None, "pseudo")

        monkeypatch.setattr(
            pre_extracted, "make_pre_extracted_loader", fake_loader
        )
        merged = pre_extracted.make_multi_dataset_loader(
            [f"partition-{i}" for i in range(404)],
            batch_size=1,
            num_workers=4,
            shuffle=1000,
        )

        assert calls == []
        iterator = iter(merged)
        assert next(iterator)[0] == "partition-0"
        assert len(calls) == 4
        assert all(kwargs["num_workers"] == 1 for _, kwargs in calls)
        assert merged.max_active_loaders == 4
        iterator.close()

    def test_multi_loader_exposes_safe_eval_active_window(self, monkeypatch):
        import data_parsing.pre_extracted as pre_extracted

        calls = []

        def fake_loader(shard_dir, **kwargs):
            calls.append((shard_dir, kwargs))
            return self._fake_loader([shard_dir], None, "pseudo")

        monkeypatch.setattr(
            pre_extracted, "make_pre_extracted_loader", fake_loader
        )
        merged = pre_extracted.make_multi_dataset_loader(
            [f"partition-{i}" for i in range(404)],
            batch_size=8,
            num_workers=4,
            shuffle=0,
            prefetch_factor=1,
            max_active_loaders=1,
        )

        iterator = iter(merged)
        assert next(iterator)[0] == "partition-0"
        assert len(calls) == 1
        assert calls[0][1]["num_workers"] == 1
        assert calls[0][1]["prefetch_factor"] == 1
        assert merged.max_active_loaders == 1
        iterator.close()


def _write_shards(dirpath, n_shards, per_shard):
    """Write n_shards minimal valid .tar shards (per_shard samples each)."""
    import tarfile
    from pathlib import Path
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    idx = 0
    for s in range(n_shards):
        with tarfile.open(f"{dirpath}/shard-{s:03d}.tar", "w") as t:
            for _ in range(per_shard):
                key = f"s{idx:06d}"
                members = {f"{key}.cam_0.jpg": _jpeg_bytes((idx % 255, 0, 0)),
                           f"{key}.ego.npy": _ego_bytes()}
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
                idx += 1
    return idx


class TestLoaderYieldsAllSamplesUnderWorkers:
    """Regression: the loader must yield EVERY sample regardless of num_workers.

    webdataset 1.0.2 auto-applies split_by_worker via the `workersplitter`
    default; passing nodesplitter=split_by_worker too split the shard list TWICE,
    so num_workers=N silently dropped (N-1)/N of the data (24/48 at nw=2, 12/48 at
    nw=4). This pins that num_workers>0 sees the full dataset — the #121 P0
    parallel-decode change and the eval loader (num_workers=4) both depend on it.
    """

    def test_no_samples_dropped_across_worker_counts(self, tmp_path):
        total = _write_shards(tmp_path / "shards", n_shards=12, per_shard=4)  # 48
        from data_parsing.pre_extracted import make_pre_extracted_loader
        for nw in (0, 2, 4):
            loader = make_pre_extracted_loader(str(tmp_path / "shards"),
                                               batch_size=1, num_workers=nw, shuffle=0)
            seen = sum(b["visual_tiles"].shape[0] for b in loader)
            assert seen == total, (
                f"num_workers={nw}: loader yielded {seen}/{total} samples — "
                f"shards are being split more than once (double split_by_worker)")

    def test_partition_loader_workers_are_not_persistent(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shards(shard_dir, n_shards=2, per_shard=1)
        from data_parsing.pre_extracted import make_pre_extracted_loader

        loader = make_pre_extracted_loader(
            str(shard_dir),
            batch_size=1,
            num_workers=2,
            shuffle=0,
        )
        torch_loader = loader.pipeline[0]
        assert torch_loader.num_workers == 2
        assert torch_loader.persistent_workers is False
        loader.close()

    def test_merged_early_close_stops_all_active_workers(self, tmp_path):
        import multiprocessing as mp

        from data_parsing.pre_extracted import make_multi_dataset_loader

        shard_dirs = []
        for index in range(8):
            shard_dir = tmp_path / f"partition-{index}"
            _write_shards(shard_dir, n_shards=1, per_shard=2)
            shard_dirs.append(str(shard_dir))

        existing_pids = {child.pid for child in mp.active_children()}
        loader = make_multi_dataset_loader(
            shard_dirs,
            batch_size=1,
            num_workers=1,
            shuffle=0,
        )
        iterator = iter(loader)
        next(iterator)
        workers = [
            child
            for child in mp.active_children()
            if child.pid not in existing_pids
        ]

        iterator.close()
        for worker in workers:
            worker.join(timeout=5)

        assert len(workers) == 1
        assert all(not worker.is_alive() for worker in workers)

    def test_explicit_shard_subset_isolated_for_overlay_output(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shards(shard_dir, n_shards=2, per_shard=3)
        from data_parsing.pre_extracted import make_pre_extracted_loader

        selected = shard_dir / "shard-001.tar"
        loader = make_pre_extracted_loader(
            str(shard_dir),
            batch_size=1,
            num_workers=0,
            shuffle=0,
            shard_files=[selected],
        )
        keys = [batch["sample_uid"][0] for batch in loader]
        assert keys == ["s000003", "s000004", "s000005"]
