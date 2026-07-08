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
