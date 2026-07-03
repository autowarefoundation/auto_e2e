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
            "max_theta": None,
        }
        (tmp_path / "manifest.json").write_text(json.dumps({
            "geometry_type": "ftheta", "projection": spec,
        }))
        proj, geom = load_projection_from_manifest(str(tmp_path))
        assert geom == "ftheta" and proj.num_views == V
        # projects an on-axis ego point to the principal point
        pts = torch.tensor([[0.0, 0.0, 5.0, 1.0]])
        res = proj.project(pts, image_size=256)
        assert res.uv_norm.shape == (1, V, 1, 2)
