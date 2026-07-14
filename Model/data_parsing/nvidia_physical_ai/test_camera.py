"""Tests for data_parsing.nvidia_physical_ai.camera.

Covers the #116 regression: load_camera_frame must decode a single frame
via lazy, seek-based reads (SeekVideoReader / PyAV) and must NEVER read an
entire mp4 file into memory (Path.read_bytes()) just to extract one frame.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data_parsing.nvidia_physical_ai.camera import (
    CAMERA_NAMES,
    load_camera_frame,
    make_map_tile,
)


def _make_synthetic_clip(path: Path, num_frames: int = 20, fps: int = 10,
                          size: int = 32) -> None:
    """Encode a tiny synthetic mp4 where frame i has pixel value (i*10) % 256.

    Lets tests assert on decoded *content*, not just shape, without shipping
    a real fixture video.
    """
    av = pytest.importorskip("av")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        val = (i * 10) % 256
        frame = av.VideoFrame.from_ndarray(
            np.full((size, size, 3), val, dtype=np.uint8), format="rgb24",
        )
        frame = frame.reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def synthetic_clip_dir(tmp_path):
    """Build a fake NVIDIA-AV-shaped data_root with one clip, one camera."""
    data_root = tmp_path / "nvidia_av"
    cam_name = CAMERA_NAMES[0]
    cam_dir = data_root / "camera" / cam_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    clip_uuid = "test-clip-0001"
    clip_path = cam_dir / f"{clip_uuid}.{cam_name}.mp4"
    _make_synthetic_clip(clip_path, num_frames=20, fps=10, size=32)

    # 20 frames @ 10fps -> 100ms apart, in microseconds.
    timestamps_us = (np.arange(20) * 100_000).astype(np.int64)

    return {
        "data_root": data_root,
        "cam_name": cam_name,
        "clip_uuid": clip_uuid,
        "camera_timestamps": {cam_name: timestamps_us},
    }


class TestLoadCameraFrameMemory:
    """The #116 regression guard: no full-file read for a single frame."""

    def test_does_not_read_full_file_into_memory(self, synthetic_clip_dir):
        """Path.read_bytes() must never be called by load_camera_frame.

        This is the direct regression check for #116: the previous
        implementation did `io.BytesIO(video_path.read_bytes())`, eagerly
        materializing the entire encoded clip in memory to decode a single
        frame. The fix uses a lazy, seekable file handle instead.
        """
        d = synthetic_clip_dir
        original_read_bytes = Path.read_bytes
        call_count = {"n": 0}

        def spy_read_bytes(self):
            call_count["n"] += 1
            return original_read_bytes(self)

        with patch.object(Path, "read_bytes", spy_read_bytes):
            load_camera_frame(
                data_root=d["data_root"],
                clip_uuid=d["clip_uuid"],
                egomotion_timestamp_us=500_000,
                camera_names=[d["cam_name"]],
                camera_timestamps=d["camera_timestamps"],
            )

        assert call_count["n"] == 0, (
            "load_camera_frame called Path.read_bytes() — this reads the "
            "ENTIRE mp4 into memory just to decode one frame (regression "
            "of #116)."
        )

    def test_multiple_calls_do_not_accumulate_full_reads(self, synthetic_clip_dir):
        """Simulates several __getitem__ calls against the same clip (the
        common case — many valid sample_idx per clip) and confirms none of
        them read the full file."""
        d = synthetic_clip_dir
        original_read_bytes = Path.read_bytes
        call_count = {"n": 0}

        def spy_read_bytes(self):
            call_count["n"] += 1
            return original_read_bytes(self)

        with patch.object(Path, "read_bytes", spy_read_bytes):
            for ts_us in (100_000, 500_000, 900_000, 1_300_000):
                load_camera_frame(
                    data_root=d["data_root"],
                    clip_uuid=d["clip_uuid"],
                    egomotion_timestamp_us=ts_us,
                    camera_names=[d["cam_name"]],
                    camera_timestamps=d["camera_timestamps"],
                )

        assert call_count["n"] == 0


class TestLoadCameraFrameCorrectness:
    """Functional correctness must be unchanged by the memory fix."""

    def test_output_shape_and_dtype(self, synthetic_clip_dir):
        d = synthetic_clip_dir
        result = load_camera_frame(
            data_root=d["data_root"],
            clip_uuid=d["clip_uuid"],
            egomotion_timestamp_us=500_000,
            camera_names=[d["cam_name"]],
            camera_timestamps=d["camera_timestamps"],
        )
        assert result.shape == (1, 3, 32, 32)
        assert result.dtype == torch.uint8

    def test_decoded_content_matches_encoded_frame(self, synthetic_clip_dir):
        """Frame at t=500ms is frame index 5, encoded with value (5*10)%256=50.

        Confirms the seek-based decode lands on the correct frame, not just
        that *a* frame was decoded.
        """
        d = synthetic_clip_dir
        result = load_camera_frame(
            data_root=d["data_root"],
            clip_uuid=d["clip_uuid"],
            egomotion_timestamp_us=500_000,
            camera_names=[d["cam_name"]],
            camera_timestamps=d["camera_timestamps"],
        )
        mean_val = result.float().mean().item()
        # Compression introduces minor noise; allow a tolerance band.
        assert 35 < mean_val < 65, (
            f"Decoded frame content looks wrong (mean={mean_val}); "
            f"expected near 50 for frame index 5."
        )

    def test_different_timestamps_yield_different_frames(self, synthetic_clip_dir):
        """Sanity check that seeking actually moves — two different
        timestamps should decode two visibly different synthetic frames."""
        d = synthetic_clip_dir
        early = load_camera_frame(
            data_root=d["data_root"], clip_uuid=d["clip_uuid"],
            egomotion_timestamp_us=0,
            camera_names=[d["cam_name"]], camera_timestamps=d["camera_timestamps"],
        )
        late = load_camera_frame(
            data_root=d["data_root"], clip_uuid=d["clip_uuid"],
            egomotion_timestamp_us=1_800_000,  # frame index ~18
            camera_names=[d["cam_name"]], camera_timestamps=d["camera_timestamps"],
        )
        assert not torch.allclose(early.float(), late.float(), atol=5.0)

    def test_missing_video_file_raises(self, synthetic_clip_dir, tmp_path):
        d = synthetic_clip_dir
        with pytest.raises(FileNotFoundError):
            load_camera_frame(
                data_root=d["data_root"],
                clip_uuid="nonexistent-clip",
                egomotion_timestamp_us=0,
                camera_names=[d["cam_name"]],
                camera_timestamps=d["camera_timestamps"],
            )


class TestMakeMapTile:
    def test_zero_tile_matches_reference_shape(self):
        ref = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        tile = make_map_tile(ref)
        assert tile.shape == ref.shape
        assert tile.dtype == ref.dtype
        assert torch.all(tile == 0)
