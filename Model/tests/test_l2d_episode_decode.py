"""The per-episode L2D video reader picks the frames LeRobot would."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("torch")

from data_processing.reasoning_label_generation import parallel_pack as pp  # noqa: E402


def _video(path: Path, count: int, fps: int = 10) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(count):
            image = np.full((48, 64, 3), (index * 20) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _fake_dataset(tmp_path: Path, start_s: float, key: str):
    video = tmp_path / "videos" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    _video(video, count=40)
    meta = SimpleNamespace(
        fps=10,
        episodes={3: {"episode_index": 3, f"videos/{key}/from_timestamp": start_s}},
        get_video_file_path=lambda episode, video_key: Path("videos/file-000.mp4"),
    )
    timestamps = [index / 10 for index in range(20)]

    class _Rows:
        def select_columns(self, _columns):
            return self

        def __getitem__(self, _rows):
            return {"timestamp": timestamps}

    return SimpleNamespace(
        lerobot_dataset=SimpleNamespace(meta=meta, root=tmp_path, hf_dataset=_Rows(), tolerance_s=1e-4),
        _episode_ranges={3: (0, len(timestamps))},
    )


def test_episode_frames_are_offset_by_the_episode_start(tmp_path, monkeypatch):
    # The episode starts 1 s into a shared file: its frame 2 is file frame 12.
    monkeypatch.setattr(pp, "_DS", _fake_dataset(tmp_path, 1.0, "cam"))
    frames = pp._l2d_episode_video_frames(3, "cam", {2, 5}, lambda index, tensor: float(tensor.mean()))
    assert sorted(frames) == [2, 5]
    assert frames[2] == pytest.approx((12 * 20 % 255) / 255.0, abs=0.03)
    assert frames[5] == pytest.approx((15 * 20 % 255) / 255.0, abs=0.03)


def test_missing_frames_fail_instead_of_packing_short(tmp_path, monkeypatch):
    # The file ends 1 s into this episode, so its frame 15 does not exist.
    monkeypatch.setattr(pp, "_DS", _fake_dataset(tmp_path, 3.0, "cam"))
    with pytest.raises(ValueError, match="lacks frames"):
        pp._l2d_episode_video_frames(3, "cam", {5, 15}, lambda index, tensor: index)


def test_frames_outside_the_episode_are_refused(tmp_path, monkeypatch):
    # Index 20 would read the next episode's video in the shared file.
    monkeypatch.setattr(pp, "_DS", _fake_dataset(tmp_path, 1.0, "cam"))
    with pytest.raises(IndexError, match="leave the episode"):
        pp._l2d_episode_video_frames(3, "cam", {19, 20}, lambda index, tensor: index)


def test_frames_off_their_timestamps_are_refused(tmp_path, monkeypatch):
    dataset = _fake_dataset(tmp_path, 1.0, "cam")
    dataset.lerobot_dataset.tolerance_s = 0.0
    timestamps = [index / 10 + 0.04 for index in range(20)]
    dataset.lerobot_dataset.hf_dataset.__class__.__getitem__ = lambda self, rows: {"timestamp": timestamps}
    monkeypatch.setattr(pp, "_DS", dataset)
    with pytest.raises(ValueError, match="off their timestamps"):
        pp._l2d_episode_video_frames(3, "cam", {2}, lambda index, tensor: index)
