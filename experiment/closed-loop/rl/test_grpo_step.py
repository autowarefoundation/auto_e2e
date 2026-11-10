"""Tests for the frame helper of grpo_step.py. Needs AlpaSim's packages (GPU host venv).

    python -m pytest experiment/closed-loop/rl/test_grpo_step.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for path in (REPO / "Model", REPO / "Model/plugins/alpasim_driver", REPO / "Model/plugins/alpasim_driver/tools"):
    sys.path.insert(0, str(path))
pytest.importorskip("alpasim_utils")

from grpo_step import to_local  # noqa: E402


def test_to_local_places_rig_plans_in_the_scene() -> None:
    yaw = math.pi / 2
    local_from_rig = np.eye(4)
    local_from_rig[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    local_from_rig[:2, 3] = [10.0, 5.0]
    positions = np.array([[[0.0, 0.0], [4.0, 0.0], [4.0, 1.0]]])
    headings = np.array([[0.0, 0.0, 0.5]])
    local = to_local(positions, headings, local_from_rig)
    assert np.allclose(local[0, :, :2], [[10.0, 5.0], [10.0, 9.0], [9.0, 9.0]])
    assert np.allclose(local[0, :, 2], [yaw, yaw, yaw + 0.5])
