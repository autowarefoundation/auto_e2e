"""Unit-conversion regression tests for L2D egomotion (speed km/h, heading deg).

The raw yaak-ai/L2D vehicle state stores speed in km/h and heading in degrees
(verified against meta/stats.json: speed max ~171.8, heading range 0..360).
Downstream physics (integrate_trajectory) and the curvature formula require SI
units. If the conversion regresses, the integrated ego trajectory — and hence
ADE/FDE — explode (yaw_rate becomes ~57x too large, speed ~3.6x). These tests
lock the conversion in.
"""

from __future__ import annotations

import numpy as np

from data_parsing.l2d.egomotion import _derive_signals


def _states(speed_kmh=100.0, heading_deg_start=90.0, heading_sweep_deg=2.0,
            accel=0.5, T=130):
    vs = np.zeros((T, 8), dtype=np.float32)
    vs[:, 0] = speed_kmh
    vs[:, 1] = heading_deg_start + np.linspace(0, heading_sweep_deg, T)
    vs[:, 6] = accel
    return vs


def test_speed_converted_kmh_to_ms():
    sig = _derive_signals(_states(speed_kmh=100.0))
    # 100 km/h == 27.78 m/s, not 100.
    assert abs(float(sig[:, 0].mean()) - 27.78) < 0.1


def test_yaw_rate_is_physical_not_degrees():
    # A 2-degree heading sweep over 13 s is a tiny yaw rate. If heading were left
    # in degrees, yaw_rate would be ~57x larger (and real L2D shards showed 30 rad/s).
    sig = _derive_signals(_states(heading_sweep_deg=2.0, T=130))
    assert abs(float(sig[1:, 2].mean())) < 0.05, "yaw_rate must be radians/s, small"


def test_curvature_reasonable_magnitude():
    # Gentle highway turn -> curvature near zero (large radius), not O(1) 1/m.
    sig = _derive_signals(_states())
    assert abs(float(sig[1:, 3].mean())) < 0.01


def test_accel_passthrough_unchanged():
    sig = _derive_signals(_states(accel=1.25))
    assert abs(float(sig[:, 1].mean()) - 1.25) < 1e-4


def test_zero_speed_curvature_guarded():
    vs = _states(speed_kmh=0.0)
    sig = _derive_signals(vs)
    # No div-by-zero blow-up when stopped.
    assert np.all(np.isfinite(sig[:, 3]))
    assert float(np.abs(sig[:, 3]).max()) == 0.0
