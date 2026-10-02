"""Offline tests for navigation-input robustness harness (#157)."""

from __future__ import annotations

import numpy as np

from evaluation.navigation_robustness import (
    DEFAULT_MODES,
    apply_navigation_perturbation,
    build_corridor_scenes,
    route_follow_predict,
    run_navigation_robustness,
)


def _toy_batch(b: int = 4, t: int = 8):
    map_context = np.ones((b, 3, 16, 16), dtype=np.float32)
    route_mask = np.zeros((b, 2, 16, 16), dtype=np.float32)
    route_mask[:, 0, 8, :] = 1.0
    # GT: straight ahead
    gt = np.stack(
        [np.stack([np.linspace(0, 10, t), np.zeros(t)], axis=-1) for _ in range(b)],
        axis=0,
    )
    return map_context, route_mask, gt


def test_perturbations_change_tensors():
    m, r, _ = _toy_batch()
    m2, r2 = apply_navigation_perturbation(m, r, "blank")
    assert np.allclose(m2, 0) and np.allclose(r2, 0)
    m3, r3 = apply_navigation_perturbation(m, r, "map_only")
    assert np.allclose(m3, m) and np.allclose(r3, 0)
    m4, r4 = apply_navigation_perturbation(m, r, "route_only")
    assert np.allclose(m4, 0) and np.allclose(r4, r)


def test_robustness_report_deltas():
    m, r, gt = _toy_batch()

    def predict(map_ctx, route):
        # Baseline uses route; blank route → zero trajectory (worse ADE).
        scale = 1.0 if route.sum() > 0 else 0.0
        return gt * scale

    report = run_navigation_robustness(
        m, r, gt, predict, modes=("map_route", "blank", "map_only")
    )
    by_mode = {c.mode: c for c in report.conditions}
    assert by_mode["map_route"].ade_delta_m == 0.0
    assert by_mode["blank"].ade_m > by_mode["map_route"].ade_m
    assert by_mode["blank"].ade_delta_m > 0
    assert set(DEFAULT_MODES)  # sanity: modes exported


def test_corridor_scenes_blank_hurts_ade():
    m, r, gt = build_corridor_scenes(batch=3, timesteps=16, hw=32)
    report = run_navigation_robustness(
        m, r, gt,
        lambda mm, rr: route_follow_predict(mm, rr, gt),
        modes=("map_route", "blank", "map_only", "route_only"),
    )
    by_mode = {c.mode: c for c in report.conditions}
    assert by_mode["map_route"].ade_m < by_mode["blank"].ade_m
    assert by_mode["blank"].ade_delta_m > 0
    assert by_mode["map_only"].ade_m > by_mode["map_route"].ade_m
