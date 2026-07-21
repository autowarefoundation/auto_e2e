"""The WM + reasoning branches must be STRICTLY ADDITIVE at init (#13/#98).

The full 3-branch model should never START worse than the imitation-only
baseline: enabling the World Model and reasoning must be a no-op on the
trajectory at initialization, so training can only improve from the imitation
starting point (the branches "open" as they learn). This is enforced by:
  - reasoning: zero-init coupling (alpha=0),
  - WM visual_history: zero-init visual_history_proj + detached planner input.

If someone removes the zero-init or wires a non-zero-init projection, the full
model would diverge from the imitation baseline at init (the regression that made
the 3-branch eval ADE 2.02m vs imitation 1.77m) — these tests fail loudly then.

Mock backbone, CPU, no shards.
"""

from __future__ import annotations

import torch

B, V, T, F = 2, 6, 4, 4


def _inputs(device):
    return dict(
        visual=torch.randn(B, V, 3, 256, 256, device=device),
        map_input=torch.randn(B, 3, 256, 256, device=device),
        vis_hist=torch.zeros(B, 896, device=device),
        ego=torch.randn(B, 256, device=device),
        history_frames=torch.randn(B, T, V, 3, 256, 256, device=device),
        future_frames=torch.randn(B, F, V, 3, 256, 256, device=device),
    )


def test_wm_visual_history_is_noop_at_init(build_mock_model):
    """At init, feeding the WM window (dense visual_history) must produce the SAME
    trajectory as feeding zeros — zero-init visual_history_proj makes it a no-op."""
    torch.manual_seed(0)
    model = build_mock_model(
        num_views=V, device=torch.device("cpu"), enable_world_model=True).eval()
    inp = _inputs(torch.device("cpu"))

    with torch.no_grad():
        # With the WM window: planner's visual_history is the dense WM output.
        out_wm = model(inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
                       mode="infer", history_frames=inp["history_frames"],
                       future_frames=inp["future_frames"])
        traj_wm = out_wm[0] if isinstance(out_wm, tuple) else out_wm
        # Without a window: planner's visual_history path differs, but the
        # projection is zero so the CONTRIBUTION of visual_history is zero either
        # way. Compare against the same model with a zeroed visual_history proj
        # contribution by passing no window (rolling buffer → also projected by
        # the same zero-init layer).
        out_zero = model(inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
                         mode="infer")
        traj_zero = out_zero[0] if isinstance(out_zero, tuple) else out_zero

    # Zero-init visual_history_proj ⇒ visual_history contributes 0 to the planner
    # context in BOTH cases, so the trajectories are identical at init.
    assert torch.allclose(traj_wm, traj_zero, atol=1e-6), (
        "WM visual_history is not a no-op at init — visual_history_proj must be "
        "zero-init so the WM branch is strictly additive")


def test_full_3branch_branches_are_noop_at_init(build_mock_model):
    """Within ONE full 3-branch model, running with the branches "active" (WM
    window fed so visual_history is dense + reasoning latent produced) yields the
    SAME trajectory as running the imitation-equivalent path (no window). The
    extra branches contribute nothing to the trajectory at init.

    (We compare within a single model, NOT against a separately-built imitation
    model: building the imitation model draws a different RNG sequence, so its
    SHARED planner weights differ — that would test RNG order, not additivity.)
    """
    torch.manual_seed(0)
    full = build_mock_model(
        num_views=V, device=torch.device("cpu"),
        enable_world_model=True, enable_reasoning=True,
        reasoning_mode="pooled_latent").eval()

    inp = _inputs(torch.device("cpu"))
    with torch.no_grad():
        # Branches active: WM window → dense visual_history; reasoning head runs.
        out_active = full(inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
                          mode="infer", history_frames=inp["history_frames"],
                          future_frames=inp["future_frames"])
        traj_active = out_active[0] if isinstance(out_active, tuple) else out_active
        # Imitation-equivalent path through the SAME model: no window.
        out_imit = full(inp["visual"], inp["map_input"], inp["vis_hist"], inp["ego"],
                        mode="infer")
        traj_imit = out_imit[0] if isinstance(out_imit, tuple) else out_imit

    # Zero-init visual_history_proj + zero-init reasoning coupling ⇒ both branches
    # contribute 0 to the planner context at init, so the trajectory is identical
    # whether or not the branches are fed. The full model never STARTS worse than
    # imitation; training can only open the branches from there.
    assert torch.allclose(traj_active, traj_imit, atol=1e-6), (
        "full 3-branch trajectory diverges from the imitation path at init — the "
        "WM/reasoning branches are not strictly additive (check zero-inits)")
