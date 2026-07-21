"""Guards the sample_id JOIN invariant (#98 labeling audit).

generate_reasoning_labels builds L2D with include_world_model_windows=True and
enumerates range(len(ds)) as the sample_ids the teacher labels. data_processing
must enumerate the SAME sample set to JOIN labels onto the right frames. That
holds today only because the World-Model window margins never exceed the
egomotion margins, so len(L2DDataset) is WM-independent. This is a pure-logic
regression guard: if someone grows the WM window past the egomotion margin, the
sample sets would diverge and every reasoning label would silently attach to the
WRONG frame — this test fails loudly first.
"""

from __future__ import annotations

from data_parsing.l2d.egomotion import _HISTORY_TIMESTEPS, _FUTURE_TIMESTEPS
from data_parsing.l2d.world_model_windows import required_margins, stride_for_hz


def test_wm_margins_do_not_exceed_egomotion_margins():
    # Defaults used by L2DDataset.__init__ (wm_num_frames=4, wm_hz=1, source_hz=10).
    stride = stride_for_hz(10.0, 1.0)
    wm_past, wm_future = required_margins(4, stride)
    assert wm_past <= _HISTORY_TIMESTEPS, (
        f"WM past margin {wm_past} > egomotion {_HISTORY_TIMESTEPS}: enabling WM "
        "windows would SHRINK len(L2DDataset), so generate (WM=True) and any "
        "data_processing enumeration would diverge -> every reasoning label "
        "attaches to the WRONG frame. Keep WM margins <= egomotion margins (or "
        "make both sides use identical WM settings and bump prompt_version).")
    assert wm_future <= _FUTURE_TIMESTEPS, (
        f"WM future margin {wm_future} > egomotion {_FUTURE_TIMESTEPS}: same "
        "silent sample_id misalignment risk.")


def test_nvidia_horizon_reach_within_future_margin():
    """NVIDIA analogue: the front-clip horizon reach (wm_num_frames*wm_stride)
    must stay within the egomotion future margin, or get_front_clip would run off
    the end of the valid-sample window and the sample_id enumeration (which uses
    those margins) would not match data_processing."""
    # nvidia_physical_ai.egomotion imports pandas + physical_ai_av (NVIDIA-only
    # optional deps not in the core requirements / CI env). Skip when unavailable,
    # matching the other NVIDIA tests — this is a pure-constant invariant check, not
    # something that should force heavyweight optional deps into the base test env.
    import pytest
    pytest.importorskip("pandas")
    pytest.importorskip("physical_ai_av")
    from data_parsing.nvidia_physical_ai.egomotion import _FUTURE_TIMESTEPS
    wm_num_frames, wm_stride = 4, 10   # NvidiaAVDataset defaults
    reach = wm_num_frames * wm_stride
    assert reach <= _FUTURE_TIMESTEPS, (
        f"NVIDIA horizon reach {reach} > future margin {_FUTURE_TIMESTEPS}: "
        "get_front_clip's furthest horizon leaves the valid window.")
