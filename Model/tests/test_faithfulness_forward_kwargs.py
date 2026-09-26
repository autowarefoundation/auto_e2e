"""Demonstration test for the faithfulness `forward_kwargs` extension.

Proves the two gaps found when trying to wire `reasoning_intervention_delta` into
the route-conditioned, flow-matching eval loop, and that threading `forward_kwargs`
(a fixed `initial_noise` + the navigation inputs) fixes both:

1. Without a fixed `initial_noise`, a stochastic planner draws different noise in
   the coupled and bypassed runs, so the "intervention delta" is dominated by
   noise rather than by the reasoning intervention (which here is 1e-3).
2. Passing the navigation inputs (`route_mask`) must not raise — the pre-extension
   signature had no `**kwargs`, so the eval loop could not thread them at all.

Self-contained: a minimal stochastic stub, no fixture / GPU / network.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from evaluation.faithfulness import reasoning_intervention_delta

_T, _D = 8, 2
_BUMP = 1e-3  # the reasoning intervention's effect on the trajectory


class _Reactive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ReasoningHead = nn.Identity()  # present => reasoning coupled


class _StochasticStub(nn.Module):
    """trajectory = reasoning_bump + route_term + noise.

    - reasoning_bump: `_BUMP` while `Reactive_E2E.ReasoningHead` is not None (the
      intervention sets it to None to bypass);
    - noise: uses `initial_noise` if given (fixed across runs), else fresh randn;
    - route_term: shifts the operating point when `route_mask` is provided.
    """

    def __init__(self) -> None:
        super().__init__()
        self.Reactive_E2E = _Reactive()

    def forward(self, camera, map_input, vis_hist, ego, *,
                projection=None, geometry_type=None, image_transform=None,
                route_mask=None, initial_noise=None, mode="infer", **_):
        b = camera.shape[0]
        reasoning_on = self.Reactive_E2E.ReasoningHead is not None
        bump = _BUMP if reasoning_on else 0.0
        route_term = 0.0 if route_mask is None else 0.5
        noise = initial_noise if initial_noise is not None else torch.randn(b, _T, _D)
        return torch.zeros(b, _T, _D) + bump + route_term + noise


def _inputs(b: int = 2):
    return (torch.randn(b, 7, 3, 256, 256), torch.randn(b, 3, 256, 256),
            torch.randn(b, 896), torch.randn(b, 256))


def test_gap_naive_call_is_noise_dominated():
    torch.manual_seed(0)
    out = reasoning_intervention_delta(_StochasticStub(), *_inputs())
    # noise ~ N(0,1) in each run -> delta on the order of 1, not the 1e-3 signal.
    assert out["trajectory_l2"] > 0.1


def test_fix_fixed_noise_recovers_the_intervention():
    torch.manual_seed(0)
    b = 2
    fixed = torch.zeros(b, _T, _D)  # same noise threaded into both runs
    out = reasoning_intervention_delta(_StochasticStub(), *_inputs(b),
                                       initial_noise=fixed)
    # noise cancels; only the reasoning bump survives: ||[1e-3, 1e-3]|| = 1e-3*sqrt(2)
    assert out["trajectory_l2"] == pytest.approx(_BUMP * 2 ** 0.5, abs=2e-4)


def test_fix_navigation_inputs_are_threaded():
    b = 2
    fixed = torch.zeros(b, _T, _D)
    route = torch.ones(b, 10)  # the pre-extension signature would TypeError on this
    out = reasoning_intervention_delta(_StochasticStub(), *_inputs(b),
                                       route_mask=route, initial_noise=fixed)
    # route is identical in both runs -> it cancels, leaving the intervention only.
    assert out["trajectory_l2"] == pytest.approx(_BUMP * 2 ** 0.5, abs=2e-4)
