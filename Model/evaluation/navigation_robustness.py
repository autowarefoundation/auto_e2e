"""Navigation-input robustness ablations (#157).

Controlled corruptions of map / route tensors so open-loop ADE/FDE (and related
metrics) can be compared against a clean ``map+route`` baseline — without
requiring a live KITScenes download for the harness itself.

Perturbation modes (issue matrix):
    * ``map_route`` — unchanged baseline
    * ``map_only`` — zero the route mask
    * ``route_only`` — zero the map context
    * ``blank`` — zero both
    * ``shuffled`` — permute batch items of map/route (wrong scene pairing)
    * ``wrong_route`` — circular-shift route within the batch
    * ``map_dropout`` / ``route_dropout`` — Bernoulli drop per sample
    * ``yaw_perturb`` — rotate route mask by a fixed yaw (degrees)

Callers supply a ``predict_fn(map_context, route_mask) -> positions[B,T,2]``
(or precomputed trajectories) and ground-truth positions; this module applies
corruptions and aggregates ADE/FDE deltas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

NAV_ROBUSTNESS_VERSION = "navigation_robustness_v1"

DEFAULT_MODES: tuple[str, ...] = (
    "map_route",
    "map_only",
    "route_only",
    "blank",
    "shuffled",
    "wrong_route",
    "map_dropout",
    "route_dropout",
    "yaw_perturb",
)


@dataclass(frozen=True)
class RobustnessConditionResult:
    mode: str
    ade_m: float
    fde_m: float
    ade_delta_m: float
    fde_delta_m: float
    n: int
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustnessReport:
    version: str
    baseline_mode: str
    conditions: list[RobustnessConditionResult]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "baseline_mode": self.baseline_mode,
            "conditions": [asdict(c) for c in self.conditions],
        }


def _ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Mean ADE / FDE over a batch of ``[B, T, 2]`` trajectories."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"pred/gt must share shape [B,T,2]; got {pred.shape} vs {gt.shape}")
    dist = np.linalg.norm(pred - gt, axis=-1)  # [B, T]
    ade = float(dist.mean())
    fde = float(dist[:, -1].mean())
    return ade, fde


def rotate_route_yaw(route_mask: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate each sample's route mask around the image center by ``yaw_deg``."""
    from scipy.ndimage import rotate

    out = np.empty_like(route_mask)
    for i in range(route_mask.shape[0]):
        # route_mask: [B, C, H, W]
        rotated = [
            rotate(route_mask[i, c], yaw_deg, reshape=False, order=1, mode="constant", cval=0.0)
            for c in range(route_mask.shape[1])
        ]
        out[i] = np.stack(rotated, axis=0)
    return out


def apply_navigation_perturbation(
    map_context: np.ndarray,
    route_mask: np.ndarray,
    mode: str,
    *,
    rng: np.random.Generator | None = None,
    dropout_p: float = 0.5,
    yaw_deg: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return corrupted ``(map_context, route_mask)`` for ``mode``."""
    if mode not in DEFAULT_MODES:
        raise ValueError(f"Unknown robustness mode {mode!r}; expected one of {DEFAULT_MODES}")
    rng = rng or np.random.default_rng(0)
    m = np.array(map_context, dtype=np.float32, copy=True)
    r = np.array(route_mask, dtype=np.float32, copy=True)
    b = m.shape[0]
    if r.shape[0] != b:
        raise ValueError("map_context and route_mask batch sizes must match")

    if mode == "map_route":
        return m, r
    if mode == "map_only":
        r[...] = 0.0
        return m, r
    if mode == "route_only":
        m[...] = 0.0
        return m, r
    if mode == "blank":
        m[...] = 0.0
        r[...] = 0.0
        return m, r
    if mode == "shuffled":
        perm = rng.permutation(b)
        # Keep map, shuffle route → wrong pairing (and vice-versa would be similar).
        return m, r[perm]
    if mode == "wrong_route":
        if b == 1:
            r[...] = 0.0
            return m, r
        return m, np.roll(r, shift=1, axis=0)
    if mode == "map_dropout":
        drop = rng.random(b) < dropout_p
        m[drop] = 0.0
        return m, r
    if mode == "route_dropout":
        drop = rng.random(b) < dropout_p
        r[drop] = 0.0
        return m, r
    if mode == "yaw_perturb":
        try:
            r = rotate_route_yaw(r, yaw_deg)
        except ImportError:
            # Fallback without scipy: roll spatially as a coarse yaw proxy.
            shift = max(1, int(round(yaw_deg / 5.0)))
            r = np.roll(r, shift=shift, axis=-1)
        return m, r
    raise AssertionError(f"unhandled mode {mode}")


PredictFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def run_navigation_robustness(
    map_context: np.ndarray,
    route_mask: np.ndarray,
    gt_positions: np.ndarray,
    predict_fn: PredictFn,
    *,
    modes: Sequence[str] = DEFAULT_MODES,
    baseline_mode: str = "map_route",
    seed: int = 157,
    dropout_p: float = 0.5,
    yaw_deg: float = 15.0,
) -> RobustnessReport:
    """Evaluate ``predict_fn`` under each navigation-input corruption.

    ``predict_fn`` must return ego-frame positions ``[B, T, 2]`` for the given
    map/route tensors. ADE/FDE deltas are relative to ``baseline_mode``.
    """
    rng = np.random.default_rng(seed)
    results: dict[str, RobustnessConditionResult] = {}

    baseline_ade = baseline_fde = 0.0
    # Evaluate baseline first.
    ordered = [baseline_mode] + [m for m in modes if m != baseline_mode]
    for mode in ordered:
        m_p, r_p = apply_navigation_perturbation(
            map_context, route_mask, mode, rng=rng, dropout_p=dropout_p, yaw_deg=yaw_deg
        )
        pred = predict_fn(m_p, r_p)
        ade, fde = _ade_fde(pred, gt_positions)
        if mode == baseline_mode:
            baseline_ade, baseline_fde = ade, fde
        results[mode] = RobustnessConditionResult(
            mode=mode,
            ade_m=ade,
            fde_m=fde,
            ade_delta_m=ade - baseline_ade,
            fde_delta_m=fde - baseline_fde,
            n=int(gt_positions.shape[0]),
        )

    return RobustnessReport(
        version=NAV_ROBUSTNESS_VERSION,
        baseline_mode=baseline_mode,
        conditions=[results[m] for m in ordered if m in results],
    )


def build_corridor_scenes(
    *,
    batch: int = 6,
    timesteps: int = 30,
    hw: int = 64,
    length_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build left / straight / right driving scenes in KITScenes-like rasters.

    ``map_context`` is a 3-channel drivable-corridor image. ``route_mask``
    channel 0 is the selected corridor; channel 1 is the destination blob.
    GT positions follow the painted corridor in ego-frame metres.
    """
    map_context = np.zeros((batch, 3, hw, hw), dtype=np.float32)
    route_mask = np.zeros((batch, 2, hw, hw), dtype=np.float32)
    gt = np.zeros((batch, timesteps, 2), dtype=np.float64)
    kinds = ("straight", "left", "right")
    for i in range(batch):
        kind = kinds[i % 3]
        xs = np.linspace(0.0, length_m, timesteps)
        if kind == "straight":
            ys = np.zeros_like(xs)
        elif kind == "left":
            ys = 6.0 * (xs / length_m) ** 2
        else:
            ys = -6.0 * (xs / length_m) ** 2
        gt[i, :, 0] = xs
        gt[i, :, 1] = ys
        # Raster: +x forward is down the image, +y left is to the side.
        for t in range(timesteps):
            row = int(np.clip(hw // 2 + (xs[t] / length_m) * (hw // 2 - 2), 0, hw - 1))
            col = int(np.clip(hw // 2 - (ys[t] / 12.0) * (hw // 4), 0, hw - 1))
            map_context[i, 0, max(0, row - 2):row + 3, max(0, col - 4):col + 5] = 1.0
            route_mask[i, 0, max(0, row - 1):row + 2, max(0, col - 1):col + 2] = 1.0
        dest_row = int(np.clip(hw // 2 + 0.9 * (hw // 2 - 2), 0, hw - 1))
        dest_col = int(np.clip(hw // 2 - (ys[-1] / 12.0) * (hw // 4), 0, hw - 1))
        route_mask[i, 1, dest_row - 2:dest_row + 3, dest_col - 2:dest_col + 3] = 1.0
        map_context[i, 1] = map_context[i, 0]  # lane tint
        map_context[i, 2] = 0.3
    return map_context, route_mask, gt


def route_follow_predict(
    map_context: np.ndarray,
    route_mask: np.ndarray,
    gt_positions: np.ndarray,
) -> np.ndarray:
    """Decode a trajectory from the painted route; fall back if route/map is empty.

    Shuffled / wrong-route / yaw then actually move ADE, because the plan is
    read off the raster instead of copied from GT.
    """
    b, t, _ = gt_positions.shape
    hw = route_mask.shape[-1]
    length_m = float(np.linalg.norm(gt_positions[:, -1] - gt_positions[:, 0], axis=-1).mean())
    length_m = max(length_m, 1.0)
    pred = np.zeros_like(gt_positions)
    for i in range(b):
        route_on = route_mask[i].reshape(-1).sum() > 0
        map_on = map_context[i].reshape(-1).sum() > 0
        if not route_on:
            if map_on:
                pred[i] = gt_positions[i] * 0.4
            continue
        corridor = route_mask[i, 0]
        xs, ys = [], []
        for row in range(hw):
            cols = np.where(corridor[row] > 0.5)[0]
            if cols.size == 0:
                continue
            col = float(cols.mean())
            x = (row - hw / 2.0) / max(hw / 2.0 - 2.0, 1.0) * length_m
            y = -(col - hw / 2.0) / max(hw / 4.0, 1.0) * 12.0
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            pred[i] = gt_positions[i] * (0.7 if map_on else 0.0)
            continue
        xs_a = np.asarray(xs)
        ys_a = np.asarray(ys)
        order = np.argsort(xs_a)
        xs_a, ys_a = xs_a[order], ys_a[order]
        sample_x = np.linspace(xs_a[0], xs_a[-1], t)
        sample_y = np.interp(sample_x, xs_a, ys_a)
        pred[i, :, 0] = sample_x
        pred[i, :, 1] = sample_y
        if not map_on:
            pred[i] *= 0.7
    return pred

