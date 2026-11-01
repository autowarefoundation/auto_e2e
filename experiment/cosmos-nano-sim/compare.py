"""Compare Sim-Auto generations of the shifted clips with the recording and with their WSM controls.

Writes, under <out>:
- grid_t<sec>s.jpg: rows are the recording and each clip, columns the six cameras, at one time.
- blend_t<sec>s.jpg: 50% WSM control / 50% generated RGB for each clip, to check alignment.
- metrics.json:
  - psnr_vs_recording: unshifted generation against the undistorted recording, per camera and time.
  - shift_px: horizontal image shift of each shifted clip against the unshifted clip on the last frame,
    by phase correlation of the lower image half, measured once on the WSM controls (what was asked)
    and once on the generations (what the model did). A generation that follows the ego shows the same
    shift as its controls.
  - last_frame_psnr: on the last frame, each clip against the unshifted clip (what the shift changed)
    and, with --other-seed, each clip against the same clip generated with another seed (what the seed
    alone changes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TIMES_S = (0, 2, 4)
FPS = 30
TILE = (416, 240)


def frames(video: Path) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(str(video)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            return out
        out.append(frame)


def shift_px(reference: np.ndarray, moved: np.ndarray) -> float:
    """Horizontal shift of the lower image half (road and nearby objects) from reference to moved."""
    h = reference.shape[0] // 2
    a, b = (np.float32(cv2.cvtColor(x[h:], cv2.COLOR_BGR2GRAY)) for x in (reference, moved))
    (dx, _), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return round(float(dx), 1)


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True, help="inference output directory")
    p.add_argument("--other-seed", type=Path, help="the same clips generated with another seed")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.scene / "meta.json").read_text())
    keys = list(meta["cameras"].values())
    variants = meta["variants"]
    def load(root: Path, kind: str) -> dict:
        return {(v["name"], k): frames(next((root / v["name"]).glob(f"{kind}_view*_{k}.mp4")))
                for v in variants for k in keys}

    gen, ctrl = load(args.generated, "vision"), load(args.generated, "control")
    other = load(args.other_seed, "vision") if args.other_seed else None
    size = gen[variants[0]["name"], keys[0]][0].shape[1::-1]

    def recorded(key: str, frame: int) -> tuple[int, np.ndarray] | None:
        """The 10 Hz recording nearest to a 30 Hz frame, resized like the generation."""
        for f in (frame, frame - 1, frame + 1, frame - 2, frame + 2):
            path = args.scene / "gt" / key / f"{f:06d}.jpg"
            if 0 <= f < meta["frames"] and path.exists():
                return f, cv2.resize(cv2.imread(str(path)), size, interpolation=cv2.INTER_AREA)
        return None

    base = variants[0]["name"]
    metrics = {"psnr_vs_recording": {}, "shift_px": {}, "last_frame_psnr": {}, "generated_size": list(size)}
    psnr = lambda a, b: round(float(cv2.PSNR(a[-1], b[-1])), 2)  # noqa: E731
    for v in variants:
        entry = {"vs_unshifted": {k: psnr(gen[base, k], gen[v["name"], k]) for k in keys}} if v["name"] != base else {}
        if other:
            entry["vs_other_seed"] = {k: psnr(gen[v["name"], k], other[v["name"], k]) for k in keys}
        metrics["last_frame_psnr"][v["name"]] = entry
    for key in keys:
        metrics["psnr_vs_recording"][key] = {}
        for t in TIMES_S:
            if (found := recorded(key, min(t * FPS, meta["frames"] - 1))) is not None:
                f, image = found
                metrics["psnr_vs_recording"][key][f"{t}s"] = round(float(cv2.PSNR(image, gen[base, key][f])), 2)
    for v in variants[1:]:
        metrics["shift_px"][v["name"]] = {
            "lateral_m": v["lateral_m"], "yaw_deg": v["yaw_deg"],
            "control": {k: shift_px(ctrl[base, k][-1], ctrl[v["name"], k][-1]) for k in keys},
            "generated": {k: shift_px(gen[base, k][-1], gen[v["name"], k][-1]) for k in keys}}

    for t in TIMES_S:
        frame = min(t * FPS, meta["frames"] - 1)
        tile = lambda image: cv2.resize(image, TILE, interpolation=cv2.INTER_AREA)  # noqa: E731
        blank = np.zeros((TILE[1], TILE[0], 3), np.uint8)
        rows = [[label(tile(found[1]), f"recorded {key}") if (found := recorded(key, frame)) is not None else blank
                 for key in keys]]
        blends = []
        for v in variants:
            title = f"lat {v['lateral_m']:+.1f} m yaw {v['yaw_deg']:+.1f} deg"
            rows.append([label(tile(gen[v["name"], key][frame]), title) for key in keys])
            blends.append([label(tile(cv2.addWeighted(gen[v["name"], key][frame], 0.5, ctrl[v["name"], key][frame], 0.5, 0)),
                                 f"blend {title}") for key in keys])
        cv2.imwrite(str(args.out / f"grid_t{t}s.jpg"), np.vstack([np.hstack(r) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(args.out / f"blend_t{t}s.jpg"), np.vstack([np.hstack(r) for r in blends]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
