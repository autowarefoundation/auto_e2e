"""Compare Sim-Auto generations of the shifted clips with the recording and with their WSM controls.

Writes, under <out>:
- grid_t<sec>s.jpg: rows are the recording and each clip, columns the six cameras, at one time.
- blend_t<sec>s.jpg: 50% WSM control / 50% generated RGB for each clip, to check alignment.
- metrics.json:
  - psnr_vs_recording: unshifted generation against the undistorted recording, per camera and time.
  - control_chamfer_px: mean distance from each control's edges to the nearest generated edge, measured
    against the clip's own control and against the unshifted control. A generation that follows the
    shifted ego is closer to its own control than to the unshifted one.
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


def edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150) > 0


def chamfer_px(control: np.ndarray, generated: np.ndarray) -> float:
    """Mean distance from control edge pixels to the nearest generated edge pixel, on the lower half (road)."""
    h = control.shape[0] // 2
    ctrl, gen = edges(control)[h:], edges(generated)[h:]
    if not ctrl.any() or not gen.any():
        return float("nan")
    distance = cv2.distanceTransform((~gen).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[ctrl].mean())


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True, help="inference output directory")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.scene / "meta.json").read_text())
    keys = list(meta["cameras"].values())
    variants = meta["variants"]
    gen, ctrl = {}, {}
    for v in variants:
        for key in keys:
            (vision,) = (args.generated / v["name"]).glob(f"vision_view*_{key}.mp4")
            (control,) = (args.generated / v["name"]).glob(f"control_view*_{key}.mp4")
            gen[v["name"], key], ctrl[v["name"], key] = frames(vision), frames(control)
    size = gen[variants[0]["name"], keys[0]][0].shape[1::-1]

    def recorded(key: str, frame: int) -> tuple[int, np.ndarray] | None:
        """The 10 Hz recording nearest to a 30 Hz frame, resized like the generation."""
        for f in (frame, frame - 1, frame + 1, frame - 2, frame + 2):
            path = args.scene / "gt" / key / f"{f:06d}.jpg"
            if 0 <= f < meta["frames"] and path.exists():
                return f, cv2.resize(cv2.imread(str(path)), size, interpolation=cv2.INTER_AREA)
        return None

    base = variants[0]["name"]
    metrics = {"psnr_vs_recording": {}, "control_chamfer_px": {}, "generated_size": list(size)}
    for key in keys:
        metrics["psnr_vs_recording"][key] = {}
        for t in TIMES_S:
            if (found := recorded(key, min(t * FPS, meta["frames"] - 1))) is not None:
                f, image = found
                metrics["psnr_vs_recording"][key][f"{t}s"] = round(float(cv2.PSNR(image, gen[base, key][f])), 2)
    for v in variants[1:]:
        own = [chamfer_px(ctrl[v["name"], k][-1], gen[v["name"], k][-1]) for k in keys]
        unshifted = [chamfer_px(ctrl[base, k][-1], gen[v["name"], k][-1]) for k in keys]
        metrics["control_chamfer_px"][v["name"]] = {
            "lateral_m": v["lateral_m"], "yaw_deg": v["yaw_deg"],
            "own_control": dict(zip(keys, np.round(own, 2).tolist())),
            "unshifted_control": dict(zip(keys, np.round(unshifted, 2).tolist()))}

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
