"""Render WSM4 controls for every shifted clip of one scene and write one Sim-Auto request per clip.

Every request uses the same initial RGB (the recorded first frame of each camera) and the same
captions, so the clips differ only in the ego pose that the WSM controls were rendered from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WSM_DIR = "cosmos-framework/.agents/skills/cookbooks/cosmos3/simulation/auto/wsm"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/cosmos-nano-sim"))
    p.add_argument("--scene", type=Path, required=True, help="scene directory written by nuplan_to_rdshq.py")
    p.add_argument("--captions", type=Path, required=True, help="JSON mapping camera_key to its caption")
    p.add_argument("--resolution", default="480")
    p.add_argument("--seed", type=int, default=0, help="a second seed shows how much the scene varies without any shift")
    args = p.parse_args()

    meta = json.loads((args.scene / "meta.json").read_text())
    keys, frames = list(meta["cameras"].values()), meta["frames"]
    captions = json.loads(args.captions.read_text())
    wsm_root = args.scene / "wsm"
    if not wsm_root.exists():
        subprocess.run([
            str(args.root / "wsm-venv/bin/cosmos-auto-wm"),
            f"input_root={args.scene / 'rds-hq'}", f"output_root={wsm_root}",
            "version=v4", "s3_spec=no_upload", "camera_type=ftheta",
            f"camera_spec.camera_list=[{','.join(keys)}]",
            "input_fps_spec.input_pose_fps=30", "resolution_spec=resize_1280x720",
            "output_frame_spec.target_render_fps=30", f"output_frame_spec.target_chunk_frame={frames}",
            "output_frame_spec.overlap_frame=0", "output_frame_spec.max_chunk=1",
        ], cwd=args.root / WSM_DIR, check=True)

    (args.scene / "captions").mkdir(exist_ok=True)
    for key in keys:
        caption = {"caption_structured": {f"chunk_0_{frames}": {
            "caption": captions[key], "start_frame": 0, "end_frame": frames}}}
        (args.scene / "captions" / f"{key}.json").write_text(json.dumps(caption, indent=2))
        rgb = args.scene / "rgb" / f"{key}.mp4"
        if not rgb.exists():  # still video of the recorded first frame; only its frame 0 conditions the model
            subprocess.run(["ffmpeg", "-loglevel", "error", "-loop", "1", "-i", str(args.scene / "rgb" / f"{key}.jpg"),
                            "-vf", "scale=1280:720", "-r", "30", "-frames:v", str(frames),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(rgb)], check=True)

    requests = []
    for variant in meta["variants"]:
        views = []
        for key in keys:
            control = wsm_root / "world_scenario" / f"ftheta_{key}" / f"{variant['name']}_0.mp4"  # renderer layout, chunk 0
            views.append({"camera_key": key, "control_path": str(control.relative_to(args.scene)),
                          "vision_path": f"rgb/{key}.mp4", "caption_path": f"captions/{key}.json"})
        requests.append({
            "name": variant["name"], "model_mode": "image2video", "resolution": args.resolution,
            "aspect_ratio": "16,9", "fps": 30, "num_frames": frames, "num_video_frames_per_chunk": frames,
            "num_outputs": 1, "seed": args.seed, "guidance": 1, "control_guidance": 1, "normalize_cfg": False,
            "autoregressive": True, "wsm": {}, "condition_frame_indexes_vision": [0], "num_conditional_frames": 1,
            "prompt": "", "multiview": {"load_caption_from_data": True, "use_first_chunk_only": True,
                                        "condition_video_as_image": True, "views": views}})
    out = args.scene / f"requests_{args.resolution}p_seed{args.seed}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in requests))
    print(out, len(requests), "requests")


if __name__ == "__main__":
    main()
