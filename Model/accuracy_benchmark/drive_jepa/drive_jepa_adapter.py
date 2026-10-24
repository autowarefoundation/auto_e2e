"""Windows inference bridge for the released NAVSIM v1 perception-free model.

The upstream model and ViT classes execute unchanged. Only NAVSIM's sampling
type and encoder initialization are bridged; the complete planning checkpoint
is loaded strictly, so a separate pretraining checkpoint is unnecessary.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


def load_model(repo: Path, checkpoint: Path) -> torch.nn.Module:
    navsim_root = repo / "navsim_v1"
    sys.path.insert(0, str(navsim_root))
    import vjepa2.src.models.vision_transformer as vit

    enum_path = navsim_root / "navsim/common/enums.py"
    spec = importlib.util.spec_from_file_location("drive_jepa_state_enums", enum_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream enum definitions: {enum_path}")
    enums = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enums)

    def init_encoder(
        resolution, checkpoint, model_kwargs, wrapper_kwargs, register_prehook, **kwargs
    ):
        if register_prehook:
            raise ValueError(
                "Released perception-free checkpoint requires double_image"
            )
        enc = model_kwargs["encoder"]
        return vit.__dict__[enc["model_name"]](
            input_size=resolution,
            num_frames=wrapper_kwargs["img_as_video_nframes"],
            **enc,
        )

    source = (
        navsim_root / "navsim/agents/drive_jepa_perception_free/drive_jepa_model.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    bridged_imports = (
        "nuplan.",
        "navsim.common.enums",
        "vjepa2.evals.",
    )
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith(bridged_imports)
        )
    ]
    namespace = {
        "TrajectorySampling": SimpleNamespace,
        "StateSE2Index": enums.StateSE2Index,
        "init_module": init_encoder,
        "__name__": "drive_jepa_inference_model",
    }
    exec(compile(tree, str(source), "exec"), namespace)
    previous_cwd = Path.cwd()
    try:
        os.chdir(navsim_root)  # Upstream reads its YAML relative to cwd.
        model = namespace["DriveJEPAModel"](
            SimpleNamespace(num_poses=8),
            None,
            num_keyval=129,
            front_only=True,
            double_image=True,
            freeze_encoder=False,
        )
    finally:
        os.chdir(previous_cwd)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "agent._model."
    if not state or any(not key.startswith(prefix) for key in state):
        raise ValueError(
            "Expected a complete official perception-free agent checkpoint"
        )
    model.load_state_dict(
        {key[len(prefix) :]: value for key, value in state.items()}, strict=True
    )
    return model.eval()


def native_front_tensor(image: np.ndarray) -> torch.Tensor:
    """Use upstream's 28-pixel top/bottom crop then 512x256 OpenCV resize."""
    import cv2

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 56:
        raise ValueError("Expected a raw RGB front camera image")
    image = cv2.resize(image[28:-28], (512, 256))
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255)


class DriveJEPAAdapter:
    prediction_times_s = np.arange(1, 9, dtype=np.float64) * 0.5

    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required for Drive-JEPA evaluation")
        self.model = load_model(
            args.drive_jepa_root.resolve(), args.checkpoint.resolve()
        ).cuda()
        self.camera = args.camera
        self.precision = args.precision
        self.command = np.asarray(args.command, dtype=np.float32)
        self.commands = args.commands

    def predict(self, sample) -> np.ndarray:
        loader = sample["loader"]
        current = sample["frame_idx"]
        # NAVSIM's two image frames are separated by 0.5 s, not 0.1 s.
        images = [
            native_front_tensor(loader.get_camera_image(self.camera, idx))
            for idx in (current - 5, current)
        ]
        camera = torch.stack(images, dim=1).unsqueeze(0).cuda()
        command = self.commands.get(sample["sample_uid"], self.command)
        status = np.concatenate(
            [command, sample["velocity_xy"], sample["acceleration_xy"]]
        )
        status = torch.as_tensor(status, dtype=torch.float32, device="cuda").unsqueeze(
            0
        )
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                dtype=dtype,
                enabled=self.precision != "fp32",
            ),
        ):
            prediction = self.model(camera, status)["trajectory"]
        return prediction[0, :, :2].float().cpu().numpy()


def create_adapter(args):
    return DriveJEPAAdapter(args)
