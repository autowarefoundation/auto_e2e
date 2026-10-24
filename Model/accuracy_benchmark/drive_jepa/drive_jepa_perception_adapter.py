"""Official perception-based Drive-JEPA, with KITScenes native input adaptation.

Model forward functions are imported unchanged. Only redundant encoder pretrain
initialization is bypassed before loading the complete agent checkpoint strictly.
"""

import ast
import importlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch

from accuracy_benchmark.drive_jepa.drive_jepa_adapter import native_front_tensor

INPUT_POLICY = {
    "model": "official perception-based ViT-L, unchanged forward, strict complete checkpoint load",
    "camera": "raw KITScenes front-center RGB, t-0.5s and current; crop28 then512x256; native ImageNet normalization",
    "calibration": "actual front-camera intrinsic and camera-to-KITScenes-reference extrinsic",
    "projection_metadata": "native0.4 scale and32-pixel padding, normalized with actual front image shape; unused calibration slot0 aliases front",
    "ego_status": "current-relative zero pose, causal KITScenes velocity/acceleration, supplied4-component command",
    "coordinate_reference": "KITScenes recorded ego origin; upstream trained vehicle geometry retained unchanged",
    "precision": "native fp32 CUDA deformable attention",
    "v2_selection": "native independent-sample learned score; no optional past simulated states/comfort calibration context supplied",
}


def load_model(repo: Path, checkpoint: Path, version="v1"):
    if version not in ("v1", "v2"):
        raise ValueError("Expected NAVSIM v1 or v2")
    navsim_root = repo / f"navsim_{version}"
    loaded_navsim = sys.modules.get("navsim")
    if (
        loaded_navsim is not None
        and Path(next(iter(loaded_navsim.__path__))).parent.resolve()
        != navsim_root.resolve()
    ):
        raise RuntimeError(
            "Different NAVSIM versions require separate fresh inference processes"
        )
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / ".benchmark/vendor/nuplan")
    )
    sys.path.insert(0, str(navsim_root))
    import vjepa2.src.models.vision_transformer as vit

    def init_encoder(
        resolution, checkpoint, model_kwargs, wrapper_kwargs, register_prehook, **kwargs
    ):
        if register_prehook:
            raise ValueError("Expected the native two-frame ViT encoder")
        enc = model_kwargs["encoder"]
        return vit.__dict__[enc["model_name"]](
            input_size=resolution,
            num_frames=wrapper_kwargs["img_as_video_nframes"],
            **enc,
        )

    name = "navsim.agents.drive_jepa_perception_based.bevformer.simple_image_encoder"
    importlib.import_module(name.rsplit(".", 1)[0])
    source = (
        navsim_root
        / "navsim/agents/drive_jepa_perception_based/bevformer/simple_image_encoder.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom)
            and (n.module or "").startswith("vjepa2.evals.")
        )
    ]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None:
        raise ImportError(f"Cannot import upstream image encoder: {source}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, "init_module", init_encoder)
    sys.modules[name] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
        DriveJEPAConfig,
    )

    if version == "v1":
        from navsim.agents.drive_jepa_perception_based.drive_jepa_model import (
            DriveJEPAModel,
        )
    else:
        # Optional sequential simulator context is absent in the official
        # independent-sample path used here. Avoid importing/initializing the
        # simulator and metric-cache stack; retain all original model methods,
        # including forward and its optional-context branch, unchanged.
        model_name = "navsim.agents.drive_jepa_perception_based.drive_jepa_model"
        model_source = (
            navsim_root
            / "navsim/agents/drive_jepa_perception_based/drive_jepa_model.py"
        )
        tree = ast.parse(
            model_source.read_text(encoding="utf-8"), filename=str(model_source)
        )
        simulator_imports = {
            "navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator",
            "navsim.planning.metric_caching.metric_cache",
            "navsim.evaluate.pdm_score",
            "navsim.common.dataclasses",
            "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics",
        }
        tree.body = [
            n
            for n in tree.body
            if not (
                isinstance(n, ast.ImportFrom)
                and n.module in simulator_imports
                or isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id in ("proposal_sampling", "simulator")
                    for t in n.targets
                )
            )
        ]
        spec = importlib.util.spec_from_file_location(model_name, model_source)
        if spec is None:
            raise ImportError(f"Cannot import upstream v2 model: {model_source}")
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = model_module
        exec(compile(tree, str(model_source), "exec"), model_module.__dict__)
        DriveJEPAModel = model_module.DriveJEPAModel
    previous_cwd = Path.cwd()
    try:
        os.chdir(navsim_root)
        model = DriveJEPAModel(DriveJEPAConfig())
    finally:
        os.chdir(previous_cwd)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "agent._pad_model."
    if not state or any(not key.startswith(prefix) for key in state):
        raise ValueError(
            "Expected a complete official perception-based agent checkpoint"
        )
    model.load_state_dict({k[len(prefix) :]: v for k, v in state.items()}, strict=True)
    return model.eval()


def native_features(sample, command, camera):
    """Use original BEV metadata preprocessing and original front image crop.

    KITScenes camera-to-reference extrinsics supply the native projection origin.
    The current ego pose is zero in NAVSIM's current-relative pose convention.
    Only current status is consumed by the released forward function.
    """
    loader, frame = sample["loader"], sample["frame_idx"]
    calibration = loader.get_camera_calibration(camera)
    current_image = loader.get_camera_image(camera, frame)
    projection, padded_shape = projection_metadata(calibration, current_image.shape)
    # The ViT forward selects calibration slot 1 and normalizes with shape
    # slot 0. NAVSIM cameras share a shape; KITScenes cameras do not. Use the
    # real front-camera shape and calibration, with an unused slot-0 alias,
    # rather than stacking unused surround images of incompatible resolutions.
    features = {
        "lidar2img": torch.from_numpy(np.stack((projection, projection))),
        "img_shape": torch.tensor([padded_shape], dtype=torch.float32),
    }
    features.update(
        camera_feature_1=native_front_tensor(current_image),
        camera_feature_2=native_front_tensor(
            loader.get_camera_image(camera, frame - 5)
        ),
    )
    status = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            sample["velocity_xy"],
            sample["acceleration_xy"],
            np.asarray(command, dtype=np.float32),
        )
    )
    features["ego_status"] = torch.from_numpy(status).reshape(1, 11)
    return features


def projection_metadata(calibration, image_shape):
    """Original 0.4 scale / 32-pixel padding applied to actual front calibration.

    The native model uses this BEV projection metadata alongside its separate
    512x256 cropped image path; that upstream convention is preserved.
    """
    extrinsic = np.asarray(calibration.extrinsic, dtype=np.float64)
    intrinsic = np.asarray(calibration.intrinsic, dtype=np.float64)
    if (
        extrinsic.shape != (4, 4)
        or intrinsic.shape != (3, 3)
        or not np.isfinite(extrinsic).all()
        or not np.isfinite(intrinsic).all()
    ):
        raise ValueError("Invalid KITScenes camera calibration")
    viewpad = np.eye(4)
    viewpad[:3, :3] = intrinsic
    scale = np.diag([0.4, 0.4, 1.0, 1.0])
    projection = scale @ viewpad @ np.linalg.inv(extrinsic)
    height, width, channels = image_shape
    padded = [((int(v * 0.4) + 31) // 32) * 32 for v in (height, width)] + [channels]
    return projection.astype(np.float32), padded


class DriveJEPAPerceptionAdapter:
    prediction_times_s = np.arange(1, 9, dtype=np.float64) * 0.5

    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for official perception-based inference"
            )
        if getattr(args, "precision", "fp32") != "fp32":
            raise ValueError(
                "Perception-based native CUDA attention requires --precision fp32"
            )
        version = "v2" if args.checkpoint.stem.endswith("_v2") else "v1"
        self.model = load_model(
            args.drive_jepa_root.resolve(), args.checkpoint.resolve(), version
        ).cuda()
        self.camera = args.camera
        self.command = np.asarray(args.command, dtype=np.float32)
        self.commands = args.commands
        self.last_proposals = self.last_scores = None

    def predict(self, sample):
        command = self.commands.get(sample["sample_uid"], self.command)
        features = {
            k: v.unsqueeze(0).cuda()
            for k, v in native_features(sample, command, self.camera).items()
        }
        # The CUDA deformable-attention extension requires fp32. Preserve native
        # upstream fp32 evaluation rather than forcing an unsupported BF16 op.
        with torch.inference_mode():
            output = self.model(features)
        self.last_proposals = output["proposals"][0].float().cpu().numpy()
        self.last_scores = output["pdm_score"][0].float().cpu().numpy()
        return output["trajectory"][0, :, :2].float().cpu().numpy()


def create_adapter(args):
    return DriveJEPAPerceptionAdapter(args)
