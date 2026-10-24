"""Official Drive-JEPA perception-based NAVSIM v2 on KITScenes val, with 20 m commands."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import os
from collections.abc import Mapping
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tarfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data_parsing.kit_scenes.source import PinnedKITScenesDownloader
from data_processing.contract_versions import UID_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = REPO_ROOT / ".benchmark"
DRIVE_JEPA_REVISION = "548bb8215e3aae18e162a0f12f1ba83b4d3eb57e"
KITSCENES_SDK_REVISION = "7765cdec5490894266070ab46e23724b58b3da42"
KITSCENES_DATA_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
WEIGHTS_REVISION = "65e0d7284f69bf29d1a4864affcfd84ca4e97a2e"
CHECKPOINT_NAME = "drive_jepa_perception_based_agent_vitl_v2.ckpt"
CHECKPOINT_SHA256 = "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05"
EXPECTED_VAL_SAMPLES = 11035
EXPECTED_VAL_SAMPLE_UID_SHA256 = (
    "00f824202c97b47ec74561d247e8e30686e5505eef2ad2bc0acba97e41ce9925"
)
# Canonical LF serialization; the original Windows artifact used CRLF.
COMMANDS_SHA256 = "c3433171b5cb3022ec1d6ad711bcd5ece993fe2801842d0a8a96269d55a554ab"
MIRROR_REVISION = "7262c151599c1b5e4a99f0da9c3196ff9ff615fe"
DISPLACEMENT_HORIZON_STEPS = {"1s": 10, "2s": 20, "3s": 30, "4s": 40}
POPULATION_CAMERAS = (
    "camera_base_front_center",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear_left",
    "camera_ring_rear",
    "camera_ring_rear_right",
)


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


def native_front_tensor(image: np.ndarray) -> torch.Tensor:
    """Use upstream's 28-pixel top/bottom crop then 512x256 OpenCV resize."""
    import cv2

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 56:
        raise ValueError("Expected a raw RGB front camera image")
    image = cv2.resize(image[28:-28], (512, 256))
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255)


def load_model(repo: Path, checkpoint: Path):
    navsim_root = repo / "navsim_v2"
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

    model_name = "navsim.agents.drive_jepa_perception_based.drive_jepa_model"
    model_source = (
        navsim_root / "navsim/agents/drive_jepa_perception_based/drive_jepa_model.py"
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
            or (
                isinstance(n, ast.Assign)
                and any(
                    (
                        isinstance(t, ast.Name)
                        and t.id in ("proposal_sampling", "simulator")
                        for t in n.targets
                    )
                )
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
    if not state or any((not key.startswith(prefix) for key in state)):
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
    loader, frame = (sample["loader"], sample["frame_idx"])
    calibration = loader.get_camera_calibration(camera)
    current_image = loader.get_camera_image(camera, frame)
    projection, padded_shape = projection_metadata(calibration, current_image.shape)
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
        or (not np.isfinite(extrinsic).all())
        or (not np.isfinite(intrinsic).all())
    ):
        raise ValueError("Invalid KITScenes camera calibration")
    viewpad = np.eye(4)
    viewpad[:3, :3] = intrinsic
    scale = np.diag([0.4, 0.4, 1.0, 1.0])
    projection = scale @ viewpad @ np.linalg.inv(extrinsic)
    height, width, channels = image_shape
    padded = [(int(v * 0.4) + 31) // 32 * 32 for v in (height, width)] + [channels]
    return (projection.astype(np.float32), padded)


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
        self.model = load_model(
            args.drive_jepa_root.resolve(), args.checkpoint.resolve()
        ).cuda()
        self.camera = args.camera
        self.commands = args.commands
        self.last_proposals = self.last_scores = None

    def predict(self, sample):
        command = self.commands[sample["sample_uid"]]
        features = {
            k: v.unsqueeze(0).cuda()
            for k, v in native_features(sample, command, self.camera).items()
        }
        with torch.inference_mode():
            output = self.model(features)
        self.last_proposals = output["proposals"][0].float().cpu().numpy()
        self.last_scores = output["pdm_score"][0].float().cpu().numpy()
        return output["trajectory"][0, :, :2].float().cpu().numpy()


def create_adapter(args):
    return DriveJEPAPerceptionAdapter(args)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def published_digest_ids(ids):
    """Overlay replay reports hash sorted UID lines with a terminal LF.

    Reactive evaluation uses no terminal LF, so preserve both explicitly.
    """
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def check_sdk(root):
    import kitscenes

    if git_revision(root) != KITSCENES_SDK_REVISION:
        raise ValueError(f"SDK checkout must be {KITSCENES_SDK_REVISION}")
    if Path(kitscenes.__file__).resolve().parent != (root / "kitscenes").resolve():
        raise ValueError("Imported kitscenes differs from --sdk-root; set PYTHONPATH")


def contiguous_length(indices):
    for expected, actual in enumerate(indices):
        if expected != actual:
            return expected
    return len(indices)


def scene_samples(dataset, scene_id):
    """Match AutoE2E v3.5 benchmark enumeration without resizing images/maps."""
    loader = dataset.get_sensor_loader(scene_id)
    poses = dataset.get_scene(scene_id).ego_poses
    reference = loader.get_reference_timestamps()
    usable = min(
        len(poses),
        len(reference),
        *[
            contiguous_length(loader.get_frame_indices(camera))
            for camera in POPULATION_CAMERAS
        ],
    )
    if usable < 91:
        raise ValueError(
            f"Scene {scene_id} cannot satisfy 40-history/50-future selection"
        )
    poses = poses[:usable]
    stamps = np.array([p.timestamp_ns for p in poses], dtype=np.int64)
    if (
        np.any(np.diff(stamps) <= 0)
        or np.max(np.abs(stamps - reference[:usable])) > 50_000_000
    ):
        raise ValueError(f"Pose/reference timestamps are misaligned in {scene_id}")
    # Horizons and image history rely on the original 10 Hz timeline.
    nominal = stamps[0] + np.arange(usable, dtype=np.int64) * 100_000_000
    if np.max(np.abs(stamps - nominal)) > 50_000_000:
        raise ValueError(f"Scene {scene_id} does not have an aligned 10 Hz timeline")
    positions = np.array([p.translation[:2] for p in poses], dtype=np.float64)
    yaw = Rotation.from_quat([p.rotation for p in poses]).as_euler("ZYX")[:, 0]
    for frame in range(40, usable - 50):
        # Causal status: use only current/past poses, never future GT derivatives.
        dt_now = (stamps[frame] - stamps[frame - 1]) / 1e9
        dt_past = (stamps[frame - 1] - stamps[frame - 2]) / 1e9
        velocity_world = (positions[frame] - positions[frame - 1]) / dt_now
        previous_velocity = (positions[frame - 1] - positions[frame - 2]) / dt_past
        acceleration_world = (velocity_world - previous_velocity) / (
            (dt_now + dt_past) / 2
        )
        cosine, sine = np.cos(yaw[frame]), np.sin(yaw[frame])
        to_ego = np.array([[cosine, sine], [-sine, cosine]])
        target = (positions[frame + 1 : frame + 51] - positions[frame]) @ to_ego.T
        yield {
            "scene_id": scene_id,
            "frame_idx": frame,
            "timestamp_ns": int(stamps[frame]),
            "sample_uid": f"kitscenes-{UID_SCHEMA_VERSION}-{scene_id}-f{frame:06d}",
            "target_xy": target.astype(np.float32),
            "loader": loader,
            "velocity_xy": (to_ego @ velocity_world).astype(np.float32),
            "acceleration_xy": (to_ego @ acceleration_world).astype(np.float32),
        }


def resample_prediction(prediction, times):
    """Interpolate at 10 Hz from origin; never extrapolate past native support."""
    prediction = np.asarray(prediction, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if (
        prediction.shape != (len(times), 2)
        or not np.isfinite(prediction).all()
        or not np.isfinite(times).all()
        or not len(times)
        or times[0] <= 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Adapter must return finite ego FLU metres at positive increasing times"
        )
    steps = min(50, int(np.floor(times[-1] * 10 + 1e-7)))
    if steps < 10:
        raise ValueError("Adapter must predict at least 1 second")
    query = np.arange(1, steps + 1) / 10
    source_t = np.concatenate([[0.0], times])
    source_xy = np.concatenate([np.zeros((1, 2)), prediction], axis=0)
    return np.column_stack(
        [np.interp(query, source_t, source_xy[:, axis]) for axis in range(2)]
    ).astype(np.float32)


def score_predictions(predictions, targets):
    steps = predictions.shape[1]
    errors = torch.linalg.vector_norm(
        torch.from_numpy(predictions) - torch.from_numpy(targets[:, :steps]),
        dim=-1,
    )
    horizons = {
        name: count
        for name, count in DISPLACEMENT_HORIZON_STEPS.items()
        if count <= min(steps, 50)
    }
    return displacement_horizon_totals(
        errors, torch.ones_like(errors, dtype=torch.bool), horizons
    )


def displacement_horizon_totals(
    errors: torch.Tensor,
    valid: torch.Tensor,
    horizon_steps: Mapping[str, int],
) -> dict[str, tuple[float, int, float, int]]:
    """Return ADE sum/count and FDE sum/count using the Reactive mask policy.

    ADE gives each sample equal weight over its valid points. FDE requires
    the endpoint to be valid. Prediction support must be checked by the caller;
    a shorter prediction is never permission to label a longer-horizon ADE.
    """
    totals = {}
    for name, step_count in horizon_steps.items():
        usable = min(step_count, errors.shape[1])
        horizon_valid = valid[:, :usable]
        per_sample_count = horizon_valid.sum(dim=1)
        eligible_ade = per_sample_count > 0
        per_sample_error = (errors[:, :usable] * horizon_valid).sum(
            dim=1
        ) / per_sample_count.clamp_min(1)
        ade_sum = float(per_sample_error[eligible_ade].sum().item())
        ade_count = int(eligible_ade.sum().item())
        fde_sum, fde_count = 0.0, 0
        if step_count <= errors.shape[1]:
            eligible_fde = valid[:, step_count - 1]
            fde_sum = float(errors[eligible_fde, step_count - 1].sum().item())
            fde_count = int(eligible_fde.sum().item())
        totals[name] = (ade_sum, ade_count, fde_sum, fde_count)
    return totals


@contextmanager
def raw_scene(args, scene_id):
    from kitscenes import KITScenesDataset

    if args.raw_root is not None:
        yield KITScenesDataset(root=args.raw_root, split="val")
        return
    args.staging_root.mkdir(parents=True, exist_ok=True)
    # Only a directory created by this run is removed; user datasets are preserved.
    with tempfile.TemporaryDirectory(
        prefix="kitscenes_scene_", dir=args.staging_root
    ) as temporary:
        downloader = PinnedKITScenesDownloader(
            temporary, revision=args.dataset_revision
        )
        archive = downloader.archives[scene_id]
        required = 2 * archive.size_bytes + 5 * 1024**3
        if shutil.disk_usage(temporary).free < required:
            raise RuntimeError(
                f"Need at least {required / 1024**3:.1f} GiB staging space for {scene_id}"
            )
        downloader.download([scene_id], expected_split="val")
        yield KITScenesDataset(root=temporary, split="val")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clone(url, destination, revision):
    if destination.exists():
        if git_revision(destination) != revision:
            raise ValueError(
                f"Existing checkout differs; preserve it and use a fresh workspace: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", url, str(destination)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "-c",
            "core.autocrlf=false",
            "checkout",
            "--detach",
            revision,
        ],
        check=True,
    )


def setup(args):
    """Fetch exact public source/weight revisions; never reset an existing checkout."""
    clone(
        "https://github.com/KIT-MRT/kitscenes.git",
        args.sdk_root,
        KITSCENES_SDK_REVISION,
    )
    clone(
        "https://github.com/linhanwang/Drive-JEPA.git",
        args.drive_jepa_root,
        DRIVE_JEPA_REVISION,
    )
    clone(
        "https://github.com/motional/nuplan-devkit.git",
        WORKSPACE / "vendor/nuplan",
        "ce3c323af01c0d7ec5672f7832ef53f9c679aab0",
    )
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            "LinhanWang/Drive-JEPA",
            CHECKPOINT_NAME,
            repo_type="dataset",
            revision=WEIGHTS_REVISION,
            local_dir=args.checkpoint.parent,
        )
    )
    if path.resolve() != args.checkpoint.resolve():
        raise ValueError(f"Checkpoint basename must be {CHECKPOINT_NAME}")
    if file_sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint checksum differs")
    print("Setup complete. Run the same command without --setup to evaluate.")


def command_from_route(route, metadata, lookahead_m=20.0, lateral_threshold_m=2.0):
    if lookahead_m <= 0 or lateral_threshold_m < 0:
        raise ValueError("Invalid command geometry thresholds")
    if not route.valid or not metadata["route_valid"] or not metadata["map_valid"]:
        return [0.0, 0.0, 0.0, 1.0], {"reason": "invalid_route_or_map"}
    pose = metadata["sample_pose_enu"]
    position = np.array([pose["x_enu_m"], pose["y_enu_m"]], dtype=np.float64)
    yaw = float(pose["yaw_rad"])
    if not np.isfinite(position).all() or not np.isfinite(yaw):
        raise ValueError("Non-finite current ego pose")
    candidates = []
    for lane_index, lane in enumerate(route.lane_sequence):
        if lane.lane_id != metadata["current_route_lane_id"]:
            continue
        points = np.asarray(lane.centerline_enu_m, dtype=np.float64)[:, :2]
        if not np.isfinite(points).all():
            raise ValueError("Non-finite route centreline")
        for point_index, (a, b) in enumerate(zip(points[:-1], points[1:])):
            delta = b - a
            length_squared = float(delta @ delta)
            if length_squared == 0:
                continue
            fraction = np.clip(float((position - a) @ delta) / length_squared, 0, 1)
            projection = a + fraction * delta
            candidates.append(
                (
                    float(np.linalg.norm(position - projection)),
                    lane_index,
                    point_index,
                    projection,
                )
            )
    if not candidates:
        return [0.0, 0.0, 0.0, 1.0], {"reason": "active_lane_missing_or_degenerate"}
    _, lane_index, point_index, start = min(candidates, key=lambda c: c[:3])
    path = [start]
    path.extend(
        np.asarray(route.lane_sequence[lane_index].centerline_enu_m)[
            point_index + 1 :, :2
        ]
    )
    for lane in route.lane_sequence[lane_index + 1 :]:
        if (
            sum(
                float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
                for a, b in zip(path[:-1], path[1:])
            )
            >= lookahead_m
        ):
            break
        points = np.asarray(lane.centerline_enu_m, dtype=np.float64)[:, :2]
        if len(points) == 0:
            continue
        if not np.isfinite(points).all():
            raise ValueError("Non-finite route centreline")
        if np.linalg.norm(np.asarray(path[-1]) - points[0]) > 2.0:
            # Do not invent a centreline across a lane-change or unresolved gap.
            return [0.0, 0.0, 0.0, 1.0], {"reason": "discontinuous_route"}
        path.extend(points)
    remaining = float(lookahead_m)
    for a, b in zip(path[:-1], path[1:]):
        delta = np.asarray(b) - np.asarray(a)
        length = float(np.linalg.norm(delta))
        if length == 0:
            continue
        if remaining <= length:
            goal = np.asarray(a) + delta * (remaining / length)
            relative = goal - position
            lateral = float(-np.sin(yaw) * relative[0] + np.cos(yaw) * relative[1])
            slot = (
                0
                if lateral > lateral_threshold_m
                else 2
                if lateral < -lateral_threshold_m
                else 1
            )
            command = [0.0] * 4
            command[slot] = 1.0
            return command, {
                "reason": "available",
                "goal_enu_m": goal.tolist(),
                "goal_lateral_m": lateral,
            }
        remaining -= length
    return [0.0, 0.0, 0.0, 1.0], {"reason": "route_shorter_than_lookahead"}


def load_commands(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Generate commands first with --generate-commands --navigation-root PATH"
        )
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    if hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest() != COMMANDS_SHA256 or provenance.get(
        "commands_sha256"
    ) != file_sha256(path):
        raise ValueError(
            "Command checksum differs from the audited v3.5 20 m population"
        )
    if provenance.get("command_policy") != "route_centerline_20m_lateral_2m_v1":
        raise ValueError("Expected the v3.5 20 m route-command policy")
    commands = json.loads(path.read_text())
    if (
        len(commands) != EXPECTED_VAL_SAMPLES
        or published_digest_ids(commands) != EXPECTED_VAL_SAMPLE_UID_SHA256
    ):
        raise ValueError(
            "20 m commands do not match the published validation population"
        )
    for value in commands.values():
        if len(value) != 4 or set(value) - {0.0, 1.0} or sum(value) != 1:
            raise ValueError("Invalid NAVSIM one-hot command")
    return commands, provenance


def official_val_scenes(sdk_root):
    if git_revision(sdk_root) != KITSCENES_SDK_REVISION:
        raise ValueError("KITScenes SDK revision differs")
    split = (
        sdk_root
        / "kitscenes/split/generated_splits/default_geo_split_v1_0/validation.txt"
    )
    payload = split.read_bytes().replace(b"\r\n", b"\n")
    if (
        hashlib.sha256(payload).hexdigest()
        != "4a78dd31f4340e0e7953b7c117e83f80d1cd99f22a0ca127c4eb5c60cad70a21"
    ):
        raise ValueError("Official validation split has changed")
    scenes = payload.decode("ascii").split()
    if len(scenes) != 117 or len(set(scenes)) != 117:
        raise ValueError("Expected 117 unique validation scenes")
    return scenes


def generate_commands(args):
    """Convert exported vector routes using pinned published current-sample metadata.

    Route reconstruction is a separate Lanelet2 preprocessing step. This
    function never reads future ego targets or invents missing route geometry.
    """
    from huggingface_hub import hf_hub_download
    from navigation.artifacts import decode_scene_navigation

    official = set(official_val_scenes(args.sdk_root))
    missing = [
        scene
        for scene in sorted(official)
        if not (args.navigation_root / scene / "scene_navigation.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} exported scene_navigation.json files; first: {missing[0]}"
        )

    def fetch(name):
        return Path(
            hf_hub_download(
                "AutowareFoundation/auto_e2e",
                "datasets/kitscenes-val/v3.5/shards/" + name,
                revision=MIRROR_REVISION,
                local_dir=args.mirror_root,
            )
        )

    manifest_path = fetch("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    partitions = {
        p["partition_id"]: p["sample_count"]
        for p in manifest["partitions"]
        if "kitscenes-v35-val-" in p["source_uri"]
    }
    if len(partitions) != 117 or sum(partitions.values()) != EXPECTED_VAL_SAMPLES:
        raise ValueError("Published mirror validation population differs")
    entries = [
        entry
        for entry in manifest["shard_entries"]
        if any(entry["name"].startswith(pid + "-") for pid in partitions)
    ]
    if len(entries) != 117:
        raise ValueError("Expected 117 published validation shards")
    commands, scene_hashes, reasons = {}, {}, Counter()
    for index, entry in enumerate(entries, 1):
        name = entry["name"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Unsafe shard filename")
        shard = fetch(name)
        if shard.stat().st_size != entry["byte_size"]:
            raise ValueError("Shard size differs from pinned manifest")
        partition = next(pid for pid in partitions if name.startswith(pid + "-"))
        scene, route, count = None, None, 0
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                suffix = ".navigation_meta.json"
                if not member.name.endswith(suffix):
                    continue
                if not member.isfile() or member.size > 128 * 1024:
                    raise ValueError("Invalid navigation metadata member")
                uid = member.name[: -len(suffix)]
                sample_scene = uid.removeprefix("kitscenes-v1-").rsplit("-f", 1)[0]
                if sample_scene not in official or uid in commands:
                    raise ValueError("Unexpected or duplicate sample identity")
                if scene is not None and scene != sample_scene:
                    raise ValueError("Expected one scene per published shard")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Unreadable navigation metadata")
                metadata = json.loads(stream.read())
                if metadata.get("schema_version") != "sample_navigation_v2":
                    raise ValueError("Published sample metadata schema differs")
                expected_hash = metadata["scene_navigation_sha256"]
                if scene is None:
                    scene = sample_scene
                    if scene in scene_hashes:
                        raise ValueError("Duplicate scene partition")
                    path = args.navigation_root / scene / "scene_navigation.json"
                    if file_sha256(path) != expected_hash:
                        raise ValueError(
                            f"Exported navigation hash differs from published v3.5: {scene}"
                        )
                    _, route = decode_scene_navigation(path.read_bytes())
                    scene_hashes[scene] = expected_hash
                elif scene_hashes[scene] != expected_hash:
                    raise ValueError("Published scene hashes disagree")
                commands[uid], details = command_from_route(route, metadata)
                reasons[details["reason"]] += 1
                count += 1
        if count != partitions[partition]:
            raise ValueError("Shard sample count differs from pinned manifest")
        print(f"Generated {index}/117 scenes, {len(commands)} commands", flush=True)
    if (
        set(scene_hashes) != official
        or len(commands) != EXPECTED_VAL_SAMPLES
        or published_digest_ids(commands) != EXPECTED_VAL_SAMPLE_UID_SHA256
    ):
        raise ValueError("Generated command population differs")
    payload = (
        json.dumps(commands, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if hashlib.sha256(payload).hexdigest() != COMMANDS_SHA256:
        raise ValueError("Generated 20 m labels differ from the reference evaluation")
    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.command_file, commands)
    write_json(
        args.command_file.with_suffix(".provenance.json"),
        {
            "command_policy": "route_centerline_20m_lateral_2m_v1",
            "command_order": ["left", "straight", "right", "unknown"],
            "commands_sha256": file_sha256(args.command_file),
            "commands_lf_sha256": COMMANDS_SHA256,
            "scene_count": 117,
            "sample_count": len(commands),
            "published_sample_uid_sha256": published_digest_ids(commands),
            "mirror_revision": MIRROR_REVISION,
            "manifest_sha256": file_sha256(manifest_path),
            "scene_navigation_sha256": scene_hashes,
            "all_scene_navigation_hashes_match_v35": True,
            "command_counts": dict(
                Counter(
                    ["left", "straight", "right", "unknown"][v.index(1.0)]
                    for v in commands.values()
                )
            ),
            "reason_counts": dict(reasons),
            "ego_reference": "KITScenes recorded ego-pose origin",
            "route_provenance": "offline whole-scene map-matched v3.5 routes; exact published hashes verified",
            "uses_future_ground_truth_for_command_extraction": False,
        },
    )
    print(f"Saved verified commands: {args.command_file}")


def evaluate(args):
    sys.path.insert(0, str(args.sdk_root.resolve()))
    check_sdk(args.sdk_root)
    if git_revision(args.drive_jepa_root) != DRIVE_JEPA_REVISION:
        raise ValueError("Drive-JEPA source revision differs")
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint checksum differs")
    official = official_val_scenes(args.sdk_root)
    selected = sorted(args.scene or official)
    if (
        len(official) != 117
        or len(set(selected)) != len(selected)
        or set(selected) - set(official)
    ):
        raise ValueError("Scenes must be unique official val UUIDs")
    args.commands, command_provenance = load_commands(args.command_file)
    identity = {
        "schema": "drive_jepa_v2_route20_v1",
        "sdk_revision": KITSCENES_SDK_REVISION,
        "dataset_revision": KITSCENES_DATA_REVISION,
        "drive_jepa_revision": DRIVE_JEPA_REVISION,
        "weights_revision": WEIGHTS_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "auto_e2e_revision": git_revision(REPO_ROOT),
        "selected_scene_ids": selected,
        "evaluator_sha256": file_sha256(Path(__file__)),
        "commands_sha256": file_sha256(args.command_file),
        "commands_lf_sha256": COMMANDS_SHA256,
        "command_provenance": command_provenance,
        "precision": "fp32",
        "input_policy": INPUT_POLICY,
        "status_policy": "causal_backward_pose_differences",
        "trajectory_policy": "ego_flu_xy_linear_interpolation_origin_no_extrapolation",
        "raw_revision_verification": "user_supplied_unverified"
        if args.raw_root
        else "archive_sha256",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != identity:
        raise ValueError("Output belongs to another run; choose a new --output")
    write_json(run_path, identity)
    adapter = DriveJEPAPerceptionAdapter(args)
    totals = {name: np.zeros(4) for name in DISPLACEMENT_HORIZON_STEPS}
    all_uids = []
    for index, scene in enumerate(selected, 1):
        path = args.output / f"{scene}.npz"
        if not path.exists():
            records, predictions, targets = [], [], []
            with raw_scene(args, scene) as dataset:
                for sample in scene_samples(dataset, scene):
                    if sample["sample_uid"] not in args.commands:
                        raise ValueError(
                            "Sample is absent from the audited 20 m commands"
                        )
                    predictions.append(
                        resample_prediction(
                            adapter.predict(sample), adapter.prediction_times_s
                        )
                    )
                    targets.append(sample["target_xy"])
                    records.append((sample["sample_uid"], sample["timestamp_ns"]))
            if not records:
                raise ValueError(f"No samples in {scene}")
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                prediction_xy=np.stack(predictions),
                target_xy=np.stack(targets),
                sample_uid=np.array([r[0] for r in records]),
                timestamp_ns=np.array([r[1] for r in records], dtype=np.int64),
            )
            temporary.replace(path)
        with np.load(path, allow_pickle=False) as data:
            for horizon, values in score_predictions(
                data["prediction_xy"], data["target_xy"]
            ).items():
                totals[horizon] += values
            all_uids.extend(data["sample_uid"].tolist())
        if len(all_uids) != len(set(all_uids)):
            raise ValueError("Duplicate sample UIDs")
        metrics = {
            f"{metric}_{horizon}_m": float(values[offset] / values[offset + 1])
            for horizon, values in totals.items()
            for metric, offset in (("ade", 0), ("fde", 2))
        }
        metrics.update(ade_5s_m=None, fde_5s_m=None)
        full = args.scene is None and index == len(selected)
        verified = (
            full
            and len(all_uids) == EXPECTED_VAL_SAMPLES
            and published_digest_ids(all_uids) == EXPECTED_VAL_SAMPLE_UID_SHA256
        )
        report = {
            **identity,
            "source_split": "val",
            "completed_scene_count": index,
            "sample_count": len(all_uids),
            "published_sample_uid_sha256": published_digest_ids(all_uids),
            "exact_sample_identity_verified": verified,
            "prediction_support_seconds": 4,
            "evaluation_role": "subset"
            if args.scene
            else ("complete" if full else "in_progress"),
            "metrics": {"trajectory": metrics},
        }
        write_json(args.output / "report.json", report)
        if full and not verified:
            raise ValueError("Full validation sample count or UID digest differs")
        print(
            f"Completed {scene}: {index}/{len(selected)} scenes, {len(all_uids)} samples",
            flush=True,
        )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--setup",
        action="store_true",
        help="Download pinned sources and the official v2 checkpoint, then exit",
    )
    action.add_argument(
        "--generate-commands",
        action="store_true",
        help="Convert exported v3.5 vector routes into 20 m commands, then exit",
    )
    parser.add_argument(
        "--navigation-root",
        type=Path,
        help="Directory containing <scene_uuid>/scene_navigation.json exports",
    )
    parser.add_argument(
        "--command-file",
        type=Path,
        default=WORKSPACE / "route_inputs_20m/commands.json",
    )
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=WORKSPACE / "route_inputs/mirror",
        help="Local cache for published v3.5 metadata shards",
    )
    parser.add_argument("--sdk-root", type=Path, default=WORKSPACE / "vendor/kitscenes")
    parser.add_argument(
        "--drive-jepa-root", type=Path, default=WORKSPACE / "vendor/Drive-JEPA"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=WORKSPACE / "weights" / CHECKPOINT_NAME
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Existing SDK data/val/<uuid> layout; otherwise stream scenes",
    )
    parser.add_argument("--staging-root", type=Path, default=WORKSPACE / "staging")
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "results/drive_jepa_v2_20m"
    )
    parser.add_argument(
        "--scene",
        action="append",
        help="Val UUID for a labelled subset; repeat to select several",
    )
    args = parser.parse_args(argv)
    if args.generate_commands and (args.navigation_root is None or args.scene):
        parser.error(
            "--generate-commands requires --navigation-root and the full validation split"
        )
    args.camera = "camera_base_front_center"
    args.precision = "fp32"
    args.dataset_revision = KITSCENES_DATA_REVISION
    return args


def main():
    args = parse_args()
    if args.setup:
        setup(args)
    elif args.generate_commands:
        generate_commands(args)
    else:
        print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
