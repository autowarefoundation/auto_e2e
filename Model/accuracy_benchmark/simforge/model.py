"""SIMFORGE-D1.5 inference on the Alpamayo 1.5 reference code.

Only this module imports torch, transformers and ``alpamayo1_5``. The prompt,
tokenisation and rollout follow the reference path: the prompt comes from
``alpamayo1_5.helper.create_message`` with camera ids and frame numbers, images
are preprocessed by the model's Qwen3-VL processor (never resized here), and
sampling uses the pinned values in ``contracts``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from accuracy_benchmark.simforge.contracts import (
    ALPAMAYO_REVISION,
    CAMERA_FRAMES,
    FUTURE_STEPS,
    MAX_GENERATION_LENGTH,
    NUM_TRAJECTORIES,
    NUM_TRAJECTORY_SETS,
    QWEN_PROCESSOR_REPO_ID,
    SEED,
    SIMFORGE_ATTN_IMPLEMENTATION,
    SIMFORGE_DTYPE,
    SIMFORGE_MODEL_FILES,
    SIMFORGE_MODEL_REPO_ID,
    SIMFORGE_MODEL_REVISION,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from accuracy_benchmark.simforge.samples import InputSample


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path, *, verify_digests: bool = True) -> dict[str, Any]:
    """Every pinned file must exist with the recorded size and (optionally) SHA-256."""
    problems: list[str] = []
    for entry in SIMFORGE_MODEL_FILES:
        path = snapshot / entry.path
        if not path.is_file():
            problems.append(f"missing {entry.path}")
            continue
        if path.stat().st_size != entry.size:
            problems.append(f"{entry.path}: size {path.stat().st_size} != {entry.size}")
            continue
        if verify_digests and file_sha256(path) != entry.sha256:
            problems.append(f"{entry.path}: sha256 differs from the pinned value")
    if problems:
        raise RuntimeError(f"snapshot {snapshot} is not the pinned {SIMFORGE_MODEL_REVISION}: " + "; ".join(problems))
    return {
        "repo_id": SIMFORGE_MODEL_REPO_ID,
        "revision": SIMFORGE_MODEL_REVISION,
        "snapshot_dir": str(snapshot),
        "files_verified": len(SIMFORGE_MODEL_FILES),
        "sha256_verified": verify_digests,
        "config_sha256": file_sha256(snapshot / "config.json"),
    }


def download_snapshot(cache_dir: Path | None, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            SIMFORGE_MODEL_REPO_ID,
            revision=SIMFORGE_MODEL_REVISION,
            cache_dir=None if cache_dir is None else str(cache_dir),
            token=token,
        )
    )


def ensure_alpamayo(source: Path) -> dict[str, Any]:
    """Import ``alpamayo1_5`` from a checkout that must be at the pinned commit."""
    src = source.resolve()
    if not (src / "alpamayo1_5" / "__init__.py").is_file():
        raise FileNotFoundError(f"{src} does not contain the alpamayo1_5 package; run the setup command")
    head = subprocess.run(
        ["git", "-C", str(src.parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if head != ALPAMAYO_REVISION:
        raise RuntimeError(f"Alpamayo checkout is at {head or 'an unknown revision'}, expected {ALPAMAYO_REVISION}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import alpamayo1_5  # noqa: F401

    return {"alpamayo_revision": ALPAMAYO_REVISION, "alpamayo_src": str(src)}


def reference_messages(helper: Any, frames: Sequence[Any], camera_indices: Any, frames_per_camera: int) -> list[dict[str, Any]]:
    """Reference prompt for per-image tensors; mixed sizes are swapped in after the text is built."""
    import torch

    tensors = list(frames)
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) == 1:
        return helper.create_message(
            frames=torch.stack(tensors), camera_indices=camera_indices, num_frames_per_camera=frames_per_camera
        )
    placeholder = torch.zeros((len(tensors), 3, 1, 1), dtype=torch.uint8)
    messages = helper.create_message(
        frames=placeholder, camera_indices=camera_indices, num_frames_per_camera=frames_per_camera
    )
    position = 0
    for item in messages[1]["content"]:
        if item.get("type") == "image":
            item["image"] = tensors[position]
            position += 1
    if position != len(tensors):
        raise RuntimeError("reference prompt image slots do not match the frames")
    return messages


class SimForgeRunner:
    """Loads the pinned model once and predicts one input sample at a time."""

    def __init__(self, *, snapshot: Path, alpamayo_src: Path, device: str, cache_dir: Path | None = None) -> None:
        self.snapshot = snapshot
        self.alpamayo_src = alpamayo_src
        self.device = device
        self.cache_dir = cache_dir
        self.model: Any = None
        self.processor: Any = None
        self._helper: Any = None
        self.provenance: dict[str, Any] = {}

    def load(self) -> None:
        import torch

        verify_snapshot(self.snapshot)
        code = ensure_alpamayo(self.alpamayo_src)
        from alpamayo1_5 import helper
        from alpamayo1_5.config import Alpamayo1_5Config
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        config_dict, _ = Alpamayo1_5Config.get_config_dict(str(self.snapshot))
        pinned_attn = config_dict.get("attn_implementation")
        if pinned_attn != SIMFORGE_ATTN_IMPLEMENTATION:
            raise RuntimeError(f"pinned config requests {pinned_attn!r}, expected {SIMFORGE_ATTN_IMPLEMENTATION!r}")
        vlm_path = config_dict.get("vlm_name_or_path")
        if vlm_path and not Path(vlm_path).is_absolute() and (self.snapshot / vlm_path).is_dir():
            config_dict["vlm_name_or_path"] = str(self.snapshot / vlm_path)
        config = Alpamayo1_5Config.from_dict(config_dict, attn_implementation=SIMFORGE_ATTN_IMPLEMENTATION)
        model = Alpamayo1_5.from_pretrained(
            str(self.snapshot),
            config=config,
            dtype=getattr(torch, SIMFORGE_DTYPE),
            attn_implementation=SIMFORGE_ATTN_IMPLEMENTATION,
        )
        self.model = model.to(self.device).eval()
        self.processor = helper.get_processor(model.tokenizer)
        self._helper = helper
        image_processor = getattr(self.processor, "image_processor", None)
        self.provenance = {
            **code,
            "processor_repo_id": QWEN_PROCESSOR_REPO_ID,
            "attention": {
                "vlm": getattr(model.vlm.config, "_attn_implementation", None),
                "expert": getattr(model.expert.config, "_attn_implementation", None),
            },
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "image_min_pixels": getattr(image_processor, "min_pixels", None),
            "image_max_pixels": getattr(image_processor, "max_pixels", None),
            "patch_size": getattr(image_processor, "patch_size", None),
            "merge_size": getattr(image_processor, "merge_size", None),
        }

    def build_inputs(self, sample: InputSample) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        if self.model is None:
            raise RuntimeError("call load() first")
        from accuracy_benchmark.simforge.samples import decode_rgb

        per_slot = [[decode_rgb(payload) for payload in slot] for slot in sample.slot_frames()]
        frames = [
            torch.from_numpy(frame).permute(2, 0, 1).contiguous() for slot in per_slot for frame in slot
        ]
        camera_indices = torch.tensor(list(sample.config.slots), dtype=torch.int64)
        messages = reference_messages(self._helper, frames, camera_indices, CAMERA_FRAMES)
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": tokenized,
            "ego_history_xyz": torch.from_numpy(sample.ego_xyz).float()[None, None],
            "ego_history_rot": torch.from_numpy(sample.ego_rot).float()[None, None],
        }
        grid = tokenized.get("image_grid_thw")
        patch = int(getattr(getattr(self.processor, "image_processor", None), "patch_size", 0) or 0)
        stats: dict[str, Any] = {
            "input_token_count": int(tokenized["input_ids"].shape[1]),
            "image_count": len(frames),
            "raw_image_hw_per_slot": [[int(slot[0].shape[0]), int(slot[0].shape[1])] for slot in per_slot],
            "image_grid_thw": None if grid is None else grid.tolist(),
            "processor_image_hw_per_image": (
                None if grid is None or not patch else [[int(r[1]) * patch, int(r[2]) * patch] for r in grid.tolist()]
            ),
        }
        return self._helper.to_device(model_inputs, self.device), stats

    def predict(self, sample: InputSample) -> dict[str, Any]:
        import time

        import torch

        if self.model is None:
            raise RuntimeError("call load() first")
        model_inputs, stats = self.build_inputs(sample)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type, dtype=getattr(torch, SIMFORGE_DTYPE)):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=TOP_P,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                num_traj_samples=NUM_TRAJECTORIES,
                num_traj_sets=NUM_TRAJECTORY_SETS,
                diffusion_kwargs=None,
                max_generation_length=MAX_GENERATION_LENGTH,
                return_extra=True,
            )
        elapsed = time.perf_counter() - started
        xyz = pred_xyz.float().cpu().numpy()
        rot = pred_rot.float().cpu().numpy()
        expected = (1, NUM_TRAJECTORY_SETS, NUM_TRAJECTORIES, FUTURE_STEPS, 3)
        if xyz.shape != expected:
            raise RuntimeError(f"unexpected trajectory shape {xyz.shape}, expected {expected}")
        texts: dict[str, list[str]] = {}
        for key, values in (extra or {}).items():
            array = np.asarray(values)
            if array.dtype.kind in {"U", "S", "O"}:
                texts[key] = [str(item) for item in array[0].reshape(-1)]
        peak = torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**20 if device_type == "cuda" else None
        return {
            "pred_xyz": xyz.reshape(NUM_TRAJECTORIES, FUTURE_STEPS, 3).astype(np.float32),
            "pred_rot": rot.reshape(NUM_TRAJECTORIES, FUTURE_STEPS, 3, 3).astype(np.float32),
            "generated_text": texts,
            "elapsed_s": round(elapsed, 3),
            "gpu_peak_allocated_mib": None if peak is None else round(float(peak), 1),
            **stats,
        }
