"""Pinned contract for the SIMFORGE-D1.5 KITScenes evaluation (issue #210).

Every constant here was checked against the Alpamayo 1.5 reference code
(revision ``ALPAMAYO_REVISION``) and the pinned SIMFORGE-D1.5 checkpoint.
Changing any of them changes the benchmark, so they are not CLI options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_parsing.kit_scenes.source import KITSCENES_DATA_REVISION, KITSCENES_REPO_ID, KITSCENES_SDK_REVISION

__all__ = [
    "ALPAMAYO_REPO_URL",
    "ALPAMAYO_REVISION",
    "BENCHMARK_LABEL",
    "CAMERA_CONFIGS",
    "CameraConfig",
    "CameraView",
    "EXPERIMENTAL_LABEL",
    "KITSCENES_DATA_REVISION",
    "KITSCENES_REPO_ID",
    "KITSCENES_SDK_REVISION",
    "SIMFORGE_MODEL_FILES",
    "SIMFORGE_MODEL_REPO_ID",
    "SIMFORGE_MODEL_REVISION",
    "PinnedFile",
    "camera_config",
    "model_contract",
]

# ---------------------------------------------------------------------------
# Model and reference code
# ---------------------------------------------------------------------------

SIMFORGE_MODEL_REPO_ID = "SIMFORGE-AI/SIMFORGE-D1.5"
SIMFORGE_MODEL_REVISION = "ec31596b99d05f0a63904ce624821652d1f9e047"
SIMFORGE_ATTN_IMPLEMENTATION = "flash_attention_2"  # VLM; the expert runs SDPA (pinned config)
SIMFORGE_DTYPE = "bfloat16"
QWEN_PROCESSOR_REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"

ALPAMAYO_REPO_URL = "https://github.com/NVlabs/alpamayo1.5.git"
ALPAMAYO_REVISION = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"


@dataclass(frozen=True)
class PinnedFile:
    path: str
    sha256: str
    size: int


# Every file of the pinned checkpoint (weight shards included). Verified before
# any run, so a silently updated or partially downloaded snapshot cannot be used.
SIMFORGE_MODEL_FILES: tuple[PinnedFile, ...] = (
    PinnedFile("LICENSE", "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df", 2619),
    PinnedFile("NOTICE", "76f08d1da8ebe90e27a3d87f9baf083a5a69b071c4f76ad96f557d29f090eb45", 1327),
    PinnedFile("README.md", "c371c4b563c5509829689244890231adbf837eadd920d4feab1a04aa2ad12a16", 6448),
    PinnedFile("config.json", "649e4a8821d7a3b438e042192b5750c208b8ae633ffd6bfda955b011f2076c48", 3515),
    PinnedFile("model-00001-of-00003.safetensors", "cc1621e05fd349172bff6e4f8c5db7e61bd0d9a18aafdb612ea2491a4415aa75", 4992544600),
    PinnedFile("model-00002-of-00003.safetensors", "9d8003dde045ae2ba823f089c75647f2fa2dfb00c6a5c430f736a113f3faa421", 4991133080),
    PinnedFile("model-00003-of-00003.safetensors", "7e396b564bab9ac05577911e92c7ab85a7caa78f212c02fcdb308f904f4a5374", 4445722516),
    PinnedFile("model.safetensors.index.json", "01800587f8cc4f0b59a49db0a77fa179e63b69ef805655850557ad865b3141c7", 96625),
    PinnedFile("vlm/chat_template.json", "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce", 5499),
    PinnedFile("vlm/config.json", "769c3565dd94c824391ca69c088123e9f1720462dde9b593edacafd3345863d4", 1492),
    PinnedFile("vlm/merges.txt", "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3", 1671839),
    PinnedFile("vlm/preprocessor_config.json", "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", 390),
    PinnedFile("vlm/tokenizer.json", "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7", 7032403),
    PinnedFile("vlm/tokenizer_config.json", "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399", 10868),
    PinnedFile("vlm/video_preprocessor_config.json", "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", 385),
    PinnedFile("vlm/vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2776833),
)

# ---------------------------------------------------------------------------
# Temporal contract (10 Hz reference timeline)
# ---------------------------------------------------------------------------

TIME_STEP_S = 0.1
EGO_HISTORY_STEPS = 16  # t0-1.5 .. t0
CAMERA_FRAMES = 4  # t0-0.3 .. t0
FUTURE_STEPS = 64  # t0+0.1 .. t0+6.4
SCORING_HORIZONS_S: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
# Strict: a deviation equal to the tolerance fails. The 10 Hz grid deviates by
# at most 0.05 ms on the pinned val scenes, far below the 50 ms half interval.
TIMESTAMP_TOLERANCE_NS = 10_000_000

# Population rule of issue #210: frames 40 + 90k with >= 30 genuine future frames.
POPULATION_START_FRAME = 40
POPULATION_STRIDE_FRAMES = 90
POPULATION_MIN_FUTURE_STEPS = 30
POPULATION_ID = "issue210-kitscenes-val-anchor40-stride90-min3s"
EXPECTED_SCENE_COUNT = 117
EXPECTED_ANCHOR_COUNT = 202
EXPECTED_HORIZON_COUNTS: dict[str, int] = {"1s": 202, "2s": 202, "3s": 202, "4s": 200, "5s": 186}

# Sampling of the Alpamayo rollout (validated values; not configurable).
NUM_TRAJECTORIES = 6
NUM_TRAJECTORY_SETS = 1
TOP_P = 0.98
TOP_K = None
TEMPERATURE = 0.6
SEED = 42
MAX_GENERATION_LENGTH = 256

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

# Native SimForge telephoto slots. They are 30 degree cameras; KITScenes cameras
# are 63-88 degree wide-angle sensors, so only a calibrated virtual crop may
# feed them.
TELEPHOTO_SLOTS: frozenset[int] = frozenset({4, 6})
VIRTUAL_TELE_HFOV_DEG = 30.0
VIRTUAL_TELE_ASPECT_RATIO = 3504 / 2272  # the ring camera aspect ratio


@dataclass(frozen=True)
class CameraView:
    """One SimForge slot: a KITScenes camera, natively or as a calibrated virtual crop."""

    kit_camera: str
    slot: int
    virtual_tele: bool = False


@dataclass(frozen=True)
class CameraConfig:
    name: str
    description: str
    views: tuple[CameraView, ...]
    benchmark: bool

    def __post_init__(self) -> None:
        slots = [view.slot for view in self.views]
        if slots != sorted(set(slots)):
            raise ValueError(f"{self.name}: slots must be unique and increasing: {slots}")
        for view in self.views:
            if view.virtual_tele != (view.slot in TELEPHOTO_SLOTS):
                raise ValueError(f"{self.name}: slot {view.slot} must be virtual_tele exactly on slots 4 and 6")
        if self.benchmark and any(view.virtual_tele for view in self.views):
            raise ValueError(f"{self.name}: a benchmark configuration cannot contain virtual views")

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(view.slot for view in self.views)

    @property
    def kit_cameras(self) -> tuple[str, ...]:
        return tuple(view.kit_camera for view in self.views)

    @property
    def run_label(self) -> str:
        return BENCHMARK_LABEL if self.benchmark else EXPERIMENTAL_LABEL

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "benchmark": self.benchmark,
            "views": [
                {
                    "simforge_slot": view.slot,
                    "kit_camera": view.kit_camera,
                    "kind": "virtual_telephoto_crop" if view.virtual_tele else "native",
                }
                for view in self.views
            ],
        }


BENCHMARK_LABEL = "issue210-kitscenes-val-simforge-d1.5-benchmark"
EXPERIMENTAL_LABEL = "experimental 7Cam-VT adapted-camera ablation (not a benchmark result)"

CAMERA_CONFIGS: dict[str, CameraConfig] = {
    "3cam": CameraConfig(
        name="3cam",
        description="common-comparison 3-camera benchmark: front-left, front, front-right",
        views=(
            CameraView("camera_ring_front_left", 0),
            CameraView("camera_ring_front", 1),
            CameraView("camera_ring_front_right", 2),
        ),
        benchmark=True,
    ),
    "7cam-vt": CameraConfig(
        name="7cam-vt",
        description=(
            "experimental 7Cam-VT adapted-camera ablation: six ring cameras plus calibrated 30 degree "
            "virtual telephoto crops (slot 4 from camera_ring_rear, slot 6 from camera_base_front_center); "
            "not an exact reproduction of SimForge's physical telephoto sensors"
        ),
        views=(
            CameraView("camera_ring_front_left", 0),
            CameraView("camera_ring_front", 1),
            CameraView("camera_ring_front_right", 2),
            CameraView("camera_ring_rear_left", 3),
            CameraView("camera_ring_rear", 4, virtual_tele=True),
            CameraView("camera_ring_rear_right", 5),
            CameraView("camera_base_front_center", 6, virtual_tele=True),
        ),
        benchmark=False,
    ),
}


def camera_config(name: str) -> CameraConfig:
    try:
        return CAMERA_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"unknown camera config {name!r}; expected one of {sorted(CAMERA_CONFIGS)}") from error


def model_contract() -> dict[str, Any]:
    """Recorded verbatim in every run manifest."""
    return {
        "model_repo_id": SIMFORGE_MODEL_REPO_ID,
        "model_revision": SIMFORGE_MODEL_REVISION,
        "alpamayo_repo_url": ALPAMAYO_REPO_URL,
        "alpamayo_revision": ALPAMAYO_REVISION,
        "qwen_processor_repo_id": QWEN_PROCESSOR_REPO_ID,
        "attn_implementation_vlm": SIMFORGE_ATTN_IMPLEMENTATION,
        "attn_implementation_expert": "sdpa",
        "dtype": SIMFORGE_DTYPE,
        "num_trajectories": NUM_TRAJECTORIES,
        "num_trajectory_sets": NUM_TRAJECTORY_SETS,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_generation_length": MAX_GENERATION_LENGTH,
    }
