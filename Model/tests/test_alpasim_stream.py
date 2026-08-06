"""Unit and parity tests for AlpaSim stream parser and driver plugin.

Guards tensor shapes, dtypes, ImageNet normalization, egomotion buffer formatting,
camera projection math, and driver plugin contracts between live streaming input
(PredictionInput) and offline KitScenes pre-extracted datasets.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from PIL import Image

_PLUGINS_DIR = Path(__file__).resolve().parents[1] / "plugins"
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from alpasim_driver.config import AutoE2EAlpaSimConfig  # noqa: E402
from alpasim_driver.plugin import (  # noqa: E402
    AutoE2EDriver,
    ModelPrediction,
    PredictionInput as PluginPredictionInput,
)
from data_parsing.alpasim_stream.parser import (  # noqa: E402
    CAMERA_NAMES as PARSER_CAMERA_NAMES,
    AlpasimStreamParser,
    PredictionInput,
)
try:
    from data_parsing.kit_scenes.camera import (  # noqa: E402
        CAMERA_NAMES as KITSCENES_CAMERA_NAMES,
        compute_camera_projection_matrices,
    )
except ImportError:
    KITSCENES_CAMERA_NAMES = PARSER_CAMERA_NAMES
    compute_camera_projection_matrices: Any = None  # type: ignore[no-redef]

from data_parsing.pre_extracted import (  # noqa: E402
    _VISUAL_HISTORY_DIM,
    _decode_image as _decode_pre_extracted_image,
)




class MockAutoE2EModel(torch.nn.Module):
    def forward(self, tensors):
        return {
            "trajectory_points": torch.zeros((1, 64, 2)),
            "headings": torch.zeros((1, 64))
        }

torch.serialization.add_safe_globals([MockAutoE2EModel])

@pytest.fixture
def dummy_checkpoint(tmp_path) -> str:
    ckpt_path = tmp_path / "dummy_random.ckpt"
    torch.save(MockAutoE2EModel(), ckpt_path)
    return str(ckpt_path)

@pytest.fixture
def sample_rgb_images() -> Dict[str, Image.Image]:

    """Generate 7 synthetic PIL images for KitScenes camera topology.

    Returns a mapping from KitScenes camera names to 256x256 RGB images.
    """
    images: Dict[str, Image.Image] = {}
    for idx, cam_name in enumerate(PARSER_CAMERA_NAMES):
        color = (idx * 30, (idx * 50) % 255, (255 - idx * 30) % 255)
        images[cam_name] = Image.new("RGB", (256, 256), color)
    return images


@pytest.fixture
def sample_numpy_frames() -> Dict[str, np.ndarray]:
    """Generate 7 synthetic uint8 numpy arrays for KitScenes camera topology.

    Returns a mapping from KitScenes camera names to ``(256, 256, 3)`` arrays.
    """
    frames: Dict[str, np.ndarray] = {}
    for idx, cam_name in enumerate(PARSER_CAMERA_NAMES):
        array = np.full((256, 256, 3), (idx * 35) % 256, dtype=np.uint8)
        frames[cam_name] = array
    return frames


@pytest.fixture
def sample_jpeg_bytes(sample_rgb_images: Dict[str, Image.Image]) -> Dict[str, bytes]:
    """Generate 7 synthetic JPEG byte blobs for KitScenes camera topology.

    Returns a mapping from KitScenes camera names to JPEG bytes.
    """
    encoded: Dict[str, bytes] = {}
    for cam_name, img in sample_rgb_images.items():
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded[cam_name] = buf.getvalue()
    return encoded


@pytest.fixture
def valid_prediction_input(
    sample_rgb_images: Dict[str, Image.Image],
) -> PredictionInput:
    """Return a valid happy-path dict ``PredictionInput`` payload."""
    return {
        "cameras": sample_rgb_images,
        "speed": 12.5,
        "acceleration": 0.5,
        "command": 1,
    }


@pytest.fixture
def stream_sequence_10hz(
    sample_numpy_frames: Dict[str, np.ndarray],
) -> List[PredictionInput]:
    """Generate a 70-frame sequence (7 seconds at 10 Hz) of streaming inputs.

    Simulates realistic ego motion accelerating from 0.0 to 14.0 m/s.
    """
    sequence: List[PredictionInput] = []
    for step in range(70):
        speed = float(step * 0.2)
        acceleration = 0.2
        sequence.append(
            {
                "cameras": sample_numpy_frames,
                "speed": speed,
                "acceleration": acceleration,
                "command": 1,
            }
        )
    return sequence


class TestAlpasimStreamParserFixturesAndBasicShape:
    """Verify basic shape, dtype, and input decoding of AlpasimStreamParser."""

    def test_happy_path_tensor_shapes(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify tensor shapes produced by ``parse_observation``.

        Expected shapes:
          - ``visual_tiles``: ``[1, 7, 3, 256, 256]``
          - ``egomotion_history``: ``[1, 256]``
          - ``visual_history``: ``[1, 896]``
          - ``map_context``: ``[1, 3, 256, 256]``
          - ``route_mask``: ``[1, 2, 256, 256]``
          - ``map_valid``: ``[1]``
          - ``route_valid``: ``[1]``
        """
        parser = AlpasimStreamParser()
        tensors = parser.parse_observation(valid_prediction_input)

        assert tensors["visual_tiles"].shape == (1, 7, 3, 256, 256)
        assert tensors["egomotion_history"].shape == (1, 256)
        assert tensors["visual_history"].shape == (1, _VISUAL_HISTORY_DIM)
        assert tensors["map_context"].shape == (1, 3, 256, 256)
        assert tensors["route_mask"].shape == (1, 2, 256, 256)
        assert tensors["map_valid"].shape == (1,)
        assert tensors["route_valid"].shape == (1,)

    def test_happy_path_tensor_dtypes(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify tensor data types produced by ``parse_observation``.

        Float Tensors must be ``torch.float32``; validity flags must be ``torch.bool``.
        """
        parser = AlpasimStreamParser()
        tensors = parser.parse_observation(valid_prediction_input)

        assert tensors["visual_tiles"].dtype == torch.float32
        assert tensors["egomotion_history"].dtype == torch.float32
        assert tensors["visual_history"].dtype == torch.float32
        assert tensors["map_context"].dtype == torch.float32
        assert tensors["route_mask"].dtype == torch.float32
        assert tensors["map_valid"].dtype == torch.bool
        assert tensors["route_valid"].dtype == torch.bool

    def test_input_types_pil_numpy_bytes(
        self,
        sample_rgb_images: Dict[str, Image.Image],
        sample_numpy_frames: Dict[str, np.ndarray],
        sample_jpeg_bytes: Dict[str, bytes],
    ) -> None:
        """Verify ``_decode_image`` supports PIL Image, numpy array, and JPEG bytes."""
        parser = AlpasimStreamParser()

        t1 = parser.parse_observation(
            {"cameras": sample_rgb_images, "speed": 5.0, "acceleration": 0.0, "command": 0}
        )["visual_tiles"]
        t2 = parser.parse_observation(
            {"cameras": sample_numpy_frames, "speed": 5.0, "acceleration": 0.0, "command": 0}
        )["visual_tiles"]
        t3 = parser.parse_observation(
            {"cameras": sample_jpeg_bytes, "speed": 5.0, "acceleration": 0.0, "command": 0}
        )["visual_tiles"]

        assert t1.shape == (1, 7, 3, 256, 256)
        assert t2.shape == (1, 7, 3, 256, 256)
        assert t3.shape == (1, 7, 3, 256, 256)


class TestOfflineKitScenesParity:
    """Parity assertions between AlpasimStreamParser and offline KitScenes datasets."""

    def test_image_normalization_parity(
        self, sample_jpeg_bytes: Dict[str, bytes]
    ) -> None:
        """Verify stream parser image normalization equals offline ``pre_extracted`` decode.

        Both paths run ImageNet Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
        Tolerance: bit-for-bit or ``atol=1e-5`` since floating-point ops are deterministic.
        """
        parser = AlpasimStreamParser()
        cam_key = PARSER_CAMERA_NAMES[0]
        jpeg_data = sample_jpeg_bytes[cam_key]

        live_tile = parser._decode_image(jpeg_data)
        offline_tile = _decode_pre_extracted_image(jpeg_data)

        assert torch.allclose(live_tile, offline_tile, atol=1e-5, rtol=1e-5), (
            "Live stream image decode must produce tensors identical to offline decode."
        )

    def test_image_normalization_range_mean_std(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify decoded image values follow ImageNet mean/std normalization ranges.

        Normalized values for RGB [0, 255] must lie approximately in [-2.12, 2.64].
        """
        parser = AlpasimStreamParser()
        cam_key = PARSER_CAMERA_NAMES[0]

        tile = parser._decode_image(sample_rgb_images[cam_key])
        assert tile.min() >= -2.5
        assert tile.max() <= 3.0

        # Uniform gray image (128, 128, 128) -> ToTensor = ~0.50196
        gray_img = Image.new("RGB", (256, 256), (128, 128, 128))
        gray_tile = parser._decode_image(gray_img)
        # Channel 0: (0.50196 - 0.485) / 0.229 ≈ 0.074
        assert torch.isclose(
            gray_tile[0].mean(), torch.tensor(0.074, dtype=torch.float32), atol=1e-2
        )

    def test_egomotion_history_formatting_and_dimensions(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify egomotion history layout matches offline 64-step x 4-signal format.

        Offline egomotion vector has shape ``(256,)`` (64 timesteps x 4 signals).
        Signals per timestep: ``[speed, acceleration, yaw_rate, curvature]``.
        The live parser emits ``[1, 256]``.
        """
        parser = AlpasimStreamParser()
        tensors = parser.parse_observation(valid_prediction_input)
        ego_hist = tensors["egomotion_history"]

        assert ego_hist.shape == (1, 256)
        assert ego_hist.dtype == torch.float32

        # Check last timestep in the history buffer (indices 252..255)
        last_timestep_ego = ego_hist[0, -4:]
        assert torch.isclose(
            last_timestep_ego[0], torch.tensor(12.5, dtype=torch.float32)
        )
        assert torch.isclose(
            last_timestep_ego[1], torch.tensor(0.5, dtype=torch.float32)
        )
        assert last_timestep_ego[2].item() == 0.0  # yaw_rate default
        assert last_timestep_ego[3].item() == 0.0  # curvature default

    def test_egomotion_10hz_sequence_sliding_window(
        self, stream_sequence_10hz: List[PredictionInput]
    ) -> None:
        """Verify egomotion deque buffer accumulates a 64-step sliding window at 10 Hz.

        After feeding 70 frames (7 s), the buffer must hold step 6 to step 69.
        Step 0 (speed 0.0) must be evicted.
        """
        parser = AlpasimStreamParser()

        last_tensors: Dict[str, torch.Tensor] = {}
        for observation in stream_sequence_10hz:
            last_tensors = parser.parse_observation(observation)

        ego_hist = last_tensors["egomotion_history"][0].reshape(64, 4)

        # Step 69 speed = 69 * 0.2 = 13.8 m/s
        expected_latest_speed = 69 * 0.2
        actual_latest_speed = ego_hist[-1, 0].item()
        assert torch.isclose(
            torch.tensor(actual_latest_speed),
            torch.tensor(expected_latest_speed),
            atol=1e-4,
        )

        # Earliest step in 64-step window is step 6 (6 * 0.2 = 1.2 m/s)
        expected_oldest_speed = 6 * 0.2
        actual_oldest_speed = ego_hist[0, 0].item()
        assert torch.isclose(
            torch.tensor(actual_oldest_speed),
            torch.tensor(expected_oldest_speed),
            atol=1e-4,
        )

    def test_camera_topology_parity(self) -> None:
        """Verify AlpasimStreamParser camera topology matches KitScenes camera contract."""
        assert PARSER_CAMERA_NAMES == KITSCENES_CAMERA_NAMES
        assert len(PARSER_CAMERA_NAMES) == 7



class TestEdgeCasesAndDiscrepancies:
    """Test edge cases and document implementation discrepancies found during investigation."""

    def test_edge_case_missing_camera_frame(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify behavior when a camera frame is missing from PredictionInput.

        If a camera is missing, parser inserts ``torch.zeros(3, 256, 256)``.
        """
        parser = AlpasimStreamParser()
        partial_cams = dict(sample_rgb_images)
        missing_cam = "camera_ring_rear_left"
        del partial_cams[missing_cam]

        tensors = parser.parse_observation(
            {"cameras": partial_cams, "speed": 10.0, "acceleration": 0.0, "command": 1}
        )

        missing_idx = PARSER_CAMERA_NAMES.index(missing_cam)
        missing_tile = tensors["visual_tiles"][0, missing_idx]

        assert missing_tile.shape == (3, 256, 256)
        assert missing_tile.dtype == torch.float32
        assert (missing_tile == 0.0).all(), (
            "Missing camera view must produce zero tensor as fallback."
        )

    def test_edge_case_out_of_range_ego_values(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify parser handles extreme / negative speed and acceleration values."""
        parser = AlpasimStreamParser()

        tensors = parser.parse_observation(
            {
                "cameras": sample_rgb_images,
                "speed": -15.0,
                "acceleration": 250.0,
                "command": -1,
            }
        )

        last_ego = tensors["egomotion_history"][0, -4:]
        assert last_ego[0].item() == -15.0
        assert last_ego[1].item() == 250.0

    def test_edge_case_malformed_command(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify parser behavior when command field is non-integer or unexpected type."""
        parser = AlpasimStreamParser()

        input_data: Dict[str, object] = {
            "cameras": sample_rgb_images,
            "speed": 0.0,
            "acceleration": 0.0,
            "command": None,
        }
        tensors = parser.parse_observation(input_data)  # type: ignore[arg-type]
        assert tensors["visual_tiles"].shape == (1, 7, 3, 256, 256)

    def test_config_camera_names_match_parser(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify AutoE2EAlpaSimConfig.camera_names match AlpasimStreamParser.CAMERA_NAMES.

        Passing inputs keyed by config camera names should successfully populate frames.
        """
        config = AutoE2EAlpaSimConfig(checkpoint_path='dummy_random.ckpt')
        config_cams = config.camera_names  # ['cam_front', 'cam_front_left', ...]

        assert list(config_cams) == list(PARSER_CAMERA_NAMES), (
            "Config camera names should match parser camera names."
        )

        # Build prediction input using config's camera names
        cams_with_config_keys = {
            name: img for name, img in zip(config_cams, sample_rgb_images.values())
        }

        parser = AlpasimStreamParser()
        tensors = parser.parse_observation(
            {"cameras": cams_with_config_keys, "speed": 10.0, "acceleration": 0.0, "command": 1}
        )

        # Frames should not be empty since the camera names match
        visual_tiles = tensors["visual_tiles"]
        assert not (visual_tiles == 0.0).all(), (
            "Frames should not be empty since the camera names match."
        )

    def test_camera_params_present_in_stream_parser(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify AlpasimStreamParser output dictionary contains 'camera_params'.

        It should provide dummy camera parameters matching the expected shape.
        """
        parser = AlpasimStreamParser()
        tensors = parser.parse_observation(valid_prediction_input)

        assert "camera_params" in tensors, (
            "AlpasimStreamParser should emit camera_params in output dict."
        )
        assert tensors["camera_params"].shape == (1, 7, 3, 4)
        assert tensors["camera_params"].dtype == torch.float32

    def test_package_init_exports_autoe2e_model(self) -> None:
        """Verify alpasim_driver package exports AutoE2EAlpaSimModel (aliased to AutoE2EDriver)."""
        import alpasim_driver
        from alpasim_driver import AutoE2EAlpaSimModel

        assert AutoE2EAlpaSimModel is AutoE2EDriver
        assert hasattr(alpasim_driver, "AutoE2EAlpaSimConfig")


class TestAlpasimDriverPlugin:
    """Verify AlpaSim driver plugin AutoE2EDriver interface and prediction return."""

    def test_driver_plugin_initialization(self, dummy_checkpoint: str) -> None:
        """Verify AutoE2EDriver initializes parser and device correctly."""
        driver = AutoE2EDriver(model_checkpoint=dummy_checkpoint, allow_mock=True)
        assert isinstance(driver.parser, AlpasimStreamParser)
        assert isinstance(driver.device, torch.device)

    def test_driver_plugin_predict_happy_path(
        self, sample_rgb_images: Dict[str, Image.Image],
        dummy_checkpoint: str
    ) -> None:
        """Verify AutoE2EDriver.predict accepts PluginPredictionInput and returns ModelPrediction.

        Expected output:
          - ``trajectory_points``: numpy array of shape ``(64, 2)`` and float32.
          - ``headings``: numpy array of shape ``(64,)`` and float32.
        """
        driver = AutoE2EDriver(model_checkpoint=dummy_checkpoint, allow_mock=True)
        pred_input = PluginPredictionInput(
            cameras=sample_rgb_images,
            speed=8.0,
            acceleration=0.1,
            command=1,
        )

        result = driver.predict(pred_input)

        assert isinstance(result, ModelPrediction)
        assert isinstance(result.trajectory_points, np.ndarray)
        assert isinstance(result.headings, np.ndarray)
        assert result.trajectory_points.shape == (64, 2)
        assert result.headings.shape == (64,)
        assert result.trajectory_points.dtype == np.float32
        assert result.headings.dtype == np.float32

    def test_driver_plugin_strict_mock_disallowed(self) -> None:
        """Verify that initializing with allow_mock=False fails fast when using mock dependencies."""
        from alpasim_driver.plugin import IS_MOCK_MODE
        if IS_MOCK_MODE:
            with pytest.raises(ImportError, match="allow_mock=False"):
                AutoE2EDriver(model_checkpoint="nonexistent.ckpt", allow_mock=False)
