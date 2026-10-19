from typing import Any, Dict, TypedDict
import collections
import io
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_HISTORY_STEPS = 64
_HISTORY_SIGNALS = 4
_VISUAL_HISTORY_DIM = 896

CAMERA_NAMES = [
    "camera_base_front_center",
    "camera_ring_front",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
]

class PredictionInput(TypedDict):
    cameras: Dict[str, Any]
    speed: float
    acceleration: float
    command: int

class AlpasimStreamParser:
    """Parses live AlpaSim frames into the exact tensor format produced by pre_extracted.py."""
    def __init__(self) -> None:
        self._egomotion_buffer: collections.deque[list[float]] = collections.deque(maxlen=_HISTORY_STEPS)
        for _ in range(_HISTORY_STEPS):
            self._egomotion_buffer.append([0.0, 0.0, 0.0, 0.0])
            
    def _decode_image(self, data: Any) -> torch.Tensor:
        """Decode and normalize image exactly as the offline loader."""
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        img = Image.open(data) if isinstance(data, (str, io.BytesIO)) else data
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((256, 256), resample=Image.Resampling.BILINEAR)
        return _TRANSFORM(img)

    def parse_observation(self, observation: PredictionInput) -> Dict[str, torch.Tensor]:
        """Convert a live PredictionInput into the pipeline's expected batch tensors.
        
        Returns:
            Dict containing:
                - visual_tiles: ``[1, 7, 3, 256, 256]``
                - egomotion_history: ``[1, 256]``
                - visual_history: ``[1, 896]``
                - map_context: ``[1, 3, 256, 256]``
                - route_mask: ``[1, 2, 256, 256]``
                - map_valid: ``[1]``
                - route_valid: ``[1]``
        """
        frames = []
        for cam_name in CAMERA_NAMES:
            frame_data = observation["cameras"].get(cam_name)
            if frame_data is None:
                frames.append(torch.zeros(3, 256, 256))
            else:
                frames.append(self._decode_image(frame_data))
        visual_tiles = torch.stack(frames).unsqueeze(0)

        current_ego = [float(observation["speed"]), float(observation["acceleration"]), 0.0, 0.0]
        self._egomotion_buffer.append(current_ego)
        
        ego_history_np = np.array(self._egomotion_buffer, dtype=np.float32).flatten()
        egomotion_history = torch.from_numpy(ego_history_np).unsqueeze(0)

        visual_history = torch.zeros(1, _VISUAL_HISTORY_DIM, dtype=torch.float32)

        map_context = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        route_mask = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
        map_valid = torch.tensor([False], dtype=torch.bool)
        route_valid = torch.tensor([False], dtype=torch.bool)

        camera_params = torch.eye(4)[:3].unsqueeze(0).repeat(7, 1, 1).unsqueeze(0).to(torch.float32)

        return {
            "visual_tiles": visual_tiles,
            "egomotion_history": egomotion_history,
            "visual_history": visual_history,
            "map_context": map_context,
            "route_mask": route_mask,
            "map_valid": map_valid,
            "route_valid": route_valid,
            "camera_params": camera_params,
        }
