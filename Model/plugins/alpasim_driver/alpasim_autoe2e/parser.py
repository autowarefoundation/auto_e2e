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

# taken from calib.json
_KIT_SCENES_PROJECTION_MATRICES = {
    "camera_base_front_center": [[128.755274, -131.199077, 1.006187, -52.390168], [127.524094, -0.908146, -249.183598, -149.383547], [0.999974, 0.007067, 0.000368, -0.421340]],
    "camera_ring_front": [[129.386719, -132.323571, 0.863261, -25.981988], [128.731816, 1.228626, -207.188816, -64.536050], [0.999854, 0.016975, 0.001910, -0.207052]],
    "camera_ring_front_left": [[180.400433, 47.059563, -0.395239, -26.819215], [62.003890, 109.554406, -208.719258, -64.554699], [0.486470, 0.873665, -0.007472, -0.209015]],
    "camera_ring_front_right": [[-48.124751, -179.391707, 0.367795, -26.278596], [66.340567, -109.558404, -207.977877, -64.484149], [0.522690, -0.852522, -0.000715, -0.206905]],
    "camera_ring_rear": [[-131.283609, 132.521569, -0.505595, -26.582819], [-128.653595, -3.802507, -207.379122, -64.645621], [-0.999840, -0.017437, 0.003915, -0.203476]],
    "camera_ring_rear_left": [[48.199949, 180.302700, -1.179315, -26.830027], [-65.506368, 107.897468, -208.745168, -64.714219], [-0.518128, 0.855260, -0.008521, -0.205967]],
    "camera_ring_rear_right": [[-179.652773, -48.899689, 1.256024, -25.914482], [-63.016182, -113.192575, -207.988609, -64.629469], [-0.474510, -0.880249, 0.000901, -0.203962]]
}
class PredictionInput(TypedDict):
    cameras: Dict[str, Any]
    speed: float
    acceleration: float
    command: int
    yaw_rate: float
    curvature: float

class AlpasimStreamParser:
    """Parses live AlpaSim frames into the exact tensor format produced by pre_extracted.py."""
    def __init__(self, camera_names: list[str]) -> None:
        self.camera_names = camera_names
        self._egomotion_buffer: collections.deque[list[float]] = collections.deque(maxlen=_HISTORY_STEPS)
        for _ in range(_HISTORY_STEPS):
            self._egomotion_buffer.append([0.0] * _HISTORY_SIGNALS)
            
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
        for cam_name in self.camera_names:
            frame_data = observation["cameras"].get(cam_name)
            if frame_data is None:
                frames.append(torch.zeros(3, 256, 256))
            else:
                frames.append(self._decode_image(frame_data))
        visual_tiles = torch.stack(frames).unsqueeze(0)

        current_ego = [0.0] * _HISTORY_SIGNALS
        current_ego[0] = float(observation["speed"])
        current_ego[1] = float(observation["acceleration"])
        current_ego[2] = float(observation.get("yaw_rate", 0.0))
        current_ego[3] = float(observation.get("curvature", 0.0))
        self._egomotion_buffer.append(current_ego)
        
        ego_history_np = np.array(self._egomotion_buffer, dtype=np.float32).flatten()
        egomotion_history = torch.from_numpy(ego_history_np).unsqueeze(0)

        visual_history = torch.zeros(1, _VISUAL_HISTORY_DIM, dtype=torch.float32)

        map_context = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        route_mask = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
        map_valid = torch.tensor([False], dtype=torch.bool)
        route_valid = torch.tensor([False], dtype=torch.bool)

        matrices = [_KIT_SCENES_PROJECTION_MATRICES[name] for name in self.camera_names]
        camera_params = torch.tensor(matrices, dtype=torch.float32).unsqueeze(0)

        return {
            "camera_tiles": visual_tiles,
            "egomotion_history": egomotion_history,
            "visual_history": visual_history,
            "map_context": map_context,
            "route_mask": route_mask,
            "map_valid": map_valid,
            "route_valid": route_valid,
            "camera_params": camera_params,
        }
