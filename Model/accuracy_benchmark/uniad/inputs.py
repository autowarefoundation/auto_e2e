# SPDX-License-Identifier: Apache-2.0
"""Original JPEG loader and unchanged UniAD model-input/state-reset boundary."""

import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
from .protocol import normalize_and_pad


def input_kwargs(frame, scene_id, box_class):
    """Same tensors/metadata as the evaluated runner; CUDA imports are deferred."""
    import torch

    arrays = load_raw_kitscenes_frame(frame)
    padded, original_shapes = normalize_and_pad(
        arrays["images_rgb"], arrays["image_shapes"]
    )
    padded_shape = tuple(padded.shape[1:])
    image = torch.from_numpy(padded.transpose(0, 3, 1, 2).copy())[None].cuda()
    metas = dict(
        scene_token=scene_id,
        sample_idx=f"{scene_id}:{frame['frame']}",
        can_bus=arrays["can_bus"].copy(),
        lidar2img=arrays["lidar2img"].copy(),
        l2g_r_mat=arrays["l2g_r_mat"].copy(),
        l2g_t=arrays["l2g_t"].copy(),
        pts_filename="camera_only_no_lidar.bin",
        img_shape=[padded_shape] * 6,
        ori_shape=original_shapes,
        pad_shape=[padded_shape] * 6,
        box_type_3d=box_class,
        scale_factor=1.0,
        flip=False,
    )
    # These stock auxiliary-evaluation placeholders contain no GT maps/future.
    # Their collision/IoU metrics are not reported as benchmark measurements.
    return dict(
        img=[image],
        img_metas=[[metas]],
        l2g_t=torch.as_tensor(arrays["l2g_t"], dtype=torch.float32, device="cuda"),
        l2g_r_mat=torch.as_tensor(
            arrays["l2g_r_mat"], dtype=torch.float32, device="cuda"
        ),
        timestamp=[
            torch.tensor(frame["timestamp"], dtype=torch.float64, device="cuda")
        ],
        command=torch.tensor([2], dtype=torch.long, device="cuda"),
        gt_lane_labels=[torch.tensor([[0, 1, 2, 3]], device="cuda")],
        gt_lane_masks=[torch.zeros((1, 4, 200, 200), device="cuda")],
        gt_segmentation=[torch.zeros((1, 7, 200, 200), device="cuda")],
        gt_instance=[torch.zeros((1, 7, 200, 200), device="cuda")],
        gt_occ_img_is_valid=[torch.ones((1, 9), dtype=torch.bool, device="cuda")],
    )


def load_raw_kitscenes_frame(frame):
    records = frame["raw_kitscenes_images"]
    shapes = np.asarray(frame["image_shapes"])
    if (
        len(records) != 6
        or shapes.shape != (6, 2)
        or not np.issubdtype(shapes.dtype, np.integer)
        or np.any(shapes <= 0)
        or np.any(shapes > 8192)
    ):
        raise ValueError("Expected six bounded native image extents")
    canvas = np.zeros(
        (6, int(shapes[:, 0].max()), int(shapes[:, 1].max()), 3), dtype=np.uint8
    )
    for i, (record, (height, width)) in enumerate(zip(records, shapes)):
        path = Path(record["path"])
        if path.is_symlink() or not 0 < path.stat().st_size <= 64 * 2**20:
            raise ValueError("Invalid original image file")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("Original image checksum changed")
        with Image.open(io.BytesIO(blob)) as opened:
            if opened.size != (width, height):
                raise ValueError("Original image dimensions changed")
            canvas[i, :height, :width] = np.asarray(opened.convert("RGB"))
    arrays = dict(images_rgb=canvas, image_shapes=shapes.astype(np.int64))
    for name, shape in (
        ("lidar2img", (6, 4, 4)),
        ("can_bus", (18,)),
        ("l2g_r_mat", (3, 3)),
        ("l2g_t", (3,)),
    ):
        value = np.asarray(frame[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid frame metadata: " + name)
        arrays[name] = value
    return arrays


def reset_scene(model, first_meta):
    # Stock forward_test updates scene_token before its first-frame check.
    # Initialize its deltas explicitly and discard every cross-scene tracking state.
    model.prev_frame_info = dict(
        prev_bev=None,
        scene_token=None,
        prev_pos=first_meta["can_bus"][:3].copy(),
        prev_angle=float(first_meta["can_bus"][-1]),
    )
    model.test_track_instances = None
    model.prev_bev = None
    model.scene_token = None
    model.timestamp = None
    model.l2g_r_mat = None
    model.l2g_t = None
    model.track_base.clear()
