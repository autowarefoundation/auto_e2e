"""Strict import of the official BEVFormer V2 R50 t1 checkpoint."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


BEVFORMER_V2_T1_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1WJDxzfQMrzohLSVZDzvmTa19wfLGWT6e&export=download&confirm=t"
)
BEVFORMER_V2_T1_CHECKPOINT_SHA256 = (
    "a498acf289307f5bf47501b650e7b171"
    "fa9dfcb326430ee62055fdef7c4d3291"
)
BEVFORMER_V2_T1_CHECKPOINT_MIRROR_KEY = (
    "pretrained/bevformer-v2/"
    "r50-t1-epoch24-a498acf289307f5bf47501b650e7b171"
    "fa9dfcb326430ee62055fdef7c4d3291.pth"
)
BEVFORMER_V2_SOURCE_REPOSITORY = (
    "https://github.com/fundamentalvision/BEVFormer"
)
BEVFORMER_V2_WEIGHT_LICENSE_SPDX = "NOASSERTION"
BEVFORMER_V2_TRAINING_DATA_LICENSE_SPDX = "CC-BY-NC-SA-4.0"


def bevformer_v2_t1_checkpoint_mirror_uri(
    account_id: str,
    *,
    cluster_name: str = "auto-e2e-platform",
) -> str:
    """Resolve the account-local content-addressed checkpoint mirror."""
    if re.fullmatch(r"[0-9]{12}", account_id) is None:
        raise ValueError("AWS account ID must contain exactly 12 digits")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", cluster_name) is None:
        raise ValueError("cluster name is not valid in an S3 bucket name")
    bucket = f"{cluster_name}-checkpoints-{account_id}"
    return f"s3://{bucket}/{BEVFORMER_V2_T1_CHECKPOINT_MIRROR_KEY}"


@dataclass(frozen=True)
class BEVFormerV2InitializationReport:
    source_sha256: str
    source_url: str
    source_repository: str
    weight_license_spdx: str
    training_data_license_spdx: str
    loaded_tensor_count: int
    loaded_element_count: int
    resized_tensors: tuple[str, ...]
    adapted_tensors: tuple[str, ...]
    omitted_components: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_bev_queries(
    source: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    source_side = math.isqrt(source.shape[0])
    if source_side * source_side != source.shape[0]:
        raise ValueError("source BEV query count is not a square grid")
    channels = source.shape[1]
    image = source.reshape(source_side, source_side, channels)
    # Official: [row=Y ascending, col=X ascending].
    # AutoE2E: [row=X descending, col=Y descending].
    image = image.flip(0, 1).transpose(0, 1)
    image = image.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        image,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0).reshape(
        height * width,
        channels,
    ).to(dtype=source.dtype)


def _resize_position_embedding(
    source: torch.Tensor,
    *,
    length: int,
) -> torch.Tensor:
    values = source.flip(0).transpose(0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        values,
        size=length,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).transpose(0, 1).to(dtype=source.dtype)


def _adapt_camera_embeddings(
    source: torch.Tensor,
    *,
    num_views: int,
) -> torch.Tensor:
    # nuScenes camera indices do not identify the nuPlan/KIT camera rig.
    # Preserve the learned embedding scale without assigning a wrong camera ID.
    return source.mean(dim=0, keepdim=True).expand(num_views, -1).clone()


def _state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("BEVFormer checkpoint must contain a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state.items()
    ):
        raise ValueError("BEVFormer checkpoint state_dict is invalid")
    return state


def load_bevformer_v2_t1_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = BEVFORMER_V2_T1_CHECKPOINT_SHA256,
) -> BEVFormerV2InitializationReport:
    """Import every compatible camera-BEV tensor and report adaptations."""
    path = Path(checkpoint_path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "BEVFormer checkpoint SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    source = _state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    reactive: Any = getattr(model, "Reactive_E2E", model)
    if getattr(reactive.Backbone, "backbone_name", None) != "res_net_50":
        raise ValueError("BEVFormer V2 initialization requires res_net_50")
    feature_fusion: Any = reactive.FeatureFusion
    if getattr(feature_fusion, "architecture", None) != "bevformer_v2_t1":
        raise ValueError(
            "BEVFormer V2 initialization requires bevformer_v2_t1 fusion"
        )
    view_fusion: Any = feature_fusion.view_fusion
    target = model.state_dict()
    root = "Reactive_E2E." if hasattr(model, "Reactive_E2E") else ""
    updates: dict[str, torch.Tensor] = {}
    resized: list[str] = []
    adapted: list[str] = []

    def add(target_name: str, source_name: str, value=None) -> None:
        full_target = root + target_name
        if full_target not in target:
            raise ValueError(f"missing BEVFormer target tensor {full_target}")
        if source_name not in source:
            raise ValueError(f"missing BEVFormer source tensor {source_name}")
        candidate = source[source_name] if value is None else value
        if candidate.shape != target[full_target].shape:
            raise ValueError(
                f"BEVFormer tensor shape mismatch for {full_target}: "
                f"{tuple(candidate.shape)} != {tuple(target[full_target].shape)}"
            )
        updates[full_target] = candidate.to(dtype=target[full_target].dtype)

    backbone_prefix = root + "Backbone.backbone."
    backbone_targets = sorted(
        name for name in target if name.startswith(backbone_prefix)
    )
    if not backbone_targets:
        raise ValueError("BEVFormer target ResNet state is empty")
    for full_target in backbone_targets:
        suffix = full_target.removeprefix(backbone_prefix)
        add(
            f"Backbone.backbone.{suffix}",
            f"img_backbone.{suffix}",
        )

    for index in range(3):
        for field in ("weight", "bias"):
            add(
                (
                    "FeatureFusion.feature_pyramid."
                    f"lateral_convs.{index}.{field}"
                ),
                f"img_neck.lateral_convs.{index}.conv.{field}",
            )
    for index in range(4):
        for field in ("weight", "bias"):
            add(
                (
                    "FeatureFusion.feature_pyramid."
                    f"fpn_convs.{index}.{field}"
                ),
                f"img_neck.fpn_convs.{index}.conv.{field}",
            )

    query_source = "pts_bbox_head.bev_embedding.weight"
    query_target = "FeatureFusion.view_fusion.bev_queries.weight"
    resized_query = _resize_bev_queries(
        source[query_source],
        height=view_fusion.bev_h,
        width=view_fusion.bev_w,
    )
    add(query_target, query_source, resized_query)
    resized.append(query_target)

    for target_axis, source_axis, length in (
        ("row", "col", view_fusion.bev_h),
        ("col", "row", view_fusion.bev_w),
    ):
        source_name = (
            f"pts_bbox_head.positional_encoding.{source_axis}_embed.weight"
        )
        target_name = (
            f"FeatureFusion.view_fusion.{target_axis}_embed.weight"
        )
        add(
            target_name,
            source_name,
            _resize_position_embedding(source[source_name], length=length),
        )
        resized.append(target_name)

    add(
        "FeatureFusion.view_fusion.level_embeddings",
        "pts_bbox_head.transformer.level_embeds",
    )
    camera_source = "pts_bbox_head.transformer.cams_embeds"
    camera_target = "FeatureFusion.view_fusion.camera_embeddings"
    add(
        camera_target,
        camera_source,
        _adapt_camera_embeddings(
            source[camera_source],
            num_views=view_fusion.num_views,
        ),
    )
    adapted.append(camera_target)

    for index, _layer in enumerate(view_fusion.layers):
        target_layer = f"FeatureFusion.view_fusion.layers.{index}"
        source_layer = (
            f"pts_bbox_head.transformer.encoder.layers.{index}"
        )
        for field in (
            "sampling_offsets.weight",
            "sampling_offsets.bias",
            "attention_weights.weight",
            "attention_weights.bias",
            "value_proj.weight",
            "value_proj.bias",
            "output_proj.weight",
            "output_proj.bias",
        ):
            add(
                f"{target_layer}.self_attention.{field}",
                f"{source_layer}.attentions.0.{field}",
            )
            cross_source = (
                f"{source_layer}.attentions.1.output_proj"
                if field.startswith("output_proj.")
                else (
                    f"{source_layer}.attentions.1."
                    f"deformable_attention.{field.rsplit('.', 1)[0]}"
                )
            )
            if field.startswith("output_proj."):
                cross_source_name = (
                    f"{cross_source}.{field.rsplit('.', 1)[1]}"
                )
            else:
                cross_source_name = (
                    f"{cross_source}.{field.rsplit('.', 1)[1]}"
                )
            add(
                f"{target_layer}.cross_attention.{field}",
                cross_source_name,
            )
        for target_slot, source_slot in ((0, "0.0"), (3, "1")):
            for field in ("weight", "bias"):
                add(
                    f"{target_layer}.ffn.{target_slot}.{field}",
                    (
                        f"{source_layer}.ffns.0.layers."
                        f"{source_slot}.{field}"
                    ),
                )
        for norm_index in range(3):
            for field in ("weight", "bias"):
                add(
                    (
                        f"{target_layer}.norms."
                        f"{norm_index}.{field}"
                    ),
                    (
                        f"{source_layer}.norms."
                        f"{norm_index}.{field}"
                    ),
                )

    incompatible = model.load_state_dict(updates, strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(
            f"unexpected BEVFormer initialization targets: {unexpected}"
        )
    camera_parameters = {
        root + f"Backbone.{name}"
        for name, _parameter in reactive.Backbone.named_parameters()
    } | {
        root + f"FeatureFusion.{name}"
        for name, _parameter in feature_fusion.named_parameters()
    }
    intentionally_new = {
        root + "FeatureFusion.view_fusion.pseudo_projection",
    }
    uninitialized = camera_parameters - set(updates) - intentionally_new
    if uninitialized:
        raise ValueError(
            "BEVFormer camera parameters lack initialization coverage: "
            f"{sorted(uninitialized)}"
        )
    return BEVFormerV2InitializationReport(
        source_sha256=actual_sha256,
        source_url=BEVFORMER_V2_T1_CHECKPOINT_URL,
        source_repository=BEVFORMER_V2_SOURCE_REPOSITORY,
        weight_license_spdx=BEVFORMER_V2_WEIGHT_LICENSE_SPDX,
        training_data_license_spdx=(
            BEVFORMER_V2_TRAINING_DATA_LICENSE_SPDX
        ),
        loaded_tensor_count=len(updates),
        loaded_element_count=sum(value.numel() for value in updates.values()),
        resized_tensors=tuple(resized),
        adapted_tensors=tuple(adapted),
        omitted_components=("detector_and_perspective_heads",),
    )
