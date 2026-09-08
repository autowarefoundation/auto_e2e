from collections import deque
from dataclasses import dataclass
from operator import index
from typing import SupportsIndex, cast

import torch
import torch.nn as nn
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
)
from .auxiliary_heads import (
    BEVSegmentationHead,
    RouteReconstructionHead,
)
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .fused_feature_pooling import FusedFeaturePooling
from .trajectory_planning import build_planner
from .map_encoder import (
    build_map_bev_fusion,
    build_split_navigation_encoder,
)
from .temporal_memory import build_temporal_memory
from .reasoning.horizon_reasoning_head import HorizonReasoningHead


@dataclass(frozen=True)
class _PreparedCameraFPNFrame:
    pyramid: tuple[torch.Tensor, ...]
    timestamps_us: tuple[int, ...]
    expected_history_timestamps_us: tuple[tuple[int, ...], ...]
    signature: tuple[
        tuple[tuple[int, ...], torch.dtype, torch.device],
        ...,
    ]
    stream_ids: tuple[str, ...]
    front_companion_mode: bool
    needs_prime: bool


class _CameraFPNCacheOwnerState:
    def __init__(self) -> None:
        self.token = object()

    def invalidate(self) -> None:
        self.token = object()


class _CameraFPNCacheLoadHook:
    def __init__(self, owner_state: _CameraFPNCacheOwnerState) -> None:
        self.owner_state = owner_state

    def __call__(
        self,
        _module: nn.Module,
        _incompatible_keys,
    ) -> None:
        self.owner_state.invalidate()


class StatefulCameraFPNCache:
    """Inference FIFO that tracks scene and exact T8 frame continuity.

    One inference stream owns each cache. Concurrent forwards must use
    separate cache instances. Batched streams advance in lockstep; one lane
    discontinuity invalidates and re-primes the complete batch.
    Timestamps must be source timestamps sampled exactly at frame_interval_us.
    Callers must not synthesize continuity across dropped source frames.
    """

    def __init__(
        self,
        history_frames: int = 7,
        *,
        frame_interval_us: int = REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
        owner_token: object,
    ) -> None:
        if history_frames <= 0 or frame_interval_us <= 0:
            raise ValueError("camera FPN cache timing must be positive")
        if owner_token is None:
            raise ValueError("camera FPN cache owner token is required")
        self.history_frames = int(history_frames)
        self.frame_interval_us = int(frame_interval_us)
        self._owner_token = owner_token
        self._frames: deque[tuple[torch.Tensor, ...]] = deque(
            maxlen=self.history_frames
        )
        self._frame_timestamps_us: deque[tuple[int, ...]] = deque(
            maxlen=self.history_frames
        )
        self._batch_size: int | None = None
        self._num_views: int | None = None
        self._stream_ids: tuple[str, ...] | None = None
        self._front_companion_mode: bool | None = None
        self._last_committed_timestamps_us: tuple[int, ...] | None = None
        self._signature: tuple[
            tuple[tuple[int, ...], torch.dtype, torch.device],
            ...,
        ] | None = None

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(self._frames)

    @property
    def frame_timestamps_us(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._frame_timestamps_us)

    @property
    def is_full(self) -> bool:
        return (
            len(self._frames) == self.history_frames
            and len(self._frame_timestamps_us) == self.history_frames
        )

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def num_views(self) -> int | None:
        return self._num_views

    @property
    def stream_ids(self) -> tuple[str, ...] | None:
        return self._stream_ids

    @property
    def last_committed_timestamps_us(self) -> tuple[int, ...] | None:
        return self._last_committed_timestamps_us

    def _clear_frames(self) -> None:
        self._frames.clear()
        self._frame_timestamps_us.clear()
        self._batch_size = None
        self._num_views = None
        self._front_companion_mode = None
        self._signature = None

    def reset(self) -> None:
        self._clear_frames()
        self._stream_ids = None
        self._last_committed_timestamps_us = None

    def _validate_owner(self, owner_token: object | None) -> None:
        if self._owner_token is not owner_token:
            raise ValueError(
                "camera FPN cache belongs to a different ReactiveE2E instance"
            )

    def bind_stream(
        self,
        stream_ids,
        *,
        batch_size: int,
    ) -> bool:
        """Bind to ordered batch streams, resetting on any identity change."""
        if batch_size <= 0:
            raise ValueError("camera FPN cache batch size must be positive")
        if isinstance(stream_ids, str):
            normalized = (stream_ids,)
        else:
            try:
                normalized = tuple(stream_ids)
            except TypeError as exc:
                raise TypeError(
                    "camera FPN cache stream IDs must be a sequence"
                ) from exc
        if (
            len(normalized) != batch_size
            or any(
                not isinstance(value, str) or not value
                for value in normalized
            )
        ):
            raise ValueError(
                "camera FPN cache requires one non-empty stream ID per sample"
            )
        if self._stream_ids == normalized:
            return False
        self.reset()
        self._stream_ids = normalized
        return True

    @staticmethod
    def _normalize_timestamps(
        timestamps_us,
        *,
        batch_size: int,
    ) -> tuple[int, ...]:
        if isinstance(timestamps_us, torch.Tensor):
            if timestamps_us.ndim == 0:
                candidates = (timestamps_us.item(),)
            else:
                candidates = tuple(
                    timestamps_us.detach().cpu().reshape(-1).tolist()
                )
        elif isinstance(timestamps_us, int) and not isinstance(
            timestamps_us,
            bool,
        ):
            candidates = (timestamps_us,)
        else:
            try:
                candidates = tuple(timestamps_us)
            except TypeError as exc:
                raise TypeError(
                    "camera FPN timestamps must be an integer sequence"
                ) from exc
        if len(candidates) != batch_size:
            raise ValueError(
                "camera FPN cache requires one timestamp per sample"
            )
        normalized = []
        for value in candidates:
            if isinstance(value, bool):
                raise ValueError("camera FPN timestamps must be integers")
            try:
                normalized.append(index(cast(SupportsIndex, value)))
            except TypeError as exc:
                raise ValueError(
                    "camera FPN timestamps must be integers"
                ) from exc
        return tuple(normalized)

    @staticmethod
    def _normalize(
        pyramid,
        *,
        batch_size: int,
        num_views: int,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[tuple[tuple[int, ...], torch.dtype, torch.device], ...],
    ]:
        if batch_size <= 0 or num_views <= 0:
            raise ValueError("camera FPN cache dimensions must be positive")
        normalized = tuple(feature.detach() for feature in pyramid)
        if len(normalized) != 4:
            raise ValueError("camera FPN cache requires four feature levels")
        expected_items = batch_size * num_views
        if any(
            feature.ndim != 4 or feature.shape[0] != expected_items
            for feature in normalized
        ):
            raise ValueError(
                "camera FPN cache feature shape differs from stream contract"
            )
        signature = tuple(
            (
                tuple(int(value) for value in feature.shape[1:]),
                feature.dtype,
                feature.device,
            )
            for feature in normalized
        )
        return normalized, signature

    def _set_or_validate_signature(
        self,
        *,
        batch_size: int,
        num_views: int,
        signature,
    ) -> None:
        if self._stream_ids is None:
            raise RuntimeError(
                "camera FPN cache must be bound to a stream before use"
            )
        if self._signature is None:
            self._batch_size = batch_size
            self._num_views = num_views
            self._signature = signature
            return
        if (
            self._batch_size != batch_size
            or self._num_views != num_views
            or self._signature != signature
        ):
            raise ValueError(
                "camera FPN cache cannot mix stream dimensions, dtype, or device"
            )

    def _expected_history_timestamps(
        self,
        current_timestamps_us: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                current_timestamp
                + frame_offset * self.frame_interval_us
                for current_timestamp in current_timestamps_us
            )
            for frame_offset in range(-self.history_frames, 0)
        )

    def _prepare_frame(
        self,
        pyramid,
        *,
        stream_ids,
        timestamps_us,
        batch_size: int,
        num_views: int,
        front_companion_mode: bool,
        owner_token: object,
    ) -> _PreparedCameraFPNFrame:
        """Validate continuity before any cached history is consumed."""
        self._validate_owner(owner_token)
        if not isinstance(front_companion_mode, bool):
            raise TypeError(
                "camera FPN cache Front companion mode must be boolean"
            )
        self.bind_stream(stream_ids, batch_size=batch_size)
        normalized_timestamps = self._normalize_timestamps(
            timestamps_us,
            batch_size=batch_size,
        )
        if (
            self._last_committed_timestamps_us is not None
            and any(
                current <= committed
                for current, committed in zip(
                    normalized_timestamps,
                    self._last_committed_timestamps_us,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "camera FPN cache requires strictly newer timestamps"
            )
        normalized_pyramid, signature = self._normalize(
            pyramid,
            batch_size=batch_size,
            num_views=num_views,
        )
        expected_history_timestamps = self._expected_history_timestamps(
            normalized_timestamps
        )
        if self._signature is not None and (
            self._batch_size != batch_size
            or self._num_views != num_views
            or self._signature != signature
            or self._front_companion_mode != front_companion_mode
        ):
            self._clear_frames()
        if (
            len(self._frames) != len(self._frame_timestamps_us)
            or (
                self._frames
                and (
                    not self.is_full
                    or self.frame_timestamps_us
                    != expected_history_timestamps
                )
            )
        ):
            self._clear_frames()
        if self._stream_ids is None:
            raise RuntimeError("camera FPN cache lost its stream binding")
        return _PreparedCameraFPNFrame(
            pyramid=normalized_pyramid,
            timestamps_us=normalized_timestamps,
            expected_history_timestamps_us=expected_history_timestamps,
            signature=signature,
            stream_ids=self._stream_ids,
            front_companion_mode=front_companion_mode,
            needs_prime=not self.is_full,
        )

    def _replace_history(
        self,
        pyramids,
        *,
        prepared: _PreparedCameraFPNFrame,
        batch_size: int,
        num_views: int,
    ) -> None:
        candidates = tuple(pyramids)
        timestamp_candidates = prepared.expected_history_timestamps_us
        if len(candidates) != self.history_frames:
            raise ValueError(
                "camera FPN cache prime requires a complete history"
            )
        if (
            self._stream_ids != prepared.stream_ids
            or (
                self._last_committed_timestamps_us is not None
                and any(
                    current <= committed
                    for current, committed in zip(
                        prepared.timestamps_us,
                        self._last_committed_timestamps_us,
                        strict=True,
                    )
                )
            )
        ):
            self._clear_frames()
            raise RuntimeError(
                "camera FPN cache changed before history prime"
            )
        normalized = []
        normalized_timestamps = []
        signature = None
        for pyramid, frame_timestamps in zip(
            candidates,
            timestamp_candidates,
            strict=True,
        ):
            frame, frame_signature = self._normalize(
                pyramid,
                batch_size=batch_size,
                num_views=num_views,
            )
            normalized_timestamps.append(self._normalize_timestamps(
                frame_timestamps,
                batch_size=batch_size,
            ))
            if signature is None:
                signature = frame_signature
            elif signature != frame_signature:
                raise ValueError(
                    "camera FPN cache frames have inconsistent feature shapes"
                )
            normalized.append(frame)
        if signature != prepared.signature:
            raise ValueError(
                "camera FPN history signature differs from current frame"
            )
        self._clear_frames()
        self._batch_size = batch_size
        self._num_views = num_views
        self._front_companion_mode = prepared.front_companion_mode
        self._signature = signature
        self._frames.extend(normalized)
        self._frame_timestamps_us.extend(normalized_timestamps)

    def _commit_frame(
        self,
        prepared: _PreparedCameraFPNFrame,
        *,
        batch_size: int,
        num_views: int,
    ) -> None:
        if (
            self._stream_ids != prepared.stream_ids
            or (
                self._last_committed_timestamps_us is not None
                and any(
                    current <= committed
                    for current, committed in zip(
                        prepared.timestamps_us,
                        self._last_committed_timestamps_us,
                        strict=True,
                    )
                )
            )
            or not self.is_full
            or self._front_companion_mode
            != prepared.front_companion_mode
            or self.frame_timestamps_us
            != prepared.expected_history_timestamps_us
        ):
            self._clear_frames()
            raise RuntimeError(
                "camera FPN cache changed before frame commit"
            )
        self._set_or_validate_signature(
            batch_size=batch_size,
            num_views=num_views,
            signature=prepared.signature,
        )
        self._frames.append(prepared.pyramid)
        self._frame_timestamps_us.append(prepared.timestamps_us)
        self._last_committed_timestamps_us = prepared.timestamps_us


class ReactiveE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                 is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, 
                 egomotion_dim=256,
                 visual_history_dim=896,
                 map_type="rasterized", map_context_channels=3,
                 route_channels=2, enable_route_conditioning=True,
                 route_encoder_hidden_channels=96,
                 map_fusion_mode="residual", map_fusion_kwargs=None,
                 temporal_memory_mode="no_memory", temporal_memory_kwargs=None,
                 planner_mode="gru", planner_kwargs=None,
                 enable_reasoning=False, reasoning_mode="none",
                 reasoning_kwargs=None,
                 enable_bev_segmentation=False,
                 bev_segmentation_classes=8,
                 enable_route_reconstruction=False,
                 auxiliary_output_size=None):
        super(ReactiveE2E, self).__init__()

        # Camera backbone feature extractor
        camera_architecture = str(
            (view_fusion_kwargs or {}).get("architecture", "legacy")
        )
        self.camera_input_size: int | None = None
        self.front_camera_index: int | None = None
        self.front_camera_input_size: int | None = None
        self.camera_architecture = camera_architecture
        if camera_architecture in {"bevformer_v2_t1", "bevformer_v2_t8"}:
            camera_input_size = (view_fusion_kwargs or {}).get(
                "image_size",
                256,
            )
            if (
                not isinstance(camera_input_size, int)
                or isinstance(camera_input_size, bool)
                or camera_input_size <= 0
            ):
                raise ValueError(
                    "BEVFormer camera image_size must be a positive integer"
                )
            self.camera_input_size = camera_input_size
            front_camera_index = (view_fusion_kwargs or {}).get(
                "front_camera_index",
                0,
            )
            front_camera_input_size = (view_fusion_kwargs or {}).get(
                "front_image_size",
                1024,
            )
            if (
                not isinstance(front_camera_index, int)
                or isinstance(front_camera_index, bool)
                or not 0 <= front_camera_index < num_views
                or not isinstance(front_camera_input_size, int)
                or isinstance(front_camera_input_size, bool)
                or front_camera_input_size < camera_input_size
                or front_camera_input_size % camera_input_size
            ):
                raise ValueError(
                    "BEVFormer front camera contract is invalid"
                )
            self.front_camera_index = front_camera_index
            self.front_camera_input_size = front_camera_input_size
        self.Backbone = Backbone(
            backbone=backbone,
            is_pretrained=is_pretrained,
            input_profile=(
                "bevformer_v2"
                if camera_architecture in {
                    "bevformer_v2_t1",
                    "bevformer_v2_t8",
                }
                else "imagenet"
            ),
        )

        # Multi-scale feature fusion with view unification.
        # view_fusion_kwargs forwards bev_h/bev_w/pc_range/image_size to BEV fusion.
        self.FeatureFusion = FeatureFusion(
            num_views=num_views,
            backbone_channels=self.Backbone.feature_channels,
            embed_dim=embed_dim,
            fusion_mode="bev",
            image_feature_size=image_feature_size,
            view_fusion_kwargs=view_fusion_kwargs,
        )
        self._camera_bev_frozen = False
        self._adapt_temporal_running_stats = False
        self._keep_backbone_batch_norm_eval = False
        self._initialize_camera_fpn_cache_owner()
        self._register_camera_fpn_cache_load_hooks()

        self.planner_mode = planner_mode
        self.FusedFeaturePooling = (
            FusedFeaturePooling(embed_dim=embed_dim)
            if planner_mode == "bezier"
            else None
        )

        # For BEV fusion mode the spatial size is bev_h × bev_w (potentially non-square).
        # Read each dim with a default so a PARTIAL view_fusion_kwargs (e.g. only
        # pc_range) doesn't KeyError — the `or` only fires for None/empty, and the
        # defaults must match BEVViewFusion's own (450×300).
        vfk = view_fusion_kwargs or {}
        map_output_h = vfk.get("bev_h", 450)
        map_output_w = vfk.get("bev_w", 300)

 
        if map_context_channels <= 0 or route_channels <= 0:
            raise ValueError("navigation channel counts must be positive")
        self.map_context_channels = int(map_context_channels)
        self.route_channels = int(route_channels)
        self.enable_route_conditioning = bool(enable_route_conditioning)

        self.NavigationEncoder = build_split_navigation_encoder(
            map_type,
            map_channels=self.map_context_channels,
            route_channels=self.route_channels,
            embed_dim=embed_dim,
            output_h=map_output_h,
            output_w=map_output_w,
            route_hidden_channels=route_encoder_hidden_channels,
        )
 
        # Map BEV fusion: combines image BEV features with map BEV features
        self.MapBEVFusion = build_map_bev_fusion(
            map_fusion_mode,
            embed_dim=embed_dim,
            **(map_fusion_kwargs or {}),
        )
        self.BEVSegmentationHead = (
            BEVSegmentationHead(
                embed_dim=embed_dim,
                num_classes=bev_segmentation_classes,
                output_size=auxiliary_output_size,
            )
            if enable_bev_segmentation
            else None
        )
        self.RouteReconstructionHead = (
            RouteReconstructionHead(
                embed_dim=embed_dim,
                route_channels=route_channels,
                output_size=auxiliary_output_size,
            )
            if enable_route_reconstruction
            else None
        )

        # Temporal Memory — compresses/fuses [B, T, feat] sequence histories into contexts
        self.TemporalMemory = build_temporal_memory(
            temporal_memory_mode,
            visual_dim=visual_history_dim,
            egomotion_dim=egomotion_dim,
            **(temporal_memory_kwargs or {}),
        )

        # Reasoning branch (1 Hz, opt-in, default OFF): horizon-aware,
        # action-relevant reasoning over the effective visual history + ego
        # context produced by TemporalMemory. Runs AFTER TemporalMemory so
        # ego_ctx is available (see Design/horizon_reasoning_architecture.md).
        # Feeds the planner through a ZERO-INIT coupling (reasoning_mode), a
        # strict no-op at init so the reactive baseline is byte-identical.
        self.enable_reasoning = enable_reasoning
        self.reasoning_mode = reasoning_mode if enable_reasoning else "none"
        self.ReasoningHead = None
        if enable_reasoning:
            rkw = dict(reasoning_kwargs or {})
            if rkw.get("route_context_dim") is not None:
                raise ValueError(
                    "route_context_dim is not supported by the initial #149 "
                    "contract; route conditioning is Reactive-only"
                )
            rkw.setdefault("visual_history_dim", visual_history_dim)
            rkw.setdefault("ego_context_dim", egomotion_dim)
            self.ReasoningHead = HorizonReasoningHead(**rkw)

        # Trajectory decoder — swappable via planner_mode (gru, flow_matching).
        # reasoning_mode wires the zero-init reasoning coupling inside the planner.
        self.TrajectoryPlanner = build_planner(
            planner_mode,
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            num_signals=num_signals,
            egomotion_dim=egomotion_dim,
            visual_history_dim=visual_history_dim,
            reasoning_mode=self.reasoning_mode,
            **(planner_kwargs or {}),
        )

        # NOTE: future visual-state prediction now lives in the World Model
        # branch (WorldActionModel.predict_future, JEPA). The old ReactiveE2E-owned
        # FutureState module was instantiated here but NEVER called in forward — a
        # gradient-dead parameter block — so it is removed. See auto_e2e.py.

    def freeze_camera_bev(
        self,
        *,
        adapt_temporal_running_stats: bool = False,
    ) -> None:
        """Freeze checkpoint weights while retaining the new front gate."""
        self._camera_bev_frozen = True
        self._adapt_temporal_running_stats = bool(
            adapt_temporal_running_stats
        )
        for module in (self.Backbone, self.FeatureFusion):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.camera_architecture in {
            "bevformer_v2_t1",
            "bevformer_v2_t8",
        }:
            front_gate = getattr(
                self.FeatureFusion.view_fusion,
                "front_residual_gate",
                None,
            )
            if not isinstance(front_gate, nn.Parameter):
                raise ValueError("BEVFormer front residual gate is missing")
            front_gate.requires_grad_(True)

    def enable_bev_finetuning(self) -> None:
        """Fine-tune camera BEV weights without rank-local backbone BN drift."""
        if self._camera_bev_frozen:
            raise ValueError("frozen camera BEV cannot be fine-tuned")
        self._keep_backbone_batch_norm_eval = True
        self._set_backbone_batch_norm_eval()

    def _set_backbone_batch_norm_eval(self) -> None:
        for module in self.Backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_keep_backbone_batch_norm_eval", False):
            self._set_backbone_batch_norm_eval()
        if self._camera_bev_frozen:
            self.Backbone.eval()
            self.FeatureFusion.eval()
            temporal_fusion = getattr(
                self.FeatureFusion,
                "temporal_fusion",
                None,
            )
            if temporal_fusion is not None:
                temporal_fusion.train(
                    mode and self._adapt_temporal_running_stats
                )
        return self

    @property
    def _camera_fpn_cache_owner(self) -> object:
        return self._camera_fpn_cache_owner_state.token

    def _initialize_camera_fpn_cache_owner(self) -> None:
        self._camera_fpn_cache_owner_state = _CameraFPNCacheOwnerState()

    def _register_camera_fpn_cache_load_hooks(self) -> None:
        registered_modules: set[int] = set()
        load_hook = _CameraFPNCacheLoadHook(
            self._camera_fpn_cache_owner_state
        )
        for root in (self.Backbone, self.FeatureFusion):
            for module in root.modules():
                module_id = id(module)
                if module_id in registered_modules:
                    continue
                registered_modules.add(module_id)
                module.register_load_state_dict_post_hook(
                    load_hook
                )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        try:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        finally:
            self._camera_fpn_cache_owner_state.invalidate()

    def create_stateful_camera_fpn_cache(
        self,
    ) -> StatefulCameraFPNCache:
        """Create an inference-only FPN cache for one T8 camera stream."""
        if self.camera_architecture != "bevformer_v2_t8":
            raise ValueError("stateful camera FPN cache requires BEVFormer T8")
        return StatefulCameraFPNCache(
            history_frames=7,
            owner_token=self._camera_fpn_cache_owner,
        )

    def _encode_base_camera_pyramid(self, camera_tiles):
        if camera_tiles.ndim != 5:
            raise ValueError(
                "camera_tiles must have shape [B,V,3,H,W]"
            )
        batch_size, num_views, channels, height, width = camera_tiles.shape
        if channels != 3:
            raise ValueError("camera_tiles must contain three RGB channels")
        if (
            self.camera_input_size is None
            or (height, width)
            != (self.camera_input_size, self.camera_input_size)
        ):
            raise ValueError(
                "camera tile size differs from BEVFormer image_size contract"
            )
        features = self.Backbone(camera_tiles.reshape(
            batch_size * num_views,
            channels,
            height,
            width,
        ))
        return self.FeatureFusion.build_bevformer_pyramid(features)

    def _replace_cached_front_pyramid(
        self,
        pyramid,
        front_camera_fpn_tile,
        *,
        camera_tiles,
        batch_size: int,
        num_views: int,
    ):
        if (
            self.front_camera_index is None
            or self.camera_input_size is None
        ):
            raise ValueError(
                "front camera FPN tile requires BEVFormer V2 fusion"
            )
        if tuple(front_camera_fpn_tile.shape) != (
            batch_size,
            3,
            self.camera_input_size,
            self.camera_input_size,
        ):
            raise ValueError(
                "front camera FPN tile differs from the base image contract"
            )
        if (
            front_camera_fpn_tile.dtype != camera_tiles.dtype
            or front_camera_fpn_tile.device != camera_tiles.device
        ):
            raise ValueError(
                "front camera FPN tile dtype and device must match camera tiles"
            )
        front_features = self.Backbone(front_camera_fpn_tile)
        front_pyramid = self.FeatureFusion.build_bevformer_pyramid(
            front_features
        )
        if len(pyramid) != len(front_pyramid):
            raise RuntimeError(
                "front camera FPN levels differ from the base pyramid"
            )
        replaced = []
        for base_level, front_level in zip(
            pyramid,
            front_pyramid,
            strict=True,
        ):
            if (
                base_level.ndim != 4
                or base_level.shape[0] != batch_size * num_views
                or front_level.shape
                != (batch_size, *base_level.shape[1:])
            ):
                raise RuntimeError(
                    "front camera FPN shape differs from the base pyramid"
                )
            level_by_view = base_level.reshape(
                batch_size,
                num_views,
                *base_level.shape[1:],
            ).clone()
            level_by_view[:, self.front_camera_index] = front_level
            replaced.append(level_by_view.flatten(0, 1))
        return tuple(replaced)

    def encode_camera_bev(
        self,
        camera_tiles,
        *,
        projection=None,
        geometry_type=None,
        image_transform=None,
        front_camera_tile=None,
        front_projection=None,
        front_image_transform=None,
        _return_base_pyramid=False,
    ):
        """Encode camera tiles without reading navigation inputs."""
        if (
            _return_base_pyramid
            and self.camera_architecture != "bevformer_v2_t8"
        ):
            raise ValueError(
                "base camera FPN output requires BEVFormer T8"
            )
        fusion_kwargs = (
            {"return_base_pyramid": True}
            if _return_base_pyramid
            else {}
        )
        if camera_tiles.ndim != 5:
            raise ValueError(
                "camera_tiles must have shape [B,V,3,H,W]"
            )
        batch_size, num_views, channels, height, width = camera_tiles.shape
        if channels != 3:
            raise ValueError("camera_tiles must contain three RGB channels")
        if (
            self.camera_input_size is not None
            and (height, width)
            != (self.camera_input_size, self.camera_input_size)
        ):
            raise ValueError(
                "camera tile size differs from BEVFormer image_size contract"
            )
        if front_camera_tile is None and self.camera_input_size is None:
            features = self.Backbone(
                camera_tiles.reshape(
                    batch_size * num_views,
                    channels,
                    height,
                    width,
                )
            )
            return self.FeatureFusion(
                features,
                batch_size,
                num_views,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
                **fusion_kwargs,
            )
        if front_camera_tile is None:
            if (
                front_projection is not None
                or front_image_transform is not None
            ):
                raise ValueError(
                    "front camera geometry requires a native front camera"
                )
            features = self.Backbone(camera_tiles.reshape(
                batch_size * num_views,
                channels,
                height,
                width,
            ))
            return self.FeatureFusion(
                features,
                batch_size,
                num_views,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
                **fusion_kwargs,
            )
        if (
            self.front_camera_index is None
            or self.front_camera_input_size is None
        ):
            raise ValueError(
                "front camera input requires BEVFormer V2 fusion"
            )
        if tuple(front_camera_tile.shape) != (
            batch_size,
            3,
            self.front_camera_input_size,
            self.front_camera_input_size,
        ):
            raise ValueError(
                "front camera tile differs from BEVFormer contract"
            )
        features = self.Backbone(camera_tiles.reshape(
            batch_size * num_views,
            channels,
            height,
            width,
        ))
        # The native front pass is additive. CAM_F0 remains present in the
        # ordinary all-view T8 encoder on every frame.
        front_features = self.Backbone(front_camera_tile)
        return self.FeatureFusion(
            features,
            batch_size,
            num_views,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
            front_features=front_features,
            front_projection=front_projection,
            front_image_transform=front_image_transform,
            **fusion_kwargs,
        )

    def _encode_camera_bev_with_base_pyramid(
        self,
        camera_tiles,
        *,
        projection=None,
        geometry_type=None,
        image_transform=None,
        front_camera_tile=None,
        front_projection=None,
        front_image_transform=None,
    ):
        output = self.encode_camera_bev(
            camera_tiles,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
            front_camera_tile=front_camera_tile,
            front_projection=front_projection,
            front_image_transform=front_image_transform,
            _return_base_pyramid=True,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "BEVFormer T8 did not return its base camera FPN"
            )
        return output

    def _encode_history_camera_bevs(
        self,
        camera_history_tiles,
        *,
        history_projections=None,
        geometry_type=None,
        image_transform=None,
    ):
        if self.camera_architecture != "bevformer_v2_t8":
            if camera_history_tiles is not None:
                raise ValueError(
                    "camera history requires BEVFormer V2 T8"
                )
            return []
        if camera_history_tiles is None:
            if history_projections is not None:
                raise ValueError(
                    "history projections require camera history tiles"
                )
            return []
        if camera_history_tiles.ndim != 6:
            raise ValueError(
                "camera_history_tiles must have shape [B,7,V,3,H,W]"
            )
        batch_size, history_count, num_views, channels, height, width = (
            camera_history_tiles.shape
        )
        if (
            history_count != 7
            or channels != 3
            or self.camera_input_size is None
            or (height, width) != (
                self.camera_input_size,
                self.camera_input_size,
            )
        ):
            raise ValueError("camera history differs from the T8 contract")
        if history_projections is not None and len(
            history_projections
        ) != history_count:
            raise ValueError("T8 requires one projection per history frame")
        backbone_training = self.Backbone.training
        fusion_training = self.FeatureFusion.training
        temporal_fusion = getattr(
            self.FeatureFusion,
            "temporal_fusion",
            None,
        )
        temporal_training = (
            temporal_fusion.training
            if temporal_fusion is not None
            else None
        )
        history_bevs = []
        self.Backbone.eval()
        self.FeatureFusion.eval()
        try:
            with torch.no_grad():
                for history_index in range(history_count):
                    tiles = camera_history_tiles[:, history_index]
                    features = self.Backbone(tiles.reshape(
                        batch_size * num_views,
                        channels,
                        height,
                        width,
                    ))
                    projection = (
                        history_projections[history_index]
                        if history_projections is not None
                        else None
                    )
                    history_bevs.append(self.FeatureFusion(
                        features,
                        batch_size,
                        num_views,
                        projection=projection,
                        geometry_type=geometry_type,
                        image_transform=image_transform,
                    ).detach())
        finally:
            self.Backbone.train(backbone_training)
            self.FeatureFusion.train(fusion_training)
            if temporal_fusion is not None:
                temporal_fusion.train(bool(temporal_training))
        return history_bevs

    def _prime_stateful_camera_fpn_cache(
        self,
        cache: StatefulCameraFPNCache,
        camera_history_tiles,
        *,
        prepared: _PreparedCameraFPNFrame,
        expected_num_views: int,
    ) -> None:
        if camera_history_tiles.ndim != 6:
            raise ValueError(
                "camera_history_tiles must have shape [B,7,V,3,H,W]"
            )
        batch_size, history_count, num_views, channels, height, width = (
            camera_history_tiles.shape
        )
        if num_views != expected_num_views:
            raise ValueError(
                "camera history view count differs from current camera tiles"
            )
        if (
            history_count != cache.history_frames
            or channels != 3
            or self.camera_input_size is None
            or (height, width) != (
                self.camera_input_size,
                self.camera_input_size,
            )
        ):
            raise ValueError("camera history differs from the T8 contract")
        pyramids = []
        with torch.no_grad():
            for history_index in range(history_count):
                pyramids.append(self._encode_base_camera_pyramid(
                    camera_history_tiles[:, history_index]
                ))
        cache._replace_history(
            pyramids,
            prepared=prepared,
            batch_size=batch_size,
            num_views=num_views,
        )

    def _encode_stateful_history_camera_bevs(
        self,
        cache: StatefulCameraFPNCache,
        *,
        batch_size: int,
        num_views: int,
        expected_timestamps_us,
        history_projections=None,
        geometry_type=None,
        image_transform=None,
    ):
        if cache.history_frames != 7:
            raise ValueError("BEVFormer T8 requires a seven-frame FPN cache")
        if not cache.is_full:
            raise RuntimeError(
                "stateful BEVFormer T8 requires a complete FPN history"
            )
        if (
            cache.batch_size != batch_size
            or cache.num_views != num_views
        ):
            raise ValueError(
                "camera FPN cache dimensions differ from current stream"
            )
        if cache.frame_timestamps_us != expected_timestamps_us:
            raise RuntimeError(
                "camera FPN cache timestamps differ from current history"
            )
        if (
            history_projections is not None
            and len(history_projections) != cache.history_frames
        ):
            raise ValueError("T8 requires one projection per history frame")
        history_bevs = []
        with torch.no_grad():
            for cache_index, pyramid in enumerate(cache.frames):
                projection = (
                    history_projections[cache_index]
                    if history_projections is not None
                    else None
                )
                history_bevs.append(
                    self.FeatureFusion.fuse_bevformer_pyramid(
                        pyramid,
                        batch_size,
                        num_views,
                        projection=projection,
                        geometry_type=geometry_type,
                        image_transform=image_transform,
                    ).detach()
                )
        return history_bevs

    def _fuse_temporal_camera_bevs(
        self,
        current_image_bev,
        history_bevs,
    ):
        if self.camera_architecture != "bevformer_v2_t8":
            if history_bevs:
                raise ValueError("non-T8 camera path received history BEVs")
            return current_image_bev
        if not history_bevs:
            history_bevs = [
                current_image_bev.detach()
                for _ in range(7)
            ]
        elif len(history_bevs) != 7:
            raise ValueError(
                "BEVFormer T8 requires either zero or seven history BEVs"
            )
        # Upstream BEVFormer@66b65f3 transformerV2.py:308-324 places the
        # current frame last in the ordered (-7,-6,-5,-4,-3,-2,-1,0) list.
        return self.FeatureFusion.fuse_temporal_bevs([
            *history_bevs,
            current_image_bev,
        ])

    def forward(self, camera_tiles, map_context, visual_history,
                egomotion_history, route_mask=None, map_valid=None,
                route_valid=None,
                projection=None, geometry_type=None, image_transform=None,
                camera_history_tiles=None, history_projections=None,
                camera_fpn_cache=None,
                camera_fpn_stream_ids=None,
                camera_fpn_timestamps_us=None,
                front_camera_tile=None, front_camera_fpn_tile=None,
                front_camera_fpn_available=None,
                front_projection=None,
                front_image_transform=None,
                mode="train", return_auxiliary=False,
                compute_bev_segmentation=True,
                compute_route_reconstruction=True,
                bev_only=False,
                **kwargs):
        """
        Run the reactive end-to-end autonomous-driving pipeline.


        Args:
            camera_tiles: (B, V, 3, H, W) — V real camera images.
            map_context: (B, C_map, H_map, W_map) — semantic BEV map.
            route_mask: (B, C_route, H_map, W_map) — selected route.
            map_valid / route_valid: (B,) explicit sample validity.
                ``enable_route_conditioning=False`` forces the route gate off.
            visual_history: (B, T, visual_history_dim) or (B, visual_history_dim).
            egomotion_history: (B, T, egomotion_dim) or (B, egomotion_dim).
            projection: Optional CameraProjectionModel operator — the geometry
                ABI (Pinhole / FTheta / Pseudo). No [B,V,3,4] matrix argument.
            geometry_type: Optional explicit geometry label passed to BEV fusion.
            image_transform: Optional ImageTransform for the model-input frame.
            front_camera_tile: Optional native-resolution front image tensor.
            front_projection: Optional one-view projection for the native front.
            camera_history_tiles: Optional seven-frame camera history ordered
                oldest to newest, shaped ``[B,7,V,3,H,W]``.
            history_projections: Optional seven projection operators mapping
                current ego coordinates into each historical image set.
            camera_fpn_cache: Optional per-stream stateful T8 FPN cache.
            camera_fpn_stream_ids: Ordered scene or stream identity per sample.
            camera_fpn_timestamps_us: Source timestamps sampled exactly every
                500 ms. Do not synthesize continuity across dropped frames.
            front_camera_fpn_tile: Optional base-resolution Front image packed
                identically to historical Front frames for cache insertion.
            front_camera_fpn_available: Per-sample validity for the exact
                base-resolution Front companion.
            mode: "train" returns enabled auxiliary predictions.
            return_auxiliary: also return enabled auxiliary predictions during
                inference, for offline Dashboard artifact generation.

        Returns:
            trajectory (B, num_timesteps * num_signals), or
            ``(trajectory, aux_outputs)`` when auxiliary outputs were requested.
        """
        B = camera_tiles.shape[0]

        # --- Camera branch ---
        prepared_camera_fpn_frame = None
        if camera_fpn_cache is not None:
            if not isinstance(
                camera_fpn_cache,
                StatefulCameraFPNCache,
            ):
                raise TypeError(
                    "camera_fpn_cache must be a StatefulCameraFPNCache"
                )
            if (
                self.camera_architecture != "bevformer_v2_t8"
                or mode == "train"
                or self.training
                or self.Backbone.training
                or self.FeatureFusion.training
                or torch.is_grad_enabled()
            ):
                raise ValueError(
                    "stateful camera FPN cache requires eval-mode "
                    "BEVFormer T8 with gradients disabled"
                )
            camera_fpn_cache._validate_owner(
                self._camera_fpn_cache_owner
            )
            if camera_fpn_stream_ids is None:
                raise ValueError(
                    "stateful camera FPN cache requires stream IDs"
                )
            if camera_fpn_timestamps_us is None:
                raise ValueError(
                    "stateful camera FPN cache requires timestamps"
                )
            if (front_camera_tile is None) != (
                front_camera_fpn_tile is None
            ):
                raise ValueError(
                    "stateful camera FPN cache requires native and exact "
                    "base-resolution Front tiles together"
                )
            if front_camera_tile is not None:
                if front_camera_fpn_available is None:
                    raise ValueError(
                        "stateful camera FPN cache requires Front companion "
                        "availability"
                    )
                front_available = torch.as_tensor(
                    front_camera_fpn_available,
                    device=camera_tiles.device,
                    dtype=torch.bool,
                ).reshape(-1)
                if front_available.numel() != B or not bool(
                    front_available.all().item()
                ):
                    raise ValueError(
                        "stateful camera FPN cache requires an exact Front "
                        "companion for every sample"
                    )
            current_image_bev, current_base_pyramid = (
                self._encode_camera_bev_with_base_pyramid(
                    camera_tiles,
                    projection=projection,
                    geometry_type=geometry_type,
                    image_transform=image_transform,
                    front_camera_tile=front_camera_tile,
                    front_projection=front_projection,
                    front_image_transform=front_image_transform,
                )
            )
            if front_camera_fpn_tile is not None:
                current_base_pyramid = (
                    self._replace_cached_front_pyramid(
                        current_base_pyramid,
                        front_camera_fpn_tile,
                        camera_tiles=camera_tiles,
                        batch_size=B,
                        num_views=int(camera_tiles.shape[1]),
                    )
                )
            prepared_camera_fpn_frame = (
                camera_fpn_cache._prepare_frame(
                    current_base_pyramid,
                    stream_ids=camera_fpn_stream_ids,
                    timestamps_us=camera_fpn_timestamps_us,
                    batch_size=B,
                    num_views=int(camera_tiles.shape[1]),
                    front_companion_mode=(
                        front_camera_fpn_tile is not None
                    ),
                    owner_token=self._camera_fpn_cache_owner,
                )
            )
            if prepared_camera_fpn_frame.needs_prime:
                if camera_history_tiles is None:
                    raise ValueError(
                        "stateful camera FPN cache requires raw history "
                        "after reset or discontinuity"
                    )
                self._prime_stateful_camera_fpn_cache(
                    camera_fpn_cache,
                    camera_history_tiles,
                    prepared=prepared_camera_fpn_frame,
                    expected_num_views=int(camera_tiles.shape[1]),
                )
            history_bevs = self._encode_stateful_history_camera_bevs(
                camera_fpn_cache,
                batch_size=B,
                num_views=int(camera_tiles.shape[1]),
                expected_timestamps_us=(
                    prepared_camera_fpn_frame
                    .expected_history_timestamps_us
                ),
                history_projections=history_projections,
                geometry_type=geometry_type,
                image_transform=image_transform,
            )
        else:
            if (
                camera_fpn_stream_ids is not None
                or camera_fpn_timestamps_us is not None
                or front_camera_fpn_tile is not None
                or front_camera_fpn_available is not None
            ):
                raise ValueError(
                    "stateful camera FPN arguments require a cache"
                )
            history_bevs = self._encode_history_camera_bevs(
                camera_history_tiles,
                history_projections=history_projections,
                geometry_type=geometry_type,
                image_transform=image_transform,
            )
            current_image_bev = self.encode_camera_bev(
                camera_tiles,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
                front_camera_tile=front_camera_tile,
                front_projection=front_projection,
                front_image_transform=front_image_transform,
            )
        image_bev = self._fuse_temporal_camera_bevs(
            current_image_bev,
            history_bevs,
        )
        emit_auxiliary = mode == "train" or bool(return_auxiliary)
        aux_outputs = {}
        if (
            self.BEVSegmentationHead is not None
            and emit_auxiliary
            and compute_bev_segmentation
        ):
            aux_outputs["bev_segmentation_logits"] = (
                self.BEVSegmentationHead(image_bev)
            )
        if bev_only:
            if (
                not emit_auxiliary
                or not compute_bev_segmentation
                or compute_route_reconstruction
                or "bev_segmentation_logits" not in aux_outputs
            ):
                raise ValueError(
                    "BEV-only forward requires only BEV segmentation output"
                )
            return image_bev.new_zeros((B, 0)), aux_outputs

        # --- Reactive-only navigation branch ---
        if (
            map_context.ndim != 4
            or map_context.shape[0] != B
            or map_context.shape[1] != self.map_context_channels
        ):
            raise ValueError(
                "map_context must have shape "
                f"[B,{self.map_context_channels},H,W]"
            )
        if route_mask is None:
            route_mask = map_context.new_zeros(
                B,
                self.route_channels,
                map_context.shape[-2],
                map_context.shape[-1],
            )
        if (
            route_mask.ndim != 4
            or route_mask.shape[:2] != (B, self.route_channels)
            or route_mask.shape[-2:] != map_context.shape[-2:]
        ):
            raise ValueError(
                "route_mask must share map spatial dimensions and have "
                f"{self.route_channels} channels"
            )

        def validity_gate(value, valid, *, default, name):
            if valid is None:
                valid = torch.full(
                    (B,),
                    default,
                    dtype=torch.bool,
                    device=value.device,
                )
            valid = torch.as_tensor(valid, device=value.device)
            if valid.shape not in ((B,), (B, 1)):
                raise ValueError(f"{name} must have shape [B] or [B,1]")
            return value * valid.reshape(B, 1, 1, 1).to(value.dtype)

        gated_map = validity_gate(
            map_context,
            map_valid,
            default=True,
            name="map_valid",
        )
        if not self.enable_route_conditioning:
            route_valid = torch.zeros(
                B,
                dtype=torch.bool,
                device=route_mask.device,
            )
        gated_route = validity_gate(
            route_mask,
            route_valid,
            default=False,
            name="route_valid",
        )
        navigation_bev, route_contribution = self.NavigationEncoder(
            gated_map,
            gated_route,
            return_route_contribution=True,
        )

        # --- Fuse image BEV + navigation BEV ---
        fused_features = self.MapBEVFusion(image_bev, navigation_bev)
        if (
            self.RouteReconstructionHead is not None
            and emit_auxiliary
            and compute_route_reconstruction
        ):
            aux_outputs["route_reconstruction_logits"] = (
                self.RouteReconstructionHead(route_contribution)
            )

        planner_features = (
            self.FusedFeaturePooling(fused_features)
            if self.FusedFeaturePooling is not None
            else fused_features
        )

        # --- Temporal Memory ---
        visual_ctx, ego_ctx = self.TemporalMemory(visual_history, egomotion_history)

        # --- Reasoning branch (1 Hz, opt-in) ---
        # Runs on the EFFECTIVE context TemporalMemory produced, so the reasoning
        # head and the planner see the same visual/ego signal. Its latent /
        # horizon tokens feed the planner through the zero-init coupling.
        reasoning_pred = None
        reasoning_latent = None
        reasoning_horizon_tokens = None
        if self.ReasoningHead is not None:
            reasoning_pred = self.ReasoningHead(
                visual_ctx, ego_ctx,
            )
            reasoning_latent = reasoning_pred.reasoning_latent
            reasoning_horizon_tokens = reasoning_pred.horizon_tokens

        # --- Trajectory Prediction ---
        trajectory = self.TrajectoryPlanner(
            planner_features, visual_ctx, ego_ctx,
            reasoning_latent=reasoning_latent,
            reasoning_horizon_tokens=reasoning_horizon_tokens,
            **kwargs,
        )
        if reasoning_pred is not None and mode == "train":
            aux_outputs["reasoning_pred"] = reasoning_pred
        result = (trajectory, aux_outputs) if aux_outputs else trajectory

        if camera_fpn_cache is not None:
            if prepared_camera_fpn_frame is None:
                raise RuntimeError(
                    "stateful camera FPN cache received no prepared frame"
                )
            camera_fpn_cache._commit_frame(
                prepared_camera_fpn_frame,
                batch_size=B,
                num_views=int(camera_tiles.shape[1]),
            )
        return result
