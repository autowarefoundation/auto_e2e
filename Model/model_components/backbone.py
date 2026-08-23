import torch
import torch.nn as nn
from .backbones import build_backbone


class Backbone(nn.Module):
    def __init__(
        self,
        backbone="swin_v2_tiny",
        is_pretrained: bool = True,
        input_profile: str = "imagenet",
    ):
        super().__init__()
        if input_profile not in {"imagenet", "bevformer_v2"}:
            raise ValueError(f"unknown backbone input profile {input_profile!r}")

        # Pre-trained backbone (pluggable)
        self.backbone = build_backbone(backbone, pretrained=is_pretrained)
        self.backbone_name = backbone
        self.input_profile = input_profile
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_bevformer_bgr_mean",
            torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1),
            persistent=False,
        )

        # Discover per-stage channel counts. feature_info is the timm convention
        # and is preferred (no forward pass), but a custom backbone may not
        # expose it; fall back to a one-shot dummy forward and read tensor shapes.
        self.feature_channels = self._infer_feature_channels()
        self.backbone_channels = sum(self.feature_channels)

    def _infer_feature_channels(self):
        info = getattr(self.backbone, "feature_info", None)
        if info is not None:
            try:
                channels = [stage["num_chs"] for stage in info]
            except (TypeError, KeyError, IndexError):
                channels = None
            if channels:
                return channels

        # Fallback: probe the backbone with a dummy input and sum the channel
        # dim of each returned feature map. Assumes channels-first output
        # (B, C, H, W) — the dominant convention for non-timm backbones.
        try:
            param = next(self.backbone.parameters())
            device, dtype = param.device, param.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32

        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 256, 256, device=device, dtype=dtype)
                out = self.backbone(dummy)
        finally:
            self.backbone.train(was_training)

        if not isinstance(out, (list, tuple)) or len(out) == 0:
            raise ValueError(
                f"Backbone '{self.backbone_name}' did not return a non-empty "
                "list of feature maps; cannot infer channel counts."
            )
        for f in out:
            if f.dim() != 4:
                raise ValueError(
                    f"Backbone '{self.backbone_name}' returned a feature map "
                    f"with shape {tuple(f.shape)}; expected 4D tensor."
                )
        return [f.shape[1] for f in out]

    def forward(self, image):
        if self.input_profile == "bevformer_v2":
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(
                    "BEVFormer backbone input must have shape [B,3,H,W]"
                )
            mean = self._imagenet_mean.to(dtype=image.dtype)
            std = self._imagenet_std.to(dtype=image.dtype)
            bgr_mean = self._bevformer_bgr_mean.to(dtype=image.dtype)
            image = (
                (image * std + mean).flip(dims=(1,)) * 255.0
                - bgr_mean
            )
        features = self.backbone(image)

        # Detect channel layout per-feature-map from the actual tensor shape
        # rather than the backbone name. timm's Swin variants return
        # channels-last (B, H, W, C); ResNet/ConvNeXt return channels-first
        # (B, C, H, W). Compare each feature map's shape against the channel
        # count discovered at construction time and permute when needed.
        permuted = []
        for f, expected_c in zip(features, self.feature_channels):
            if f.dim() == 4 and f.shape[1] != expected_c and f.shape[-1] == expected_c:
                permuted.append(f.permute(0, 3, 1, 2))
            else:
                permuted.append(f)
        return permuted
