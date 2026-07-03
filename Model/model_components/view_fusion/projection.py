"""Differentiable camera projection operators for BEV view fusion.

The contract of ``BEVViewFusion`` with geometry is NOT a fixed ``[B, V, 3, 4]``
matrix. It is a *projection operator*: something that maps ego-frame BEV
reference points to sampling coordinates on each camera's model-input image
plane, plus a visibility mask. A pinhole ``K @ T`` matrix is only ONE such
operator (its linear fast path). Fisheye (f-theta) cameras need a non-linear
operator, and a calibration-free run needs a learnable pseudo operator. All
expose the same :meth:`project_ego_to_image` so the fusion module never branches
on camera model.

Design (per the Issue #77 discussion):
    - The public contract is :class:`CameraProjectionModel`, not ``K @ T``. A
      pinhole matrix lives *inside* :class:`PinholeProjection` as a fast path.
    - Native fisheye is a first-class :class:`FThetaProjection`; we do not force
      it through a pinhole (which would lose FOV). Rectification, when wanted, is
      expressed as an :class:`ImageTransform` + a ``rectified_pinhole`` operator,
      not by pretending a fisheye is a pinhole.
    - Projection is defined against the *model-input* image frame, so the image
      side (resize/crop/pad/rectification) is part of the contract via
      :class:`ImageTransform` — a projection that is correct on the raw image but
      not on the resized tensor would sample the wrong pixels.

Coordinate convention: reference points are in the ego/vehicle frame,
X=forward, Y=left, Z=up, in metres. :meth:`project_ego_to_image` returns pixel
coordinates normalized to ``[0, 1]`` by the model-input width/height, a
visibility mask, the per-point depth/range, and a metadata dict recording the
geometry that produced the result.

Reference:
    - BEVFormer (Li et al., ECCV 2022): spatial cross-attention. Its essence is
      "each BEV query samples relevant regions from each camera view"; the
      projection being a pinhole matrix is an implementation detail, not the
      contract.
    - NVIDIA PhysicalAI-AV sensor model: native f-theta (fisheye), motivating a
      non-linear projection operator rather than a pinhole-only ABI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch

# Geometry labels. A projection operator carries exactly one of these so
# experiment metadata can record, honestly, which geometry produced a run.
GEOMETRY_PINHOLE = "pinhole"
GEOMETRY_RECTIFIED_PINHOLE = "rectified_pinhole"
GEOMETRY_FTHETA = "ftheta"
GEOMETRY_PSEUDO = "pseudo"

VALID_GEOMETRY_TYPES = (
    GEOMETRY_PINHOLE,
    GEOMETRY_RECTIFIED_PINHOLE,
    GEOMETRY_FTHETA,
    GEOMETRY_PSEUDO,
)

# Points with camera-frame depth below this are treated as behind the camera.
_DEPTH_EPS = 1e-5


@dataclass(frozen=True)
class ImageTransform:
    """Describes the model-input image frame a projection must target.

    A projection is only useful if its pixel coordinates match the tensor that
    actually enters the backbone. If the raw image was resized/cropped/padded (or
    rectified from a fisheye), the projection's ``uv`` must be expressed in that
    final frame. This dataclass carries the information needed to normalize
    projected pixels into ``[0, 1]`` for ``grid_sample`` and to record, honestly,
    what image-side processing was applied.

    Attributes:
        model_input_size: ``(width, height)`` of the tensor fed to the backbone.
            uv is normalized by this (per-axis), so non-square inputs are handled.
        original_size: ``(width, height)`` of the raw image, if known (metadata).
        rectification: label of any rectification applied to the raw image before
            the backbone (e.g. "ftheta_to_pinhole"), or None. Metadata/honesty
            only; the geometry itself lives in the projection operator.
    """

    model_input_size: tuple[int, int]
    original_size: Optional[tuple[int, int]] = None
    rectification: Optional[str] = None

    @classmethod
    def square(cls, size: int) -> "ImageTransform":
        """Convenience for the common square model input (e.g. 256x256 shards)."""
        return cls(model_input_size=(int(size), int(size)))

    @property
    def wh(self) -> tuple[float, float]:
        return float(self.model_input_size[0]), float(self.model_input_size[1])


def _as_image_transform(image_transform) -> ImageTransform:
    """Coerce a bare int/float (square size) into an :class:`ImageTransform`."""
    if isinstance(image_transform, ImageTransform):
        return image_transform
    return ImageTransform.square(int(image_transform))


@dataclass
class ProjectionResult:
    """Output of projecting ego-frame reference points onto camera images.

    Shapes use ``M`` for the flattened reference-point count (``N * num_z`` in
    BEVViewFusion terms) so operators stay agnostic to the BEV grid layout; the
    fusion module reshapes ``M -> (N, num_z)`` afterwards.

    Attributes:
        uv_norm: ``[Bp, V, M, 2]`` pixel coordinates normalized to ``[0, 1]``.
            ``Bp`` is the operator's batch dim (real ``B`` for calibrated
            cameras, ``1`` for the batch-independent pseudo operator, which then
            broadcasts across the batch in the sampling loop).
        valid_mask: ``[Bp, V, M]`` bool — True where the point is in front of the
            camera AND lands within the image bounds.
        depth: ``[Bp, V, M]`` per-point depth/range along the optical axis
            (metres for calibrated cameras). Diagnostics / future depth
            supervision; not required by the sampler.
        metadata: what geometry produced this result (geometry_type,
            model_input_size, rectification). Recorded so experiment logs are
            honest about pinhole vs f-theta vs pseudo.
    """

    uv_norm: torch.Tensor
    valid_mask: torch.Tensor
    depth: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class CameraProjectionModel(Protocol):
    """The BEV-fusion geometry contract: ego points -> image sampling coords.

    Implementations (:class:`PinholeProjection`, :class:`FThetaProjection`,
    :class:`PseudoProjection`) differ only in the camera model; the fusion module
    treats them uniformly.
    """

    geometry_type: str

    @property
    def num_views(self) -> int:
        ...

    def project_ego_to_image(
        self, points_ego: torch.Tensor, image_transform
    ) -> ProjectionResult:
        ...


def _homogenize(points_ego: torch.Tensor) -> torch.Tensor:
    """``[M, 3]`` ego points -> ``[M, 4]`` homogeneous (append a ones column)."""
    if points_ego.shape[-1] == 4:
        return points_ego  # already homogeneous
    ones = torch.ones(
        *points_ego.shape[:-1], 1, device=points_ego.device, dtype=points_ego.dtype
    )
    return torch.cat([points_ego, ones], dim=-1)


def _finalize_linear(projected, image_transform, geometry_type) -> ProjectionResult:
    """Perspective-divide, per-axis normalize by model-input size, build mask.

    Shared by the pinhole fast path. ``projected`` is ``[Bp, V, M, 3]`` =
    ``[u*d, v*d, d]`` in the model-input pixel frame.
    """
    it = _as_image_transform(image_transform)
    w, h = it.wh
    depth = projected[..., 2]                         # [Bp, V, M]
    valid_depth = depth > _DEPTH_EPS                  # in front of the camera
    depth_safe = depth.clamp(min=_DEPTH_EPS).unsqueeze(-1)  # avoid div-by-zero
    uv = projected[..., :2] / depth_safe              # pixel coords [Bp, V, M, 2]
    wh = torch.tensor([w, h], device=uv.device, dtype=uv.dtype)
    uv_norm = uv / wh
    in_bounds = (
        (uv_norm[..., 0] >= 0) & (uv_norm[..., 0] <= 1)
        & (uv_norm[..., 1] >= 0) & (uv_norm[..., 1] <= 1)
    )
    mask = valid_depth & in_bounds
    meta = {"geometry_type": geometry_type, "model_input_size": it.model_input_size,
            "rectification": it.rectification}
    return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth, metadata=meta)


class PinholeProjection:
    """Linear ego-to-pixel projection: the ``K @ T`` fast path.

    Construct either from a combined ego-to-pixel matrix (``matrix`` =
    ``intrinsic @ extrinsic``, ``[B, V, 3, 4]``, already scaled to the
    model-input image) or from separate intrinsics ``K`` ``[B, V, 3, 3]`` and an
    ego->camera transform ``T_camera_ego`` ``[B, V, 4, 4]`` via :meth:`from_KT`.
    Keeping K and T separable lets a rectified fisheye be expressed as
    ``PinholeProjection.from_KT(K_rectified, T, geometry_type="rectified_pinhole")``
    instead of pretending a fisheye is a pinhole.
    """

    def __init__(self, matrix, geometry_type: str = GEOMETRY_PINHOLE,
                 K=None, T_camera_ego=None):
        if matrix.dim() != 4 or matrix.shape[-2:] != (3, 4):
            raise ValueError(
                f"PinholeProjection matrix must be [B, V, 3, 4], got {tuple(matrix.shape)}"
            )
        if geometry_type not in (GEOMETRY_PINHOLE, GEOMETRY_RECTIFIED_PINHOLE):
            raise ValueError(
                f"PinholeProjection geometry_type must be 'pinhole' or "
                f"'rectified_pinhole', got {geometry_type!r}"
            )
        self.matrix = matrix              # [B, V, 3, 4] combined (used by project)
        self.K = K                        # [B, V, 3, 3] or None (introspection)
        self.T_camera_ego = T_camera_ego  # [B, V, 4, 4] or None
        self.geometry_type = geometry_type

    @classmethod
    def from_KT(cls, K, T_camera_ego, geometry_type: str = GEOMETRY_PINHOLE):
        """Build from separate intrinsics ``K`` [B,V,3,3] and ego->camera
        transform ``T_camera_ego`` [B,V,4,4]; keeps both for introspection."""
        if K.dim() != 4 or K.shape[-2:] != (3, 3):
            raise ValueError(f"K must be [B, V, 3, 3], got {tuple(K.shape)}")
        if T_camera_ego.dim() != 4 or T_camera_ego.shape[-2:] != (4, 4):
            raise ValueError(
                f"T_camera_ego must be [B, V, 4, 4], got {tuple(T_camera_ego.shape)}"
            )
        # matrix = K @ T[:, :, :3, :]  -> [B, V, 3, 4]. Coerce T to K's dtype/device
        # first so mismatched inputs don't crash the einsum.
        T3 = T_camera_ego[:, :, :3, :].to(dtype=K.dtype, device=K.device)
        matrix = torch.einsum("bvij,bvjk->bvik", K, T3)
        return cls(matrix, geometry_type=geometry_type, K=K, T_camera_ego=T_camera_ego)

    @property
    def num_views(self) -> int:
        return self.matrix.shape[1]

    def to(self, device) -> "PinholeProjection":
        def _mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return PinholeProjection(
            self.matrix.to(device), geometry_type=self.geometry_type,
            K=_mv(self.K), T_camera_ego=_mv(self.T_camera_ego),
        )

    def to_spec(self) -> dict:
        """Serialize to a JSON-able manifest spec (batch dim dropped)."""
        return {
            "type": self.geometry_type,
            "matrix": self.matrix[0].detach().cpu().tolist(),  # [V, 3, 4]
        }

    def project_ego_to_image(self, points_ego, image_transform) -> ProjectionResult:
        """Project ego points ``[M, 3]`` (or homogeneous ``[M, 4]``) onto each
        camera. ``B, V`` come from the matrix — runtime ``V`` is derived here."""
        pts = _homogenize(points_ego)
        proj = self.matrix.to(device=pts.device, dtype=pts.dtype)
        # out[b, v, m, i] = sum_j proj[b, v, i, j] * points[m, j]
        projected = torch.einsum("bvij,mj->bvmi", proj, pts)
        return _finalize_linear(projected, image_transform, self.geometry_type)


class PseudoProjection:
    """Learnable calibration-free fallback (shape-testing / ablation only).

    This is NOT real geometry. It is a learnable spatial prior that lets the
    module run without calibration. A single shared ``[3, 4]`` matrix is expanded
    to ``V`` views at projection time, so one instance serves any view count.
    Pixel coordinates are squashed with ``sigmoid`` (the raw matrix is unbounded)
    rather than normalized by the model-input size.

    The learnable tensor is owned by :class:`BEVViewFusion` (a leaf Parameter, so
    the optimizer sees it); this operator wraps it per forward. Callers must
    explicitly request the pseudo path — the fusion module never falls into it
    silently on behalf of a caller that meant to pass real calibration.
    """

    geometry_type = GEOMETRY_PSEUDO

    def __init__(self, matrix: torch.Tensor, num_views: int):
        # Accept exactly [3, 4] or a leading-1 [1, 3, 4]; anything else (e.g. a
        # per-view [V, 3, 4]) is a misuse — the prior is view-independent — and
        # would silently mis-reshape, so reject it explicitly.
        if tuple(matrix.shape) not in ((3, 4), (1, 3, 4)):
            raise ValueError(
                f"PseudoProjection matrix must be [3, 4] or [1, 3, 4], "
                f"got {tuple(matrix.shape)}"
            )
        self.matrix = matrix
        self.num_views = num_views

    def project_ego_to_image(self, points_ego, image_transform) -> ProjectionResult:
        pts = _homogenize(points_ego)
        # Expand the shared [3, 4] prior to [1, V, 3, 4]: batch dim 1 broadcasts
        # across the real batch in the sampling loop (prior is batch- and
        # view-independent by construction).
        proj = self.matrix.reshape(3, 4).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 4]
        proj = proj.expand(1, self.num_views, 3, 4).to(device=pts.device, dtype=pts.dtype)
        projected = torch.einsum("bvij,mj->bvmi", proj, pts)  # [1, V, M, 3]

        depth = projected[..., 2]
        valid_depth = depth > _DEPTH_EPS
        depth_safe = depth.clamp(min=_DEPTH_EPS).unsqueeze(-1)
        uv = projected[..., :2] / depth_safe
        # Unbounded pseudo outputs → sigmoid to keep coords in (0, 1). in-bounds
        # is then trivially satisfied, so the mask reduces to the depth check.
        uv_norm = uv.sigmoid()
        it = _as_image_transform(image_transform)
        meta = {"geometry_type": self.geometry_type,
                "model_input_size": it.model_input_size, "rectification": None}
        return ProjectionResult(uv_norm=uv_norm, valid_mask=valid_depth,
                                depth=depth, metadata=meta)


class FThetaProjection:
    """Non-linear f-theta (fisheye) projection, native to NVIDIA PhysicalAI-AV.

    Maps ego points to pixels WITHOUT flattening the fisheye to a pinhole, so a
    wide-FOV camera keeps its full field of view (no rectification FOV loss). The
    forward polynomial maps the incidence angle ``theta`` (angle between the
    camera-frame ray and the optical +Z axis) to a pixel radius::

        r(theta) = c0 + c1*theta + c2*theta^2 + ...
        u = cx + r * (x_cam / rho),  v = cy + r * (y_cam / rho)

    where ``rho = sqrt(x_cam^2 + y_cam^2)``. Matches the SDK's
    ``FThetaCameraModel.ray2pixel``.

    Parameters (all pre-scaled to the model-input image by the parser):
        t_camera_ego: ``[B, V, 4, 4]`` ego->camera rigid transform.
        fw_poly: ``[K]`` or ``[B, V, K]`` forward polynomial coefficients
            (ascending powers of theta), radius in pixels.
        cx, cy: principal point in pixels, scalar or ``[B, V]``.
        max_theta: incidence-angle FOV cutoff (radians); the upper bound of the
            valid theta range. Points beyond it are masked (outside the lens).
        bw_poly: optional backward polynomial (radius -> theta) for inverse/remap
            (e.g. rectification); not used by the forward projection.
    """

    geometry_type = GEOMETRY_FTHETA

    def __init__(self, t_camera_ego, fw_poly, cx, cy, max_theta=None, bw_poly=None):
        if t_camera_ego.dim() != 4 or t_camera_ego.shape[-2:] != (4, 4):
            raise ValueError(
                f"FThetaProjection t_camera_ego must be [B, V, 4, 4], "
                f"got {tuple(t_camera_ego.shape)}"
            )
        self.t_camera_ego = t_camera_ego
        self.fw_poly = fw_poly
        self.cx = cx
        self.cy = cy
        self.max_theta = max_theta   # upper bound of the valid theta range (FOV)
        self.bw_poly = bw_poly       # radius -> theta, for inverse/remap only

    @property
    def num_views(self) -> int:
        return self.t_camera_ego.shape[1]

    def to(self, device) -> "FThetaProjection":
        def _mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return FThetaProjection(
            self.t_camera_ego.to(device), _mv(self.fw_poly),
            _mv(self.cx), _mv(self.cy), max_theta=_mv(self.max_theta),
            bw_poly=_mv(self.bw_poly),
        )

    def to_spec(self) -> dict:
        """Serialize to a JSON-able manifest spec (batch dim dropped).

        Fields differ in their canonical *batched* rank, so a rank alone cannot
        tell apart e.g. a per-view ``fw_poly [V, K]`` (keep whole) from a batched
        ``cx [B, V]`` (drop batch). Drop the leading dim only when the field is
        at its batched rank; otherwise keep it whole (a scalar becomes a float).
        """
        def _ser(x, batched_rank):
            if not torch.is_tensor(x):
                return x
            if x.dim() == 0:
                return x.item()
            if x.dim() == batched_rank:
                return x[0].detach().cpu().tolist()   # drop leading batch dim
            return x.detach().cpu().tolist()          # unbatched — keep whole
        return {
            "type": self.geometry_type,
            "t_camera_ego": _ser(self.t_camera_ego, 4),  # [B,V,4,4] -> [V,4,4]
            "fw_poly": _ser(self.fw_poly, 3),            # [B,V,K] -> [V,K]; [V,K]/[K] kept
            "cx": _ser(self.cx, 2),                      # [B,V] -> [V]; [V] kept
            "cy": _ser(self.cy, 2),
            "max_theta": _ser(self.max_theta, 2),        # scalar/list, never a raw tensor
        }

    def _radius(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the forward polynomial r(theta) via Horner's method.

        ``fw_poly`` is ascending-power coefficients. Accepts a shared ``[K]``
        vector, a per-view ``[V, K]``, or a batched ``[B, V, K]`` — all are
        normalized to broadcast against ``theta`` ``[B, V, M]``, so the operator
        is robust to however it was constructed (e.g. reloaded from a manifest).
        A shared ``[K]`` is applied identically to every view/point.
        """
        coeffs = torch.as_tensor(self.fw_poly, device=theta.device, dtype=theta.dtype)
        if coeffs.dim() == 1:
            # shared [K] — plain Horner, broadcasts over all of theta.
            r = torch.zeros_like(theta)
            for c in reversed(coeffs.unbind(0)):
                r = r * theta + c
            return r
        if coeffs.dim() == 2:
            coeffs = coeffs.unsqueeze(0)   # [V, K] -> [1, V, K]
        elif coeffs.dim() != 3:
            raise ValueError(
                f"fw_poly must be [K], [V, K] or [B, V, K], got {tuple(coeffs.shape)}"
            )
        # [B, V, K] -> Horner over the last dim, broadcasting on M via [B, V, 1, K].
        coeffs = coeffs.unsqueeze(2)  # [B, V, 1, K]
        r = torch.zeros_like(theta)
        for k in reversed(range(coeffs.shape[-1])):
            r = r * theta + coeffs[..., k]
        return r

    def project_ego_to_image(self, points_ego, image_transform) -> ProjectionResult:
        pts = _homogenize(points_ego)
        it = _as_image_transform(image_transform)
        w, h = it.wh
        T = self.t_camera_ego.to(device=pts.device, dtype=pts.dtype)
        # camera-frame points: [B, V, M, 4] then drop homogeneous w.
        cam = torch.einsum("bvij,mj->bvmi", T, pts)[..., :3]
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        rho = torch.sqrt(x * x + y * y).clamp(min=_DEPTH_EPS)
        theta = torch.atan2(rho, z)                     # incidence angle from +Z
        r = self._radius(theta)                         # pixel radius
        cx = torch.as_tensor(self.cx, device=cam.device, dtype=cam.dtype)
        cy = torch.as_tensor(self.cy, device=cam.device, dtype=cam.dtype)
        if cx.dim() > 0:
            cx = cx.unsqueeze(-1)  # [B, V] -> [B, V, 1] to broadcast on M
            cy = cy.unsqueeze(-1)
        u = cx + r * (x / rho)
        v = cy + r * (y / rho)
        uv_norm = torch.stack([u / w, v / h], dim=-1)

        depth = z                                       # optical-axis depth
        in_bounds = (
            (uv_norm[..., 0] >= 0) & (uv_norm[..., 0] <= 1)
            & (uv_norm[..., 1] >= 0) & (uv_norm[..., 1] <= 1)
        )
        # A fisheye sees rays beyond the +Z hemisphere (theta up to its real FOV,
        # which can exceed 90°), so do NOT gate on z > 0 — that would reimpose a
        # 180° ceiling and defeat the native f-theta operator. Gate on the lens
        # FOV (max_theta) when known; otherwise fall back to the +Z hemisphere as
        # a safe default (we cannot validate arbitrary wide rays without a bound).
        if self.max_theta is not None:
            # Accept a Python scalar, a list (e.g. reloaded from a manifest), or a
            # tensor. A per-view bound must broadcast against theta [B, V, M], so
            # add the M axis (mirrors cx/cy); a scalar broadcasts as-is.
            max_theta = torch.as_tensor(self.max_theta, device=theta.device, dtype=theta.dtype)
            if max_theta.dim() == 1:
                max_theta = max_theta.reshape(1, -1)   # [V] -> [1, V]
            if max_theta.dim() > 0:
                max_theta = max_theta.unsqueeze(-1)    # [B, V] -> [B, V, 1]
            mask = in_bounds & (theta <= max_theta)
        else:
            mask = in_bounds & (z > _DEPTH_EPS)
        meta = {"geometry_type": self.geometry_type,
                "model_input_size": it.model_input_size, "rectification": it.rectification}
        return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth, metadata=meta)
