"""Differentiable camera projection operators for BEV view fusion.

The BEV fusion module's contract with geometry is NOT a fixed ``[B, V, 3, 4]``
matrix. It is a *projection operator*: something that maps ego-frame BEV
reference points to sampling coordinates plus a visibility mask on each camera's
model-input image plane. A pinhole ``K @ T`` matrix is only ONE such operator
(the linear fast path). Fisheye (f-theta) cameras need a non-linear operator,
and a calibration-free run needs a learnable pseudo operator. All three expose
the same :meth:`project` so ``BEVViewFusion`` never branches on camera model.

Coordinate convention (shared with :class:`BEVViewFusion`): reference points are
in the ego/vehicle frame, X=forward, Y=left, Z=up, in metres. :meth:`project`
returns pixel coordinates normalized to ``[0, 1]`` by ``image_size`` (the
model-input image is square, produced by the shard packing / backbone
transform), a visibility ``mask`` (in front of camera AND within image bounds),
and the per-point ``depth`` for diagnostics.

``geometry_type`` is an explicit, honest label of what geometry produced the
result — "pinhole", "rectified_pinhole", "ftheta", or "pseudo". A caller that
wants the calibration-free path must ASK for "pseudo"; the fusion module never
silently invents geometry.

Reference:
    - BEVFormer (Li et al., ECCV 2022): spatial cross-attention. Its essence is
      "each BEV query samples relevant regions from each camera view"; the
      projection being a pinhole matrix is an implementation detail, not the
      contract.
    - NVIDIA PhysicalAI-AV sensor model: native f-theta (fisheye), motivating a
      non-linear projection operator rather than a pinhole-only ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
            (metres for calibrated cameras). Kept for diagnostics and possible
            future depth supervision; not required by the sampler.
    """

    uv_norm: torch.Tensor
    valid_mask: torch.Tensor
    depth: Optional[torch.Tensor] = None


def _finalize_pinhole(projected: torch.Tensor, image_size: float) -> ProjectionResult:
    """Perspective-divide, normalize by ``image_size`` and build the mask.

    Shared by :class:`PinholeProjection` and :class:`FThetaProjection`'s
    rectified fallback. ``projected`` is ``[Bp, V, M, 3]`` = ``[u*d, v*d, d]``.
    """
    depth = projected[..., 2]                         # [Bp, V, M]
    valid_depth = depth > _DEPTH_EPS                  # in front of the camera
    depth_safe = depth.clamp(min=_DEPTH_EPS).unsqueeze(-1)  # avoid div-by-zero
    uv = projected[..., :2] / depth_safe              # pixel coords [Bp, V, M, 2]
    uv_norm = uv / image_size
    in_bounds = (
        (uv_norm[..., 0] >= 0) & (uv_norm[..., 0] <= 1)
        & (uv_norm[..., 1] >= 0) & (uv_norm[..., 1] <= 1)
    )
    mask = valid_depth & in_bounds
    return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth)


class PinholeProjection:
    """Linear ego-to-pixel projection: the ``K @ T`` fast path.

    Wraps a combined ``[B, V, 3, 4]`` ego-to-pixel matrix (intrinsic @ extrinsic,
    already scaled to the model-input image size by the dataset parser). This is
    the operator that reproduces the historical ``camera_params`` behaviour, now
    named for what it is. For a fisheye camera rectified to a pinhole model, pass
    ``geometry_type="rectified_pinhole"`` so metadata stays honest about the FOV
    loss that rectification implies.
    """

    def __init__(self, matrix: torch.Tensor, geometry_type: str = GEOMETRY_PINHOLE):
        if matrix.dim() != 4 or matrix.shape[-2:] != (3, 4):
            raise ValueError(
                f"PinholeProjection matrix must be [B, V, 3, 4], got {tuple(matrix.shape)}"
            )
        if geometry_type not in (GEOMETRY_PINHOLE, GEOMETRY_RECTIFIED_PINHOLE):
            raise ValueError(
                f"PinholeProjection geometry_type must be 'pinhole' or "
                f"'rectified_pinhole', got {geometry_type!r}"
            )
        self.matrix = matrix
        self.geometry_type = geometry_type

    @property
    def num_views(self) -> int:
        return self.matrix.shape[1]

    def to(self, device) -> "PinholeProjection":
        return PinholeProjection(self.matrix.to(device), geometry_type=self.geometry_type)

    def to_spec(self) -> dict:
        """Serialize to a JSON-able manifest spec (batch dim dropped)."""
        return {
            "type": self.geometry_type,
            "matrix": self.matrix[0].detach().cpu().tolist(),  # [V, 3, 4]
        }

    def project(self, points_ego_homo: torch.Tensor, image_size: float) -> ProjectionResult:
        """Project homogeneous ego points ``[M, 4]`` onto each camera.

        Returns coords/mask shaped ``[B, V, M, ...]`` where ``B, V`` come from
        the projection matrix — this is where runtime ``V`` is derived, not from
        any construction-time ``num_views``.
        """
        proj = self.matrix.to(points_ego_homo.dtype)
        # out[b, v, m, i] = sum_j proj[b, v, i, j] * points[m, j]
        projected = torch.einsum("bvij,mj->bvmi", proj, points_ego_homo)
        return _finalize_pinhole(projected, image_size)


class PseudoProjection:
    """Learnable calibration-free fallback (shape-testing / ablation only).

    This is NOT real geometry. It is a learnable spatial prior that lets the
    module run without calibration. A single shared ``[3, 4]`` matrix is expanded
    to ``V`` views at projection time, so one instance serves any view count.
    Pixel coordinates are squashed with ``sigmoid`` (the raw matrix is unbounded)
    rather than divided by ``image_size``.

    The learnable tensor is owned by :class:`BEVViewFusion` (so it is a leaf on
    the module and the optimizer sees it); this operator just wraps it per
    forward. Callers must explicitly request ``geometry_type="pseudo"`` — the
    fusion module does not fall into this path silently on behalf of a caller
    that meant to pass real calibration.
    """

    geometry_type = GEOMETRY_PSEUDO

    def __init__(self, matrix: torch.Tensor, num_views: int):
        # matrix: shared [3, 4] learnable prior (a leaf Parameter owned upstream).
        if matrix.shape[-2:] != (3, 4):
            raise ValueError(
                f"PseudoProjection matrix must end in [3, 4], got {tuple(matrix.shape)}"
            )
        self.matrix = matrix
        self.num_views = num_views

    def project(self, points_ego_homo: torch.Tensor, image_size: float) -> ProjectionResult:
        # Expand the shared [3, 4] prior to [1, V, 3, 4]: batch dim 1 broadcasts
        # across the real batch in the sampling loop (the prior is batch- and
        # view-independent by construction).
        proj = self.matrix.reshape(3, 4).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 4]
        proj = proj.expand(1, self.num_views, 3, 4).to(points_ego_homo.dtype)
        projected = torch.einsum("bvij,mj->bvmi", proj, points_ego_homo)  # [1, V, M, 3]

        depth = projected[..., 2]
        valid_depth = depth > _DEPTH_EPS
        depth_safe = depth.clamp(min=_DEPTH_EPS).unsqueeze(-1)
        uv = projected[..., :2] / depth_safe
        # Unbounded pseudo outputs → sigmoid to keep coords in (0, 1). in-bounds
        # is then trivially satisfied, so the mask reduces to the depth check.
        uv_norm = uv.sigmoid()
        mask = valid_depth
        return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth)


class FThetaProjection:
    """Non-linear f-theta (fisheye) projection, native to NVIDIA PhysicalAI-AV.

    Maps ego points to pixels WITHOUT flattening the fisheye to a pinhole, so a
    wide-FOV camera keeps its full field of view (no rectification FOV loss). The
    forward polynomial maps the incidence angle ``theta`` (angle between the
    camera-frame ray and the optical +Z axis) to a pixel radius::

        r(theta) = c0 + c1*theta + c2*theta^2 + ...
        u = cx + r * (x_cam / rho),  v = cy + r * (y_cam / rho)

    where ``rho = sqrt(x_cam^2 + y_cam^2)``. This matches the SDK's
    ``FThetaCameraModel.ray2pixel`` family.

    Parameters (all pre-scaled to the model-input image by the parser):
        t_camera_ego: ``[B, V, 4, 4]`` ego->camera rigid transform.
        fw_poly: ``[K]`` or ``[B, V, K]`` forward polynomial coefficients
            (ascending powers of theta), radius in pixels.
        cx, cy: principal point in pixels, scalar or ``[B, V]``.
        max_theta: optional incidence-angle cutoff (radians); points beyond it
            are masked out (outside the lens FOV).
    """

    geometry_type = GEOMETRY_FTHETA

    def __init__(self, t_camera_ego, fw_poly, cx, cy, max_theta=None):
        if t_camera_ego.dim() != 4 or t_camera_ego.shape[-2:] != (4, 4):
            raise ValueError(
                f"FThetaProjection t_camera_ego must be [B, V, 4, 4], "
                f"got {tuple(t_camera_ego.shape)}"
            )
        self.t_camera_ego = t_camera_ego
        self.fw_poly = fw_poly
        self.cx = cx
        self.cy = cy
        self.max_theta = max_theta

    @property
    def num_views(self) -> int:
        return self.t_camera_ego.shape[1]

    def to(self, device) -> "FThetaProjection":
        def _mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return FThetaProjection(
            self.t_camera_ego.to(device), _mv(self.fw_poly),
            _mv(self.cx), _mv(self.cy), max_theta=_mv(self.max_theta),
        )

    def to_spec(self) -> dict:
        """Serialize to a JSON-able manifest spec (batch dim dropped)."""
        def _l(x):
            return x[0].detach().cpu().tolist() if torch.is_tensor(x) and x.dim() > 0 else x
        return {
            "type": self.geometry_type,
            "t_camera_ego": self.t_camera_ego[0].detach().cpu().tolist(),  # [V,4,4]
            "fw_poly": _l(self.fw_poly),                                   # [V,K]
            "cx": _l(self.cx),                                             # [V]
            "cy": _l(self.cy),                                             # [V]
            "max_theta": self.max_theta,
        }

    def _radius(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the forward polynomial r(theta) via Horner's method.

        ``fw_poly`` is ascending-power coefficients; broadcast either as a shared
        ``[K]`` vector or per-view ``[B, V, K]`` (unsqueezed to ``[B, V, 1, K]``).
        """
        coeffs = self.fw_poly.to(theta.dtype)
        if coeffs.dim() == 1:
            powers = coeffs  # [K]
            r = torch.zeros_like(theta)
            for c in reversed(powers.unbind(0)):
                r = r * theta + c
            return r
        # per-view [B, V, K] -> Horner over the last dim, broadcasting on M.
        coeffs = coeffs.unsqueeze(2)  # [B, V, 1, K]
        r = torch.zeros_like(theta)
        for k in reversed(range(coeffs.shape[-1])):
            r = r * theta + coeffs[..., k]
        return r

    def project(self, points_ego_homo: torch.Tensor, image_size: float) -> ProjectionResult:
        T = self.t_camera_ego.to(points_ego_homo.dtype)
        # camera-frame points: [B, V, M, 4] then drop homogeneous w.
        cam = torch.einsum("bvij,mj->bvmi", T, points_ego_homo)[..., :3]
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        rho = torch.sqrt(x * x + y * y).clamp(min=_DEPTH_EPS)
        theta = torch.atan2(rho, z)                     # incidence angle from +Z
        r = self._radius(theta)                         # pixel radius
        cx = self.cx if torch.is_tensor(self.cx) else torch.as_tensor(self.cx, device=cam.device, dtype=cam.dtype)
        cy = self.cy if torch.is_tensor(self.cy) else torch.as_tensor(self.cy, device=cam.device, dtype=cam.dtype)
        if torch.is_tensor(cx) and cx.dim() > 0:
            cx = cx.unsqueeze(-1)  # [B, V] -> [B, V, 1] to broadcast on M
            cy = cy.unsqueeze(-1)
        u = cx + r * (x / rho)
        v = cy + r * (y / rho)
        uv_norm = torch.stack([u, v], dim=-1) / image_size

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
            max_theta = self.max_theta
            if torch.is_tensor(max_theta):
                max_theta = max_theta.to(device=theta.device, dtype=theta.dtype)
            mask = in_bounds & (theta <= max_theta)
        else:
            mask = in_bounds & (z > _DEPTH_EPS)
        return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth)
