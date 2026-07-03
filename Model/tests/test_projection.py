"""Unit tests for the camera projection operator ABI (projection.py).

These exercise the operators in isolation (no backbone, no fusion) so a geometry
bug is localized to the projection math rather than the sampling loop.
"""

import pytest
import torch

from model_components.view_fusion.projection import (
    GEOMETRY_FTHETA,
    GEOMETRY_PSEUDO,
    GEOMETRY_RECTIFIED_PINHOLE,
    FThetaProjection,
    PinholeProjection,
    ProjectionResult,
    PseudoProjection,
)


def _homo(points):
    """[M, 3] ego points -> [M, 4] homogeneous."""
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype)
    return torch.cat([points, ones], dim=-1)


class TestPinholeProjection:
    def test_shape_and_view_count(self, device):
        proj = PinholeProjection(torch.randn(2, 5, 3, 4, device=device))
        assert proj.num_views == 5
        pts = _homo(torch.randn(7, 3, device=device))
        res = proj.project_ego_to_image(pts, 256)
        assert isinstance(res, ProjectionResult)
        assert res.uv_norm.shape == (2, 5, 7, 2)
        assert res.valid_mask.shape == (2, 5, 7)
        assert res.depth.shape == (2, 5, 7)

    def test_center_projects_to_image_center(self, device):
        # fx=fy=112, cx=cy=112, z passthrough, 224.
        # ego point on the optical axis (x=y=0, z=2) -> pixel (112,112) -> 0.5,0.5.
        cam = torch.zeros(1, 1, 3, 4, device=device)
        cam[0, 0, 0, 0] = 112.0
        cam[0, 0, 0, 2] = 112.0
        cam[0, 0, 1, 1] = 112.0
        cam[0, 0, 1, 2] = 112.0
        cam[0, 0, 2, 2] = 1.0
        res = PinholeProjection(cam).project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 2.0]], device=device)), 224)
        assert res.valid_mask[0, 0, 0]
        assert torch.allclose(res.uv_norm[0, 0, 0], torch.tensor([0.5, 0.5], device=device), atol=1e-4)

    def test_behind_camera_masked(self, device):
        cam = torch.zeros(1, 1, 3, 4, device=device)
        cam[0, 0, 0, 0] = 112.0
        cam[0, 0, 1, 1] = 112.0
        cam[0, 0, 2, 2] = -1.0    # negate z -> depth < 0
        cam[0, 0, 2, 3] = -100.0
        res = PinholeProjection(cam).project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 2.0]], device=device)), 224)
        assert not res.valid_mask.any()

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="3, 4"):
            PinholeProjection(torch.randn(2, 5, 4, 4))

    def test_rejects_bad_geometry_label(self):
        with pytest.raises(ValueError, match="geometry_type"):
            PinholeProjection(torch.randn(1, 1, 3, 4), geometry_type=GEOMETRY_FTHETA)

    def test_rectified_pinhole_label_allowed(self):
        proj = PinholeProjection(torch.randn(1, 1, 3, 4), geometry_type=GEOMETRY_RECTIFIED_PINHOLE)
        assert proj.geometry_type == GEOMETRY_RECTIFIED_PINHOLE


class TestPseudoProjection:
    def test_view_count_agnostic(self, device):
        shared = torch.randn(3, 4, device=device)
        for v in (1, 4, 7, 8):
            res = PseudoProjection(shared, num_views=v).project_ego_to_image(
                _homo(torch.randn(5, 3, device=device)), 256)
            assert res.uv_norm.shape == (1, v, 5, 2)   # batch-independent prior
        assert PseudoProjection(shared, num_views=8).geometry_type == GEOMETRY_PSEUDO

    def test_coords_in_unit_range(self, device):
        # sigmoid keeps pseudo coords within (0, 1) even for unbounded matrices.
        res = PseudoProjection(torch.randn(3, 4, device=device) * 100, num_views=3).project_ego_to_image(
            _homo(torch.randn(6, 3, device=device)), 256)
        assert (res.uv_norm >= 0).all() and (res.uv_norm <= 1).all()

    def test_gradient_flows_to_shared_matrix(self, device):
        shared = torch.randn(3, 4, device=device, requires_grad=True)
        res = PseudoProjection(shared, num_views=4).project_ego_to_image(
            _homo(torch.randn(5, 3, device=device)), 256)
        res.uv_norm.sum().backward()
        assert shared.grad is not None and shared.grad.abs().max() > 0

    def test_rejects_per_view_matrix(self, device):
        # A [V,3,4] tensor is a misuse (the prior is view-independent) and would
        # crash cryptically at reshape; reject it at construction.
        with pytest.raises(ValueError, match=r"\[3, 4\]"):
            PseudoProjection(torch.zeros(2, 3, 4, device=device), num_views=4)

    def test_accepts_leading_one_matrix(self, device):
        res = PseudoProjection(torch.randn(1, 3, 4, device=device), num_views=3).project_ego_to_image(
            _homo(torch.randn(4, 3, device=device)), 256)
        assert res.uv_norm.shape == (1, 3, 4, 2)


class TestFThetaProjection:
    def _identity_transform(self, device, v=1):
        T = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, v, 4, 4).contiguous()
        return T

    def test_on_axis_maps_to_principal_point(self, device):
        # theta=0 on the optical axis -> radius r(0)=fw_poly[0]; with fw_poly[0]=0
        # the point lands exactly at (cx, cy).
        T = self._identity_transform(device)
        fw_poly = torch.tensor([0.0, 200.0], device=device)  # r = 200*theta
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0)
        # ego point straight ahead along +Z (optical axis): x=y=0, z=5
        res = proj.project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 5.0]], device=device)), 256)
        assert res.valid_mask[0, 0, 0]
        assert torch.allclose(res.uv_norm[0, 0, 0], torch.tensor([0.5, 0.5], device=device), atol=1e-4)

    def test_off_axis_radius_grows_with_theta(self, device):
        T = self._identity_transform(device)
        fw_poly = torch.tensor([0.0, 200.0], device=device)
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0)
        # a point off-axis in +x should map to u > cx (right of centre)
        res = proj.project_ego_to_image(_homo(torch.tensor([[1.0, 0.0, 5.0]], device=device)), 256)
        assert res.uv_norm[0, 0, 0, 0] > 0.5

    def test_max_theta_masks_wide_rays(self, device):
        T = self._identity_transform(device)
        fw_poly = torch.tensor([0.0, 100.0], device=device)
        # a point nearly perpendicular to the axis has theta ~ pi/2; cap below it.
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0, max_theta=0.1)
        res = proj.project_ego_to_image(_homo(torch.tensor([[10.0, 0.0, 0.5]], device=device)), 256)
        assert not res.valid_mask.any()

    def test_behind_camera_masked(self, device):
        T = self._identity_transform(device)
        fw_poly = torch.tensor([0.0, 100.0], device=device)
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0)
        res = proj.project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, -5.0]], device=device)), 256)
        assert not res.valid_mask.any()

    def test_wide_fov_admits_rays_beyond_hemisphere(self, device):
        """With max_theta > 90 deg, a ray with z < 0 (theta > 90 deg) must be
        admissible — the native fisheye must NOT be capped at a 180 deg FOV."""
        T = self._identity_transform(device)
        # small radius so the wide ray still lands inside the image bounds.
        fw_poly = torch.tensor([0.0, 20.0], device=device)
        # ~100 deg FOV half-angle; a ray at theta ~ 95 deg has z < 0.
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0, max_theta=1.8)
        # x large, z slightly negative -> theta = atan2(rho, z) in (90, 180) deg.
        res = proj.project_ego_to_image(_homo(torch.tensor([[1.0, 0.0, -0.05]], device=device)), 256)
        assert res.valid_mask.any(), \
            "max_theta fisheye wrongly rejected a valid ray past the +Z hemisphere"
        # and the same ray is rejected once it exceeds the FOV cap.
        narrow = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0, max_theta=1.0)
        res2 = narrow.project_ego_to_image(_homo(torch.tensor([[1.0, 0.0, -0.05]], device=device)), 256)
        assert not res2.valid_mask.any(), "ray beyond max_theta should be masked"

    def test_rejects_bad_transform_shape(self):
        with pytest.raises(ValueError, match="4, 4"):
            FThetaProjection(torch.randn(1, 1, 3, 4), torch.tensor([0.0, 1.0]), 1.0, 1.0)

    def test_tensor_max_theta_moves_with_to_and_projects(self, device):
        """A tensor max_theta must follow .to(device) and be usable in project()
        without a device mismatch."""
        T = self._identity_transform(device)
        fw_poly = torch.tensor([0.0, 100.0], device=device)
        # Construct on CPU with a CPU tensor max_theta, then move to device.
        proj = FThetaProjection(
            T.cpu(), fw_poly.cpu(), cx=128.0, cy=128.0,
            max_theta=torch.tensor(1.0),
        ).to(device)
        assert proj.max_theta.device.type == device.type
        # project() must run (comparison theta <= max_theta on the same device).
        res = proj.project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 5.0]], device=device)), 256)
        assert res.uv_norm.shape == (1, 1, 1, 2)

    def test_per_view_max_theta_broadcasts(self, device):
        """A per-view [B, V] max_theta must broadcast against theta [B, V, M]."""
        T = self._identity_transform(device, v=3)
        fw_poly = torch.tensor([0.0, 20.0], device=device)
        # Different FOV per camera; shape [B=1, V=3].
        max_theta = torch.tensor([[0.1, 1.8, 1.8]], device=device)
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0, max_theta=max_theta)
        pts = _homo(torch.tensor([[1.0, 0.0, -0.05]], device=device))  # wide ray
        res = proj.project_ego_to_image(pts, 256)  # must not raise
        assert res.valid_mask.shape == (1, 3, 1)
        # cam 0 (max_theta 0.1) rejects the wide ray; cams 1,2 (1.8) admit it.
        assert not res.valid_mask[0, 0, 0]
        assert res.valid_mask[0, 1, 0] and res.valid_mask[0, 2, 0]

    def test_to_spec_shared_poly_and_tensor_max_theta_json_able(self, device):
        """to_spec must keep a shared [K] polynomial whole and emit a JSON-able
        max_theta (not a raw tensor)."""
        import json
        T = self._identity_transform(device, v=2)
        fw_poly = torch.tensor([0.0, 300.0, -5.0, 0.1], device=device)  # shared [K]
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0,
                                max_theta=torch.tensor(1.5, device=device))
        spec = proj.to_spec()
        assert spec["fw_poly"] == [0.0, 300.0, -5.0, 0.1], "shared poly truncated"
        json.dumps(spec)  # must not raise (tensor max_theta scalarized)

    def test_cpu_operator_projects_cuda_points(self, device):
        """A CPU operator must project CUDA points (params coerced to device)."""
        if device.type != "cuda":
            pytest.skip("needs CUDA")
        T = torch.eye(4).reshape(1, 1, 4, 4)  # CPU
        fw_poly = torch.tensor([0.0, 100.0])  # CPU
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0)  # all CPU
        pts = _homo(torch.tensor([[0.0, 0.0, 5.0]], device=device))  # CUDA
        res = proj.project_ego_to_image(pts, 256)  # must not raise
        assert res.uv_norm.device.type == "cuda"
