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
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
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
        # Seed deterministically: the pseudo path passes coords through sigmoid,
        # whose gradient vanishes where it saturates, so an unseeded random draw
        # can make d(sum)/d(shared) round to ~0 and flake. A small matrix keeps
        # projected values near 0 (sigmoid's high-gradient region).
        torch.manual_seed(0)
        shared = (torch.randn(3, 4, device=device) * 0.05).requires_grad_(True)
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
        # Full polynomial preserved (not truncated to the first coefficient).
        # float32 round-trip, so compare approximately.
        assert isinstance(spec["fw_poly"], list) and len(spec["fw_poly"]) == 4, \
            "shared poly truncated"
        assert spec["fw_poly"] == pytest.approx([0.0, 300.0, -5.0, 0.1], abs=1e-5)
        json.dumps(spec)  # must not raise (tensor max_theta scalarized)

    def test_radius_accepts_shared_and_per_view_poly(self, device):
        """_radius must handle a shared [K], per-view [V,K], and batched [B,V,K]
        fw_poly identically for an on-axis point (round-2 review regression)."""
        T = self._identity_transform(device, v=3)
        pt = _homo(torch.tensor([[0.0, 0.0, 5.0]], device=device))  # on-axis
        shared = FThetaProjection(T, torch.tensor([0.0, 200.0], device=device),
                                  cx=128.0, cy=128.0)
        per_view = FThetaProjection(T, torch.tensor([[0.0, 200.0]] * 3, device=device),
                                    cx=128.0, cy=128.0)
        batched = FThetaProjection(T, torch.tensor([[[0.0, 200.0]] * 3], device=device),
                                   cx=128.0, cy=128.0)
        outs = [p.project_ego_to_image(pt, 256).uv_norm for p in (shared, per_view, batched)]
        for o in outs:
            assert o.shape == (1, 3, 1, 2)
            assert torch.allclose(o[0, 0, 0], torch.tensor([0.5, 0.5], device=device), atol=1e-4)

    def test_radius_rejects_bad_poly_rank(self, device):
        T = self._identity_transform(device)
        bad = torch.zeros(1, 1, 1, 2, device=device)  # 4-D fw_poly
        proj = FThetaProjection(T, bad, cx=128.0, cy=128.0)
        with pytest.raises(ValueError, match="fw_poly"):
            proj.project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 5.0]], device=device)), 256)

    def test_flu_ego_forward_maps_to_optical_center(self, device):
        """The convention boundary that actually matters: an ego-FLU point on the
        +X (forward) axis, pushed through the FLU->RDF (R_EGO_FLU_TO_CAM_OPT)
        transform, must land at the optical center with depth>0 — i.e. ego
        forward == camera +Z. Uses the SDK axis matrix, not identity T."""
        from data_parsing.nvidia_physical_ai.calibration import R_EGO_FLU_TO_CAM_OPT

        # t_camera_ego = FLU-ego -> camera-optical (RDF), no translation.
        T = torch.eye(4, device=device)
        T[:3, :3] = torch.tensor(R_EGO_FLU_TO_CAM_OPT, dtype=torch.float32, device=device)
        T = T.reshape(1, 1, 4, 4)
        # r(theta)=200*theta so an on-axis (theta=0) point lands exactly at (cx,cy).
        fw_poly = torch.tensor([0.0, 200.0], device=device)
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0)
        # ego-FLU point straight AHEAD: x=+5 (forward), y=0 (no left), z=0 (no up).
        ego_forward = _homo(torch.tensor([[5.0, 0.0, 0.0]], device=device))
        res = proj.project_ego_to_image(ego_forward, 256)
        assert res.valid_mask[0, 0, 0], "ego-forward should be visible (depth>0)"
        assert res.depth[0, 0, 0] > 0, "ego-forward must have positive camera depth"
        assert torch.allclose(res.uv_norm[0, 0, 0],
                              torch.tensor([0.5, 0.5], device=device), atol=1e-4), \
            "ego-FLU forward must project to the optical center after FLU->RDF"

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


class TestBuildFThetaFromCalibration:
    """build_ftheta_projection wires native (W,H) and a real FOV bound (max_theta
    from r2th) — points 1 & 4 of the reviewer's feedback."""

    class _Model:
        def __init__(self, w, h):
            import numpy as np
            self.width, self.height = w, h
            self.principal_point = np.array([w / 2.0, h / 2.0])
            # forward theta->radius and its inverse radius->theta, sized so the
            # farthest image corner maps to a realistic FOV (< pi). For a 1920x1080
            # frame the corner radius is ~1101 px; slope ~1/900 -> theta ~1.22 rad.
            self.th2r = np.polynomial.Polynomial([0.0, 900.0])   # r = 900*theta
            self.r2th = np.polynomial.Polynomial([0.0, 1 / 900.0])  # theta = r/900

    class _Intr:
        def __init__(self, models):
            self.camera_models = models

    class _Extr:
        def __init__(self, poses):
            self.sensor_poses = poses

    def _pose(self):
        import scipy.spatial.transform as spt
        import numpy as np
        return spt.RigidTransform.from_components(
            rotation=spt.Rotation.identity(), translation=np.zeros(3))

    def test_native_wh_and_max_theta_from_r2th(self):
        pytest.importorskip("scipy")
        from data_parsing.nvidia_physical_ai.calibration import build_ftheta_projection
        # Non-square native frame: normalization must use native (W,H), not 256.
        names = ["cam_a", "cam_b"]
        models = {n: self._Model(1920, 1080) for n in names}
        poses = {n: self._pose() for n in names}
        proj = build_ftheta_projection(self._Intr(models), self._Extr(poses), names)
        # image_wh carries the native size, per view.
        assert tuple(proj.image_wh.shape) == (1, 2, 2)
        assert float(proj.image_wh[0, 0, 0]) == 1920.0
        assert float(proj.image_wh[0, 0, 1]) == 1080.0
        # max_theta derived from r2th at the corner radius (finite, sane FOV).
        assert proj.max_theta is not None
        mt = proj.max_theta.reshape(-1)
        assert (mt > 0).all() and (mt < 3.15).all()
        # On-axis ego-forward projects to the principal point (0.5, 0.5) in the
        # native frame regardless of the non-square aspect.
        res = proj.project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 5.0]])), 256)
        assert torch.allclose(res.uv_norm[0, 0, 0], torch.tensor([0.5, 0.5]), atol=1e-4)

    def test_no_r2th_leaves_max_theta_none(self):
        pytest.importorskip("scipy")
        from data_parsing.nvidia_physical_ai.calibration import build_ftheta_projection
        m = self._Model(800, 600)
        del m.r2th  # lens without a backward polynomial -> no derivable FOV bound
        proj = build_ftheta_projection(
            self._Intr({"c": m}), self._Extr({"c": self._pose()}), ["c"])
        assert proj.max_theta is None  # falls back to +Z hemisphere
