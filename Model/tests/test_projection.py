"""Unit tests for the camera projection operator ABI (projection.py).

These exercise the operators in isolation (no backbone, no fusion) so a geometry
bug is localized to the projection math rather than the sampling loop.
"""

import math

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

    def test_from_KT_matches_combined_matrix_with_rotation(self, device):
        """from_KT(K, T) must equal PinholeProjection(K @ T[:3]) for a NON-trivial
        rotation+translation, and project a known point to the hand-computed pixel
        — catching a K/T order swap or a wrong T slice."""
        # K: fx=fy=100, principal point (128,128).
        K = torch.tensor([[[[100.0, 0.0, 128.0],
                            [0.0, 100.0, 128.0],
                            [0.0, 0.0, 1.0]]]], device=device)          # [1,1,3,3]
        # T: 90 deg yaw about camera Z + translation, ego->camera.
        c, s = 0.0, 1.0  # cos/sin(90deg)
        T = torch.tensor([[[[c, -s, 0.0, 0.3],
                            [s, c, 0.0, -0.2],
                            [0.0, 0.0, 1.0, 2.0],
                            [0.0, 0.0, 0.0, 1.0]]]], device=device)     # [1,1,4,4]
        proj_kt = PinholeProjection.from_KT(K, T)
        combined = torch.einsum("bvij,bvjk->bvik", K, T[:, :, :3, :])
        proj_m = PinholeProjection(combined)
        pt = _homo(torch.tensor([[0.5, -0.4, 3.0]], device=device))
        r_kt = proj_kt.project_ego_to_image(pt, 256)
        r_m = proj_m.project_ego_to_image(pt, 256)
        assert torch.allclose(r_kt.uv_norm, r_m.uv_norm, atol=1e-5), \
            "from_KT composition must equal the pre-combined matrix"
        # Hand-computed expected pixel: cam = T @ [x,y,z,1]; uvd = K @ cam[:3].
        cam = (T[0, 0] @ torch.tensor([0.5, -0.4, 3.0, 1.0], device=device))[:3]
        uvd = K[0, 0] @ cam
        u_exp = (uvd[0] / uvd[2]) / 256.0
        v_exp = (uvd[1] / uvd[2]) / 256.0
        assert torch.allclose(r_kt.uv_norm[0, 0, 0],
                              torch.stack([u_exp, v_exp]), atol=1e-5)


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
        """_radius must handle shared [K], per-view [V,K] and batched [B,V,K]
        fw_poly. Uses an OFF-axis point (theta>0) with DISTINCT per-view
        coefficients so coefficient order / Horner / per-view broadcasting are
        actually exercised (an on-axis theta=0 point would zero every term and
        prove nothing)."""
        T = self._identity_transform(device, v=3)
        pt = _homo(torch.tensor([[1.0, 0.0, 5.0]], device=device))  # off-axis
        rho, z = 1.0, 5.0
        theta = math.atan2(rho, z)
        # shared: every view uses r = 200*theta -> identical radius.
        shared = FThetaProjection(T, torch.tensor([0.0, 200.0], device=device),
                                  cx=128.0, cy=128.0)
        out_s = shared.project_ego_to_image(pt, 256).uv_norm
        assert out_s.shape == (1, 3, 1, 2)
        r_exp = 200.0 * theta
        u_exp = (128.0 + r_exp * (1.0 / rho)) / 256.0
        for v in range(3):
            assert out_s[0, v, 0, 0].item() == pytest.approx(u_exp, abs=1e-4)

        # per-view: distinct slopes 100/200/300 -> radii must differ per view and
        # match each view's own polynomial (proves per-view broadcasting).
        per_view = FThetaProjection(
            T, torch.tensor([[0.0, 100.0], [0.0, 200.0], [0.0, 300.0]], device=device),
            cx=128.0, cy=128.0)
        out_p = per_view.project_ego_to_image(pt, 256).uv_norm
        for v, slope in enumerate((100.0, 200.0, 300.0)):
            u_v = (128.0 + slope * theta * (1.0 / rho)) / 256.0
            assert out_p[0, v, 0, 0].item() == pytest.approx(u_v, abs=1e-4)
        # batched [1,V,K] must equal the per-view result.
        batched = FThetaProjection(
            T, torch.tensor([[[0.0, 100.0], [0.0, 200.0], [0.0, 300.0]]], device=device),
            cx=128.0, cy=128.0)
        out_b = batched.project_ego_to_image(pt, 256).uv_norm
        assert torch.allclose(out_b, out_p, atol=1e-5)

    def test_radius_rejects_bad_poly_rank(self, device):
        T = self._identity_transform(device)
        bad = torch.zeros(1, 1, 1, 2, device=device)  # 4-D fw_poly
        proj = FThetaProjection(T, bad, cx=128.0, cy=128.0)
        with pytest.raises(ValueError, match="fw_poly"):
            proj.project_ego_to_image(_homo(torch.tensor([[0.0, 0.0, 5.0]], device=device)), 256)

    def test_flu_ego_forward_maps_to_optical_center(self, device):
        """The convention boundary that actually matters: an ego-FLU point on the
        +X (forward) axis, pushed through the FLU->RDF transform, must land at the
        optical center with depth>0 — i.e. ego forward == camera +Z.

        The FLU->RDF rotation (x_cam=-y_ego, y_cam=-z_ego, z_cam=x_ego) is encoded
        INLINE here — independently of the source R_EGO_FLU_TO_CAM_OPT — so the
        test both validates the operator against the convention and does not drag
        in the dataset module's heavy deps (pandas/scipy) on CI.
        """
        # t_camera_ego = FLU-ego -> camera-optical (RDF), no translation.
        R = torch.tensor([[0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        T = torch.eye(4, device=device)
        T[:3, :3] = R
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

        # Pin down BOTH other axes so a sign flip on left/up cannot pass (the
        # forward axis alone underconstrains R_EGO_FLU_TO_CAM_OPT). ego-LEFT
        # (+Y) -> camera -X -> image left of center (u < 0.5); ego-UP (+Z) ->
        # camera -Y -> image top (v < 0.5).
        ego_left = _homo(torch.tensor([[5.0, 1.0, 0.0]], device=device))
        res_left = proj.project_ego_to_image(ego_left, 256)
        assert res_left.uv_norm[0, 0, 0, 0] < 0.5, \
            "ego-left (+Y) must project left of center (camera X=right)"
        ego_up = _homo(torch.tensor([[5.0, 0.0, 1.0]], device=device))
        res_up = proj.project_ego_to_image(ego_up, 256)
        assert res_up.uv_norm[0, 0, 0, 1] < 0.5, \
            "ego-up (+Z) must project above center (camera Y=down)"

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
        pytest.importorskip("pandas")  # calibration import pulls the dataset pkg
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
        # max_theta must EQUAL r2th at the exact corner radius (r=r_max/900), not
        # just be "some sane angle" — a wrong r2th eval would still be < pi.
        r_max = math.hypot(1920 / 2.0, 1080 / 2.0)   # corner from principal point
        expected_mt = r_max / 900.0                   # r2th slope in _Model
        mt = proj.max_theta.reshape(-1)
        assert mt[0].item() == pytest.approx(expected_mt, abs=1e-3)
        # Off-center point: normalization must use the NATIVE (W,H) per axis, and
        # the radius must be the UNSCALED native polynomial. Hand-compute both.
        # Identity pose -> ego==optical here; use an optical-frame point.
        pt = _homo(torch.tensor([[1.0, 0.5, 5.0]]))
        res = proj.project_ego_to_image(pt, 256)
        rho = math.hypot(1.0, 0.5)
        theta = math.atan2(rho, 5.0)
        r = 900.0 * theta                             # native pixels, unscaled
        u_exp = (960.0 + r * (1.0 / rho)) / 1920.0    # per-axis native normalize
        v_exp = (540.0 + r * (0.5 / rho)) / 1080.0
        assert res.uv_norm[0, 0, 0, 0].item() == pytest.approx(u_exp, abs=1e-4)
        assert res.uv_norm[0, 0, 0, 1].item() == pytest.approx(v_exp, abs=1e-4)

    def test_no_r2th_leaves_max_theta_none(self):
        pytest.importorskip("scipy")
        pytest.importorskip("pandas")  # calibration import pulls the dataset pkg
        from data_parsing.nvidia_physical_ai.calibration import build_ftheta_projection
        m = self._Model(800, 600)
        del m.r2th  # lens without a backward polynomial -> no derivable FOV bound
        proj = build_ftheta_projection(
            self._Intr({"c": m}), self._Extr({"c": self._pose()}), ["c"])
        assert proj.max_theta is None  # falls back to +Z hemisphere

    def test_mixed_rig_unbounded_lens_still_masks_behind_camera(self):
        """A rig where one lens has r2th (finite bound) and another does not
        (inf) must STILL mask behind-camera rays on the unbounded lens — the inf
        bound must fall back to the +Z hemisphere gate, not accept everything."""
        pytest.importorskip("scipy")
        pytest.importorskip("pandas")  # calibration import pulls the dataset pkg
        from data_parsing.nvidia_physical_ai.calibration import build_ftheta_projection
        bounded = self._Model(1920, 1080)             # has r2th -> finite bound
        unbounded = self._Model(1920, 1080)
        del unbounded.r2th                            # inf bound
        names = ["bounded", "unbounded"]
        proj = build_ftheta_projection(
            self._Intr({"bounded": bounded, "unbounded": unbounded}),
            self._Extr({n: self._pose() for n in names}), names)
        assert proj.max_theta is not None             # at least one lens is bounded
        # Point directly BEHIND the rig (optical z<0, on-axis): must be masked on
        # BOTH views, including the unbounded (inf) one.
        behind = _homo(torch.tensor([[0.0, 0.0, -5.0]]))
        res = proj.project_ego_to_image(behind, 256)
        assert not res.valid_mask[0, 0, 0], "bounded lens must mask behind-camera"
        assert not res.valid_mask[0, 1, 0], \
            "unbounded (inf) lens must still mask behind-camera via z>0 fallback"

    def test_radius_monotonic_over_fov(self):
        """A calibrated forward polynomial must be radially monotonic over its FOV
        so two incidence angles never alias to the same pixel radius. This is a
        geometry contract that shape/sidedness tests do not catch."""
        T = torch.eye(4).reshape(1, 1, 4, 4)
        fw_poly = torch.tensor([0.0, 900.0])          # r = 900*theta (monotonic)
        proj = FThetaProjection(T, fw_poly, cx=960.0, cy=540.0, image_wh=(1920.0, 1080.0))
        thetas = torch.linspace(0.0, 1.2, 200)
        r = proj._radius(thetas.reshape(1, 1, -1)).reshape(-1)
        assert (r[1:] - r[:-1] >= 0).all(), "r(theta) must be non-decreasing over the FOV"
