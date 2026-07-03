"""Build camera projection operators from NVIDIA PhysicalAI-AV calibration.

The dataset ships real calibration as two features (via the SDK):
  - ``camera_intrinsics``  -> ``physical_ai_av.calibration.CameraIntrinsics``
    holding one ``FThetaCameraModel`` per camera (native fisheye).
  - ``sensor_extrinsics``  -> ``SensorExtrinsics`` holding a
    ``scipy...RigidTransform`` sensor->rig(ego) pose per sensor.

This module converts those into the projection operators BEV fusion consumes,
WITHOUT flattening the fisheye to a pinhole (no FOV loss): an
:class:`FThetaProjection` per rig, plus the ego->camera transform.

Frame conventions (critical):
  - BEV reference points are in the ego frame X=forward, Y=left, Z=up (FLU),
    per BEVViewFusion's contract.
  - The SDK camera frame is X=right, Y=down, Z=forward (out of the lens), per
    ``camera_models.CameraModel.ray2pixel``'s docstring.
  - The SDK extrinsic is a sensor->rig(ego) RigidTransform; the rig frame for
    this dataset is the standard AV convention X=forward, Y=left, Z=up.
  We therefore compose:  T_camopt<-ego = R_rig->camopt @ inv(sensor_pose),
  where R_rig->camopt maps ego-FLU axes to the camera optical (RDF-like) axes:
      x_camopt = -y_ego   (right      = -left)
      y_camopt = -z_ego   (down       = -up)
      z_camopt =  x_ego   (forward    =  forward)
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..calibration import scale_intrinsic  # noqa: F401  (shared, used by pinhole path)

# Ego(FLU) -> camera-optical(RDF) axis permutation, as a 3x3 rotation.
#   x_cam = -y_ego, y_cam = -z_ego, z_cam = x_ego
R_EGO_FLU_TO_CAM_OPT = np.array(
    [[0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0],
     [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def _ego_to_camera_transform(sensor_pose, sensor_frame_is_optical: bool = True) -> np.ndarray:
    """Compose the 4x4 ego->camera-optical transform for one sensor.

    ``sensor_pose`` is the SDK's sensor->rig(ego) RigidTransform; ``inv`` of it is
    ego->sensor. The subtlety is what the SDK's per-camera *sensor* frame is:

    - ``sensor_frame_is_optical=True`` (DEFAULT): the camera's sensor frame IS
      the optical frame the FThetaCameraModel expects (X=right, Y=down,
      Z=forward). This is the standard AV convention where intrinsics and
      extrinsics are shipped to compose directly, so ``inv(sensor_pose)`` already
      maps ego->optical and NO extra axis rotation is applied. Applying one here
      would double-rotate and systematically skew every projection.
    - ``sensor_frame_is_optical=False``: the sensor frame is rig-aligned FLU
      (X=forward, Y=left, Z=up), so we additionally rotate FLU->optical.

    The SDK does not document which convention its ``sensor_extrinsics`` use and
    ships no camera<->sensor rotation, so the default follows the direct-compose
    convention. This MUST be validated on real data (project a known forward ego
    point and confirm it lands near the forward camera's image centre, depth>0)
    before the f-theta path is trusted quantitatively (see #77).
    """
    M = np.asarray(sensor_pose.as_matrix(), dtype=np.float64)  # sensor->ego, 4x4
    ego_to_sensor = np.linalg.inv(M)                           # ego->sensor
    if sensor_frame_is_optical:
        return ego_to_sensor                                   # already ego->optical
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = R_EGO_FLU_TO_CAM_OPT
    return R @ ego_to_sensor                                   # FLU sensor -> optical


def _corner_max_theta(model) -> float | None:
    """FOV bound (radians) = the incidence angle at the farthest image corner.

    Uses the SDK's backward polynomial ``r2th`` (pixel radius -> theta) evaluated
    at the corner radius from the principal point. This is a principled per-lens
    FOV cutoff derived from the published calibration, so wide f-theta rays are
    admitted up to the real lens FOV and no farther. Returns None if the model
    exposes no usable ``r2th`` (then the operator falls back to the +Z
    hemisphere, i.e. f-theta without an explicit FOV bound).
    """
    r2th = getattr(model, "r2th", None)
    if r2th is None:
        return None
    w, h = int(model.width), int(model.height)
    cx, cy = float(model.principal_point[0]), float(model.principal_point[1])
    # Farthest corner distance from the principal point, in native pixels.
    corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
    r_max = max(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 for px, py in corners)
    try:
        theta = float(r2th(r_max))
    except Exception:
        return None
    # A sane FOV bound is positive and within a hemisphere-plus; reject
    # degenerate polynomial extrapolation.
    if not (0.0 < theta < math.pi):
        return None
    return theta


def build_ftheta_projection(
    intrinsics,
    extrinsics,
    camera_names,
    polynomial_degree: int = 4,
    sensor_frame_is_optical: bool = True,
):
    """Construct a native-frame :class:`FThetaProjection` from SDK calibration.

    The f-theta parameters stay in the camera's NATIVE pixel resolution; the
    operator normalizes projected ``(u, v)`` per-axis by that native ``(W, H)``.
    Under a plain resize to the model input this normalization is EXACT (the
    model size cancels), so a non-square / anisotropic resize needs no radius
    approximation — the isotropic radial polynomial is never mean-scaled.

    A per-lens FOV bound ``max_theta`` is derived from the SDK's ``r2th`` at the
    image corner, so wide f-theta rays are admitted up to the real lens FOV.

    Args:
        intrinsics: ``physical_ai_av.calibration.CameraIntrinsics``.
        extrinsics: ``physical_ai_av.calibration.SensorExtrinsics``.
        camera_names: ordered list of camera ids (slot order == visual_tiles).
        polynomial_degree: f-theta forward-polynomial degree floor (the SDK's may
            be longer; the full polynomial is preserved).
        sensor_frame_is_optical: whether the SDK sensor frame is already the
            camera optical frame (see :func:`_ego_to_camera_transform`).

    Returns:
        FThetaProjection with batch dim 1 ([1, V, ...]); stored as a per-dataset
        rig constant. ``max_theta`` is None if no lens exposes a usable r2th.
    """
    from model_components.view_fusion.projection import FThetaProjection

    if not camera_names:
        raise ValueError("camera_names must be non-empty to build a projection.")

    V = len(camera_names)
    t_camera_ego = np.zeros((1, V, 4, 4), dtype=np.float32)
    cx = np.zeros((1, V), dtype=np.float32)
    cy = np.zeros((1, V), dtype=np.float32)
    image_wh = np.zeros((1, V, 2), dtype=np.float32)

    # Native forward polynomials (no scaling); size fw_poly to the LONGEST so no
    # coefficient is dropped. polynomial_degree is only a floor.
    coefs = [np.asarray(intrinsics.camera_models[n].th2r.coef, dtype=np.float32)
             for n in camera_names]
    K = max(polynomial_degree + 1, max(len(c) for c in coefs))
    fw_poly = np.zeros((1, V, K), dtype=np.float32)

    max_thetas = []
    any_bound = False
    for i, name in enumerate(camera_names):
        model = intrinsics.camera_models[name]
        pose = extrinsics.sensor_poses[name]
        t_camera_ego[0, i] = _ego_to_camera_transform(pose, sensor_frame_is_optical)
        # np.polynomial.Polynomial.coef is ascending powers (matches our Horner).
        fw_poly[0, i, : len(coefs[i])] = coefs[i]
        cx[0, i] = float(model.principal_point[0])   # native pixels, unscaled
        cy[0, i] = float(model.principal_point[1])
        image_wh[0, i, 0] = float(model.width)
        image_wh[0, i, 1] = float(model.height)
        mt = _corner_max_theta(model)
        max_thetas.append(mt if mt is not None else float("inf"))
        any_bound = any_bound or (mt is not None)

    # Per-view FOV bound when at least one lens exposes r2th (lenses without one
    # get +inf = unbounded); else None so the operator uses the +Z hemisphere
    # fallback (f-theta without an explicit FOV bound).
    max_theta = torch.tensor([max_thetas], dtype=torch.float32) if any_bound else None

    return FThetaProjection(
        t_camera_ego=torch.from_numpy(t_camera_ego),
        fw_poly=torch.from_numpy(fw_poly),
        cx=torch.from_numpy(cx),
        cy=torch.from_numpy(cy),
        image_wh=torch.from_numpy(image_wh),
        max_theta=max_theta,
    )
