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


def _ftheta_pixel_scale(model, target_wh: tuple[int, int]) -> tuple[float, float, float]:
    """Isotropic pixel scale from a camera's native size to target_wh.

    The f-theta radius polynomial (th2r) is isotropic in native pixels, so a
    single scale is only exact under an isotropic (aspect-preserving) resize.
    The shard packing resizes to a square target; if the camera is non-square
    this is an approximation, so we scale by the mean of the two axis scales and
    surface the anisotropy. Real-data geometry validation is required before
    trusting f-theta projection quantitatively (see #77).
    """
    import warnings

    native_w, native_h = int(model.width), int(model.height)
    tw, th = target_wh
    sx, sy = tw / native_w, th / native_h
    if abs(sx - sy) / max(sx, sy) > 1e-3:
        warnings.warn(
            f"Anisotropic resize ({native_w}x{native_h} -> {tw}x{th}) applied to an "
            f"isotropic f-theta model; radius scaling uses the mean scale and is "
            f"approximate. Validate on real data before quantitative use (#77).",
            RuntimeWarning,
            stacklevel=2,
        )
    return (sx + sy) / 2.0, sx, sy


def build_ftheta_projection(
    intrinsics,
    extrinsics,
    camera_names,
    target_wh: tuple[int, int] = (256, 256),
    polynomial_degree: int = 4,
    sensor_frame_is_optical: bool = True,
):
    """Construct an :class:`FThetaProjection` scaled to the model-input frame.

    The SDK's f-theta parameters are in native camera resolution; shards are
    packed at ``target_wh``, and BEV fusion normalizes pixel coords by that
    size. We therefore scale principal point and radius polynomial to
    ``target_wh`` here so the projection matches the resized image.

    Args:
        intrinsics: ``physical_ai_av.calibration.CameraIntrinsics``.
        extrinsics: ``physical_ai_av.calibration.SensorExtrinsics``.
        camera_names: ordered list of camera ids (slot order == visual_tiles).
        target_wh: model-input (width, height); must match the shard image_size.
        polynomial_degree: f-theta forward-polynomial degree (SDK default 4).
        sensor_frame_is_optical: whether the SDK sensor frame is already the
            camera optical frame (see :func:`_ego_to_camera_transform`).

    Returns:
        FThetaProjection with batch dim 1 ([1, V, ...]); stored as a per-dataset
        rig constant.
    """
    from model_components.view_fusion.projection import FThetaProjection

    if not camera_names:
        raise ValueError("camera_names must be non-empty to build a projection.")

    V = len(camera_names)
    t_camera_ego = np.zeros((1, V, 4, 4), dtype=np.float32)
    cx = np.zeros((1, V), dtype=np.float32)
    cy = np.zeros((1, V), dtype=np.float32)

    # Gather each camera's scaled forward polynomial first; size fw_poly to the
    # LONGEST so no coefficient is silently dropped (Horner handles any K). The
    # polynomial_degree arg is only a floor — real SDK polynomials may be longer.
    coefs = []
    for name in camera_names:
        model = intrinsics.camera_models[name]
        r_scale, _, _ = _ftheta_pixel_scale(model, target_wh)
        coefs.append(np.asarray(model.th2r.coef, dtype=np.float32) * r_scale)
    K = max(polynomial_degree + 1, max(len(c) for c in coefs))
    fw_poly = np.zeros((1, V, K), dtype=np.float32)

    for i, name in enumerate(camera_names):
        model = intrinsics.camera_models[name]
        pose = extrinsics.sensor_poses[name]
        _, sx, sy = _ftheta_pixel_scale(model, target_wh)
        t_camera_ego[0, i] = _ego_to_camera_transform(pose, sensor_frame_is_optical)
        # np.polynomial.Polynomial.coef is ascending powers (matches our Horner).
        fw_poly[0, i, : len(coefs[i])] = coefs[i]
        cx[0, i] = float(model.principal_point[0]) * sx
        cy[0, i] = float(model.principal_point[1]) * sy

    return FThetaProjection(
        t_camera_ego=torch.from_numpy(t_camera_ego),
        fw_poly=torch.from_numpy(fw_poly),
        cx=torch.from_numpy(cx),
        cy=torch.from_numpy(cy),
    )
