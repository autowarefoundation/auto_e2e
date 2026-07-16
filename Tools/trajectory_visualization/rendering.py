import cv2
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class MapGeometry:
    """
    Data class representing the geometry and properties of the top-down Map view.
    
    Args:
        meters_per_pixel_x (float): Resolution along the x-axis in meters per pixel.
        meters_per_pixel_y (float): Resolution along the y-axis in meters per pixel.
        ego_pixel_x (float): The x-coordinate (in pixels) of the ego vehicle on the map.
        ego_pixel_y (float): The y-coordinate (in pixels) of the ego vehicle on the map.
        rotation_rad (float): The rotation of the map in radians.
    """
    meters_per_pixel_x: float
    meters_per_pixel_y: float
    ego_pixel_x: float
    ego_pixel_y: float
    rotation_rad: float

    def __post_init__(self):
        if self.meters_per_pixel_x <= 0 or self.meters_per_pixel_y <= 0:
            raise ValueError(f"Invalid map geometry: resolution must be positive, got ({self.meters_per_pixel_x}, {self.meters_per_pixel_y})")

def get_camera_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes the full 3x4 camera projection matrix P = K[R|t].

    Args:
        K (np.ndarray): 3x3 intrinsic camera matrix.
        R (np.ndarray): 3x3 rotation matrix from world to camera frame.
        t (np.ndarray): 3x1 translation vector from world to camera frame.

    Returns:
        np.ndarray: The 3x4 camera projection matrix.
    """
    A = np.hstack((R, t))
    projection_matrix = K @ A
    return projection_matrix

def project_BEV_to_CameraView(trajectory_m: torch.Tensor, projection_matrix: np.ndarray) -> np.ndarray:
    """
    Projects 3D ground coordinates from the BEV frame into the 2D camera view.
    
    Args:
        trajectory_m (torch.Tensor): Tensor of shape (N, 2) containing X, Z ground plane coordinates in meters.
        projection_matrix (np.ndarray): 3x4 camera projection matrix.
        
    Returns:
        np.ndarray: Array of shape (N, 2) containing the projected 2D (u, v) pixel coordinates.
                    Points projected behind the camera or with z <= 0.1 are set to [-1, -1].
    """
    N = trajectory_m.shape[0]
    points_3d = np.ones((4, N), dtype=np.float32)
    points_3d[0, :] = trajectory_m[:, 0].numpy()  # x: right
    points_3d[1, :] = 1.5                         # y: down (ground)
    points_3d[2, :] = trajectory_m[:, 1].numpy()  # z: front

    points_2d_hom = projection_matrix @ points_3d

    valid_mask = points_2d_hom[2, :] > 0.1
    
    points_2d = np.zeros((N, 2), dtype=np.float32)
    points_2d[valid_mask, 0] = points_2d_hom[0, valid_mask] / points_2d_hom[2, valid_mask]
    points_2d[valid_mask, 1] = points_2d_hom[1, valid_mask] / points_2d_hom[2, valid_mask]
    
    points_2d[~valid_mask] = -1

    return points_2d

def render_trajectory_on_camera_view(
    camera_image: np.ndarray,
    left_2d: np.ndarray,
    right_2d: np.ndarray,
    color: tuple = (0, 255, 0),
    outline_thickness: int = 2,
) -> np.ndarray:
    """
    Overlays a perspective-correct 3D trajectory ribbon onto a 2D camera view.
    
    Args:
        camera_image (np.ndarray): The 2D front camera image array (H, W, 3).
        left_2d (np.ndarray): The projected 2D coordinates of the left boundary of the trajectory.
        right_2d (np.ndarray): The projected 2D coordinates of the right boundary of the trajectory.
        color (tuple): The RGB/BGR color tuple for the trajectory ribbon. Defaults to green.
        outline_thickness (int): Thickness of the drawn outline for the trajectory edges.
        
    Returns:
        np.ndarray: The new camera image array with the trajectory ribbon drawn over it.
    """
    img_with_traj = camera_image.copy()
    h, w = img_with_traj.shape[:2]
    
    N = left_2d.shape[0]
    
    chunks = []
    current_chunk_indices = []
    for i in range(N):
        valid = (left_2d[i, 0] != -1 and right_2d[i, 0] != -1)
        if valid:
            current_chunk_indices.append(i)
        else:
            if len(current_chunk_indices) > 1:
                chunks.append(current_chunk_indices)
            current_chunk_indices = []
    if len(current_chunk_indices) > 1:
        chunks.append(current_chunk_indices)
        
    if not chunks:
        return img_with_traj
        
    accumulator = np.zeros((h, w), dtype=np.float32)
    
    for chunk in chunks:
        mask_solid = np.zeros((h, w), dtype=np.uint8)
        for k in range(len(chunk) - 1):
            idx1 = chunk[k]
            idx2 = chunk[k+1]
            quad = np.array([left_2d[idx1], left_2d[idx2], right_2d[idx2], right_2d[idx1]], dtype=np.int32)
            cv2.fillPoly(mask_solid, [quad], (1,))
            
        mask_odd_even = np.zeros((h, w), dtype=np.uint8)
        poly_pts = []
        for i in chunk:
            poly_pts.append(left_2d[i])
        for i in reversed(chunk):
            poly_pts.append(right_2d[i])
        poly_pts_arr = np.array(poly_pts, dtype=np.int32)
        cv2.fillPoly(mask_odd_even, [poly_pts_arr], (1,))
        
        intersection_hole = ((mask_solid == 1) & (mask_odd_even == 0)).astype(np.float32)
        
        accumulator += mask_solid.astype(np.float32)
        accumulator += intersection_hole
            
    base_alpha = 0.3
    alpha_map = np.clip(accumulator * base_alpha, 0.0, 1.0)[..., None]
    
    color_img = np.full((h, w, 3), color, dtype=np.float32)
    img_with_traj = (color_img * alpha_map + img_with_traj.astype(np.float32) * (1.0 - alpha_map)).astype(np.uint8)
    
    for chunk in chunks:
        left_pts = np.array([left_2d[i] for i in chunk], dtype=np.int32)
        right_pts = np.array([right_2d[i] for i in chunk], dtype=np.int32)
        
        cv2.polylines(img_with_traj, [left_pts], isClosed=False, color=color, thickness=outline_thickness, lineType=cv2.LINE_AA)
        cv2.polylines(img_with_traj, [right_pts], isClosed=False, color=color, thickness=outline_thickness, lineType=cv2.LINE_AA)
        
        if len(chunk) > 1:
            idx_start = chunk[0]
            idx_end = chunk[-1]
            cv2.line(img_with_traj, tuple(map(int, left_2d[idx_start])), tuple(map(int, right_2d[idx_start])), color, outline_thickness, cv2.LINE_AA)
            cv2.line(img_with_traj, tuple(map(int, left_2d[idx_end])), tuple(map(int, right_2d[idx_end])), color, outline_thickness, cv2.LINE_AA)
        
    return img_with_traj

def generate_grid(
    prediction_xy: torch.Tensor | None = None, 
    target_xy: torch.Tensor | None = None,
    prediction_color: tuple | None = (140, 255, 0),
    target_color: tuple | None = (255, 80, 120)
    ) -> np.ndarray:
    """
    Generates a 2D plotting grid and draws the predicted and (optionally) actual trajectories.
    
    Args:
        prediction_xy (torch.Tensor | None): Tensor of shape (N, 2) representing the predicted trajectory in meters.
        target_xy (torch.Tensor | None): Tensor of shape (M, 2) representing the target ground truth trajectory in meters.
        prediction_color (tuple | None): Color of the prediction trajectory line. Defaults to light green.
        target_color (tuple | None): Color of the target trajectory line. Defaults to pinkish red.
        
    Returns:
        np.ndarray: The resulting 1080x480 grid image containing the plotted trajectories.
    """
    if prediction_color is None:
        prediction_color = (140, 255, 0)
    if target_color is None:
        target_color = (255, 80, 120)
    width, height = 480, 1080
    bg_color = (19, 12, 6)
    grid_color = (66, 32, 23)
    text_color = (230, 230, 240)
    ego_color = (255, 255, 255)
    
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    margin_left, margin_right = 50, 20
    margin_top, margin_bottom = 60, 50
    
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    x_min, x_max = -20.0, 20.0
    y_min, y_max = -10.0, 80.0
    
    def to_px(x_m, y_m):
        px = margin_left + (x_m - x_min) / (x_max - x_min) * plot_w
        py = margin_top + plot_h - (y_m - y_min) / (y_max - y_min) * plot_h
        return int(px), int(py)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    for x_tick in range(int(x_min), int(x_max) + 1, 10):
        px, py = to_px(x_tick, y_min)
        cv2.line(img, (px, margin_top), (px, margin_top + plot_h), grid_color, 2, cv2.LINE_8)
        text = str(x_tick)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.putText(img, text, (px - text_size[0]//2, margin_top + plot_h + 15), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
    for y_tick in range(0, int(y_max) + 1, 20):
        px, py = to_px(x_min, y_tick)
        cv2.line(img, (margin_left, py), (margin_left + plot_w, py), grid_color, 2, cv2.LINE_8)
        text = str(y_tick)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.putText(img, text, (margin_left - text_size[0] - 5, py + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
    cv2.rectangle(img, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), text_color, 1, cv2.LINE_8)
    
    x_label = "Lateral (m)"
    x_label_size = cv2.getTextSize(x_label, font, 0.5, 1)[0]
    cv2.putText(img, x_label, (margin_left + plot_w//2 - x_label_size[0]//2, height - 15), font, 0.5, text_color, 1, cv2.LINE_AA)
    
    y_label = "Longitudinal (m)"
    y_label_size = cv2.getTextSize(y_label, font, 0.5, 1)[0]
    temp_img = np.full((y_label_size[1] + 10, y_label_size[0] + 10, 3), bg_color, dtype=np.uint8)
    cv2.putText(temp_img, y_label, (5, y_label_size[1] + 5), font, 0.5, text_color, 1, cv2.LINE_AA)
    rotated_temp = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    ry, rx, _ = rotated_temp.shape
    start_y = margin_top + plot_h//2 - ry//2
    start_x = 5
    img[start_y:start_y+ry, start_x:start_x+rx] = rotated_temp
    
    font_title = cv2.FONT_HERSHEY_SIMPLEX
    title = "Trajectory Prediction"
    title_size = cv2.getTextSize(title, font_title, 0.6, 1)[0]
    start_x_title = margin_left + plot_w//2 - title_size[0]//2
    cv2.putText(img, title, (start_x_title, margin_top - 35), font_title, 0.6, (115, 229, 0), 1, cv2.LINE_AA)
    
    # Draw Legend
    # 1. Prediction (Left)
    if prediction_xy is not None:
        pred_text = "Prediction"
        cv2.circle(img, (margin_left + 15, margin_top - 14), 6, prediction_color, -1, cv2.LINE_AA)
        cv2.putText(img, pred_text, (margin_left + 30, margin_top - 10), font, 0.5, prediction_color, 1, cv2.LINE_AA)
    
    # 2. Target / Ground Truth (Right)
    if target_xy is not None:
        tgt_text = "Ground Truth"
        tgt_size = cv2.getTextSize(tgt_text, font, 0.5, 1)[0]
        tgt_x = margin_left + plot_w - 30 - tgt_size[0]
        cv2.putText(img, tgt_text, (tgt_x, margin_top - 10), font, 0.5, target_color, 1, cv2.LINE_AA)
        cv2.circle(img, (margin_left + plot_w - 15, margin_top - 14), 6, target_color, -1, cv2.LINE_AA)
    
    plot_area = img[margin_top:margin_top+plot_h, margin_left:margin_left+plot_w]
    
    def to_px_local(x_m, y_m):
        px = (x_m - x_min) / (x_max - x_min) * plot_w
        py = plot_h - (y_m - y_min) / (y_max - y_min) * plot_h
        return int(px), int(py)

    if target_xy is not None:
        pts = []
        for i in range(target_xy.shape[0]):
            pts.append(to_px_local(float(target_xy[i, 0]), float(target_xy[i, 1])))
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(plot_area, [pts_arr], isClosed=False, color=target_color, thickness=3, lineType=cv2.LINE_AA)

    if prediction_xy is not None:
        pts = []
        for i in range(prediction_xy.shape[0]):
            pts.append(to_px_local(float(prediction_xy[i, 0]), float(prediction_xy[i, 1])))
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(plot_area, [pts_arr], isClosed=False, color=prediction_color, thickness=3, lineType=cv2.LINE_AA)
        
    ego_px, ego_py = to_px_local(0, 0)
    px_per_m_x = plot_w / (x_max - x_min)
    px_per_m_y = plot_h / (y_max - y_min)
    ego_w = int(2.0 * px_per_m_x)
    ego_h = int(3.0 * px_per_m_y)

    tip = (ego_px, int(ego_py - ego_h / 3))
    left_base = (int(ego_px - ego_w / 2), int(ego_py + ego_h / 3))
    right_base = (int(ego_px + ego_w / 2), int(ego_py + ego_h / 3))
    triangle_pts = np.array([tip, right_base, left_base], np.int32).reshape((-1, 1, 2))
    
    cv2.fillPoly(plot_area, [triangle_pts], ego_color, cv2.LINE_AA)
    cv2.polylines(plot_area, [triangle_pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    return img

def concatenate_grid_and_camera(grid_img: np.ndarray, cam_img: np.ndarray) -> np.ndarray:
    """
    Horizontally concatenates the BEV grid image and the camera view, scaling the camera view to match heights.
    
    Args:
        grid_img (np.ndarray): The generated 2D grid image (typically 1080px high).
        cam_img (np.ndarray): The perspective front camera image to be concatenated.
        
    Returns:
        np.ndarray: A single wider image horizontally concatenating both inputs.
    """
    grid_h, grid_w = grid_img.shape[:2]
    cam_h, cam_w = cam_img.shape[:2]
    
    scale = grid_h / cam_h
    new_cam_w = int(cam_w * scale)
    cam_resized = cv2.resize(cam_img, (new_cam_w, grid_h))

    return np.hstack((grid_img, cam_resized))

def meters_to_pixels_trajectory(trajectory_m: torch.Tensor, geometry: MapGeometry) -> torch.Tensor:
    """
    Converts a trajectory from metric coordinates (meters) to map pixel coordinates based on the map geometry.
    
    Args:
        trajectory_m (torch.Tensor): Tensor of shape (N, 2) representing the metric (X, Y) trajectory.
        geometry (MapGeometry): MapGeometry object defining the map resolution and ego origin.
        
    Returns:
        torch.Tensor: Tensor of shape (N, 2) representing the transformed (X, Y) pixel coordinates on the map tile.
    """
    import math
    trajectory_px = torch.zeros_like(trajectory_m)
    
    cos_theta = math.cos(geometry.rotation_rad)
    sin_theta = math.sin(geometry.rotation_rad)
    
    x_rot = trajectory_m[:, 0] * cos_theta - trajectory_m[:, 1] * sin_theta
    y_rot = trajectory_m[:, 0] * sin_theta + trajectory_m[:, 1] * cos_theta
    
    trajectory_px[:, 0] = geometry.ego_pixel_x + (x_rot / geometry.meters_per_pixel_x)
    trajectory_px[:, 1] = geometry.ego_pixel_y - (y_rot / geometry.meters_per_pixel_y)
    
    return trajectory_px

def overlay_the_trajectory_with_map(
        trajectory_px: torch.Tensor,
        map_image: np.ndarray,
        geometry: MapGeometry,
        color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Overlays a trajectory line onto the map tile, drawing both an inner line and a black outline.
    
    Args:
        trajectory_px (torch.Tensor): Tensor of shape (N, 2) containing the pixel coordinates of the trajectory.
        map_image (np.ndarray): The 2D map image tile array.
        geometry (MapGeometry): Map geometry used to determine dynamic zoom/line thickness scaling.
        color (tuple): The color of the inner trajectory line. Defaults to green.
        
    Returns:
        np.ndarray: The map image with the thick bordered trajectory overlaid.
    """
    bgr_color = color
    black_color = (0, 0, 0)
    map_with_trajectory = map_image.copy()

    pixel_points_float = [(x.item(), y.item()) for x, y in trajectory_px]
    pixel_points = np.array(pixel_points_float, np.int32)
    pts = pixel_points.reshape((-1, 1, 2))

    zoom_scale = 0.4 / geometry.meters_per_pixel_x
    linewidth = int(1 * zoom_scale)
    outline_width = max(1, int(1 * zoom_scale))

    cv2.polylines(map_with_trajectory, [pts], isClosed=False, color=black_color, thickness=linewidth + outline_width * 2, lineType=cv2.LINE_AA)
    cv2.polylines(map_with_trajectory, [pts], isClosed=False, color=bgr_color, thickness=linewidth, lineType=cv2.LINE_AA)
    return map_with_trajectory

def render_trajectory_map_tile(
    map_image: np.ndarray,
    geometry: MapGeometry,
    prediction_xy: torch.Tensor | None = None,
    target_xy: torch.Tensor | None = None,
    prediction_color: tuple | None = (0, 255, 0),
    target_color: tuple | None = (255, 80, 120),
    is_approximate: bool = False
) -> np.ndarray:
    """
    Draws the complete map tile view including the ego vehicle triangle, prediction trajectory, and target trajectory.
    
    Args:
        map_image (np.ndarray): The base map image array.
        geometry (MapGeometry): The map geometric metadata.
        prediction_xy (torch.Tensor | None): The predicted trajectory (N, 2) in meters.
        target_xy (torch.Tensor | None): The target ground truth trajectory (M, 2) in meters.
        prediction_color (tuple | None): Color of the prediction line. Defaults to green.
        target_color (tuple | None): Color of the target line. Defaults to pinkish red.
        is_approximate (bool): If True, overlays an 'APPROXIMATE' watermark text over the map.
        
    Returns:
        np.ndarray: The completed map tile rendering.
    """
    if prediction_color is None:
        prediction_color = (0, 255, 0)
    if target_color is None:
        target_color = (255, 80, 120)
    map_with_trajectory = map_image.copy()
    if prediction_xy is not None:
        trajectory_px = meters_to_pixels_trajectory(prediction_xy, geometry)
        map_with_trajectory = overlay_the_trajectory_with_map(
            trajectory_px, map_with_trajectory, geometry, prediction_color
        )
    
    if target_xy is not None:
        target_pts_tensor = meters_to_pixels_trajectory(target_xy, geometry)
        target_pts_float = [(x.item(), y.item()) for x, y in target_pts_tensor]
        target_pts = np.array(target_pts_float, np.int32).reshape((-1, 1, 2))
        
        zoom_scale = 0.4 / geometry.meters_per_pixel_x
        linewidth = int(1 * zoom_scale)
        outline_width = max(1, int(1 * zoom_scale))
        
        cv2.polylines(map_with_trajectory, [target_pts], False, (0, 0, 0), linewidth + outline_width * 2, lineType=cv2.LINE_AA)
        cv2.polylines(map_with_trajectory, [target_pts], False, target_color, linewidth, lineType=cv2.LINE_AA)

    import math
    dx = -math.sin(geometry.rotation_rad)
    dy = -math.cos(geometry.rotation_rad)
    rx = math.cos(geometry.rotation_rad)
    ry = -math.sin(geometry.rotation_rad)

    x0, y0 = geometry.ego_pixel_x, geometry.ego_pixel_y
    zoom_scale = 0.4 / geometry.meters_per_pixel_x
    L = 8.0 * zoom_scale
    W = 4.0 * zoom_scale

    tip = (int(x0 + L * dx), int(y0 + L * dy))
    left_back = (int(x0 - L * dx + W * rx), int(y0 - L * dy + W * ry))
    right_back = (int(x0 - L * dx - W * rx), int(y0 - L * dy - W * ry))

    poly_points = np.array([tip, right_back, left_back], np.int32).reshape((-1, 1, 2))
    
    agent_color = (126, 27, 232)
    outline_width = max(1, int(1 * zoom_scale))
    cv2.fillPoly(map_with_trajectory, [poly_points], agent_color, cv2.LINE_8)
    cv2.polylines(map_with_trajectory, [poly_points], isClosed=True, color=(0, 0, 0), thickness=outline_width, lineType=cv2.LINE_8)

    if is_approximate:
        overlay = map_with_trajectory.copy()
        watermark_text = "APPROXIMATE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
        text_x = (map_with_trajectory.shape[1] - text_size[0]) // 2
        text_y = (map_with_trajectory.shape[0] + text_size[1]) // 2
        cv2.putText(overlay, watermark_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, map_with_trajectory, 0.6, 0, map_with_trajectory)
    return map_with_trajectory

def render_trajectory_on_a_grid(
    prediction_xy: torch.Tensor | None = None,
    target_xy: torch.Tensor | None = None,
    prediction_color: tuple | None = (140, 255, 0),
    target_color: tuple | None = (255, 80, 120)
) -> np.ndarray:
    """
    High-level wrapper to generate the 2D grid plot representing the trajectories.
    
    Args:
        prediction_xy (torch.Tensor | None): Predicted trajectory points in meters.
        target_xy (torch.Tensor | None): Target trajectory points in meters.
        prediction_color (tuple | None): Color of the prediction line. Defaults to light green.
        target_color (tuple | None): Color of the target line. Defaults to pinkish red.
        
    Returns:
        np.ndarray: The generated 2D grid image rendering.
    """
    if prediction_color is None:
        prediction_color = (140, 255, 0)
    if target_color is None:
        target_color = (255, 80, 120)
    grid_with_trajectory = generate_grid(
        prediction_xy=prediction_xy, 
        target_xy=target_xy,
        prediction_color=prediction_color,
        target_color=target_color
    )

    return grid_with_trajectory

def complete_front_camera_view_with_trajectory(
    front_camera_image: np.ndarray,
    prediction_xy: torch.Tensor | None = None,
    K: np.ndarray | None = None,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
    P: np.ndarray | None = None,
    prediction_color: tuple | None = (0, 255, 0),
    target_xy: torch.Tensor | None = None,
    target_color: tuple | None = (255, 80, 120),
    is_approximate: bool = False
) -> np.ndarray:
    """
    Renders the complete front camera view, projecting the 3D metric trajectories onto the 2D image plane.
    
    Args:
        front_camera_image (np.ndarray): The raw 2D front camera image.
        prediction_xy (torch.Tensor | None): Predicted trajectory points in meters (X, Y).
        K (np.ndarray | None): Intrinsic camera matrix (3x3).
        R (np.ndarray | None): Rotation matrix (3x3).
        t (np.ndarray | None): Translation vector (3x1).
        P (np.ndarray | None): Full pre-computed projection matrix (3x4).
        prediction_color (tuple | None): Color of the prediction ribbon. Defaults to green.
        target_xy (torch.Tensor | None): Target trajectory points in meters (X, Y).
        target_color (tuple | None): Color of the target ribbon. Defaults to pinkish red.
        is_approximate (bool): If True, falls back to a synthetic camera model if no calibration is provided.
        
    Returns:
        np.ndarray: The final 3D projected perspective front camera image rendering.
    """
    if prediction_color is None:
        prediction_color = (0, 255, 0)
    if target_color is None:
        target_color = (255, 80, 120)
    from .kinematics import get_trajectory_boundaries_3d
    
    if P is None and (K is None or R is None or t is None):
        if not is_approximate:
            raise ValueError("Camera rendering requires either a verified projection matrix (P) or synthetic calibration (K, R, t). Neither was provided. Set is_approximate=True to use default synthetic calibration.")
        
        h, w = front_camera_image.shape[:2]
        K = np.array([[1000, 0, w/2], [0, 1000, h/2], [0, 0, 1]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        
    if P is None and not is_approximate:
        raise ValueError("Using synthetic/separated calibration (K, R, t) instead of a verified projection matrix requires passing is_approximate=True.")

    if P is not None:
        projection_matrix = P
    else:
        assert K is not None and R is not None and t is not None
        projection_matrix = get_camera_projection_matrix(K, R, t)
    
    cam_with_traj = front_camera_image.copy()

    if target_xy is not None:
        t_left_m, t_right_m = get_trajectory_boundaries_3d(target_xy, width_m=1.8)
        t_left_2d = project_BEV_to_CameraView(t_left_m, projection_matrix)
        t_right_2d = project_BEV_to_CameraView(t_right_m, projection_matrix)
        cam_with_traj = render_trajectory_on_camera_view(
            cam_with_traj, t_left_2d, t_right_2d, color=target_color, outline_thickness=3
        )

    if prediction_xy is not None:
        left_m, right_m = get_trajectory_boundaries_3d(prediction_xy, width_m=1.8)

        left_2d = project_BEV_to_CameraView(left_m, projection_matrix)
        right_2d = project_BEV_to_CameraView(right_m, projection_matrix)

        cam_with_traj = render_trajectory_on_camera_view(
            cam_with_traj, left_2d, right_2d, color=prediction_color, outline_thickness=3
        )

    if is_approximate:
        overlay = cam_with_traj.copy()
        watermark_text = "APPROXIMATE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0
        thickness = 8
        text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
        
        text_x = (cam_with_traj.shape[1] - text_size[0]) // 2
        text_y = (cam_with_traj.shape[0] + text_size[1]) // 2
        
        cv2.putText(overlay, watermark_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, cam_with_traj, 0.6, 0, cam_with_traj)

    return cam_with_traj
