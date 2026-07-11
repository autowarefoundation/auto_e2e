import torch
import cv2
import numpy as np
import math
from typing import Optional

_DT = 0.1  # 10 Hz
_FUTURE_TIMESTEPS = 64
MAP_W = 640
MAP_H = 360

class Visualization:

    @staticmethod
    def accel_and_curv_to_meters_trajectory(
            action_sequence: torch.Tensor,
            current_speed: float,
            future_timesteps: int,
            initial_heading: float = 0.0
    ) -> torch.Tensor:
        """
        Converts an action sequence of acceleration and curvature into a 2D trajectory in meters.

        Args:
            action_sequence (torch.Tensor): Flattened tensor of [acceleration, curvature] actions.
            current_speed (float): Initial speed of the vehicle in m/s.
            future_timesteps (int): Number of timesteps to predict.
            initial_heading (float, optional): Initial heading angle in radians. Defaults to 0.0.

        Returns:
            torch.Tensor: A tensor of shape (future_timesteps + 1, 2) containing [x, y] coordinates in meters.
        """
        # change the trajectory format
        action_sequence = torch.reshape(action_sequence, (future_timesteps, 2))

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = torch.zeros((future_timesteps + 1, 2))
        trajectory_m[0, :] = 0

        # 1.1 velocity is needed for integration
        v = float(current_speed)

        # 1.2 Yaw angle is needed to derive 2D acceleration
        yaw = float(initial_heading)

        for i in range(future_timesteps):
            accel = action_sequence[i, 0].item()
            curv = action_sequence[i, 1].item()

            v = v + (accel * _DT)
            yaw = yaw + (v * curv * _DT)

            # The format is [X Y]. Sign convention for yaw is + = CCW
            trajectory_m[i + 1, 0] = trajectory_m[i, 0] - (v * math.sin(yaw) * _DT)
            trajectory_m[i + 1, 1] = trajectory_m[i, 1] + (v * math.cos(yaw) * _DT)

        return trajectory_m

    @staticmethod
    def meters_to_pixels_trajectory(trajectory_m: torch.Tensor, resolution_m_px: float, map_image: np.ndarray) -> torch.Tensor:
        """
        Converts a trajectory in meters to pixel coordinates on a map image.

        Args:
            trajectory_m (torch.Tensor): Tensor of [x, y] coordinates in meters.
            resolution_m_px (float): Resolution of the map image in meters per pixel.
            map_image (np.ndarray): The map image used to determine the center offset.

        Returns:
            torch.Tensor: Tensor of pixel coordinates [u, v] for the trajectory.
        """
        h, w = map_image.shape[:2]

        trajectory_px = torch.zeros_like(trajectory_m)
        trajectory_px[:, 0] = (w / 2) + (trajectory_m[:, 0] / resolution_m_px)
        trajectory_px[:, 1] = (h / 2) - (trajectory_m[:, 1] / resolution_m_px)

        return trajectory_px

    @staticmethod
    def overlay_the_trajectory_with_map(
            trajectory_px: torch.Tensor,
            map_image: np.ndarray,
            color: tuple = (0, 255, 0),
            initial_heading: float = 0.0,
            resolution_m_px: float = 0.4
    ) -> np.ndarray:
        """
        Overlays a trajectory in pixel coordinates onto a map image, adding an agent marker.

        Args:
            trajectory_px (torch.Tensor): Tensor of pixel coordinates for the trajectory.
            map_image (np.ndarray): The base map image to draw on.
            color (tuple, optional): Color of the trajectory line in BGR. Defaults to (0, 255, 0).
            initial_heading (float, optional): Initial heading of the ego vehicle. Defaults to 0.0.
            resolution_m_px (float, optional): Map resolution in meters per pixel. Defaults to 0.4.

        Returns:
            np.ndarray: A new image array with the trajectory and agent marker drawn.
        """
        bgr_color = color
        black_color = (0, 0, 0)

        map_with_trajectory = map_image.copy()

        # Convert PyTorch tensor points to float first to avoid quantization errors in angle math
        pixel_points_float = [(x.item(), y.item()) for x, y in trajectory_px]
        pixel_points = np.array(pixel_points_float, np.int32)
        pts = pixel_points.reshape((-1, 1, 2))

        # Scaling based on zoom level (assuming base resolution is 0.4 m/px when resized to 1280x720)
        zoom_scale = 0.4 / resolution_m_px

        linewidth = int(1 * zoom_scale)
        outline_width = max(1, int(1 * zoom_scale))

        # Draw trajectory line with OpenCV (AA = Anti-Aliased for smooth edges)
        cv2.polylines(map_with_trajectory, [pts], isClosed=False, color=black_color, thickness=linewidth + outline_width * 2, lineType=cv2.LINE_AA)
        cv2.polylines(map_with_trajectory, [pts], isClosed=False, color=bgr_color, thickness=linewidth, lineType=cv2.LINE_AA)

        # Agent marker: sleek arrowhead pointing in the initial heading
        dx = -math.sin(initial_heading)
        dy = -math.cos(initial_heading)
        rx = math.cos(initial_heading)
        ry = -math.sin(initial_heading)

        x0, y0 = pixel_points[0]
        L = 8.0 * zoom_scale
        W = 4.0 * zoom_scale

        tip = (int(x0 + L * dx), int(y0 + L * dy))
        left_back = (int(x0 - L * dx + W * rx), int(y0 - L * dy + W * ry))
        right_back = (int(x0 - L * dx - W * rx), int(y0 - L * dy - W * ry))

        poly_points = np.array([tip, right_back, left_back], np.int32).reshape((-1, 1, 2))
        
        # Draw thick black outline then filled color inside for the agent marker
        agent_color = (126, 27, 232) #purple
        cv2.fillPoly(map_with_trajectory, [poly_points], agent_color, cv2.LINE_8)
        cv2.polylines(map_with_trajectory, [poly_points], isClosed=True, color=black_color, thickness=outline_width, lineType=cv2.LINE_8)

        return map_with_trajectory

    @staticmethod
    def render_trajectory_map_tile(
        action_sequence: torch.Tensor,
        current_speed: float,
        map_image: np.ndarray,
        resolution_m_px: float,
        color: tuple = (0, 255, 0),
        initial_heading: float = 0.0
    ) -> np.ndarray:
        """
        Integrates predicted trajectory into metric coordinates and
        draws them onto the raw BEV map tile.

        Args:
            action_sequence: (128, ) flattened (64, 2) tensor of predicted [acceleration, curvature].
            current_speed: Scalar float from the egomotion history.
            map_image: A map tile, not normalized (BGR numpy array).
            resolution_m_px: The metric resolution of the map image.

        Returns:
            A new Numpy array with the trajectory drawn on it.
        """

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(
            action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading
        )

        # 2. Map coordinates (m) to pixels (px)

        trajectory_px = Visualization.meters_to_pixels_trajectory(trajectory_m, resolution_m_px, map_image)

        # 3. Overlay the trajectory onto the map tile

        map_with_trajectory = Visualization.overlay_the_trajectory_with_map(trajectory_px, map_image, color, initial_heading, resolution_m_px)
        
        return map_with_trajectory

    @staticmethod
    def render_trajectory_on_a_grid(
        action_sequence: torch.Tensor,
        current_speed: float,
        actual_action_sequence: Optional[torch.Tensor] = None,
        prediction_color: tuple = (140, 255, 0),
        actual_trajectory_color: tuple = (255, 80, 120)
    ) -> np.ndarray:
        """
        Renders predicted (and optionally actual) action sequences onto a standardized BEV grid visualization.

        Args:
            action_sequence (torch.Tensor): Predicted trajectory [acceleration, curvature].
            current_speed (float): Current speed of the ego vehicle.
            actual_action_sequence (Optional[torch.Tensor]): Ground truth trajectory.
            prediction_color (tuple): Color for predicted trajectory.
            actual_trajectory_color (tuple): Color for actual trajectory.

        Returns:
            np.ndarray: An image array of the grid visualization.
        """
        pred_m = Visualization.accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading=0.0)
                
        act_m = None
        if actual_action_sequence is not None:
            act_m = Visualization.accel_and_curv_to_meters_trajectory(actual_action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading=0.0)

        grid_with_trajectory = Visualization.generate_grid(
            prediction_m=pred_m, 
            actual_trajectory_m=act_m,
            prediction_color=prediction_color,
            actual_trajectory_color=actual_trajectory_color
        )

        return grid_with_trajectory

    @staticmethod
    def generate_grid(
        prediction_m: torch.Tensor, 
        actual_trajectory_m: Optional[torch.Tensor] = None,
        prediction_color: tuple = (140, 255, 0),
        actual_trajectory_color: tuple = (255, 80, 120)
        ) -> np.ndarray:
        """
        Generates a 2D plotting grid and draws the predicted and (optionally) actual trajectories.

        Args:
            prediction_m (torch.Tensor): Predicted trajectory in meters [x, y].
            actual_trajectory_m (Optional[torch.Tensor], optional): Ground truth trajectory in meters [x, y]. Defaults to None.
            prediction_color (tuple, optional): Color for the predicted trajectory. Defaults to (140, 255, 0).
            actual_trajectory_color (tuple, optional): Color for the actual trajectory. Defaults to (255, 80, 120).

        Returns:
            np.ndarray: The rendered grid image with axes, labels, and trajectories.
        """
        # Configuration
        width, height = 480, 1080
        bg_color = (19, 12, 6)         # Very dark blue #060c13 (BGR)
        grid_color = (66, 32, 23)      # Faint deep purple/blue #172042 (BGR)
        text_color = (230, 230, 240)   # Crisp light blue-white
        ego_color = (255, 255, 255)    # Solid white
        
        # Create image
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Coordinate mapping
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
        
        # X ticks
        for x_tick in range(int(x_min), int(x_max) + 1, 10):
            px, py = to_px(x_tick, y_min)
            cv2.line(img, (px, margin_top), (px, margin_top + plot_h), grid_color, 1, cv2.LINE_AA)
            text = str(x_tick)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.putText(img, text, (px - text_size[0]//2, margin_top + plot_h + 15), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
        # Y ticks
        for y_tick in range(0, int(y_max) + 1, 20):
            px, py = to_px(x_min, y_tick)
            cv2.line(img, (margin_left, py), (margin_left + plot_w, py), grid_color, 1, cv2.LINE_AA)
            text = str(y_tick)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.putText(img, text, (margin_left - text_size[0] - 5, py + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
        # Box around plot
        cv2.rectangle(img, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), text_color, 1, cv2.LINE_AA)
        
        # X label
        x_label = "Lateral (m)"
        x_label_size = cv2.getTextSize(x_label, font, 0.5, 1)[0]
        cv2.putText(img, x_label, (margin_left + plot_w//2 - x_label_size[0]//2, height - 15), font, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Y label (rotated)
        y_label = "Longitudinal (m)"
        y_label_size = cv2.getTextSize(y_label, font, 0.5, 1)[0]
        temp_img = np.full((y_label_size[1] + 10, y_label_size[0] + 10, 3), bg_color, dtype=np.uint8)
        cv2.putText(temp_img, y_label, (5, y_label_size[1] + 5), font, 0.5, text_color, 1, cv2.LINE_AA)
        rotated_temp = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        ry, rx, _ = rotated_temp.shape
        start_y = margin_top + plot_h//2 - ry//2
        start_x = 5
        img[start_y:start_y+ry, start_x:start_x+rx] = rotated_temp
        
        # Title
        font_title = cv2.FONT_HERSHEY_SIMPLEX
        title = "Trajectory Prediction"
        title_size = cv2.getTextSize(title, font_title, 0.6, 1)[0]
        start_x = margin_left + plot_w//2 - title_size[0]//2
        cv2.putText(img, title, (start_x, margin_top - 20), font_title, 0.6, (115, 229, 0), 1, cv2.LINE_AA)
        
        # --- Plot Canvas for Clipping ---
        plot_canvas = img[(margin_top+1):(margin_top+plot_h-1), (margin_left+1):(margin_left+plot_w-1)].copy()
        
        def to_px_local(x_m, y_m):
            px = (x_m - x_min) / (x_max - x_min) * plot_w
            py = plot_h - (y_m - y_min) / (y_max - y_min) * plot_h
            return int(px), int(py)

        # Draw Actual Trajectory
        if actual_trajectory_m is not None:
            pts = []
            for i in range(actual_trajectory_m.shape[0]):
                pts.append(to_px_local(float(actual_trajectory_m[i, 0]), float(actual_trajectory_m[i, 1])))
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_canvas, [pts_arr], isClosed=False, color=actual_trajectory_color, thickness=4, lineType=cv2.LINE_AA)
            
        # Draw Predicted Trajectory
        if prediction_m is not None:
            pts = []
            for i in range(prediction_m.shape[0]):
                pts.append(to_px_local(float(prediction_m[i, 0]), float(prediction_m[i, 1])))
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_canvas, [pts_arr], isClosed=False, color=prediction_color, thickness=6, lineType=cv2.LINE_AA)
            
        # Draw Ego Vehicle (Filled triangle with outline)
        ego_px, ego_py = to_px_local(0, 0)
        px_per_m_x = plot_w / (x_max - x_min)
        px_per_m_y = plot_h / (y_max - y_min)
        ego_w = int(2.0 * px_per_m_x)
        ego_h = int(3.0 * px_per_m_y)

        tip = (ego_px, int(ego_py - ego_h / 3))
        left_base = (int(ego_px - ego_w / 2), int(ego_py + ego_h / 3))
        right_base = (int(ego_px + ego_w / 2), int(ego_py + ego_h / 3))
        triangle_pts = np.array([tip, right_base, left_base], np.int32).reshape((-1, 1, 2))
        
        cv2.fillPoly(plot_canvas, [triangle_pts], ego_color, cv2.LINE_AA)
        cv2.polylines(plot_canvas, [triangle_pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        # Paste clipped region back
        img[(margin_top+1):(margin_top+plot_h-1), (margin_left+1):(margin_left+plot_w-1)] = plot_canvas
        
        return img
        
    @staticmethod
    def get_camera_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Computes the full 3x4 camera projection matrix P = K[R|t].

        Args:
            K (np.ndarray): 3x3 camera intrinsic matrix.
            R (np.ndarray): 3x3 rotation matrix (world to camera).
            t (np.ndarray): 3x1 translation vector.

        Returns:
            np.ndarray: The 3x4 projection matrix.
        """
        # Construct Extrinsic matrix [R | t] (3x4)
        A = np.hstack((R, t))

        projection_matrix = K @ A

        return projection_matrix

    @staticmethod
    def project_BEV_to_CameraView(trajectory_m: torch.Tensor, projection_matrix: np.ndarray) -> np.ndarray:
        """
        Projects 3D ground coordinates from the BEV frame into the 2D camera view.

        Args:
            trajectory_m (torch.Tensor): 2D trajectory coordinates in meters (x, z).
            projection_matrix (np.ndarray): 3x4 camera projection matrix.

        Returns:
            np.ndarray: Array of shape (N, 2) containing 2D pixel coordinates. Points behind the camera are mapped to [-1, -1].
        """
        N = trajectory_m.shape[0]
        # Coordinates: x = right, y = down, z = front
        # Assuming ground is at y = 1.5m relative to the camera
        # Later this might be changed so that it is derived from the extrinsics matrix
        points_3d = np.ones((4, N), dtype=np.float32)
        points_3d[0, :] = trajectory_m[:, 0].numpy()  # x: right
        points_3d[1, :] = 1.5                         # y: down (ground)
        points_3d[2, :] = trajectory_m[:, 1].numpy()  # z: front

        # Project to 2D
        points_2d_hom = projection_matrix @ points_3d # (3, 4) @ (4, N) = (3, N)

        # Normalize by depth (z)
        valid_mask = points_2d_hom[2, :] > 0.1
        
        points_2d = np.zeros((N, 2), dtype=np.float32)
        points_2d[valid_mask, 0] = points_2d_hom[0, valid_mask] / points_2d_hom[2, valid_mask]
        points_2d[valid_mask, 1] = points_2d_hom[1, valid_mask] / points_2d_hom[2, valid_mask]
        
        # Set invalid points behind camera to -1
        points_2d[~valid_mask] = -1

        return points_2d

    @staticmethod
    def get_cumulative_distances(trajectory_m: torch.Tensor) -> np.ndarray:
        """
        Calculates the cumulative path distance along a 2D trajectory.

        Args:
            trajectory_m (torch.Tensor): Trajectory points in meters.

        Returns:
            np.ndarray: 1D array of cumulative distances from the start of the trajectory.
        """
        pts = trajectory_m.numpy()
        diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        dists = np.zeros(pts.shape[0], dtype=np.float32)
        dists[1:] = np.cumsum(diffs)
        return dists

    @staticmethod
    def get_trajectory_boundaries_3d(trajectory_m: torch.Tensor, width_m: float = 1.8) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the left and right 3D boundary lines of a trajectory based on a fixed vehicle width.

        Args:
            trajectory_m (torch.Tensor): Centerline trajectory in meters.
            width_m (float, optional): Total width of the trajectory path in meters. Defaults to 1.8.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Left and right boundary trajectories in meters.
        """
        pts = trajectory_m.numpy()
        N = pts.shape[0]
        
        left_bound = np.zeros((N, 2), dtype=np.float32)
        right_bound = np.zeros((N, 2), dtype=np.float32)
        
        for i in range(N):
            if i < N - 1:
                d = pts[i+1] - pts[i]
            else:
                d = pts[i] - pts[i-1]
                
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                d = np.array([0, 1])
            else:
                d = d / norm
                
            n = np.array([-d[1], d[0]])
            
            left_bound[i] = pts[i] + n * (width_m / 2.0)
            right_bound[i] = pts[i] - n * (width_m / 2.0)
            
        return torch.from_numpy(left_bound), torch.from_numpy(right_bound)

    @staticmethod
    def render_trajectory_on_camera_view(
        camera_image: np.ndarray,
        left_2d: np.ndarray,
        right_2d: np.ndarray,
        color: tuple = (0, 255, 0),
        outline_thickness: int = 2,
    ) -> np.ndarray:
        """
        Overlays a perspective-correct 3D trajectory onto a 2D camera view.
        Uses a semi-opaque fill that mathematically accumulates opacity exactly at self-intersections.

        Args:
            camera_image (np.ndarray): Base camera image.
            left_2d (np.ndarray): Projected left boundary points in pixels.
            right_2d (np.ndarray): Projected right boundary points in pixels.
            color (tuple, optional): RGB/BGR color of the trajectory. Defaults to (0, 255, 0).
            outline_thickness (int, optional): Thickness of the opaque edge lines. Defaults to 2.

        Returns:
            np.ndarray: The camera image with the trajectory seamlessly overlaid.
        """
        img_with_traj = camera_image.copy()
        h, w = img_with_traj.shape[:2]
        
        N = left_2d.shape[0]
        
        # 1. Collect contiguous chunks of valid projected points
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
            
        # 2. Accumulate opacity cleanly using a topological mask difference for self-intersections
        accumulator = np.zeros((h, w), dtype=np.float32)
        
        for chunk in chunks:
            # Mask 1: segment-by-segment (solid fill, no holes, NO seams inside since it's an OR mask)
            mask_solid = np.zeros((h, w), dtype=np.uint8)
            for k in range(len(chunk) - 1):
                idx1 = chunk[k]
                idx2 = chunk[k+1]
                quad = np.array([left_2d[idx1], left_2d[idx2], right_2d[idx2], right_2d[idx1]], dtype=np.int32)
                cv2.fillPoly(mask_solid, [quad], (1,))
                
            # Mask 2: perimeter polygon (has holes at self-intersections due to OpenCV odd-even fill rule)
            mask_odd_even = np.zeros((h, w), dtype=np.uint8)
            poly_pts = []
            for i in chunk:
                poly_pts.append(left_2d[i])
            for i in reversed(chunk):
                poly_pts.append(right_2d[i])
            poly_pts_arr = np.array(poly_pts, dtype=np.int32)
            cv2.fillPoly(mask_odd_even, [poly_pts_arr], (1,))
            
            # The self-intersection hole is exactly where mask_solid has filled it but mask_odd_even left it blank!
            intersection_hole = ((mask_solid == 1) & (mask_odd_even == 0)).astype(np.float32)
            
            accumulator += mask_solid.astype(np.float32)
            accumulator += intersection_hole
                
        # Blend color with 70% transparency (alpha=0.3)
        base_alpha = 0.3
        alpha_map = np.clip(accumulator * base_alpha, 0.0, 1.0)[..., None]
        
        color_img = np.full((h, w, 3), color, dtype=np.float32)
        img_with_traj = (color_img * alpha_map + img_with_traj.astype(np.float32) * (1.0 - alpha_map)).astype(np.uint8)
        
        # 3. Draw fully opaque edges without black outline
        for chunk in chunks:
            left_pts = np.array([left_2d[i] for i in chunk], dtype=np.int32)
            right_pts = np.array([right_2d[i] for i in chunk], dtype=np.int32)
            
            # Draw left and right edges
            cv2.polylines(img_with_traj, [left_pts], isClosed=False, color=color, thickness=outline_thickness, lineType=cv2.LINE_AA)
            cv2.polylines(img_with_traj, [right_pts], isClosed=False, color=color, thickness=outline_thickness, lineType=cv2.LINE_AA)
            
            # Cap the ends
            if len(chunk) > 1:
                idx_start = chunk[0]
                idx_end = chunk[-1]
                cv2.line(img_with_traj, tuple(map(int, left_2d[idx_start])), tuple(map(int, right_2d[idx_start])), color, outline_thickness, cv2.LINE_AA)
                cv2.line(img_with_traj, tuple(map(int, left_2d[idx_end])), tuple(map(int, right_2d[idx_end])), color, outline_thickness, cv2.LINE_AA)
            
        return img_with_traj
    
    @staticmethod
    def complete_front_camera_view_with_trajectory(
        action_sequence: torch.Tensor,
        current_speed: float,
        front_camera_image: np.ndarray,
        K: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        color: Optional[tuple] = None
    ) -> np.ndarray:
        """
        End-to-end function to transform an action sequence into 3D boundaries and overlay them on the camera view.

        Args:
            action_sequence (torch.Tensor): Action sequence of [acceleration, curvature].
            current_speed (float): Vehicle speed in m/s.
            front_camera_image (np.ndarray): The base camera image.
            K (Optional[np.ndarray], optional): Camera intrinsics. Defaults to None.
            R (Optional[np.ndarray], optional): Camera rotation matrix. Defaults to None.
            t (Optional[np.ndarray], optional): Camera translation vector. Defaults to None.
            P (Optional[np.ndarray], optional): Full projection matrix (overrides K, R, t). Defaults to None.
            color (Optional[tuple], optional): Color of the trajectory. Defaults to None.

        Returns:
            np.ndarray: The final image with the projected trajectory.
        """
        if color is None:
            color = (0, 255, 0)
            
        # 1. Generate trajectory in BEV (meters)
        traj_m = Visualization.accel_and_curv_to_meters_trajectory(
            action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading=0.0
        )

        # 2. Project trajectory to Camera View
        if P is not None:
            projection_matrix = P
        else:
            if K is None or R is None or t is None:
                raise ValueError("Either P or (K, R, t) must be provided.")
            projection_matrix = Visualization.get_camera_projection_matrix(K, R, t)
        
        left_m, right_m = Visualization.get_trajectory_boundaries_3d(traj_m, width_m=1.8)

        left_2d = Visualization.project_BEV_to_CameraView(left_m, projection_matrix)
        right_2d = Visualization.project_BEV_to_CameraView(right_m, projection_matrix)

        # 3. Draw on camera image
        cam_with_traj = Visualization.render_trajectory_on_camera_view(
            front_camera_image, left_2d, right_2d, color=color, outline_thickness=3
        )

        return cam_with_traj

    @staticmethod
    def concatenate_grid_and_camera(grid_img: np.ndarray, cam_img: np.ndarray) -> np.ndarray:
        """
        Horizontally concatenates the BEV grid image and the camera view, scaling the camera image to fit vertically.

        Args:
            grid_img (np.ndarray): BEV grid visualization image.
            cam_img (np.ndarray): Camera view image.

        Returns:
            np.ndarray: Horizontally stitched combined image.
        """
        grid_h, grid_w = grid_img.shape[:2]
        cam_h, cam_w = cam_img.shape[:2]
        
        scale = grid_h / cam_h
        new_cam_w = int(cam_w * scale)
        cam_resized = cv2.resize(cam_img, (new_cam_w, grid_h))

        return np.hstack((grid_img, cam_resized))