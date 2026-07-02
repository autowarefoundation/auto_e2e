import torch
import cv2
import numpy as np
import math

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
            initial_heading: float = 0.0,
            radius_m: float = 800.0
    ) -> torch.Tensor:

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
        bgr_color = color
        black_color = (0, 0, 0)

        map_with_trajectory = map_image.copy()

        # Convert PyTorch tensor points to float first to avoid quantization errors in angle math
        pixel_points_float = [(x.item(), y.item()) for x, y in trajectory_px]
        pixel_points = np.array(pixel_points_float, np.int32)
        pts = pixel_points.reshape((-1, 1, 2))

        # Scaling based on zoom level (assuming base resolution is 0.4 m/px when resized to 1280x720)
        zoom_scale = 0.4 / resolution_m_px

        linewidth = 1
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
        current_speed: float
    ) -> np.ndarray:
        # 1. Convert trajectory to [x y] in meters
        trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(
            action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading=0.0
        )
        # 2. Generate Grid and overlay trajectory
        grid_with_trajectory = Visualization.generate_grid(prediction_m=trajectory_m)

        return grid_with_trajectory

    @staticmethod
    def generate_grid(prediction_m: torch.Tensor, actual_trajectory_m: torch.Tensor = None) -> np.ndarray:
        # Configuration
        width, height = 480, 1080
        bg_color = (19, 12, 6)         # Very dark blue #060c13 (BGR)
        grid_color = (66, 32, 23)      # Faint deep purple/blue #172042 (BGR)
        text_color = (230, 230, 240)   # Crisp light blue-white
        pred_color = (140, 255, 0)     # Neon Green/Cyan (BGR)
        pred_glow_color = (50, 120, 0) # Darker neon green for glow
        hist_color = (255, 80, 120)    # Vibrant purple (BGR)
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
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_canvas, [pts], isClosed=False, color=hist_color, thickness=4, lineType=cv2.LINE_AA)
            
        # Draw Prediction
        if prediction_m is not None:
            pts = []
            for i in range(prediction_m.shape[0]):
                pts.append(to_px_local(float(prediction_m[i, 0]), float(prediction_m[i, 1])))
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_canvas, [pts], isClosed=False, color=pred_color, thickness=6, lineType=cv2.LINE_AA)
            
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