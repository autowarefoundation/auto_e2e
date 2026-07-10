import torch
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E
from model_components.view_fusion import PinholeProjection


def run_forward_pass(backbone, planner_mode, device, embed_dim=256, batch_size=2,
                     num_views=7, use_real_geometry=True):
    print(f"{'='*110}")
    print(f"  backbone = '{backbone}' | planner_mode = '{planner_mode}' | "
          f"batch={batch_size} | views={num_views}")
    print(f"{'='*110}\n")

    # Instantiate model. Fusion is always BEV (PR #94); the nav-map is a separate
    # map_input branch, not a camera view.
    model = AutoE2E(backbone=backbone, num_views=num_views, embed_dim=embed_dim,
                    planner_mode=planner_mode).to(device)

    # Visual Scene Input: [batch, num_views, channels, height, width]
    camera_tiles = torch.randn(batch_size, num_views, 3, 256, 256).to(device)
    # Map Input: [batch, channels, height, width]
    map_input = torch.randn(batch_size, 3, 256, 256).to(device)
    # Visual History Input: [batch, 896]
    visual_history = torch.randn(batch_size, 896).to(device)
    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Geometry: pass a projection operator, or the explicit pseudo path. A pinhole
    # matrix (intrinsic @ extrinsic) is wrapped as a PinholeProjection operator;
    # there is no camera_params matrix argument on forward.
    if use_real_geometry:
        matrix = torch.randn(batch_size, num_views, 3, 4).to(device)
        projection = PinholeProjection(matrix)
        geometry_type = "pinhole"
    else:
        projection = None
        geometry_type = "pseudo"

    trajectory = model(
        camera_tiles=camera_tiles, map_input=map_input,
        visual_history=visual_history, egomotion_history=egomotion_history,
        projection=projection, geometry_type=geometry_type, mode="infer",
    )

    print(f"Trajectory Prediction:              {trajectory.shape}")
    print("\nCOMPLETE\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    with torch.no_grad():
        for backbone in ("swin_v2_tiny", "conv_next_v2_tiny", "res_net_50"):
            for planner in ("bezier", "flow_matching"):
                run_forward_pass(backbone, planner, device)


if __name__ == "__main__":
    main()
