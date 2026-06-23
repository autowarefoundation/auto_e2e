import torch
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E

def run_forward_pass(backbone, fusion_mode, planner_mode, device, embed_dim=8, batch_size=2, num_views=8):
    print(f"{'='*110}")
    print(f"  backbone = '{backbone}' | planner_mode= '{planner_mode} | batch={batch_size} | views={num_views}")
    print(f"{'='*110}\n")

    # Instantiate model
    model = AutoE2E(backbone=backbone, num_views=num_views, embed_dim=embed_dim, fusion_mode=fusion_mode)
    model = model.to(device)

    # Visual Scene Input: [batch, num_views, channels, height, width]
    camera_tiles = torch.randn(batch_size, num_views, 3, 256, 256).to(device)

    # Map Input: [batch, channels, height, width]
    map_input = torch.randn(batch_size, 3, 256, 256).to(device)

    # Visual History Input: [batch, 896] — 64 frames × 14-dim compressed scene memory
    visual_history = torch.randn(batch_size, 896).to(device)

    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Camera parameters: [batch, num_views, 3, 4] projection matrices
    # Only used by BEV fusion; None triggers learnable pseudo-projection
    camera_params = None
    if fusion_mode == "bev":
        camera_params = torch.randn(batch_size, num_views, 3, 4).to(device)

    # Run inference - train mode means all layers are activated
    trajectory = \
        model(camera_tiles=camera_tiles, map_input=map_input, visual_history=visual_history, egomotion_history=egomotion_history,
              camera_params=camera_params, mode="infer")

    print(f"Trajectory Prediction:              {trajectory.shape}")
    print("\nCOMPLETE\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    with torch.no_grad():

        # Run a forward pass in the network with all registered backbones, fusion modes and planners
        # SWIN-V2-TINY
        run_forward_pass("swin_v2_tiny", "bezier", device)
        run_forward_pass("swin_v2_tiny", "flow_matching", device)
      

        # CONVNEXT-V2-TINY
        run_forward_pass("conv_next_v2_tiny", "bezier",device)
        run_forward_pass("conv_next_v2_tiny", "flow_matching",device)
        
        # RESNET-50
        run_forward_pass("res_net_50", "bezier",device)
        run_forward_pass("res_net_50", "flow_matching",device)
      


if __name__ == "__main__":
    main()
