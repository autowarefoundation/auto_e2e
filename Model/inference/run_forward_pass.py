import torch
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E

def run_forward_pass(backbone, fusion_mode, device, embed_dim=256, batch_size=2, num_views=8):
    print(f"{'='*80}")
    print(f"  backbone = '{backbone}' | fusion_mode = '{fusion_mode}' | batch={batch_size} | views={num_views}")
    print(f"{'='*80}\n")

    # Instantiate model
    model = AutoE2E(backbone=backbone, num_views=num_views, embed_dim=embed_dim, fusion_mode=fusion_mode)
    model = model.to(device)

    # Visual Scene Input: [batch, num_views, channels, height, width]
    visual_tiles = torch.randn(batch_size, num_views, 3, 256, 256).to(device)

    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Visual Scene History: [batch, 896]
    visual_history = torch.randn(batch_size, 896).to(device)

    # Camera parameters: [batch, num_views, 3, 4] projection matrices
    # Only used by BEV fusion; None triggers learnable pseudo-projection
    camera_params = None
    if fusion_mode == "bev":
        camera_params = torch.randn(batch_size, num_views, 3, 4).to(device)

    # Run inference - train mode means all layers are activated
    trajectory, compressed_visual_feature_vector, future_visual_features = \
        model(visual_tiles, visual_history, egomotion_history, 
              camera_params=camera_params, mode="train")

    print(f"Trajectory Prediction:              {trajectory.shape}")
    print(f"Compressed Visual Feature Vector:   {compressed_visual_feature_vector.shape}")
    print(f"Future Visual Features Prediction:")
    for i, f in enumerate(future_visual_features):
        print(f"  t+{(i+1)*1.6:.1f}s: {f.shape}")
    print(f"\nCOMPLETE\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    # Run a forward pass in the network with all registered backbones and fusion modes
    run_forward_pass("swin_v2_tiny", "concat", device)
    run_forward_pass("swin_v2_tiny", "cross_attn", device)
    run_forward_pass("swin_v2_tiny", "bev", device)
    run_forward_pass("conv_next_v2_tiny", "concat", device)
    run_forward_pass("conv_next_v2_tiny", "cross_attn", device)
    run_forward_pass("conv_next_v2_tiny", "bev", device)
    run_forward_pass("res_net_50", "concat", device)
    run_forward_pass("res_net_50", "cross_attn", device)
    run_forward_pass("res_net_50", "bev", device)


if __name__ == "__main__":
    main()
