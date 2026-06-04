import torch
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E

def run_inference(fusion_mode, device, batch_size=2, num_views=8):
    print(f"{'='*60}")
    print(f"  fusion_mode = '{fusion_mode}' | batch={batch_size} | views={num_views}")
    print(f"{'='*60}\n")

    # Instantiate model
    model = AutoE2E(num_views=num_views, fusion_mode=fusion_mode)
    model = model.to(device)

    # Visual Scene Input: [batch, num_views, channels, height, width]
    visual_tiles = torch.randn(batch_size, num_views, 3, 224, 224).to(device)

    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Visual Scene History: [batch, 896]
    visual_history = torch.randn(batch_size, 896).to(device)

    # Run inference
    trajectory, compressed_visual_feature_vector, future_visual_features = \
        model(visual_tiles, visual_history, egomotion_history)

    print(f"Trajectory Prediction:              {trajectory.shape}")
    print(f"Compressed Visual Feature Vector:   {compressed_visual_feature_vector.shape}")
    print(f"Future Visual Features Prediction:")
    for i, f in enumerate(future_visual_features):
        print(f"  t+{(i+1)*1.6:.1f}s: {f.shape}")
    print(f"\nCOMPLETE\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    # Test all registered fusion modes
    run_inference("concat", device)
    run_inference("cross_attn", device)


if __name__ == "__main__":
    main()
