import torch
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E

def main():
    # Device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference \n')

    # Model configuration
    num_views = 8  # 7 cameras + 1 map tile
    batch_size = 2

    # Instantiate model
    model = AutoE2E(num_views=num_views, fusion_mode="concat")
    model = model.to(device)

    # Visual Scene Input: [batch, num_views, channels, height, width]
    # 7 cameras + 1 map tile at 224x224 resolution
    visual_tiles = torch.randn(batch_size, num_views, 3, 224, 224).to(device)

    # Egomotion History Input: [batch, 256]
    # Speed, Acceleration, Yaw Angle, Yaw Rate for
    # 6.4s past history giving 64 x 4 samples at 10Hz
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Visual Scene History: [batch, 896]
    # Length 14 compressed visual feature vector at 10Hz
    # for 6.4s past horizon giving 64 x 14 samples
    visual_history = torch.randn(batch_size, 896).to(device)

    # Run inference
    trajectory, compressed_visual_feature_vector, future_visual_features = \
        model(visual_tiles, visual_history, egomotion_history)

    # Trajectory Prediction
    print("---")
    print("\n")
    print("Trajectory Prediction: \n")
    print(trajectory.shape, "\n")

    # Compressed Visual Feature Vector
    print("---")
    print("\n")
    print("Compressed Current Scene Visual Feature Vector: \n")
    print(compressed_visual_feature_vector.shape, "\n")

    # Future Visual Feature Prediction
    print("---")
    print("\n")
    print("Future Visual Features Prediction: \n")
    for i in range(0, len(future_visual_features)):
        print(future_visual_features[i].shape)
    print("\n")

    print("COMPLETE")

if __name__ == "__main__":
    main()
