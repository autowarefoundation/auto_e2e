import torch
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E

def main():
    # Device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference \n')
            
    # Instantiate model
    model = AutoE2E()

    # Dummy Visual Scene Input
    # 7 cameras + 1 map tile - in batch dimension
    # giving 8 effective visual inputs assuming batch
    # size of 1
    visual_tiles = torch.randn(8, 3, 224, 224)

    # Dummy Egomotion History Input
    # Speed, Acceleration, Yaw Angle, Yaw Rate for
    # 6.4s past history giving 64 x 4 samples at 10Hz
    egomotion_history = torch.randn(256)

    # Dummy Visual Scene History
    # Length 14 compressed visual feature vector at 10Hz
    # for 6.4s past horizon giving 64 x 14 samples
    visual_history = torch.randn(896)
    
    # Run inference - returns trajectory and compressed
    # visual feature vector of the current scene alongside
    # a prediction of the future visual state of the scene
    # in feature space
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