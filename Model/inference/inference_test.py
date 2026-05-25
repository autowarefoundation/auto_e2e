import torch
import sys
sys.path.append('..')
from model_components.auto_fsd import AutoFSD

def main():
    # Device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')
            
    # Instantiate model
    model = AutoFSD()

    # Dummy Visual Scene Input
    # 7 cameras + 1 map tile - in batch dimension
    # giving 8 effective visual inputs
    visual_tiles = torch.randn(8, 3, 224, 224)

    # Dummy Egomotion History Input
    # Speed, Acceleration, Yaw Angle, Yaw Rate for
    # 6.4s past history giving 64 x 4 samples at 10Hz
    egomotion_history = torch.randn(256)

    # Dummy Visual Scene History
    # Length 14 compressed visual feature vector at 10Hz
    # for 6.4s past horizon giving 896 samples
    visual_history = torch.randn(896)
    
    # Run inference - returns trajectory and compressed
    # visual feature vector of the current scene
    trajectory, compressed_visual_feature_vector = \
        model(visual_tiles, visual_history, egomotion_history)

    # Print the output tensor shapes
    print("Trajectory : ", trajectory.shape)
    print("Visual Feature Vector : ", compressed_visual_feature_vector.shape)

if __name__ == "__main__":
    main()