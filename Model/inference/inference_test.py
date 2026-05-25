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

    # Dummy Visual Input
    # 7 cameras + 1 map tile - in batch dimension
    dummy_visual_input = torch.randn(8, 3, 224, 224)

    # Dummy Egomotion History Input
    # Speed, Acceleration, Yaw Angle, Yaw Rate for
    # 6.4s past history giving 64 x 4 samples at 10Hz
    dummy_egomotion_history_input = torch.randn(256)

    
    # Run inference
    output = model(dummy_visual_input, dummy_egomotion_history_input)

    # Print the output tensor shape
    print(output.shape)

if __name__ == "__main__":
    main()