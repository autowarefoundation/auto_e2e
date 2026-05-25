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

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Run inference
    output = model(dummy_input)

    # Print the output tensor shape
    print(output.shape)

if __name__ == "__main__":
    main()