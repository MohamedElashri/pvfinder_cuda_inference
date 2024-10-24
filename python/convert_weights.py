import torch
import numpy as np
import argparse

def convert_pytorch_to_npz(pth_path, npz_path):
    # Load PyTorch state dict
    state_dict = torch.load(pth_path, map_location='cpu')
    
    # Convert each tensor to numpy array
    numpy_dict = {}
    for key, value in state_dict.items():
        # Convert tensor to numpy and ensure it's float32
        numpy_dict[key] = value.detach().cpu().numpy().astype(np.float32)
        print(f"Converting {key}: shape = {numpy_dict[key].shape}")

    # Save as npz
    np.savez(npz_path, **numpy_dict)
    print(f"Saved weights to {npz_path}")
    
    # Verify the saved weights
    loaded = np.load(npz_path)
    print("\nVerifying saved weights:")
    for key in loaded.keys():
        print(f"{key}: shape = {loaded[key].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch weights to NPZ format')
    parser.add_argument('pth_path', help='Path to PyTorch .pth file')
    parser.add_argument('npz_path', help='Path to output .npz file')
    args = parser.parse_args()
    
    convert_pytorch_to_npz(args.pth_path, args.npz_path)