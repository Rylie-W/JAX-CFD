#!/usr/bin/env python3
"""Check the structure of data files to understand their format."""

import numpy as np
import os

def check_data_file(filepath: str):
    """Check the structure of a .npz data file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking file: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
    print(f"{'='*60}")
    
    data = np.load(filepath)
    print(f"Keys in file: {list(data.keys())}")
    
    # Print detailed info for each array
    for key in data.keys():
        array = data[key]
        print(f"\nKey: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        print(f"  Min value: {np.min(array):.6f}")
        print(f"  Max value: {np.max(array):.6f}")
        print(f"  Mean value: {np.mean(array):.6f}")
        print(f"  Std value: {np.std(array):.6f}")
        
        # Print a few sample values
        if array.size > 0:
            flat_array = array.flatten()
            sample_size = min(5, len(flat_array))
            print(f"  Sample values: {flat_array[:sample_size]}")

def main():
    """Check structure of both training and predicted data files."""
    # Check smaller files first
    files_to_check = [
        "data/training_data/decaying_turbulence_v2_64x64_index_1.npz",
        "data/pict_data/decaying_turbulence_64x64_index_1.npz",
    ]
    
    for filepath in files_to_check:
        check_data_file(filepath)

if __name__ == "__main__":
    main() 