#!/usr/bin/env python3
"""Quick demo script to run segmented energy spectrum comparison."""

import os
import sys

def main():
    """Run the segmented energy spectrum analysis with different configurations."""
    
    print("🔍 Segmented Energy Spectrum Analysis Demo")
    print("="*60)
    
    # Check if data files exist
    resolutions_to_check = ['64x64', '128x128']
    available_resolutions = []
    
    for res in resolutions_to_check:
        training_file = f"data/training_data/decaying_turbulence_v2_{res}_index_1.npz"
        pred_file = f"data/pict_data/pict_from_warmup_with_comparison_{res}_index_1.npz"
        
        if os.path.exists(training_file) and os.path.exists(pred_file):
            available_resolutions.append(res)
            print(f"✅ {res} data files found")
        else:
            print(f"❌ {res} data files missing:")
            if not os.path.exists(training_file):
                print(f"    Missing: {training_file}")
            if not os.path.exists(pred_file):
                print(f"    Missing: {pred_file}")
    
    if not available_resolutions:
        print("\n❌ No data files found! Please generate training and PICT data first.")
        return
    
    print(f"\n📊 Available resolutions: {available_resolutions}")
    
    # Different analysis configurations to try
    configs = [
        {"segment_size": 500, "max_timesteps": 2000, "description": "500-step segments, first 2000 timesteps"},
        {"segment_size": 1000, "max_timesteps": 5000, "description": "1000-step segments, first 5000 timesteps"},
        {"segment_size": 2000, "max_timesteps": 10000, "description": "2000-step segments, first 10000 timesteps"}
    ]
    
    print("\n🎯 Analysis Configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['description']}")
    
    # Let user choose or run default
    choice = input(f"\nSelect configuration (1-{len(configs)}) or press Enter for default [2]: ").strip()
    
    if choice == "":
        choice = "2"
    
    try:
        config_idx = int(choice) - 1
        if 0 <= config_idx < len(configs):
            selected_config = configs[config_idx]
        else:
            print("Invalid choice, using default configuration.")
            selected_config = configs[1]  # Default to config 2
    except ValueError:
        print("Invalid input, using default configuration.")
        selected_config = configs[1]
    
    print(f"\n🚀 Running analysis with: {selected_config['description']}")
    print(f"   Segment size: {selected_config['segment_size']}")
    print(f"   Max timesteps: {selected_config['max_timesteps']}")
    
    # Modify the main analysis script to use selected configuration
    import segmented_energy_spectrum_comparison as seg_analysis
    
    # Override the configuration in the main module
    seg_analysis.SEGMENT_SIZE = selected_config['segment_size']
    seg_analysis.MAX_TIMESTEPS = selected_config['max_timesteps']
    
    # Run the analysis
    try:
        seg_analysis.main()
        print("\n✅ Analysis completed successfully!")
        print("📁 Check the 'segmented_energy_spectrum_results/' directory for results")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
