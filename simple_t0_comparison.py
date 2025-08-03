#!/usr/bin/env python3
"""Simple comparison of velocity values at t=0 between pict data and training data."""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

def compare_t0_simple(training_file: str, pict_file: str, resolution: str):
    """Simple comparison of t=0 velocity fields."""
    print(f"\n{'='*60}")
    print(f"Comparing t=0 velocities for {resolution}")
    print(f"{'='*60}")
    
    # Load data
    training_data = np.load(training_file)
    pict_data = np.load(pict_file)
    
    # Extract velocity fields at t=0
    training_u_t0 = training_data['u'][0]
    training_v_t0 = training_data['v'][0]
    pict_u_t0 = pict_data['u'][0]
    pict_v_t0 = pict_data['v'][0]
    
    print(f"Training data t=0: u={training_u_t0.shape}, v={training_v_t0.shape}")
    print(f"Pict data t=0:     u={pict_u_t0.shape}, v={pict_v_t0.shape}")
    
    # Compute differences
    diff_u = training_u_t0 - pict_u_t0
    diff_v = training_v_t0 - pict_v_t0
    
    # Check if they are identical
    is_identical_u = np.allclose(training_u_t0, pict_u_t0, rtol=1e-10, atol=1e-10)
    is_identical_v = np.allclose(training_v_t0, pict_v_t0, rtol=1e-10, atol=1e-10)
    
    print(f"\n🔍 Identical Check:")
    print(f"  U-velocity identical: {is_identical_u}")
    print(f"  V-velocity identical: {is_identical_v}")
    
    # Basic statistics
    print(f"\n📊 Training Data Statistics:")
    print(f"  U: min={np.min(training_u_t0):.6f}, max={np.max(training_u_t0):.6f}, mean={np.mean(training_u_t0):.6f}, std={np.std(training_u_t0):.6f}")
    print(f"  V: min={np.min(training_v_t0):.6f}, max={np.max(training_v_t0):.6f}, mean={np.mean(training_v_t0):.6f}, std={np.std(training_v_t0):.6f}")
    
    print(f"\n📊 Pict Data Statistics:")
    print(f"  U: min={np.min(pict_u_t0):.6f}, max={np.max(pict_u_t0):.6f}, mean={np.mean(pict_u_t0):.6f}, std={np.std(pict_u_t0):.6f}")
    print(f"  V: min={np.min(pict_v_t0):.6f}, max={np.max(pict_v_t0):.6f}, mean={np.mean(pict_v_t0):.6f}, std={np.std(pict_v_t0):.6f}")
    
    if not (is_identical_u and is_identical_v):
        print(f"\n📏 Difference Statistics:")
        print(f"  U diff: min={np.min(diff_u):.6f}, max={np.max(diff_u):.6f}, mean={np.mean(diff_u):.6f}, std={np.std(diff_u):.6f}")
        print(f"  V diff: min={np.min(diff_v):.6f}, max={np.max(diff_v):.6f}, mean={np.mean(diff_v):.6f}, std={np.std(diff_v):.6f}")
        
        print(f"\n📐 Error Metrics:")
        print(f"  RMSE U: {np.sqrt(np.mean(diff_u**2)):.6f}")
        print(f"  RMSE V: {np.sqrt(np.mean(diff_v**2)):.6f}")
        print(f"  MAE U:  {np.mean(np.abs(diff_u)):.6f}")
        print(f"  MAE V:  {np.mean(np.abs(diff_v)):.6f}")
        print(f"  Max absolute error U: {np.max(np.abs(diff_u)):.6f}")
        print(f"  Max absolute error V: {np.max(np.abs(diff_v)):.6f}")
        
        print(f"\n🔗 Correlations:")
        corr_u = np.corrcoef(training_u_t0.flatten(), pict_u_t0.flatten())[0, 1]
        corr_v = np.corrcoef(training_v_t0.flatten(), pict_v_t0.flatten())[0, 1]
        print(f"  U-velocity correlation: {corr_u:.6f}")
        print(f"  V-velocity correlation: {corr_v:.6f}")
    
    # Sample values comparison (first few grid points)
    print(f"\n🔢 Sample Values Comparison (first 3x3 grid points):")
    print(f"Training U[0:3, 0:3]:")
    print(training_u_t0[:3, :3])
    print(f"Pict U[0:3, 0:3]:")
    print(pict_u_t0[:3, :3])
    
    if not is_identical_u:
        print(f"U Difference[0:3, 0:3]:")
        print(diff_u[:3, :3])
    
    # Create simple visualization
    if not (is_identical_u and is_identical_v):
        create_simple_plots(training_u_t0, training_v_t0, pict_u_t0, pict_v_t0, 
                           diff_u, diff_v, resolution)
    else:
        print(f"\n✅ Velocity fields are identical at t=0 - no plots needed!")
    
    return {
        'resolution': resolution,
        'identical_u': is_identical_u,
        'identical_v': is_identical_v,
        'rmse_u': float(np.sqrt(np.mean(diff_u**2))) if not is_identical_u else 0.0,
        'rmse_v': float(np.sqrt(np.mean(diff_v**2))) if not is_identical_v else 0.0,
        'correlation_u': float(np.corrcoef(training_u_t0.flatten(), pict_u_t0.flatten())[0, 1]) if not is_identical_u else 1.0,
        'correlation_v': float(np.corrcoef(training_v_t0.flatten(), pict_v_t0.flatten())[0, 1]) if not is_identical_v else 1.0
    }

def create_simple_plots(training_u, training_v, pict_u, pict_v, diff_u, diff_v, resolution):
    """Create simple comparison plots."""
    try:
        # Create 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # U-velocity comparison
        im1 = axes[0, 0].imshow(training_u, cmap='RdBu_r', origin='lower')
        axes[0, 0].set_title(f'Training U (t=0)\n{resolution}')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(pict_u, cmap='RdBu_r', origin='lower')
        axes[0, 1].set_title(f'Pict U (t=0)\n{resolution}')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(diff_u, cmap='RdBu_r', origin='lower')
        axes[0, 2].set_title('U Difference\n(Training - Pict)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # V-velocity comparison
        im4 = axes[1, 0].imshow(training_v, cmap='RdBu_r', origin='lower')
        axes[1, 0].set_title('Training V (t=0)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(pict_v, cmap='RdBu_r', origin='lower')
        axes[1, 1].set_title('Pict V (t=0)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(diff_v, cmap='RdBu_r', origin='lower')
        axes[1, 2].set_title('V Difference\n(Training - Pict)')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        
        # Save with lower DPI to avoid size issues
        save_path = f't0_simple_comparison_{resolution}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Simple comparison plot saved: {save_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create plots: {str(e)}")

def main():
    """Main function."""
    resolutions = ['64x64', '128x128']
    results = {}
    
    for resolution in resolutions:
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        pict_file = f"data/pict_data/decaying_turbulence_{resolution}_index_1.npz"
        
        if not os.path.exists(training_file):
            print(f"❌ Training file not found: {training_file}")
            continue
        if not os.path.exists(pict_file):
            print(f"❌ Pict file not found: {pict_file}")
            continue
        
        try:
            result = compare_t0_simple(training_file, pict_file, resolution)
            results[resolution] = result
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
    
    # Save results
    with open('t0_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ t=0 comparison complete!")
    print("📄 Results saved to: t0_comparison_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()