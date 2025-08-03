#!/usr/bin/env python3
"""Compare and visualize the first timestep between training data and pict data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json
from pathlib import Path

def compare_first_timestep(training_file: str, pict_file: str, resolution: str):
    """Compare the first timestep between training and pict data."""
    print(f"\n{'='*60}")
    print(f"Comparing first timestep for {resolution}")
    print(f"Training: {training_file}")
    print(f"Pict: {pict_file}")
    print(f"{'='*60}")
    
    # Load data
    training_data = np.load(training_file)
    pict_data = np.load(pict_file)
    
    # Extract first timestep velocity fields
    training_u_0 = training_data['u'][0]  # First saved timestep
    training_v_0 = training_data['v'][0]
    pict_u_0 = pict_data['u'][0]          # First timestep  
    pict_v_0 = pict_data['v'][0]
    
    print(f"Training data shapes: u={training_u_0.shape}, v={training_v_0.shape}")
    print(f"Pict data shapes: u={pict_u_0.shape}, v={pict_v_0.shape}")
    
    # Compute differences
    diff_u = training_u_0 - pict_u_0
    diff_v = training_v_0 - pict_v_0
    
    # Check if identical
    is_identical_u = np.allclose(training_u_0, pict_u_0, rtol=1e-10, atol=1e-10)
    is_identical_v = np.allclose(training_v_0, pict_v_0, rtol=1e-10, atol=1e-10)
    
    print(f"\n🔍 Identical Check:")
    print(f"  U-velocity identical: {is_identical_u}")
    print(f"  V-velocity identical: {is_identical_v}")
    
    # Compute statistics
    print(f"\n📊 Training Data Statistics:")
    print(f"  U: min={np.min(training_u_0):.6f}, max={np.max(training_u_0):.6f}, mean={np.mean(training_u_0):.6f}, std={np.std(training_u_0):.6f}")
    print(f"  V: min={np.min(training_v_0):.6f}, max={np.max(training_v_0):.6f}, mean={np.mean(training_v_0):.6f}, std={np.std(training_v_0):.6f}")
    
    print(f"\n📊 Pict Data Statistics:")
    print(f"  U: min={np.min(pict_u_0):.6f}, max={np.max(pict_u_0):.6f}, mean={np.mean(pict_u_0):.6f}, std={np.std(pict_u_0):.6f}")
    print(f"  V: min={np.min(pict_v_0):.6f}, max={np.max(pict_v_0):.6f}, mean={np.mean(pict_v_0):.6f}, std={np.std(pict_v_0):.6f}")
    
    if not (is_identical_u and is_identical_v):
        print(f"\n📏 Difference Statistics:")
        print(f"  U diff: min={np.min(diff_u):.6f}, max={np.max(diff_u):.6f}, mean={np.mean(diff_u):.6f}, std={np.std(diff_u):.6f}")
        print(f"  V diff: min={np.min(diff_v):.6f}, max={np.max(diff_v):.6f}, mean={np.mean(diff_v):.6f}, std={np.std(diff_v):.6f}")
        
        # RMSE
        rmse_u = np.sqrt(np.mean(diff_u**2))
        rmse_v = np.sqrt(np.mean(diff_v**2))
        print(f"  RMSE - U: {rmse_u:.6f}, V: {rmse_v:.6f}")
        
        # Correlation
        corr_u = np.corrcoef(training_u_0.flatten(), pict_u_0.flatten())[0, 1]
        corr_v = np.corrcoef(training_v_0.flatten(), pict_v_0.flatten())[0, 1]
        print(f"  Correlation - U: {corr_u:.6f}, V: {corr_v:.6f}")
    
    # Create visualization
    create_comparison_plots(training_u_0, training_v_0, pict_u_0, pict_v_0, 
                          diff_u, diff_v, resolution, is_identical_u, is_identical_v)
    
    return {
        'resolution': resolution,
        'identical_u': is_identical_u,
        'identical_v': is_identical_v,
        'rmse_u': float(np.sqrt(np.mean(diff_u**2))) if not is_identical_u else 0.0,
        'rmse_v': float(np.sqrt(np.mean(diff_v**2))) if not is_identical_v else 0.0,
        'correlation_u': float(np.corrcoef(training_u_0.flatten(), pict_u_0.flatten())[0, 1]) if not is_identical_u else 1.0,
        'correlation_v': float(np.corrcoef(training_v_0.flatten(), pict_v_0.flatten())[0, 1]) if not is_identical_v else 1.0,
        'training_stats': {
            'u_min': float(np.min(training_u_0)), 'u_max': float(np.max(training_u_0)),
            'u_mean': float(np.mean(training_u_0)), 'u_std': float(np.std(training_u_0)),
            'v_min': float(np.min(training_v_0)), 'v_max': float(np.max(training_v_0)),
            'v_mean': float(np.mean(training_v_0)), 'v_std': float(np.std(training_v_0)),
        },
        'pict_stats': {
            'u_min': float(np.min(pict_u_0)), 'u_max': float(np.max(pict_u_0)),
            'u_mean': float(np.mean(pict_u_0)), 'u_std': float(np.std(pict_u_0)),
            'v_min': float(np.min(pict_v_0)), 'v_max': float(np.max(pict_v_0)),
            'v_mean': float(np.mean(pict_v_0)), 'v_std': float(np.std(pict_v_0)),
        }
    }

def create_comparison_plots(training_u, training_v, pict_u, pict_v, diff_u, diff_v, 
                          resolution, is_identical_u, is_identical_v):
    """Create comprehensive comparison plots."""
    
    # Create output directory
    output_dir = Path("first_timestep_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Common settings for colormaps
    u_vmin = min(np.min(training_u), np.min(pict_u))
    u_vmax = max(np.max(training_u), np.max(pict_u))
    v_vmin = min(np.min(training_v), np.min(pict_v))
    v_vmax = max(np.max(training_v), np.max(pict_v))
    
    # U-velocity comparison
    # Training U
    plt.subplot(4, 3, 1)
    im1 = plt.imshow(training_u, cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
    plt.title(f'Training U-velocity\n(First timestep after warmup)')
    plt.colorbar(im1, shrink=0.6)
    
    # Pict U  
    plt.subplot(4, 3, 2)
    im2 = plt.imshow(pict_u, cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
    plt.title(f'Pict U-velocity\n(Initial condition)')
    plt.colorbar(im2, shrink=0.6)
    
    # U difference
    plt.subplot(4, 3, 3)
    diff_u_max = max(abs(np.min(diff_u)), abs(np.max(diff_u)))
    im3 = plt.imshow(diff_u, cmap='RdBu_r', vmin=-diff_u_max, vmax=diff_u_max)
    status_u = "IDENTICAL" if is_identical_u else "DIFFERENT"
    plt.title(f'U Difference (Training - Pict)\n{status_u}')
    plt.colorbar(im3, shrink=0.6)
    
    # V-velocity comparison
    # Training V
    plt.subplot(4, 3, 4)
    im4 = plt.imshow(training_v, cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
    plt.title(f'Training V-velocity\n(First timestep after warmup)')
    plt.colorbar(im4, shrink=0.6)
    
    # Pict V
    plt.subplot(4, 3, 5)
    im5 = plt.imshow(pict_v, cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
    plt.title(f'Pict V-velocity\n(Initial condition)')
    plt.colorbar(im5, shrink=0.6)
    
    # V difference
    plt.subplot(4, 3, 6)
    diff_v_max = max(abs(np.min(diff_v)), abs(np.max(diff_v)))
    im6 = plt.imshow(diff_v, cmap='RdBu_r', vmin=-diff_v_max, vmax=diff_v_max)
    status_v = "IDENTICAL" if is_identical_v else "DIFFERENT"
    plt.title(f'V Difference (Training - Pict)\n{status_v}')
    plt.colorbar(im6, shrink=0.6)
    
    # Scatter plots for correlation analysis
    plt.subplot(4, 3, 7)
    plt.scatter(pict_u.flatten()[::10], training_u.flatten()[::10], alpha=0.5, s=1)
    plt.xlabel('Pict U-velocity')
    plt.ylabel('Training U-velocity')
    plt.title('U-velocity Correlation')
    plt.plot([u_vmin, u_vmax], [u_vmin, u_vmax], 'r--', alpha=0.7)
    if not is_identical_u:
        corr_u = np.corrcoef(training_u.flatten(), pict_u.flatten())[0, 1]
        plt.text(0.05, 0.95, f'r = {corr_u:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.subplot(4, 3, 8)
    plt.scatter(pict_v.flatten()[::10], training_v.flatten()[::10], alpha=0.5, s=1)
    plt.xlabel('Pict V-velocity')
    plt.ylabel('Training V-velocity')
    plt.title('V-velocity Correlation')
    plt.plot([v_vmin, v_vmax], [v_vmin, v_vmax], 'r--', alpha=0.7)
    if not is_identical_v:
        corr_v = np.corrcoef(training_v.flatten(), pict_v.flatten())[0, 1]
        plt.text(0.05, 0.95, f'r = {corr_v:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Histograms of differences
    plt.subplot(4, 3, 9)
    if not is_identical_u:
        plt.hist(diff_u.flatten(), bins=50, alpha=0.7, density=True)
        plt.xlabel('U Difference')
        plt.ylabel('Density')
        plt.title('Distribution of U Differences')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'U velocities are\nidentical', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('U Difference Distribution')
    
    plt.subplot(4, 3, 10)
    if not is_identical_v:
        plt.hist(diff_v.flatten(), bins=50, alpha=0.7, density=True)
        plt.xlabel('V Difference')
        plt.ylabel('Density')
        plt.title('Distribution of V Differences')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'V velocities are\nidentical', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('V Difference Distribution')
    
    # Summary statistics
    plt.subplot(4, 3, 11)
    plt.axis('off')
    stats_text = f"""Resolution: {resolution}
    
Training Data (First timestep after warmup):
• U: min={np.min(training_u):.4f}, max={np.max(training_u):.4f}
• V: min={np.min(training_v):.4f}, max={np.max(training_v):.4f}

Pict Data (Initial condition):  
• U: min={np.min(pict_u):.4f}, max={np.max(pict_u):.4f}
• V: min={np.min(pict_v):.4f}, max={np.max(pict_v):.4f}

Comparison Results:
• U identical: {is_identical_u}
• V identical: {is_identical_v}
"""
    
    if not (is_identical_u and is_identical_v):
        rmse_u = np.sqrt(np.mean(diff_u**2))
        rmse_v = np.sqrt(np.mean(diff_v**2))
        stats_text += f"""
• RMSE U: {rmse_u:.6f}
• RMSE V: {rmse_v:.6f}"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Magnitude comparison
    plt.subplot(4, 3, 12)
    training_mag = np.sqrt(training_u**2 + training_v**2)
    pict_mag = np.sqrt(pict_u**2 + pict_v**2)
    
    mag_vmin = min(np.min(training_mag), np.min(pict_mag))
    mag_vmax = max(np.max(training_mag), np.max(pict_mag))
    
    im12 = plt.imshow(training_mag - pict_mag, cmap='RdBu_r')
    plt.title('Velocity Magnitude Difference\n(Training - Pict)')
    plt.colorbar(im12, shrink=0.6)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'first_timestep_comparison_{resolution}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.close()

def main():
    """Main function to compare both resolutions."""
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
            result = compare_first_timestep(training_file, pict_file, resolution)
            results[resolution] = result
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
    
    # Save results
    output_file = 'first_timestep_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ First timestep comparison complete!")
    print(f"📄 Results saved to: {output_file}")
    print(f"📊 Visualizations saved to: first_timestep_comparison/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()