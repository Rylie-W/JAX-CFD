#!/usr/bin/env python3
"""Visualize and compare the first 5 timesteps between training data and pict data."""

import os
# Force JAX to use CPU to avoid CUDA/cuDNN issues
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path

def visualize_first_5_steps(training_file: str, pict_file: str, resolution: str):
    """Create detailed visualization of the first 5 timesteps."""
    print(f"\n{'='*60}")
    print(f"Visualizing first 5 timesteps for {resolution}")
    print(f"Training: {training_file}")
    print(f"Pict: {pict_file}")
    print(f"{'='*60}")
    
    # Load data
    training_data = np.load(training_file)
    pict_data = np.load(pict_file)
    
    # Extract first 5 timesteps
    n_steps = 5
    
    training_u = training_data['u'][:n_steps]
    training_v = training_data['v'][:n_steps]
    pict_u = pict_data['u'][:n_steps]
    pict_v = pict_data['v'][:n_steps]
    
    print(f"Training data shapes: u={training_u.shape}, v={training_v.shape}")
    print(f"Pict data shapes: u={pict_u.shape}, v={pict_v.shape}")
    
    # Create visualizations
    create_detailed_comparison(training_u, training_v, pict_u, pict_v, resolution)
    create_difference_evolution(training_u, training_v, pict_u, pict_v, resolution)
    create_statistics_comparison(training_u, training_v, pict_u, pict_v, resolution)

def create_detailed_comparison(training_u, training_v, pict_u, pict_v, resolution):
    """Create detailed field comparison for first 5 timesteps."""
    
    # Create output directory
    output_dir = Path("first_5_steps_visualization")
    output_dir.mkdir(exist_ok=True)
    
    n_steps = training_u.shape[0]
    
    # Create main comparison figure
    fig = plt.figure(figsize=(25, 20))
    
    # Calculate common color limits for better comparison
    all_u = np.concatenate([training_u.flatten(), pict_u.flatten()])
    all_v = np.concatenate([training_v.flatten(), pict_v.flatten()])
    u_vmin, u_vmax = np.percentile(all_u, [2, 98])
    v_vmin, v_vmax = np.percentile(all_v, [2, 98])
    
    for t in range(n_steps):
        # Training U-velocity
        plt.subplot(6, n_steps, t + 1)
        im = plt.imshow(training_u[t], cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
        plt.title(f'Training U\nt={t}', fontsize=12)
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # Pict U-velocity
        plt.subplot(6, n_steps, n_steps + t + 1)
        im = plt.imshow(pict_u[t], cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
        plt.title(f'Pict U\nt={t}', fontsize=12)
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # U-velocity difference
        plt.subplot(6, n_steps, 2*n_steps + t + 1)
        diff_u = training_u[t] - pict_u[t]
        diff_max = max(abs(np.min(diff_u)), abs(np.max(diff_u)))
        im = plt.imshow(diff_u, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        rmse_u = np.sqrt(np.mean(diff_u**2))
        plt.title(f'U Diff\nt={t}\nRMSE={rmse_u:.3f}', fontsize=12)
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # Training V-velocity
        plt.subplot(6, n_steps, 3*n_steps + t + 1)
        im = plt.imshow(training_v[t], cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
        plt.title(f'Training V\nt={t}', fontsize=12)
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # Pict V-velocity
        plt.subplot(6, n_steps, 4*n_steps + t + 1)
        im = plt.imshow(pict_v[t], cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
        plt.title(f'Pict V\nt={t}', fontsize=12)
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # V-velocity difference
        plt.subplot(6, n_steps, 5*n_steps + t + 1)
        diff_v = training_v[t] - pict_v[t]
        diff_max = max(abs(np.min(diff_v)), abs(np.max(diff_v)))
        im = plt.imshow(diff_v, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        rmse_v = np.sqrt(np.mean(diff_v**2))
        plt.title(f'V Diff\nt={t}\nRMSE={rmse_v:.3f}', fontsize=12)
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'detailed_comparison_{resolution}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Detailed comparison saved to: {output_file}")
    plt.close()

def create_difference_evolution(training_u, training_v, pict_u, pict_v, resolution):
    """Create difference evolution visualization."""
    
    output_dir = Path("first_5_steps_visualization")
    
    n_steps = training_u.shape[0]
    
    # Calculate statistics for each timestep
    rmse_u = []
    rmse_v = []
    corr_u = []
    corr_v = []
    energy_training = []
    energy_pict = []
    
    for t in range(n_steps):
        # RMSE
        diff_u = training_u[t] - pict_u[t]
        diff_v = training_v[t] - pict_v[t]
        rmse_u.append(np.sqrt(np.mean(diff_u**2)))
        rmse_v.append(np.sqrt(np.mean(diff_v**2)))
        
        # Correlation
        corr_u.append(np.corrcoef(training_u[t].flatten(), pict_u[t].flatten())[0, 1])
        corr_v.append(np.corrcoef(training_v[t].flatten(), pict_v[t].flatten())[0, 1])
        
        # Energy
        energy_training.append(0.5 * np.mean(training_u[t]**2 + training_v[t]**2))
        energy_pict.append(0.5 * np.mean(pict_u[t]**2 + pict_v[t]**2))
    
    # Create evolution plots
    fig = plt.figure(figsize=(20, 12))
    timesteps = range(n_steps)
    
    # RMSE evolution
    plt.subplot(2, 3, 1)
    plt.plot(timesteps, rmse_u, 'bo-', label='U-velocity', linewidth=3, markersize=8)
    plt.plot(timesteps, rmse_v, 'ro-', label='V-velocity', linewidth=3, markersize=8)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE Evolution\n(Training vs Pict)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    for i, (u, v) in enumerate(zip(rmse_u, rmse_v)):
        plt.text(i, u + 0.02, f'{u:.3f}', ha='center', va='bottom', fontsize=10)
        plt.text(i, v - 0.02, f'{v:.3f}', ha='center', va='top', fontsize=10)
    
    # Correlation evolution
    plt.subplot(2, 3, 2)
    plt.plot(timesteps, corr_u, 'bo-', label='U-velocity', linewidth=3, markersize=8)
    plt.plot(timesteps, corr_v, 'ro-', label='V-velocity', linewidth=3, markersize=8)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.title('Correlation Evolution\n(Training vs Pict)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    for i, (u, v) in enumerate(zip(corr_u, corr_v)):
        plt.text(i, u + 0.03, f'{u:.3f}', ha='center', va='bottom', fontsize=10)
        plt.text(i, v - 0.03, f'{v:.3f}', ha='center', va='top', fontsize=10)
    
    # Energy evolution
    plt.subplot(2, 3, 3)
    plt.plot(timesteps, energy_training, 'go-', label='Training', linewidth=3, markersize=8)
    plt.plot(timesteps, energy_pict, 'mo-', label='Pict', linewidth=3, markersize=8)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Kinetic Energy', fontsize=12)
    plt.title('Kinetic Energy Evolution', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Velocity magnitude comparison at each timestep
    for t in range(min(3, n_steps)):  # Show first 3 timesteps
        plt.subplot(2, 3, 4 + t)
        
        # Calculate velocity magnitudes
        training_mag = np.sqrt(training_u[t]**2 + training_v[t]**2)
        pict_mag = np.sqrt(pict_u[t]**2 + pict_v[t]**2)
        
        # Create scatter plot
        plt.scatter(pict_mag.flatten()[::10], training_mag.flatten()[::10], 
                   alpha=0.6, s=3, c='blue')
        
        # Add perfect correlation line
        max_mag = max(np.max(training_mag), np.max(pict_mag))
        plt.plot([0, max_mag], [0, max_mag], 'r--', alpha=0.7, linewidth=2)
        
        plt.xlabel('Pict Velocity Magnitude', fontsize=12)
        plt.ylabel('Training Velocity Magnitude', fontsize=12)
        plt.title(f'Velocity Magnitude\nCorrelation at t={t}', fontsize=12)
        
        # Add correlation coefficient
        corr_mag = np.corrcoef(training_mag.flatten(), pict_mag.flatten())[0, 1]
        plt.text(0.05, 0.95, f'r = {corr_mag:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'evolution_analysis_{resolution}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Evolution analysis saved to: {output_file}")
    plt.close()

def create_statistics_comparison(training_u, training_v, pict_u, pict_v, resolution):
    """Create statistics comparison table."""
    
    output_dir = Path("first_5_steps_visualization")
    
    n_steps = training_u.shape[0]
    
    # Create statistics figure
    fig = plt.figure(figsize=(15, 10))
    
    # Collect statistics
    stats_data = []
    
    for t in range(n_steps):
        # Training stats
        train_u_mean = np.mean(training_u[t])
        train_u_std = np.std(training_u[t])
        train_u_min = np.min(training_u[t])
        train_u_max = np.max(training_u[t])
        
        train_v_mean = np.mean(training_v[t])
        train_v_std = np.std(training_v[t])
        train_v_min = np.min(training_v[t])
        train_v_max = np.max(training_v[t])
        
        # Pict stats
        pict_u_mean = np.mean(pict_u[t])
        pict_u_std = np.std(pict_u[t])
        pict_u_min = np.min(pict_u[t])
        pict_u_max = np.max(pict_u[t])
        
        pict_v_mean = np.mean(pict_v[t])
        pict_v_std = np.std(pict_v[t])
        pict_v_min = np.min(pict_v[t])
        pict_v_max = np.max(pict_v[t])
        
        # Differences
        diff_u = training_u[t] - pict_u[t]
        diff_v = training_v[t] - pict_v[t]
        rmse_u = np.sqrt(np.mean(diff_u**2))
        rmse_v = np.sqrt(np.mean(diff_v**2))
        
        # Correlations
        corr_u = np.corrcoef(training_u[t].flatten(), pict_u[t].flatten())[0, 1]
        corr_v = np.corrcoef(training_v[t].flatten(), pict_v[t].flatten())[0, 1]
        
        stats_data.append({
            'timestep': t,
            'train_u_mean': train_u_mean,
            'train_u_std': train_u_std,
            'train_u_range': train_u_max - train_u_min,
            'pict_u_mean': pict_u_mean,
            'pict_u_std': pict_u_std,
            'pict_u_range': pict_u_max - pict_u_min,
            'rmse_u': rmse_u,
            'corr_u': corr_u,
            'train_v_mean': train_v_mean,
            'train_v_std': train_v_std,
            'train_v_range': train_v_max - train_v_min,
            'pict_v_mean': pict_v_mean,
            'pict_v_std': pict_v_std,
            'pict_v_range': pict_v_max - pict_v_min,
            'rmse_v': rmse_v,
            'corr_v': corr_v
        })
    
    # Create text summary
    plt.subplot(1, 1, 1)
    plt.axis('off')
    
    summary_text = f"First 5 Timesteps Comparison - {resolution}\n"
    summary_text += "="*60 + "\n\n"
    
    summary_text += "Timestep | U-velocity                    | V-velocity\n"
    summary_text += "         | RMSE    Correlation         | RMSE    Correlation\n"
    summary_text += "-"*60 + "\n"
    
    for data in stats_data:
        summary_text += f"   {data['timestep']}     | {data['rmse_u']:.4f}     {data['corr_u']:.4f}        | {data['rmse_v']:.4f}     {data['corr_v']:.4f}\n"
    
    summary_text += "\n" + "-"*60 + "\n\n"
    
    summary_text += "Detailed Statistics:\n\n"
    
    for data in stats_data:
        summary_text += f"Timestep {data['timestep']}:\n"
        summary_text += f"  Training U: mean={data['train_u_mean']:.4f}, std={data['train_u_std']:.4f}, range={data['train_u_range']:.4f}\n"
        summary_text += f"  Pict U:     mean={data['pict_u_mean']:.4f}, std={data['pict_u_std']:.4f}, range={data['pict_u_range']:.4f}\n"
        summary_text += f"  Training V: mean={data['train_v_mean']:.4f}, std={data['train_v_std']:.4f}, range={data['train_v_range']:.4f}\n"
        summary_text += f"  Pict V:     mean={data['pict_v_mean']:.4f}, std={data['pict_v_std']:.4f}, range={data['pict_v_range']:.4f}\n"
        summary_text += f"  U Comparison: RMSE={data['rmse_u']:.4f}, Correlation={data['corr_u']:.4f}\n"
        summary_text += f"  V Comparison: RMSE={data['rmse_v']:.4f}, Correlation={data['corr_v']:.4f}\n\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'statistics_summary_{resolution}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Statistics summary saved to: {output_file}")
    plt.close()

def main():
    """Main function to visualize both resolutions."""
    resolutions = ['64x64', '128x128']
    
    for resolution in resolutions:
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        pict_file = f"data/pict_data/pict_from_warmup_with_comparison_{resolution}_index_1.npz"
        
        if not os.path.exists(training_file):
            print(f"❌ Training file not found: {training_file}")
            continue
        if not os.path.exists(pict_file):
            print(f"❌ Pict file not found: {pict_file}")
            continue
        
        try:
            visualize_first_5_steps(training_file, pict_file, resolution)
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ First 5 timesteps visualization complete!")
    print(f"📊 Visualizations saved to: first_5_steps_visualization/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()