#!/usr/bin/env python3
"""Compare and visualize the first 50 timesteps between training data and pict data."""

import os
# Force JAX to use CPU to avoid CUDA/cuDNN issues
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
import json
from pathlib import Path

# Additional JAX configuration for CPU
jax.config.update('jax_platform_name', 'cpu')

def compare_first_50_timesteps(training_file: str, pict_file: str, resolution: str):
    """Compare the first 50 timesteps between training and pict data."""
    print(f"\n{'='*60}")
    print(f"Comparing first 50 timesteps for {resolution}")
    print(f"Training: {training_file}")
    print(f"Pict: {pict_file}")
    print(f"{'='*60}")
    
    # Load data
    training_data = np.load(training_file)
    pict_data = np.load(pict_file)
    
    # Extract first 50 timesteps
    n_timesteps = 50
    training_u = training_data['u'][:n_timesteps]  # First 50 timesteps after warmup
    training_v = training_data['v'][:n_timesteps]
    pict_u = pict_data['u'][:n_timesteps]          # First 50 timesteps from initial condition
    pict_v = pict_data['v'][:n_timesteps]
    
    print(f"Training data shapes: u={training_u.shape}, v={training_v.shape}")
    print(f"Pict data shapes: u={pict_u.shape}, v={pict_v.shape}")
    
    # Compute temporal statistics
    results = compute_temporal_analysis(training_u, training_v, pict_u, pict_v, resolution)
    
    # Create visualizations
    create_temporal_comparison_plots(training_u, training_v, pict_u, pict_v, results, resolution)
    
    return results

def compute_temporal_analysis(training_u, training_v, pict_u, pict_v, resolution):
    """Compute comprehensive temporal analysis."""
    print("Computing temporal analysis...")
    
    n_timesteps = training_u.shape[0]
    results = {
        'resolution': resolution,
        'n_timesteps': n_timesteps,
        'timestep_analysis': {},
        'energy_evolution': {},
        'field_statistics': {}
    }
    
    # 1. Timestep-by-timestep analysis
    print("  1. Computing timestep-by-timestep differences...")
    rmse_u_evolution = []
    rmse_v_evolution = []
    corr_u_evolution = []
    corr_v_evolution = []
    
    for t in range(n_timesteps):
        # RMSE
        diff_u = training_u[t] - pict_u[t]
        diff_v = training_v[t] - pict_v[t]
        rmse_u = float(np.sqrt(np.mean(diff_u**2)))
        rmse_v = float(np.sqrt(np.mean(diff_v**2)))
        rmse_u_evolution.append(rmse_u)
        rmse_v_evolution.append(rmse_v)
        
        # Correlation
        corr_u = float(np.corrcoef(training_u[t].flatten(), pict_u[t].flatten())[0, 1])
        corr_v = float(np.corrcoef(training_v[t].flatten(), pict_v[t].flatten())[0, 1])
        corr_u_evolution.append(corr_u)
        corr_v_evolution.append(corr_v)
    
    results['timestep_analysis'] = {
        'rmse_u_evolution': rmse_u_evolution,
        'rmse_v_evolution': rmse_v_evolution,
        'corr_u_evolution': corr_u_evolution,
        'corr_v_evolution': corr_v_evolution
    }
    
    # 2. Energy evolution analysis
    print("  2. Computing energy evolution...")
    training_energy = []
    pict_energy = []
    
    for t in range(n_timesteps):
        # Kinetic energy = 0.5 * (u^2 + v^2)
        training_ke = float(0.5 * np.mean(training_u[t]**2 + training_v[t]**2))
        pict_ke = float(0.5 * np.mean(pict_u[t]**2 + pict_v[t]**2))
        training_energy.append(training_ke)
        pict_energy.append(pict_ke)
    
    results['energy_evolution'] = {
        'training_energy': training_energy,
        'pict_energy': pict_energy,
        'energy_difference': [t - p for t, p in zip(training_energy, pict_energy)]
    }
    
    # 3. Field statistics evolution
    print("  3. Computing field statistics evolution...")
    training_u_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    training_v_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    pict_u_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    pict_v_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    
    for t in range(n_timesteps):
        # Training stats
        training_u_stats['mean'].append(float(np.mean(training_u[t])))
        training_u_stats['std'].append(float(np.std(training_u[t])))
        training_u_stats['min'].append(float(np.min(training_u[t])))
        training_u_stats['max'].append(float(np.max(training_u[t])))
        
        training_v_stats['mean'].append(float(np.mean(training_v[t])))
        training_v_stats['std'].append(float(np.std(training_v[t])))
        training_v_stats['min'].append(float(np.min(training_v[t])))
        training_v_stats['max'].append(float(np.max(training_v[t])))
        
        # Pict stats
        pict_u_stats['mean'].append(float(np.mean(pict_u[t])))
        pict_u_stats['std'].append(float(np.std(pict_u[t])))
        pict_u_stats['min'].append(float(np.min(pict_u[t])))
        pict_u_stats['max'].append(float(np.max(pict_u[t])))
        
        pict_v_stats['mean'].append(float(np.mean(pict_v[t])))
        pict_v_stats['std'].append(float(np.std(pict_v[t])))
        pict_v_stats['min'].append(float(np.min(pict_v[t])))
        pict_v_stats['max'].append(float(np.max(pict_v[t])))
    
    results['field_statistics'] = {
        'training_u': training_u_stats,
        'training_v': training_v_stats,
        'pict_u': pict_u_stats,
        'pict_v': pict_v_stats
    }
    
    # 4. Summary statistics
    print("  4. Computing summary statistics...")
    results['summary'] = {
        'avg_rmse_u': float(np.mean(rmse_u_evolution)),
        'avg_rmse_v': float(np.mean(rmse_v_evolution)),
        'avg_corr_u': float(np.mean(corr_u_evolution)),
        'avg_corr_v': float(np.mean(corr_v_evolution)),
        'initial_rmse_u': float(rmse_u_evolution[0]),
        'initial_rmse_v': float(rmse_v_evolution[0]),
        'final_rmse_u': float(rmse_u_evolution[-1]),
        'final_rmse_v': float(rmse_v_evolution[-1]),
        'initial_corr_u': float(corr_u_evolution[0]),
        'initial_corr_v': float(corr_v_evolution[0]),
        'final_corr_u': float(corr_u_evolution[-1]),
        'final_corr_v': float(corr_v_evolution[-1])
    }
    
    return results

def create_temporal_comparison_plots(training_u, training_v, pict_u, pict_v, results, resolution):
    """Create comprehensive temporal comparison plots."""
    
    # Create output directory
    output_dir = Path("first_50_timesteps_comparison")
    output_dir.mkdir(exist_ok=True)
    
    n_timesteps = training_u.shape[0]
    timesteps = np.arange(n_timesteps)
    
    # Create main comparison figure
    fig = plt.figure(figsize=(24, 20))
    
    # 1. RMSE Evolution
    plt.subplot(4, 4, 1)
    plt.plot(timesteps, results['timestep_analysis']['rmse_u_evolution'], 'b-', label='U-velocity', linewidth=2)
    plt.plot(timesteps, results['timestep_analysis']['rmse_v_evolution'], 'r-', label='V-velocity', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('RMSE')
    plt.title('RMSE Evolution\n(Training vs Pict)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Correlation Evolution
    plt.subplot(4, 4, 2)
    plt.plot(timesteps, results['timestep_analysis']['corr_u_evolution'], 'b-', label='U-velocity', linewidth=2)
    plt.plot(timesteps, results['timestep_analysis']['corr_v_evolution'], 'r-', label='V-velocity', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Correlation')
    plt.title('Correlation Evolution\n(Training vs Pict)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # 3. Energy Evolution
    plt.subplot(4, 4, 3)
    plt.plot(timesteps, results['energy_evolution']['training_energy'], 'g-', label='Training', linewidth=2)
    plt.plot(timesteps, results['energy_evolution']['pict_energy'], 'orange', label='Pict', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Kinetic Energy')
    plt.title('Kinetic Energy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Energy Difference
    plt.subplot(4, 4, 4)
    plt.plot(timesteps, results['energy_evolution']['energy_difference'], 'purple', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Energy Difference\n(Training - Pict)')
    plt.title('Energy Difference Evolution')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 5. U-velocity Standard Deviation Evolution
    plt.subplot(4, 4, 5)
    plt.plot(timesteps, results['field_statistics']['training_u']['std'], 'b-', label='Training', linewidth=2)
    plt.plot(timesteps, results['field_statistics']['pict_u']['std'], 'cyan', label='Pict', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Standard Deviation')
    plt.title('U-velocity Std Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. V-velocity Standard Deviation Evolution
    plt.subplot(4, 4, 6)
    plt.plot(timesteps, results['field_statistics']['training_v']['std'], 'r-', label='Training', linewidth=2)
    plt.plot(timesteps, results['field_statistics']['pict_v']['std'], 'pink', label='Pict', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Standard Deviation')
    plt.title('V-velocity Std Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Sample velocity fields at different timesteps
    sample_timesteps = [0, 10, 25, 49]
    for i, t in enumerate(sample_timesteps):
        plt.subplot(4, 4, 7 + i)
        
        # Show velocity magnitude difference
        training_mag = np.sqrt(training_u[t]**2 + training_v[t]**2)
        pict_mag = np.sqrt(pict_u[t]**2 + pict_v[t]**2)
        diff_mag = training_mag - pict_mag
        
        im = plt.imshow(diff_mag, cmap='RdBu_r')
        plt.title(f'Velocity Mag Diff at t={t}\n(Training - Pict)')
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
    
    # 8. Velocity field snapshots comparison
    plt.subplot(4, 4, 11)
    # Training velocity magnitude at t=0
    training_mag_0 = np.sqrt(training_u[0]**2 + training_v[0]**2)
    im = plt.imshow(training_mag_0, cmap='viridis')
    plt.title('Training Velocity Mag\nat t=0')
    plt.colorbar(im, shrink=0.6)
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    # Pict velocity magnitude at t=0
    pict_mag_0 = np.sqrt(pict_u[0]**2 + pict_v[0]**2)
    im = plt.imshow(pict_mag_0, cmap='viridis')
    plt.title('Pict Velocity Mag\nat t=0')
    plt.colorbar(im, shrink=0.6)
    plt.axis('off')
    
    # 9-10. Summary statistics
    plt.subplot(4, 4, 13)
    plt.axis('off')
    summary_text = f"""Resolution: {resolution}
    
Summary Statistics (50 timesteps):
• Avg RMSE U: {results['summary']['avg_rmse_u']:.4f}
• Avg RMSE V: {results['summary']['avg_rmse_v']:.4f}
• Avg Corr U: {results['summary']['avg_corr_u']:.4f}
• Avg Corr V: {results['summary']['avg_corr_v']:.4f}

Initial vs Final:
• RMSE U: {results['summary']['initial_rmse_u']:.4f} → {results['summary']['final_rmse_u']:.4f}
• RMSE V: {results['summary']['initial_rmse_v']:.4f} → {results['summary']['final_rmse_v']:.4f}
• Corr U: {results['summary']['initial_corr_u']:.4f} → {results['summary']['final_corr_u']:.4f}
• Corr V: {results['summary']['initial_corr_v']:.4f} → {results['summary']['final_corr_v']:.4f}
"""
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # 11. U-velocity min/max evolution
    plt.subplot(4, 4, 14)
    plt.plot(timesteps, results['field_statistics']['training_u']['min'], 'b--', label='Training Min', linewidth=1)
    plt.plot(timesteps, results['field_statistics']['training_u']['max'], 'b-', label='Training Max', linewidth=1)
    plt.plot(timesteps, results['field_statistics']['pict_u']['min'], 'c--', label='Pict Min', linewidth=1)
    plt.plot(timesteps, results['field_statistics']['pict_u']['max'], 'c-', label='Pict Max', linewidth=1)
    plt.xlabel('Timestep')
    plt.ylabel('U-velocity')
    plt.title('U-velocity Min/Max Evolution')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 12. V-velocity min/max evolution
    plt.subplot(4, 4, 15)
    plt.plot(timesteps, results['field_statistics']['training_v']['min'], 'r--', label='Training Min', linewidth=1)
    plt.plot(timesteps, results['field_statistics']['training_v']['max'], 'r-', label='Training Max', linewidth=1)
    plt.plot(timesteps, results['field_statistics']['pict_v']['min'], 'pink', label='Pict Min', linewidth=1, linestyle='--')
    plt.plot(timesteps, results['field_statistics']['pict_v']['max'], 'pink', label='Pict Max', linewidth=1)
    plt.xlabel('Timestep')
    plt.ylabel('V-velocity')
    plt.title('V-velocity Min/Max Evolution')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 13. Scatter plot correlation at different timesteps
    plt.subplot(4, 4, 16)
    colors = ['blue', 'green', 'orange', 'red']
    sample_t = [0, 15, 30, 49]
    for i, t in enumerate(sample_t):
        plt.scatter(pict_u[t].flatten()[::50], training_u[t].flatten()[::50], 
                   alpha=0.6, s=2, c=colors[i], label=f't={t}')
    plt.xlabel('Pict U-velocity')
    plt.ylabel('Training U-velocity')
    plt.title('U-velocity Correlation\nat Different Timesteps')
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save main plot
    output_file = output_dir / f'first_50_timesteps_comparison_{resolution}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Main visualization saved to: {output_file}")
    plt.close()
    
    # Create detailed velocity field evolution plot
    create_velocity_evolution_plot(training_u, training_v, pict_u, pict_v, resolution, output_dir)

def create_velocity_evolution_plot(training_u, training_v, pict_u, pict_v, resolution, output_dir):
    """Create detailed velocity field evolution plot."""
    
    # Select timesteps to show
    timesteps_to_show = [0, 5, 10, 20, 30, 40, 49]
    
    fig = plt.figure(figsize=(28, 16))
    
    for i, t in enumerate(timesteps_to_show):
        # Training U
        plt.subplot(4, len(timesteps_to_show), i + 1)
        im = plt.imshow(training_u[t], cmap='RdBu_r')
        plt.title(f'Training U\nt={t}')
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # Pict U
        plt.subplot(4, len(timesteps_to_show), len(timesteps_to_show) + i + 1)
        im = plt.imshow(pict_u[t], cmap='RdBu_r')
        plt.title(f'Pict U\nt={t}')
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # Training V
        plt.subplot(4, len(timesteps_to_show), 2 * len(timesteps_to_show) + i + 1)
        im = plt.imshow(training_v[t], cmap='RdBu_r')
        plt.title(f'Training V\nt={t}')
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # Pict V
        plt.subplot(4, len(timesteps_to_show), 3 * len(timesteps_to_show) + i + 1)
        im = plt.imshow(pict_v[t], cmap='RdBu_r')
        plt.title(f'Pict V\nt={t}')
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save evolution plot
    output_file = output_dir / f'velocity_field_evolution_{resolution}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Velocity evolution plot saved to: {output_file}")
    plt.close()

def main():
    """Main function to compare both resolutions."""
    resolutions = ['64x64', '128x128']
    all_results = {}
    
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
            results = compare_first_50_timesteps(training_file, pict_file, resolution)
            all_results[resolution] = results
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = 'first_50_timesteps_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ First 50 timesteps comparison complete!")
    print(f"📄 Results saved to: {output_file}")
    print(f"📊 Visualizations saved to: first_50_timesteps_comparison/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()