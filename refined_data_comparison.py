#!/usr/bin/env python3
"""Refined data comparison analysis with correct time sampling understanding."""

import os
# Force JAX to use CPU to avoid CUDA/cuDNN issues
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
import json
from pathlib import Path

# Additional JAX configuration for CPU
jax.config.update('jax_platform_name', 'cpu')

def load_and_align_data(training_file: str, pred_file: str, sampling_interval: int = 1):
    """Load and temporally align training and prediction data."""
    print(f"Loading training data: {training_file}")
    training_data = np.load(training_file)
    
    print(f"Loading prediction data: {pred_file}")
    pred_data = np.load(pred_file)
    
    # Extract velocity fields
    training_u = training_data['u']  # Shape: (12200, 64, 64)
    training_v = training_data['v']  # Shape: (12200, 64, 64)
    pred_u = pred_data['u']         # Shape: (244, 64, 64)  
    pred_v = pred_data['v']         # Shape: (244, 64, 64)
    
    # Subsample training data to match prediction sampling
    training_u_subsampled = training_u[::sampling_interval]  # Every 50th step
    training_v_subsampled = training_v[::sampling_interval]
    pred_u = pred_data['u'][:training_u_subsampled.shape[0]]    # Shape: (243, 64, 64) - removed last timestep  
    pred_v = pred_data['v'][:training_v_subsampled.shape[0]] 
    
    print(f"Original training shape: {training_u.shape}")
    print(f"Subsampled training shape: {training_u_subsampled.shape}")
    print(f"Prediction shape: {pred_u.shape}")
    
    # Verify alignment
    assert training_u_subsampled.shape[0] == pred_u.shape[0], f"Time alignment failed: {training_u_subsampled.shape[0]} vs {pred_u.shape[0]}"
    
    return {
        'training_u': training_u_subsampled,
        'training_v': training_v_subsampled,
        'pred_u': pred_u,
        'pred_v': pred_v,
        'resolution': training_data['resolution'],
        'time_steps': training_u_subsampled.shape[0]
    }

def compute_energy_spectrum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute energy spectrum from velocity fields."""
    # Compute 2D FFT of velocity fields
    u_fft = jnp.fft.fft2(u, axes=(-2, -1))
    v_fft = jnp.fft.fft2(v, axes=(-2, -1))
    
    # Compute energy density
    energy_density = (jnp.abs(u_fft)**2 + jnp.abs(v_fft)**2) / 2
    
    # Get spatial dimensions
    nx, ny = u.shape[-2], u.shape[-1]
    kx = jnp.fft.fftfreq(nx, 1.0) * nx  # Scale by nx for proper wavenumber
    ky = jnp.fft.fftfreq(ny, 1.0) * ny  # Scale by ny for proper wavenumber
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
    k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2)
    
    # FIXED: Radial averaging to get 1D spectrum
    k_max = int(jnp.max(k_magnitude)) + 1  # +1 to ensure non-empty range
    k_max = max(k_max, 10)  # Ensure at least 10 bins
    k_bins = jnp.linspace(0, k_max, k_max + 1)
    energy_spectrum = jnp.zeros((energy_density.shape[0], len(k_bins) - 1))
    
    for i in range(len(k_bins) - 1):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
        energy_spectrum = energy_spectrum.at[:, i].set(
            jnp.mean(energy_density * mask[None, :, :], axis=(-2, -1))
        )
    
    return energy_spectrum

def apply_selected_metrics(data: Dict) -> Dict:
    """Apply the four selected evaluation metrics."""
    results = {}
    
    training_u = data['training_u']
    training_v = data['training_v'] 
    pred_u = data['pred_u']
    pred_v = data['pred_v']
    
    print("Computing evaluation metrics...")
    
    # 1. Energy Spectrum Metric
    print("  1. Computing energy spectrum metric...")
    training_spectrum = compute_energy_spectrum(training_u, training_v)
    pred_spectrum = compute_energy_spectrum(pred_u, pred_v)
    
    # Compute log difference where both spectra are above threshold
    threshold = 0.01
    valid_mask = (training_spectrum > threshold) & (pred_spectrum > threshold)
    log_diff = jnp.abs(jnp.log(pred_spectrum) - jnp.log(training_spectrum))
    energy_metric = jnp.mean(jnp.where(valid_mask, log_diff, 0))
    results['energy_spectrum_metric'] = float(energy_metric)
    
    # 2. Spatial Correlation Metric (simplified u-x correlation)
    print("  2. Computing spatial correlation metric...")
    # Compute spatial autocorrelation for u component
    def spatial_autocorr(field):
        # Compute autocorrelation along x-axis
        autocorr = []
        for lag in range(min(10, field.shape[-1]//4)):  # Limited lags
            shifted = jnp.roll(field, lag, axis=-1)
            corr = jnp.mean(field * shifted, axis=(-2, -1))
            autocorr.append(corr)
        return jnp.array(autocorr)
    
    training_spatial_corr = spatial_autocorr(training_u)
    pred_spatial_corr = spatial_autocorr(pred_u)
    
    # Compute difference in spatial correlations
    spatial_threshold = 0.5
    spatial_diff = jnp.abs(pred_spatial_corr - training_spatial_corr)
    spatial_metric = jnp.mean(jnp.where(jnp.abs(training_spatial_corr) > spatial_threshold, 
                                       spatial_diff, 0))
    results['spatial_correlation_metric'] = float(spatial_metric)
    
    # 3. Temporal Autocorrelation
    print("  3. Computing temporal autocorrelation...")
    def temporal_autocorr(field, max_lag=20):
        # Compute temporal autocorrelation
        autocorr = []
        for lag in range(min(max_lag, field.shape[0]//4)):
            if lag < field.shape[0]:
                field_rolled = jnp.roll(field, lag, axis=0)
                # Mask out invalid entries due to rolling
                valid_range = slice(lag, None) if lag > 0 else slice(None)
                corr = jnp.mean(field[valid_range] * field_rolled[valid_range])
                autocorr.append(corr)
        return jnp.array(autocorr)
    
    training_temp_corr = temporal_autocorr(training_u)
    pred_temp_corr = temporal_autocorr(pred_u)
    
    # Store temporal autocorrelation results
    results['temporal_autocorr_training'] = training_temp_corr.tolist()
    results['temporal_autocorr_pred'] = pred_temp_corr.tolist()
    
    # 4. Temporal Correlation Metric
    print("  4. Computing temporal correlation metric...")
    temporal_threshold = 0.5
    temporal_diff = jnp.abs(pred_temp_corr - training_temp_corr)
    temporal_metric = jnp.mean(jnp.where(jnp.abs(training_temp_corr) > temporal_threshold,
                                        temporal_diff, 0))
    results['temporal_correlation_metric'] = float(temporal_metric)
    
    # Additional basic statistics
    results['rmse_u'] = float(jnp.sqrt(jnp.mean((training_u - pred_u)**2)))
    results['rmse_v'] = float(jnp.sqrt(jnp.mean((training_v - pred_v)**2)))
    results['mae_u'] = float(jnp.mean(jnp.abs(training_u - pred_u)))
    results['mae_v'] = float(jnp.mean(jnp.abs(training_v - pred_v)))
    
    return results

def create_comprehensive_plots(data: Dict, results: Dict, resolution: str, save_dir: str):
    """Create comprehensive visualization plots."""
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Energy spectrum comparison
    plt.subplot(3, 3, 1)
    training_spectrum = compute_energy_spectrum(data['training_u'], data['training_v'])
    pred_spectrum = compute_energy_spectrum(data['pred_u'], data['pred_v'])
    
    k_values = jnp.arange(training_spectrum.shape[1])
    plt.loglog(k_values[1:], jnp.mean(training_spectrum, axis=0)[1:], 'b-', label='Training', linewidth=2)
    plt.loglog(k_values[1:], jnp.mean(pred_spectrum, axis=0)[1:], 'r--', label='Predicted', linewidth=2)
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy E(k)')
    plt.title('Energy Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Temporal autocorrelation
    plt.subplot(3, 3, 2)
    if 'temporal_autocorr_training' in results:
        lags = jnp.arange(len(results['temporal_autocorr_training']))
        plt.plot(lags, results['temporal_autocorr_training'], 'b-', label='Training', linewidth=2)
        plt.plot(lags, results['temporal_autocorr_pred'], 'r--', label='Predicted', linewidth=2)
        plt.xlabel('Time lag')
        plt.ylabel('Autocorrelation')
        plt.title('Temporal Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Metrics summary
    plt.subplot(3, 3, 3)
    metric_names = ['Energy Spectrum', 'Spatial Corr', 'Temporal Corr']
    metric_values = [
        results.get('energy_spectrum_metric', 0),
        results.get('spatial_correlation_metric', 0), 
        results.get('temporal_correlation_metric', 0)
    ]
    bars = plt.bar(metric_names, metric_values, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics Summary')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 4-6: Velocity field snapshots (first time step)
    plt.subplot(3, 3, 4)
    plt.imshow(data['training_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.title('Training U-velocity (t=0)')
    
    plt.subplot(3, 3, 5)
    plt.imshow(data['pred_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar() 
    plt.title('Predicted U-velocity (t=0)')
    
    plt.subplot(3, 3, 6)
    plt.imshow(data['training_u'][0] - data['pred_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.title('U-velocity Difference (t=0)')
    
    # Plot 7-9: Error statistics over time
    plt.subplot(3, 3, 7)
    time_steps = jnp.arange(data['time_steps'])
    u_errors = jnp.sqrt(jnp.mean((data['training_u'] - data['pred_u'])**2, axis=(-2, -1)))
    v_errors = jnp.sqrt(jnp.mean((data['training_v'] - data['pred_v'])**2, axis=(-2, -1)))
    plt.plot(time_steps, u_errors, 'b-', label='U RMSE', linewidth=2)
    plt.plot(time_steps, v_errors, 'r-', label='V RMSE', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('RMSE')
    plt.title('Error Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Statistics comparison
    plt.subplot(3, 3, 8)
    stats_data = {
        'RMSE U': results.get('rmse_u', 0),
        'RMSE V': results.get('rmse_v', 0),
        'MAE U': results.get('mae_u', 0),
        'MAE V': results.get('mae_v', 0)
    }
    bars = plt.bar(stats_data.keys(), stats_data.values(), 
                  color=['lightblue', 'lightblue', 'lightgreen', 'lightgreen'])
    plt.ylabel('Error Value')
    plt.title('Error Statistics')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, stats_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 9: Data characteristics
    plt.subplot(3, 3, 9)
    char_data = {
        'Training U std': float(jnp.std(data['training_u'])),
        'Pred U std': float(jnp.std(data['pred_u'])),
        'Training V std': float(jnp.std(data['training_v'])),
        'Pred V std': float(jnp.std(data['pred_v']))
    }
    bars = plt.bar(char_data.keys(), char_data.values(),
                  color=['darkblue', 'blue', 'darkred', 'red'])
    plt.ylabel('Standard Deviation')
    plt.title('Data Characteristics')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, char_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Comprehensive Data Comparison Analysis - {resolution}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f'comprehensive_analysis_{resolution}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to: {save_path}")

def main():
    """Main function to run the refined data comparison analysis."""
    # Create results directory
    results_dir = "refined_comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define data files to compare (start with 64x64 for testing)
    resolutions = ['128x128']  # Can expand to ['64x64', '128x128', '256x256', '512x512']
    
    all_results = {}
    
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}")
        print(f"{'='*60}")
        
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        pred_file = f"data/pict_data/decaying_turbulence_{resolution}_index_1.npz"
        
        if not os.path.exists(training_file):
            print(f"Training file not found: {training_file}")
            continue
        if not os.path.exists(pred_file):
            print(f"Predicted file not found: {pred_file}")
            continue
        
        try:
            print("Loading and aligning data...")
            data = load_and_align_data(training_file, pred_file)
            
            print("Applying evaluation metrics...")
            results = apply_selected_metrics(data)
            
            all_results[resolution] = results
            
            print("Creating visualizations...")
            create_comprehensive_plots(data, results, resolution, results_dir)
            
            results_file = os.path.join(results_dir, f'detailed_results_{resolution}.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📊 Results Summary for {resolution}:")
            print(f"  Energy Spectrum Metric: {results.get('energy_spectrum_metric', 0):.6f}")
            print(f"  Spatial Correlation Metric: {results.get('spatial_correlation_metric', 0):.6f}")
            print(f"  Temporal Correlation Metric: {results.get('temporal_correlation_metric', 0):.6f}")
            print(f"  RMSE U: {results.get('rmse_u', 0):.6f}")
            print(f"  RMSE V: {results.get('rmse_v', 0):.6f}")
            print(f"  MAE U: {results.get('mae_u', 0):.6f}")
            print(f"  MAE V: {results.get('mae_v', 0):.6f}")
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    combined_results_file = os.path.join(results_dir, 'all_refined_results.json')
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Refined analysis complete!")
    print(f"📁 Results saved in: {results_dir}/")
    print(f"📄 Combined results: {combined_results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 