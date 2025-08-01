#!/usr/bin/env python3
"""Fixed data comparison analysis with resolved numerical issues."""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_align_data(training_file: str, pred_file: str, sampling_interval: int = 50):
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

def compute_energy_spectrum_fixed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute energy spectrum from velocity fields with numerical stability."""
    # Convert to JAX arrays
    u_jax = jnp.array(u)
    v_jax = jnp.array(v)
    
    # Compute 2D FFT of velocity fields
    u_fft = jnp.fft.fft2(u_jax, axes=(-2, -1))
    v_fft = jnp.fft.fft2(v_jax, axes=(-2, -1))
    
    # Compute energy density (kinetic energy in Fourier space)
    energy_density = 0.5 * (jnp.abs(u_fft)**2 + jnp.abs(v_fft)**2)
    
    # Get spatial dimensions
    nx, ny = u.shape[-2], u.shape[-1]
    
    # Create wavenumber arrays
    kx = jnp.fft.fftfreq(nx, 1.0) * nx
    ky = jnp.fft.fftfreq(ny, 1.0) * ny
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
    k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2)
    
    # Define wavenumber bins for radial averaging
    k_max = int(min(nx, ny) // 2)  # Nyquist limit
    k_bins = jnp.arange(0, k_max + 1)
    
    # Initialize energy spectrum
    energy_spectrum = jnp.zeros((energy_density.shape[0], len(k_bins)))
    
    # Radial averaging
    for i, k in enumerate(k_bins):
        if i < len(k_bins) - 1:
            mask = (k_magnitude >= k) & (k_magnitude < k + 1)
        else:
            mask = k_magnitude >= k
        
        # Sum energy in this wavenumber shell
        if jnp.sum(mask) > 0:
            energy_spectrum = energy_spectrum.at[:, i].set(
                jnp.sum(energy_density * mask[None, :, :], axis=(-2, -1))
            )
    
    return energy_spectrum

def apply_selected_metrics_fixed(data: Dict) -> Dict:
    """Apply the four selected evaluation metrics with fixed numerical issues."""
    results = {}
    
    training_u = data['training_u']
    training_v = data['training_v'] 
    pred_u = data['pred_u']
    pred_v = data['pred_v']
    
    print("Computing evaluation metrics...")
    
    # 1. Energy Spectrum Metric (Fixed)
    print("  1. Computing energy spectrum metric...")
    try:
        training_spectrum = compute_energy_spectrum_fixed(training_u, training_v)
        pred_spectrum = compute_energy_spectrum_fixed(pred_u, pred_v)
        
        # Time-averaged spectra
        training_spectrum_avg = jnp.mean(training_spectrum, axis=0)
        pred_spectrum_avg = jnp.mean(pred_spectrum, axis=0)
        
        # Compute relative log difference with numerical stability
        threshold = 1e-10  # Small threshold to avoid log(0)
        valid_mask = (training_spectrum_avg > threshold) & (pred_spectrum_avg > threshold)
        
        if jnp.sum(valid_mask) > 0:
            log_diff = jnp.abs(jnp.log(pred_spectrum_avg + threshold) - 
                              jnp.log(training_spectrum_avg + threshold))
            energy_metric = jnp.mean(log_diff[valid_mask])
        else:
            energy_metric = jnp.inf
            
        results['energy_spectrum_metric'] = float(energy_metric)
        results['training_spectrum_avg'] = training_spectrum_avg.tolist()
        results['pred_spectrum_avg'] = pred_spectrum_avg.tolist()
        
    except Exception as e:
        print(f"    Error in energy spectrum calculation: {e}")
        results['energy_spectrum_metric'] = float('inf')
        results['energy_spectrum_error'] = str(e)
    
    # 2. Spatial Correlation Metric (u-x correlation)
    print("  2. Computing spatial correlation metric...")
    def compute_spatial_autocorr(field, max_lag=10):
        """Compute spatial autocorrelation along x-axis."""
        autocorr = []
        for lag in range(min(max_lag, field.shape[-1]//4)):
            shifted = jnp.roll(field, lag, axis=-1)
            # Compute correlation coefficient
            corr = jnp.corrcoef(field.flatten(), shifted.flatten())[0, 1]
            autocorr.append(corr)
        return jnp.array(autocorr)
    
    try:
        training_spatial_corr = compute_spatial_autocorr(training_u)
        pred_spatial_corr = compute_spatial_autocorr(pred_u)
        
        # Compute difference in spatial correlations
        spatial_threshold = 0.1  # Lowered threshold
        spatial_diff = jnp.abs(pred_spatial_corr - training_spatial_corr)
        valid_spatial = jnp.abs(training_spatial_corr) > spatial_threshold
        
        if jnp.sum(valid_spatial) > 0:
            spatial_metric = jnp.mean(spatial_diff[valid_spatial])
        else:
            spatial_metric = jnp.mean(spatial_diff)
            
        results['spatial_correlation_metric'] = float(spatial_metric)
        results['training_spatial_corr'] = training_spatial_corr.tolist()
        results['pred_spatial_corr'] = pred_spatial_corr.tolist()
        
    except Exception as e:
        print(f"    Error in spatial correlation calculation: {e}")
        results['spatial_correlation_metric'] = float('inf')
    
    # 3. Temporal Autocorrelation
    print("  3. Computing temporal autocorrelation...")
    def compute_temporal_autocorr(field, max_lag=20):
        """Compute temporal autocorrelation."""
        # Spatially average the field first
        field_spatial_avg = jnp.mean(field, axis=(-2, -1))
        
        autocorr = []
        for lag in range(min(max_lag, len(field_spatial_avg)//4)):
            if lag == 0:
                corr = 1.0  # Perfect correlation at lag 0
            else:
                # Compute valid range to avoid boundary effects
                valid_length = len(field_spatial_avg) - lag
                if valid_length > 1:
                    x1 = field_spatial_avg[:valid_length]
                    x2 = field_spatial_avg[lag:lag+valid_length]
                    corr = jnp.corrcoef(x1, x2)[0, 1]
                else:
                    corr = 0.0
            autocorr.append(corr)
        return jnp.array(autocorr)
    
    try:
        training_temp_corr = compute_temporal_autocorr(training_u)
        pred_temp_corr = compute_temporal_autocorr(pred_u)
        
        # Store temporal autocorrelation results
        results['temporal_autocorr_training'] = training_temp_corr.tolist()
        results['temporal_autocorr_pred'] = pred_temp_corr.tolist()
        
    except Exception as e:
        print(f"    Error in temporal autocorrelation calculation: {e}")
        results['temporal_autocorr_training'] = []
        results['temporal_autocorr_pred'] = []
    
    # 4. Temporal Correlation Metric (Fixed)
    print("  4. Computing temporal correlation metric...")
    try:
        if 'temporal_autocorr_training' in results and 'temporal_autocorr_pred' in results:
            training_temp_corr = jnp.array(results['temporal_autocorr_training'])
            pred_temp_corr = jnp.array(results['temporal_autocorr_pred'])
            
            # Compute difference in temporal correlations
            temporal_threshold = 0.1  # Lowered threshold
            temporal_diff = jnp.abs(pred_temp_corr - training_temp_corr)
            valid_temporal = jnp.abs(training_temp_corr) > temporal_threshold
            
            if jnp.sum(valid_temporal) > 0:
                temporal_metric = jnp.mean(temporal_diff[valid_temporal])
            else:
                temporal_metric = jnp.mean(temporal_diff)
                
            results['temporal_correlation_metric'] = float(temporal_metric)
        else:
            results['temporal_correlation_metric'] = float('inf')
            
    except Exception as e:
        print(f"    Error in temporal correlation metric: {e}")
        results['temporal_correlation_metric'] = float('inf')
    
    # Additional basic statistics
    results['rmse_u'] = float(jnp.sqrt(jnp.mean((training_u - pred_u)**2)))
    results['rmse_v'] = float(jnp.sqrt(jnp.mean((training_v - pred_v)**2)))
    results['mae_u'] = float(jnp.mean(jnp.abs(training_u - pred_u)))
    results['mae_v'] = float(jnp.mean(jnp.abs(training_v - pred_v)))
    
    # Energy statistics
    training_energy = jnp.mean(0.5 * (training_u**2 + training_v**2), axis=(-2, -1))
    pred_energy = jnp.mean(0.5 * (pred_u**2 + pred_v**2), axis=(-2, -1))
    results['training_energy_timeseries'] = training_energy.tolist()
    results['pred_energy_timeseries'] = pred_energy.tolist()
    results['energy_rmse'] = float(jnp.sqrt(jnp.mean((training_energy - pred_energy)**2)))
    
    return results

def create_enhanced_plots(data: Dict, results: Dict, resolution: str, save_dir: str):
    """Create enhanced visualization plots."""
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Energy spectrum comparison
    plt.subplot(4, 3, 1)
    if 'training_spectrum_avg' in results and 'pred_spectrum_avg' in results:
        k_values = np.arange(len(results['training_spectrum_avg']))
        training_spec = np.array(results['training_spectrum_avg'])
        pred_spec = np.array(results['pred_spectrum_avg'])
        
        # Only plot where values are positive
        valid_idx = (training_spec > 0) & (pred_spec > 0)
        if np.sum(valid_idx) > 1:
            plt.loglog(k_values[valid_idx], training_spec[valid_idx], 'b-', label='Training', linewidth=2)
            plt.loglog(k_values[valid_idx], pred_spec[valid_idx], 'r--', label='Predicted', linewidth=2)
            plt.xlabel('Wavenumber k')
            plt.ylabel('Energy E(k)')
            plt.title('Energy Spectrum Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Plot 2: Temporal autocorrelation
    plt.subplot(4, 3, 2)
    if results.get('temporal_autocorr_training') and results.get('temporal_autocorr_pred'):
        training_autocorr = results['temporal_autocorr_training']
        pred_autocorr = results['temporal_autocorr_pred']
        lags = np.arange(len(training_autocorr))
        plt.plot(lags, training_autocorr, 'b-', label='Training', linewidth=2, marker='o')
        plt.plot(lags, pred_autocorr, 'r--', label='Predicted', linewidth=2, marker='s')
        plt.xlabel('Time lag')
        plt.ylabel('Autocorrelation')
        plt.title('Temporal Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Spatial autocorrelation
    plt.subplot(4, 3, 3)
    if results.get('training_spatial_corr') and results.get('pred_spatial_corr'):
        training_spatial = results['training_spatial_corr']
        pred_spatial = results['pred_spatial_corr']
        lags = np.arange(len(training_spatial))
        plt.plot(lags, training_spatial, 'b-', label='Training', linewidth=2, marker='o')
        plt.plot(lags, pred_spatial, 'r--', label='Predicted', linewidth=2, marker='s')
        plt.xlabel('Spatial lag')
        plt.ylabel('Autocorrelation')
        plt.title('Spatial Autocorrelation (U-X)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Energy evolution over time
    plt.subplot(4, 3, 4)
    if results.get('training_energy_timeseries') and results.get('pred_energy_timeseries'):
        time_steps = np.arange(len(results['training_energy_timeseries']))
        plt.plot(time_steps, results['training_energy_timeseries'], 'b-', label='Training', linewidth=2)
        plt.plot(time_steps, results['pred_energy_timeseries'], 'r--', label='Predicted', linewidth=2)
        plt.xlabel('Time step')
        plt.ylabel('Kinetic Energy')
        plt.title('Energy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 5-7: Velocity field snapshots (initial, middle, final)
    time_indices = [0, len(data['training_u'])//2, -1]
    time_labels = ['Initial', 'Middle', 'Final']
    
    for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
        plt.subplot(4, 3, 5 + i)
        diff = data['training_u'][t_idx] - data['pred_u'][t_idx]
        im = plt.imshow(diff, cmap='RdBu_r', origin='lower')
        plt.colorbar(im)
        plt.title(f'U-velocity Difference ({label})')
    
    # Plot 8: Metrics summary
    plt.subplot(4, 3, 8)
    metric_names = ['Energy\nSpectrum', 'Spatial\nCorrelation', 'Temporal\nCorrelation']
    metric_values = [
        results.get('energy_spectrum_metric', float('inf')),
        results.get('spatial_correlation_metric', 0), 
        results.get('temporal_correlation_metric', 0)
    ]
    
    # Handle infinite values for plotting
    plot_values = []
    for val in metric_values:
        if np.isfinite(val):
            plot_values.append(val)
        else:
            plot_values.append(0)  # Will be marked as "inf" in text
    
    bars = plt.bar(metric_names, plot_values, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics Summary')
    
    for bar, val in zip(bars, metric_values):
        if np.isfinite(val):
            text = f'{val:.4f}'
        else:
            text = 'inf'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                text, ha='center', va='bottom')
    
    # Plot 9: Error statistics
    plt.subplot(4, 3, 9)
    error_names = ['RMSE U', 'RMSE V', 'MAE U', 'MAE V']
    error_values = [
        results.get('rmse_u', 0),
        results.get('rmse_v', 0),
        results.get('mae_u', 0),
        results.get('mae_v', 0)
    ]
    bars = plt.bar(error_names, error_values, 
                  color=['lightblue', 'lightblue', 'lightgreen', 'lightgreen'])
    plt.ylabel('Error Value')
    plt.title('Error Statistics')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, error_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 10: Data characteristics comparison
    plt.subplot(4, 3, 10)
    char_names = ['Training\nU std', 'Pred\nU std', 'Training\nV std', 'Pred\nV std']
    char_values = [
        float(jnp.std(data['training_u'])),
        float(jnp.std(data['pred_u'])),
        float(jnp.std(data['training_v'])),
        float(jnp.std(data['pred_v']))
    ]
    bars = plt.bar(char_names, char_values,
                  color=['darkblue', 'blue', 'darkred', 'red'])
    plt.ylabel('Standard Deviation')
    plt.title('Data Characteristics')
    for bar, val in zip(bars, char_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 11: RMSE evolution over time
    plt.subplot(4, 3, 11)
    time_steps = np.arange(data['time_steps'])
    u_errors = np.sqrt(np.mean((data['training_u'] - data['pred_u'])**2, axis=(-2, -1)))
    v_errors = np.sqrt(np.mean((data['training_v'] - data['pred_v'])**2, axis=(-2, -1)))
    plt.plot(time_steps, u_errors, 'b-', label='U RMSE', linewidth=2)
    plt.plot(time_steps, v_errors, 'r-', label='V RMSE', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('RMSE')
    plt.title('Error Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Energy difference
    plt.subplot(4, 3, 12)
    if results.get('training_energy_timeseries') and results.get('pred_energy_timeseries'):
        time_steps = np.arange(len(results['training_energy_timeseries']))
        energy_diff = np.array(results['training_energy_timeseries']) - np.array(results['pred_energy_timeseries'])
        plt.plot(time_steps, energy_diff, 'g-', linewidth=2)
        plt.xlabel('Time step')
        plt.ylabel('Energy Difference')
        plt.title('Energy Difference (Training - Predicted)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Enhanced Data Comparison Analysis - {resolution}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f'enhanced_analysis_{resolution}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced visualization saved to: {save_path}")

def main():
    """Main function to run the fixed data comparison analysis."""
    # Create results directory
    results_dir = "fixed_comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define data files to compare
    resolutions = ['64x64']  # Start with 64x64, can expand later
    
    all_results = {}
    
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}")
        print(f"{'='*60}")
        
        # Define file paths
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        pred_file = f"data/pict_data/decaying_turbulence_{resolution}_index_1.npz"
        
        # Check if files exist
        if not os.path.exists(training_file):
            print(f"Training file not found: {training_file}")
            continue
        if not os.path.exists(pred_file):
            print(f"Predicted file not found: {pred_file}")
            continue
        
        try:
            # Load and align data with correct sampling
            print("Loading and aligning data...")
            data = load_and_align_data(training_file, pred_file, sampling_interval=50)
            
            # Apply evaluation metrics
            print("Applying evaluation metrics...")
            results = apply_selected_metrics_fixed(data)
            
            # Store results
            all_results[resolution] = results
            
            # Create enhanced visualizations
            print("Creating visualizations...")
            create_enhanced_plots(data, results, resolution, results_dir)
            
            # Save detailed results
            results_file = os.path.join(results_dir, f'fixed_results_{resolution}.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print comprehensive summary
            print(f"\n📊 Comprehensive Results Summary for {resolution}:")
            print(f"  Energy Spectrum Metric: {results.get('energy_spectrum_metric', 'N/A'):.6f}")
            print(f"  Spatial Correlation Metric: {results.get('spatial_correlation_metric', 0):.6f}")
            print(f"  Temporal Correlation Metric: {results.get('temporal_correlation_metric', 0):.6f}")
            print(f"  RMSE U: {results.get('rmse_u', 0):.6f}")
            print(f"  RMSE V: {results.get('rmse_v', 0):.6f}")
            print(f"  MAE U: {results.get('mae_u', 0):.6f}")
            print(f"  MAE V: {results.get('mae_v', 0):.6f}")
            print(f"  Energy RMSE: {results.get('energy_rmse', 0):.6f}")
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    combined_results_file = os.path.join(results_dir, 'all_fixed_results.json')
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Fixed analysis complete!")
    print(f"📁 Results saved in: {results_dir}/")
    print(f"📄 Combined results: {combined_results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 